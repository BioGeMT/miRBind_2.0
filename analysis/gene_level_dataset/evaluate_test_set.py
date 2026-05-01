"""
Evaluation of miRBind V4 vs TargetScan weighted context++ on the test set
(ts_considered_all_mirs).

Computes:
  - Correlation & regression: Pearson, Spearman, R², MSE, MAE
  - Classification: ROC-AUC, Average Precision (AP)
  - Head-to-head: Williams' test, DeLong's test, bootstrap 95% CIs
  - Per-miRNA breakdown with all of the above

Usage:
    python evaluate_test_set.py \
        --predictions test_results_gene_level_V4_pretrained/ts_considered_all_mirs \
        --truth_col "actual_fold_change" \
        --model_col "predicted_V4 Model - 20260113_164412" \
        --competitor_col "predicted_TargetScan Weighted Context++ Score" \
        --competitor_name "TS Weighted Context++" \
        --binary_threshold -0.05 \
        --output report.txt
"""

import argparse
import glob
import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# Statistical helpers
# =============================================================================

def pearson(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return stats.pearsonr(x, y)[0]


def spearman(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return stats.spearmanr(x, y)[0]


def r_squared(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 2:
        return np.nan
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def mse(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return np.mean((y_true[mask] - y_pred[mask]) ** 2)


def mae(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return np.mean(np.abs(y_true[mask] - y_pred[mask]))


def win_rate(y_true, model_pred, comp_pred):
    """Fraction of pairs where |model - true| < |comp - true|."""
    mask = np.isfinite(y_true) & np.isfinite(model_pred) & np.isfinite(comp_pred)
    y_true, model_pred, comp_pred = y_true[mask], model_pred[mask], comp_pred[mask]
    model_err = np.abs(y_true - model_pred)
    comp_err = np.abs(y_true - comp_pred)
    wins = np.sum(model_err < comp_err)
    ties = np.sum(model_err == comp_err)
    n = len(y_true)
    return (wins + 0.5 * ties) / n if n > 0 else np.nan


def bootstrap_corr_diff(y_true, model_pred, comp_pred, corr_func, n_boot=10000, seed=42):
    """Bootstrap 95% CI for the difference in correlation (model - competitor)."""
    mask = np.isfinite(y_true) & np.isfinite(model_pred) & np.isfinite(comp_pred)
    y_true, model_pred, comp_pred = y_true[mask], model_pred[mask], comp_pred[mask]
    n = len(y_true)
    if n < 10:
        return np.nan, np.nan, np.nan

    observed_diff = corr_func(y_true, model_pred) - corr_func(y_true, comp_pred)

    rng = np.random.RandomState(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        diffs[i] = corr_func(y_true[idx], model_pred[idx]) - corr_func(y_true[idx], comp_pred[idx])

    return observed_diff, np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)


def williams_test(y_true, pred_a, pred_b):
    """
    Williams' test for comparing two dependent Pearson correlations.
    Tests H0: r(y,a) == r(y,b) when a and b are correlated.
    Returns (t_stat, p_value, n).
    """
    mask = np.isfinite(y_true) & np.isfinite(pred_a) & np.isfinite(pred_b)
    y_true, pred_a, pred_b = y_true[mask], pred_a[mask], pred_b[mask]
    n = len(y_true)
    if n < 10:
        return np.nan, np.nan, n

    r_ya = stats.pearsonr(y_true, pred_a)[0]
    r_yb = stats.pearsonr(y_true, pred_b)[0]
    r_ab = stats.pearsonr(pred_a, pred_b)[0]

    r_bar = (r_ya + r_yb) / 2
    R = 1 - r_ya**2 - r_yb**2 - r_ab**2 + 2 * r_ya * r_yb * r_ab
    denom = (2 * (1 - r_ab) * R / (n - 1)) + ((r_bar**2) * (1 - r_ab)**3)
    if denom <= 0:
        return np.nan, np.nan, n
    t_stat = (r_ya - r_yb) * np.sqrt((n - 1) * (1 + r_ab) / denom)
    p_val = 2 * stats.t.sf(np.abs(t_stat), df=n - 3)

    return t_stat, p_val, n


# =============================================================================
# Classification metrics
# =============================================================================

def compute_binary_metrics(y_true_cont, y_pred_cont, threshold):
    """ROC-AUC and Average Precision for binarised downregulation."""
    from sklearn.metrics import roc_auc_score, average_precision_score

    mask = np.isfinite(y_true_cont) & np.isfinite(y_pred_cont)
    y_true_cont, y_pred_cont = y_true_cont[mask], y_pred_cont[mask]

    if len(y_true_cont) < 10:
        return {"roc_auc": np.nan, "ap": np.nan, "n_pos": 0, "n_neg": 0, "n": 0}

    y_binary = (y_true_cont < threshold).astype(int)
    n_pos = y_binary.sum()
    n_neg = len(y_binary) - n_pos

    if n_pos == 0 or n_neg == 0:
        return {"roc_auc": np.nan, "ap": np.nan, "n_pos": n_pos, "n_neg": n_neg, "n": len(y_binary)}

    scores = -y_pred_cont
    return {
        "roc_auc": roc_auc_score(y_binary, scores),
        "ap": average_precision_score(y_binary, scores),
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
        "n": len(y_binary),
    }


def delong_test(y_true_cont, pred_a, pred_b, threshold):
    """DeLong's test for comparing two ROC-AUCs. Returns (auc_a, auc_b, z, p)."""
    from sklearn.metrics import roc_auc_score

    mask = np.isfinite(y_true_cont) & np.isfinite(pred_a) & np.isfinite(pred_b)
    y_true_cont, pred_a, pred_b = y_true_cont[mask], pred_a[mask], pred_b[mask]

    y_binary = (y_true_cont < threshold).astype(int)
    n_pos = y_binary.sum()
    n_neg = len(y_binary) - n_pos

    if n_pos < 5 or n_neg < 5:
        return np.nan, np.nan, np.nan, np.nan

    scores_a = -pred_a
    scores_b = -pred_b

    auc_a = roc_auc_score(y_binary, scores_a)
    auc_b = roc_auc_score(y_binary, scores_b)

    pos_scores_a = scores_a[y_binary == 1]
    neg_scores_a = scores_a[y_binary == 0]
    pos_scores_b = scores_b[y_binary == 1]
    neg_scores_b = scores_b[y_binary == 0]

    v_a_pos = np.array([np.mean(pa > neg_scores_a) + 0.5 * np.mean(pa == neg_scores_a) for pa in pos_scores_a])
    v_a_neg = np.array([np.mean(pos_scores_a > na) + 0.5 * np.mean(pos_scores_a == na) for na in neg_scores_a])
    v_b_pos = np.array([np.mean(pb > neg_scores_b) + 0.5 * np.mean(pb == neg_scores_b) for pb in pos_scores_b])
    v_b_neg = np.array([np.mean(pos_scores_b > nb) + 0.5 * np.mean(pos_scores_b == nb) for nb in neg_scores_b])

    s_pos = np.cov(v_a_pos, v_b_pos) if len(v_a_pos) > 1 else np.zeros((2, 2))
    s_neg = np.cov(v_a_neg, v_b_neg) if len(v_a_neg) > 1 else np.zeros((2, 2))

    S = s_pos / n_pos + s_neg / n_neg
    var_diff = S[0, 0] + S[1, 1] - 2 * S[0, 1]

    if var_diff <= 0:
        return auc_a, auc_b, np.nan, np.nan

    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p_val = 2 * stats.norm.sf(np.abs(z))

    return auc_a, auc_b, z, p_val


def bootstrap_binary_diff(y_true_cont, pred_a, pred_b, threshold, n_boot=10000, seed=42):
    """Bootstrap 95% CIs for ROC-AUC and AP difference (model - competitor)."""
    from sklearn.metrics import roc_auc_score, average_precision_score

    mask = np.isfinite(y_true_cont) & np.isfinite(pred_a) & np.isfinite(pred_b)
    y_true_cont, pred_a, pred_b = y_true_cont[mask], pred_a[mask], pred_b[mask]

    n = len(y_true_cont)
    y_binary = (y_true_cont < threshold).astype(int)
    n_pos = y_binary.sum()
    n_neg = n - n_pos

    nan_result = {
        "roc_diff": np.nan, "roc_ci": (np.nan, np.nan),
        "ap_diff": np.nan, "ap_ci": (np.nan, np.nan),
    }

    if n < 10 or n_pos < 5 or n_neg < 5:
        return nan_result

    scores_a = -pred_a
    scores_b = -pred_b

    obs_roc_a = roc_auc_score(y_binary, scores_a)
    obs_roc_b = roc_auc_score(y_binary, scores_b)
    obs_ap_a = average_precision_score(y_binary, scores_a)
    obs_ap_b = average_precision_score(y_binary, scores_b)

    rng = np.random.RandomState(seed)
    roc_diffs = np.empty(n_boot)
    ap_diffs = np.empty(n_boot)

    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yb = y_binary[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            roc_diffs[i] = np.nan
            ap_diffs[i] = np.nan
            continue
        sa, sb = scores_a[idx], scores_b[idx]
        roc_diffs[i] = roc_auc_score(yb, sa) - roc_auc_score(yb, sb)
        ap_diffs[i] = average_precision_score(yb, sa) - average_precision_score(yb, sb)

    return {
        "roc_diff": obs_roc_a - obs_roc_b,
        "roc_ci": (np.nanpercentile(roc_diffs, 2.5), np.nanpercentile(roc_diffs, 97.5)),
        "ap_diff": obs_ap_a - obs_ap_b,
        "ap_ci": (np.nanpercentile(ap_diffs, 2.5), np.nanpercentile(ap_diffs, 97.5)),
    }


# =============================================================================
# Formatting helpers
# =============================================================================

def fmt_ci(diff, ci):
    if np.isnan(diff):
        return "N/A"
    return f"{diff:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]"


def fmt_p(p):
    if np.isnan(p):
        return "N/A"
    if p == 0:
        return "≈ 0"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


# =============================================================================
# Report generation
# =============================================================================

def find_prediction_file(pred_dir):
    """Find the predictions CSV in a directory."""
    for pattern in ["predictions_fillzero_*.csv", "predictions_*.csv", "*.csv"]:
        matches = glob.glob(os.path.join(pred_dir, pattern))
        if matches:
            return sorted(matches)[-1]
    return None


class TeeWriter:
    """Write simultaneously to stdout and a file."""
    def __init__(self, filepath):
        self.file = open(filepath, "w")
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate miRBind V4 vs TargetScan on the test set")
    parser.add_argument("--predictions", required=True,
                        help="Directory containing the prediction CSV (ts_considered_all_mirs)")
    parser.add_argument("--truth_col", required=True,
                        help="Column name for ground truth log2FC")
    parser.add_argument("--model_col", required=True,
                        help="Column name for model predictions")
    parser.add_argument("--competitor_col", required=True,
                        help="Column name for competitor predictions")
    parser.add_argument("--competitor_name", default="TS Weighted Context++",
                        help="Display name for competitor")
    parser.add_argument("--mirna_col", default="miRNA",
                        help="Column name for miRNA identifier")
    parser.add_argument("--binary_threshold", type=float, default=-0.05,
                        help="log2FC threshold for binarisation (default: -0.05)")
    parser.add_argument("--n_bootstrap", type=int, default=10000,
                        help="Number of bootstrap resamples (default: 10,000)")
    parser.add_argument("--output", default="evaluation_report.txt",
                        help="Output path for the report text file")
    args = parser.parse_args()

    # Set up tee: print to stdout and save to file
    tee = TeeWriter(args.output)
    sys.stdout = tee

    # =========================================================================
    # Load data
    # =========================================================================
    pred_file = find_prediction_file(args.predictions)
    if pred_file is None:
        print(f"ERROR: No prediction file found in {args.predictions}")
        sys.exit(1)

    df = pd.read_csv(pred_file)
    model_name = args.model_col.replace("predicted_", "")
    comp_name = args.competitor_name

    for col in [args.truth_col, args.model_col, args.competitor_col]:
        if col not in df.columns:
            print(f"ERROR: Column '{col}' not found. Available: {list(df.columns)}")
            sys.exit(1)

    y = df[args.truth_col].values.astype(float)
    m = df[args.model_col].values.astype(float)
    c = df[args.competitor_col].values.astype(float)

    n_total = len(df)
    n_nonzero_comp = np.sum(c != 0)

    print("=" * 100)
    print("EVALUATION REPORT: miRBind V4 vs TargetScan Weighted Context++")
    print("=" * 100)
    print(f"  Prediction file:   {pred_file}")
    print(f"  Total pairs:       {n_total:,}")
    print(f"  Pairs with non-zero TS score: {n_nonzero_comp:,} ({100*n_nonzero_comp/n_total:.1f}%)")
    if args.mirna_col in df.columns:
        mirnas = sorted(df[args.mirna_col].unique())
        n_genes = df.groupby(args.mirna_col).size().values
        print(f"  miRNAs:            {len(mirnas)} ({', '.join(mirnas)})")

    # =========================================================================
    # SECTION 1: Correlation & regression metrics
    # =========================================================================
    print(f"\n{'=' * 100}")
    print("SECTION 1: CORRELATION & REGRESSION METRICS")
    print("=" * 100)

    header = f"  {'Method':<35s}  {'N':>6s}  {'Pearson':>8s}  {'Spearman':>8s}  {'R²':>8s}  {'MSE':>8s}  {'MAE':>8s}"
    print(header)
    print(f"  {'-' * 90}")

    for col, name in [(args.model_col, model_name), (args.competitor_col, comp_name)]:
        p = df[col].values.astype(float)
        n = int(np.sum(np.isfinite(y) & np.isfinite(p)))
        print(f"  {name:<35s}  {n:>6,}  {pearson(y, p):>8.4f}  {spearman(y, p):>8.4f}  "
              f"{r_squared(y, p):>8.4f}  {mse(y, p):>8.4f}  {mae(y, p):>8.4f}")

    # =========================================================================
    # SECTION 2: Classification metrics
    # =========================================================================
    print(f"\n{'=' * 100}")
    print(f"SECTION 2: CLASSIFICATION METRICS (threshold: log2FC < {args.binary_threshold})")
    print("=" * 100)

    n_pos = int(np.sum(y < args.binary_threshold))
    n_neg = int(np.sum(y >= args.binary_threshold))
    print(f"  Positives (downregulated): {n_pos:,}  Negatives: {n_neg:,}  "
          f"Prevalence: {n_pos / max(1, n_pos + n_neg):.1%}")

    print(f"\n  {'Method':<35s}  {'ROC-AUC':>8s}  {'AP':>8s}")
    print(f"  {'-' * 55}")

    for col, name in [(args.model_col, model_name), (args.competitor_col, comp_name)]:
        bm = compute_binary_metrics(y, df[col].values.astype(float), args.binary_threshold)
        print(f"  {name:<35s}  {bm['roc_auc']:>8.4f}  {bm['ap']:>8.4f}")

    # =========================================================================
    # SECTION 3: Statistical comparison
    # =========================================================================
    print(f"\n{'=' * 100}")
    print(f"SECTION 3: STATISTICAL COMPARISON ({model_name} vs {comp_name})")
    print("=" * 100)

    # Win rate
    wr = win_rate(y, m, c)
    print(f"  Win rate:                {wr:.4f}")

    # Williams' test
    w_t, w_p, w_n = williams_test(y, m, c)
    print(f"  Williams' test:          t={w_t:.4f}  p={fmt_p(w_p)}  (N={w_n:,})")

    # Bootstrap CIs — correlation
    pear_diff, pear_lo, pear_hi = bootstrap_corr_diff(y, m, c, pearson, n_boot=args.n_bootstrap)
    spear_diff, spear_lo, spear_hi = bootstrap_corr_diff(y, m, c, spearman, n_boot=args.n_bootstrap)
    print(f"  Pearson diff (M - C):    {fmt_ci(pear_diff, (pear_lo, pear_hi))}")
    print(f"  Spearman diff (M - C):   {fmt_ci(spear_diff, (spear_lo, spear_hi))}")

    # DeLong's test
    dl_auc_m, dl_auc_c, dl_z, dl_p = delong_test(y, m, c, args.binary_threshold)
    if not np.isnan(dl_auc_m):
        print(f"  DeLong's test:           AUC_model={dl_auc_m:.4f}  AUC_comp={dl_auc_c:.4f}  "
              f"z={dl_z:.4f}  p={fmt_p(dl_p)}")
    else:
        print(f"  DeLong's test:           N/A (insufficient class balance)")

    # Bootstrap CIs — ROC-AUC and AP
    bin_boot = bootstrap_binary_diff(y, m, c, args.binary_threshold, n_boot=args.n_bootstrap)
    print(f"  ROC-AUC diff (M - C):    {fmt_ci(bin_boot['roc_diff'], bin_boot['roc_ci'])}")
    print(f"  AP diff (M - C):         {fmt_ci(bin_boot['ap_diff'], bin_boot['ap_ci'])}")

    # =========================================================================
    # SECTION 4: Per-miRNA breakdown
    # =========================================================================
    if args.mirna_col in df.columns:
        print(f"\n{'=' * 100}")
        print("SECTION 4: PER-miRNA BREAKDOWN")
        print("=" * 100)

        mirnas = sorted(df[args.mirna_col].unique())

        # 4a. Basic + binary metrics per miRNA
        print(f"\n  {'miRNA':<22s}  {'Method':<25s}  {'N':>6s}  {'Pearson':>8s}  "
              f"{'Spearman':>8s}  {'ROC-AUC':>8s}  {'AP':>8s}")
        print(f"  {'-' * 110}")

        for mirna in mirnas:
            sub = df[df[args.mirna_col] == mirna]
            ys = sub[args.truth_col].values.astype(float)
            for col, name in [(args.model_col, model_name), (args.competitor_col, comp_name)]:
                ps = sub[col].values.astype(float)
                bm = compute_binary_metrics(ys, ps, args.binary_threshold)
                n = int(np.sum(np.isfinite(ys) & np.isfinite(ps)))
                print(f"  {mirna:<22s}  {name:<25s}  {n:>6,}  {pearson(ys, ps):>8.4f}  "
                      f"{spearman(ys, ps):>8.4f}  {bm['roc_auc']:>8.4f}  {bm['ap']:>8.4f}")
            print()

        # 4b. Williams' test per miRNA
        print(f"\n  --- Williams' test per miRNA: {model_name} vs {comp_name} ---")
        print(f"  {'miRNA':<22s}  {'Pear_M':>8s}  {'Pear_C':>8s}  {'Diff':>8s}  "
              f"{'Williams_t':>10s}  {'p-value':>12s}  {'Win rate':>8s}")
        print(f"  {'-' * 95}")

        for mirna in mirnas:
            sub = df[df[args.mirna_col] == mirna]
            ys = sub[args.truth_col].values.astype(float)
            ms = sub[args.model_col].values.astype(float)
            cs = sub[args.competitor_col].values.astype(float)

            r_m = pearson(ys, ms)
            r_c = pearson(ys, cs)
            w_t, w_p, _ = williams_test(ys, ms, cs)
            wr = win_rate(ys, ms, cs)

            print(f"  {mirna:<22s}  {r_m:>8.4f}  {r_c:>8.4f}  {r_m - r_c:>+8.4f}  "
                  f"{w_t:>10.4f}  {fmt_p(w_p):>12s}  {wr:>8.4f}")

        # 4c. DeLong's test per miRNA
        print(f"\n  --- DeLong's test per miRNA: {model_name} vs {comp_name} ---")
        print(f"  {'miRNA':<22s}  {'AUC_M':>8s}  {'AUC_C':>8s}  {'Diff':>8s}  "
              f"{'z':>8s}  {'p-value':>12s}")
        print(f"  {'-' * 75}")

        for mirna in mirnas:
            sub = df[df[args.mirna_col] == mirna]
            ys = sub[args.truth_col].values.astype(float)
            ms = sub[args.model_col].values.astype(float)
            cs = sub[args.competitor_col].values.astype(float)

            dl_m, dl_c, dl_z, dl_p = delong_test(ys, ms, cs, args.binary_threshold)
            if not np.isnan(dl_m):
                print(f"  {mirna:<22s}  {dl_m:>8.4f}  {dl_c:>8.4f}  {dl_m - dl_c:>+8.4f}  "
                      f"{dl_z:>8.4f}  {fmt_p(dl_p):>12s}")
            else:
                print(f"  {mirna:<22s}  {'N/A':>8s}  {'N/A':>8s}  {'N/A':>8s}  "
                      f"{'N/A':>8s}  {'N/A':>12s}")

    # =========================================================================
    # Done
    # =========================================================================
    print(f"\n{'=' * 100}")
    print(f"Report saved to: {args.output}")
    print("=" * 100)

    # Restore stdout and close file
    sys.stdout = tee.stdout
    tee.close()
    print(f"Report saved to: {args.output}")


if __name__ == "__main__":
    main()
