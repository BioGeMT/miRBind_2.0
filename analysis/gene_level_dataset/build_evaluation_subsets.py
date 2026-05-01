"""
Build evaluation subsets from the test set for TargetScan baseline comparison.

1. All genes:              All pairs (TS=0 for genes without predictions)
2. TS-considered (7 miRs): Only genes where TS has a prediction for at least one of the 7 test miRNAs
3. TS-considered (all miRs): Only genes that appear in TS predictions for ANY human miRNA
4. With sites:             Only (gene, miRNA) pairs where TS has a non-zero prediction

Computes r², Pearson, and Spearman for each subset using both context++ and
weighted context++ scores.

Usage:
    python build_evaluation_subsets.py \
        --test-set data/TS8/processed/test_set_long.tsv \
        --ts-genes data/TS8/processed/test_set_ts_all_genes.txt \
        --output data/TS8/processed/eval
"""

import argparse
import pandas as pd
import numpy as np
from scipy import stats


def compute_metrics(y_true, y_pred, label=""):
    """Compute r², Pearson, Spearman between two arrays."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    n = len(y_true)

    if n < 3:
        return {"label": label, "N": n, "r2": np.nan, "pearson": np.nan, "spearman": np.nan}

    # Handle constant arrays (e.g. all predictions = 0)
    if np.std(y_pred) == 0 or np.std(y_true) == 0:
        return {"label": label, "N": n, "r2": np.nan, "pearson": np.nan, "spearman": np.nan}

    pearson_r, _ = stats.pearsonr(y_true, y_pred)
    spearman_r, _ = stats.spearmanr(y_true, y_pred)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {"label": label, "N": n, "r2": r2, "pearson": pearson_r, "spearman": spearman_r}


def print_metrics(m):
    print(f"  {m['label']:<45s}  N={m['N']:>6,}  r²={m['r2']:>8.4f}  "
          f"Pearson={m['pearson']:>7.4f}  Spearman={m['spearman']:>7.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Build evaluation subsets for TargetScan comparison"
    )
    parser.add_argument("--test-set", required=True,
                        help="test_set_long.tsv from build_test_set.py")
    parser.add_argument("--ts-genes", default=None,
                        help="File with all human gene symbols TS has modeled (all miRNAs), "
                             "one per line. Output by build_test_set.py as *_ts_all_genes.txt")
    parser.add_argument("--output", default="eval",
                        help="Output prefix for subset TSV files")
    args = parser.parse_args()

    # =========================================================================
    # Load test set
    # =========================================================================
    print("Loading test set ...")
    df = pd.read_csv(args.test_set, sep="\t")
    print(f"  {len(df):,} rows, {df['Gene_Symbol'].nunique():,} genes")
    print(f"  Columns: {list(df.columns)}")

    assert "TS_context_pp" in df.columns, "Missing TS_context_pp column"
    assert "TS_weighted_context_pp" in df.columns, "Missing TS_weighted_context_pp column"

    # =========================================================================
    # Build subsets
    # =========================================================================

    # 1. ALL GENES — everything as-is
    df_all = df.copy()

    # 2. TS-CONSIDERED (7 miRNAs) — genes where TS has a non-zero context++ for at least one of our 7 test miRNAs
    genes_with_ts_7 = set(
        df.loc[df["TS_context_pp"] != 0, "Gene_Symbol"].unique()
    )
    df_ts_considered_7 = df[df["Gene_Symbol"].isin(genes_with_ts_7)].copy()

    # 3. TS-CONSIDERED (all miRNAs) — genes that appear in TS predictions for any human miRNA
    df_ts_considered_all = None
    if args.ts_genes is not None:
        with open(args.ts_genes, "r") as f:
            all_ts_genes = set(line.strip() for line in f if line.strip())
        genes_in_test = set(df["Gene_Symbol"].unique())
        overlap = genes_in_test & all_ts_genes
        print(f"\n  TS all-miRNA gene set: {len(all_ts_genes):,} genes")
        print(f"  Test set genes:        {len(genes_in_test):,}")
        print(f"  Overlap:               {len(overlap):,}")
        print(f"  Test-only:             {len(genes_in_test - all_ts_genes):,}")
        df_ts_considered_all = df[df["Gene_Symbol"].isin(overlap)].copy()

    # 4. WITH SITES — only pairs with non-zero context++ score
    df_with_sites_ctx = df[df["TS_context_pp"] != 0].copy()
    df_with_sites_wgt = df[df["TS_weighted_context_pp"] != 0].copy()

    # =========================================================================
    # Save subsets
    # =========================================================================
    subsets = {
        "all_genes": df_all,
        "ts_considered_7mirs": df_ts_considered_7,
        "with_sites_ctx": df_with_sites_ctx,
        "with_sites_wgt": df_with_sites_wgt,
    }
    if df_ts_considered_all is not None:
        subsets["ts_considered_all_mirs"] = df_ts_considered_all

    for name, subset in subsets.items():
        path = f"{args.output}_{name}.tsv"
        subset.to_csv(path, sep="\t", index=False)
        n_ctx = (subset["TS_context_pp"] != 0).sum()
        n_wgt = (subset["TS_weighted_context_pp"] != 0).sum()
        print(f"\n  Saved {name}: {path}")
        print(f"    {len(subset):,} rows, {subset['Gene_Symbol'].nunique():,} genes")
        print(f"    Non-zero ctx++: {n_ctx:,}  Non-zero weighted: {n_wgt:,}")

    # =========================================================================
    # Compute correlations
    # =========================================================================
    print("\n" + "=" * 100)
    print("EVALUATION: TargetScan context++ score vs log2fc")
    print("=" * 100)

    fc = "log2fc"

    eval_subsets = [
        ("All genes", df_all),
        ("TS-considered (7 miRNAs)", df_ts_considered_7),
        ("TS-considered (all miRNAs)", df_ts_considered_all),
        ("With sites (ctx++)", df_with_sites_ctx),
        ("With sites (weighted)", df_with_sites_wgt),
    ]

    for name, subset in eval_subsets:
        if subset is None:
            continue
        print(f"\n--- {name} ---")
        m_ctx = compute_metrics(subset[fc].values, subset["TS_context_pp"].values,
                                f"context++ score")
        m_wgt = compute_metrics(subset[fc].values, subset["TS_weighted_context_pp"].values,
                                f"weighted context++ score")
        print_metrics(m_ctx)
        print_metrics(m_wgt)

    # =========================================================================
    # Per-miRNA breakdown for each subset
    # =========================================================================
    print("\n" + "=" * 100)
    print("PER-miRNA BREAKDOWN")
    print("=" * 100)

    mirnas = sorted(df["miRNA"].unique())

    for name, subset in [("All genes", df_all),
                          ("TS-considered (7 miRNAs)", df_ts_considered_7),
                          ("TS-considered (all miRNAs)", df_ts_considered_all),
                          ("With sites (ctx++)", df_with_sites_ctx),
                          ("With sites (weighted)", df_with_sites_wgt)]:
        if subset is None:
            continue
        print(f"\n--- {name} ---")
        print(f"  {'miRNA':<25s}  {'N':>6s}  {'r²_ctx':>8s}  {'Pear_ctx':>8s}  {'Spear_ctx':>9s}  "
              f"{'r²_wgt':>8s}  {'Pear_wgt':>8s}  {'Spear_wgt':>9s}")
        print(f"  {'-'*100}")
        for m in mirnas:
            sub_m = subset[subset["miRNA"] == m]
            if len(sub_m) == 0:
                continue
            mc = compute_metrics(sub_m[fc].values, sub_m["TS_context_pp"].values, m)
            mw = compute_metrics(sub_m[fc].values, sub_m["TS_weighted_context_pp"].values, m)
            print(f"  {m:<25s}  {mc['N']:>6,}  {mc['r2']:>8.4f}  {mc['pearson']:>8.4f}  {mc['spearman']:>9.4f}  "
                  f"{mw['r2']:>8.4f}  {mw['pearson']:>8.4f}  {mw['spearman']:>9.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()