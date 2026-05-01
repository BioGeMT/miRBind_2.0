# Reproducing the miRBind2-3UTR functional evaluation

This directory contains the scripts used to build the 50,549-pair test set
described in the miRBind2 manuscript and to evaluate miRBind2-3UTR against
the TargetScan 8.0 weighted context++ baseline.

## Pipeline overview

```
Stage 1   build_test_set.py
          → test_set.ensembl_id_joined_long.tsv
          → test_set.ensembl_id_joined_ts_all_genes.txt

Stage 2   build_evaluation_subsets.py
          → eval_ens_ts_considered_all_mirs.tsv   (the manuscript's 50,549 pairs)
          → 4 other subsets + stdout TargetScan baseline report

[Inference]  miRBind2-3UTR (V4) checkpoint
          → test_results_gene_level_V4_pretrained/ts_considered_all_mirs/*.tsv
            (same rows + column "predicted_V4 Model - 20260113_164412")

Stage 3   evaluate_test_set.py
          → evaluation_report.txt
```

## Requirements

- Python 3.10+ with `pandas`, `numpy`, `scipy`
- `bash`, `curl`, `unzip`
- `gdown` (the setup script installs it via pip if missing)

## Step 0 — Download and lay out the inputs

The Stage 1 inputs come from two sources:

1. A Google Drive zip containing the UCSC and fold-change files:
   <https://drive.google.com/file/d/1ZeRynCY0dz__hf2JeAHCxuCo4c8HOw6t/view?usp=sharing>
2. The TargetScan 8.0 (vert_80) public download files.

Both are fetched and placed into the expected directory layout by:

```bash
bash setup_stage1_inputs.sh "https://drive.google.com/file/d/1ZeRynCY0dz__hf2JeAHCxuCo4c8HOw6t/view?usp=sharing"
```

Layout:

```
project_root/
├── data/                                ← two levels above the scripts
│   ├── UCSC/hg19/3utr_sequences_hg19.txt
│   ├── UCSC/hg19/id_map
│   ├── fold_change/mirna_fcs.csv
│   └── TS8/
│       ├── Conserved_Site_Context_Scores.txt
│       ├── Nonconserved_Site_Context_Scores.txt
│       ├── Gene_info.txt
│       └── processed/
└── analysis/
    └── gene_level_dataset/                         ← run all bash scripts from here
        ├── setup_stage1_inputs.sh
        ├── run_build_test_set.sh
        └── ...
```

All run scripts assume they are invoked from the script directory and reference
data via `../../data/...`.

## Step 1 — Build the test set

```bash
bash run_build_test_set.sh
```

Joins the UCSC 3′UTR sequences, the measured log₂ fold-changes, and the
TargetScan context++ scores via the representative Ensembl transcript
designated in `Gene_info.txt`. Writes:

- `../../data/TS8/processed/test_set.ensembl_id_joined_long.tsv`
- `../../data/TS8/processed/test_set.ensembl_id_joined_ts_all_genes.txt`

## Step 2 — Build the evaluation subsets

```bash
bash run_build_evaluation_subsets.sh
```

Produces five subset TSVs under `../../data/TS8/processed/`. The one used in the
manuscript is `eval_ens_ts_considered_all_mirs.tsv` (50,549 pairs). The TS
baseline correlations are printed to stdout; redirect to keep them:

```bash
bash run_build_evaluation_subsets.sh > eval_ens_ts_baseline.txt
```

## Step 3 — Run miRBind2-3UTR inference

**Not included in this script group.** Take `eval_ens_ts_considered_all_mirs.tsv`,
run the trained miRBind2-3UTR (V4) model on each (miRNA, UTR) pair, and write
TSVs into `test_results_gene_level_V4_pretrained/ts_considered_all_mirs/` with
columns:

- `actual_fold_change` (from `log2fc`)
- `predicted_V4 Model - 20260113_164412`
- `predicted_TargetScan Weighted Context++ Score` (from `TS_weighted_context_pp`)
- `miRNA`

UTRs are capped at 3000 nt and miRNAs at 28 nt before inference.

## Step 4 — Evaluate

```bash
bash run_evaluate_test_set.sh
```

Compares miRBind2-3UTR against the TargetScan weighted context++ baseline using
Pearson, Spearman, R² (regression) and ROC-AUC, AP (classification, threshold
log₂FC < −0.05), with Williams' test, DeLong's test, and 10,000-resample
bootstrap CIs. Writes `evaluation_report.txt`.

## Files

| File | Purpose |
|---|---|
| `setup_stage1_inputs.sh` | Download Drive zip + TS8 files, lay out directories |
| `run_build_test_set.sh` | Stage 1 invocation |
| `build_test_set.py` | Stage 1 logic |
| `run_build_evaluation_subsets.sh` | Stage 2 invocation |
| `build_evaluation_subsets.py` | Stage 2 logic |
| `run_evaluate_test_set.sh` | Stage 3 invocation |
| `evaluate_test_set.py` | Stage 3 logic |
