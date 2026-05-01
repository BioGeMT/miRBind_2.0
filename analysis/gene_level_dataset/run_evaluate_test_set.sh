#!/bin/bash
# Run evaluation of miRBind V4 vs TargetScan on the test set (ts_considered_all_mirs)
#
# Adjust paths as needed for your directory layout.

PRED_DIR="test_results_gene_level_V4_pretrained/ts_considered_all_mirs"
OUTPUT="evaluation_report.txt"

python evaluate_test_set.py \
    --predictions "${PRED_DIR}" \
    --truth_col "actual_fold_change" \
    --model_col "predicted_V4 Model - 20260113_164412" \
    --competitor_col "predicted_TargetScan Weighted Context++ Score" \
    --competitor_name "TS Weighted Context++" \
    --mirna_col "miRNA" \
    --binary_threshold -0.05 \
    --n_bootstrap 10000 \
    --output "${OUTPUT}"
