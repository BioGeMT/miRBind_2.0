#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${SCRIPT_DIR}/../../code/pairwise_binding_site_model"

INPUT_FILE="shap_results.tsv"
OUTPUT_DIR="clustering_results"

cd "${CODE_DIR}"

python -m explainability.clustering \
    --input "${INPUT_FILE}" \
    --shap_col "shap_values_2d" \
    --mirna_seq_col "noncodingRNA" \
    --mirna_name_col "noncodingRNA_name" \
    --output_dir "${OUTPUT_DIR}" \
    --approach "both" \
    --reduction_method "max" \
    --clustering_method "gmm" \
    --k_global 5 \
    --k_mirna 3 \
    --min_samples_per_mirna 10 \
    --shap_filter "positive" \
    --prediction_filter "TP" \
    --true_label_col "label" \
    --pred_label_col "predicted_class"

echo "Clustering results saved to: ${OUTPUT_DIR}"
