#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${SCRIPT_DIR}/../../code/pairwise_binding_site_model"

INPUT_FILE="shap_results.tsv"
OUTPUT_DIR="importance_plots"

cd "${CODE_DIR}"

python -m explainability.aggregate \
    --input_file "${INPUT_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --axis_mode "mirna" \
    --stratify_by all TP TN FP FN \
    --reduction_method "sum" \
    --aggregation_method "mean" \
    --plot_type "bar" \
    --plot_top_n 20 \
    --shap_column "shap_values_2d" \
    --mirna_column "noncodingRNA" \
    --target_column "gene"

echo "Importance plots saved to: ${OUTPUT_DIR}"
