#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${SCRIPT_DIR}/../../code/pairwise_binding_site_model"

MODEL_PATH="model_outputs/pairwise_onehot_model_YYYYMMDD_HHMMSS.pt"
INPUT_FILE="../../data/chimeric_datasets/Manakov2022_flat/AGO2_eCLIP_Manakov2022_test.tsv"
OUTPUT_FILE="predictions_$(date +%Y%m%d_%H%M%S).tsv"

cd "${CODE_DIR}"

python -m inference.predict \
    --model_path "${MODEL_PATH}" \
    --input_file "${INPUT_FILE}" \
    --output_file "${OUTPUT_FILE}" \
    --model_type "pairwise_onehot" \
    --batch_size 32

echo "Predictions saved to: ${OUTPUT_FILE}"
