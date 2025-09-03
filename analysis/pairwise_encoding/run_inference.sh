#!/bin/bash
set -e

# Example inference script for miRNA binding site prediction
# Modify the paths below to match your setup

# Path to your trained model checkpoint
MODEL_PATH="model_outputs/pairwise_model_20250902_132244.pt"

# Input data file for inference
INPUT_FILE="../../data/chimeric_datasets/Manakov2022_flat/AGO2_eCLIP_Manakov2022_test.tsv"

# Output file - original dataset with predictions appended
OUTPUT_FILE="$(basename "$INPUT_FILE" .tsv)_with_predictions_$(date +%Y%m%d_%H%M%S).tsv"

echo "Running inference..."
echo "Model: ${MODEL_PATH}"
echo "Input: ${INPUT_FILE}"
echo "Output: ${OUTPUT_FILE}"

python inference_pairwise_model.py \
  --model_path "${MODEL_PATH}" \
  --input_file "${INPUT_FILE}" \
  --output_file "${OUTPUT_FILE}" \
  --batch_size 32 \
  --model_type "pairwise_onehot"

echo "Inference complete!"
echo "Dataset with predictions saved to: ${OUTPUT_FILE}"