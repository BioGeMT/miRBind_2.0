#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${SCRIPT_DIR}/../../code/pairwise_binding_site_model"

TRAIN_FILE="../../data/chimeric_datasets/Manakov2022_flat/AGO2_eCLIP_Manakov2022_train.tsv"
TEST_FILE_1="../../data/chimeric_datasets/Manakov2022_flat/AGO2_eCLIP_Manakov2022_test.tsv"
TEST_FILE_2="../../data/chimeric_datasets/Manakov2022_flat/AGO2_eCLIP_Manakov2022_leftout.tsv"

cd "${CODE_DIR}"

python -m training.train \
    --train_file "${TRAIN_FILE}" \
    --test_file1 "${TEST_FILE_1}" \
    --test_file2 "${TEST_FILE_2}" \
    --target_length 50 \
    --mirna_length 28 \
    --batch_size 32 \
    --num_epochs 15 \
    --val_fraction 0.1 \
    --learning_rate 0.001 \
    --embedding_dim 8 \
    --dropout_rate 0.2 \
    --filter_sizes "128,64,32" \
    --kernel_sizes "6,3,3" \
    --patience 7 \
    --output_dir "model_outputs" \
    --model "pairwise_onehot"

echo "Training complete!"
