#!/bin/bash
#SBATCH --account=ssamm10
#SBATCH --job-name=retrainCNN
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30

set -euo pipefail
trap 'echo "Error at line $LINENO: $BASH_COMMAND"; exit 1' ERR

mkdir -p results

exec > >(tee -a results/RUNME.log) 2>&1

# ===== Define variables =====

MANAKOV="AGO2_eCLIP_Manakov2022"
TRAIN_SET="${MANAKOV}_train"
TEST_SETS=(
    "${MANAKOV}_test"
    "${MANAKOV}_leftout"
)
ALL_DATASETS=(
    "${TRAIN_SET}"
    "${TEST_SETS[@]}"
)
CODE_DIR="../../code/mirbench_cnn_cofold"

mkdir -p results/encoding results/training results/predictions results/evaluation

# ===== Download or locate dataset, and encode it=====
echo
echo "Encoding datasets..."
echo
for DATASET in "${ALL_DATASETS[@]}"; do
    # Extract base dataset name and split (everything before/after last '_')
    DATASET_NAME="${DATASET%_*}"
    SPLIT="${DATASET##*_}"

    echo "Locating dataset: $DATASET_NAME, split: $SPLIT..."
    DATASET_PATH=$(python ${CODE_DIR}/get_dataset_path.py --dataset "$DATASET_NAME" --split "$SPLIT" | tail -n 1)
    
    echo "Adding dotbracket structures to ${DATASET}.tsv..."
    python ${CODE_DIR}/get_dotbracket_structure.py \
        --dataset_path "$DATASET_PATH" \
        --output_path "results/encoding/${DATASET}_dotbracket.tsv"

    echo "Encoding ${DATASET}.tsv into the 50_20_2 tensor..."
    python ${CODE_DIR}/encode_50_20_2.py \
        --i_file "results/encoding/${DATASET}_dotbracket.tsv" \
        --o_prefix "results/encoding/${DATASET}_50_20_2"
done
echo "All datasets encoded in results/encoding/ directory."

# ===== Train CNN models =====

echo
echo "Training CNN model..."
echo

echo "Training CNN model on ${MANAKOV} train set using the 50 x 20 x 2 encoding..."
python ${CODE_DIR}/train_CNN_50_20_channels.py \
    --ratio 1 \
    --data "results/encoding/${MANAKOV}_train_50_20_2_dataset.npy" \
    --labels "results/encoding/${MANAKOV}_train_50_20_2_labels.npy" \
    --num_rows "results/encoding/${MANAKOV}_train_50_20_2_num_rows.npy" \
    --model "results/training/CNN_${MANAKOV}_train_50_20_2.keras" \
    --channels 2 \
    --debug 1

echo "CNN model trained and saved in results/training/ directory."

# ===== Predict and evaluate models =====

echo
echo "Predicting and evaluating CNN model on test sets..."
echo

for DATASET in "${TEST_SETS[@]}"; do
    echo "Predicting with ${MANAKOV} CNN model on ${DATASET} set using the 50 x 20 x 2 encoding..."
    python ${CODE_DIR}/predict.py \
        --model_path "results/training/CNN_${MANAKOV}_train_50_20_2.keras" \
        --dataset "results/encoding/${DATASET}_50_20_2_dataset.npy" \
        --num_rows "results/encoding/${DATASET}_50_20_2_num_rows.npy" \
        --channels 2 \
        --output_path "results/predictions/${DATASET}_CNN_${MANAKOV}_train_50_20_2_preds.npy"

    echo "Evaluating predictions ${MANAKOV} CNN model on ${DATASET} set using the 50 x 20 x 2 encoding..."
    python ${CODE_DIR}/evaluate.py \
        --preds_path "results/predictions/${DATASET}_CNN_${MANAKOV}_train_50_20_2_preds.npy" \
        --labels_path "results/encoding/${DATASET}_50_20_2_labels.npy" \
        --output_path "results/evaluation/${DATASET}_CNN_${MANAKOV}_train_50_20_2_metrics.json" \
        --model_name "CNN_${MANAKOV}_train_50_20_2" \
        --test_set_name "${DATASET}"
done

echo "All predictions and evaluations completed. Results are in results/predictions/ and results/evaluation/ directories."

echo
echo "Script completed successfully. Check results in the results/ directory."