#!/bin/bash

#SBATCH --job-name=mirbind_test
#SBATCH --output=mirbind_test_%j.log
#SBATCH --error=mirbind_test_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00


# Print environment info
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Current working directory: $(pwd)"
echo "Python version: $(python --version 2>&1)"

# Set environment
set -e

# check arguments
if [ $# -ne 3 ]; then
    echo "usage: $0 <train_file> <test_file1> <test_file2>"
    exit 1
fi

# assign command line arguments
TRAIN_FILE="$1"
TEST_FILE1="$2"
TEST_FILE2="$3"

# Set output directory
OUTPUT_DIR="model_outputs_conservation_smol"
mkdir -p ${OUTPUT_DIR}

# Set training parameters
EMBEDDING_DIM=8
FILTER_SIZES="128,64,32"
KERNEL_SIZES="6,5,5"
DROPOUT_RATE=0.2
LEARNING_RATE=0.001
NUM_EPOCHS=30
BATCH_SIZE=32
PATIENCE=5

# Run training
echo "Starting training with conservation scores..."
python ../code/pairwise_cnn_conservation/train_pairwise_model.py \
    --train_file ${TRAIN_FILE} \
    --test_file1 ${TEST_FILE1} \
    --test_file2 ${TEST_FILE2} \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --embedding_dim ${EMBEDDING_DIM} \
    --filter_sizes ${FILTER_SIZES} \
    --kernel_sizes ${KERNEL_SIZES} \
    --dropout_rate ${DROPOUT_RATE} \
    --learning_rate ${LEARNING_RATE} \
    --patience ${PATIENCE} \
    --output_dir ${OUTPUT_DIR}

echo "Training complete!"