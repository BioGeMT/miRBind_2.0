#!/bin/bash

# Print environment info
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Current working directory: $(pwd)"
echo "Python version: $(python --version 2>&1)"

# Define paths
TRAIN_DIR="family_train"
OUTPUT_DIR="family_outputs"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Set training parameters
EMBEDDING_DIM=8
FILTER_SIZES="128,64,32"
KERNEL_SIZES="6,3,3"
DROPOUT_RATE=0.2
LEARNING_RATE=0.001
NUM_EPOCHS=30
BATCH_SIZE=32
PATIENCE=5
TARGET_LENGTH=50
MIRNA_LENGTH=28

# Run training
echo "Starting training for miRNA family models..."
python ../code/pairwise_cnn_conservation/single_family_pairwise_cnn_conservation/train_family_models.py \
    --train_dir ${TRAIN_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --target_length ${TARGET_LENGTH} \
    --mirna_length ${MIRNA_LENGTH} \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --embedding_dim ${EMBEDDING_DIM} \
    --filter_sizes ${FILTER_SIZES} \
    --kernel_sizes ${KERNEL_SIZES} \
    --dropout_rate ${DROPOUT_RATE} \
    --learning_rate ${LEARNING_RATE} \
    --patience ${PATIENCE}

echo "Training complete!"
echo "Job finished at $(date)"

# Print summary of output directory
echo ""
echo "Output directory contents:"
ls -la ${OUTPUT_DIR}/

# Count number of family models trained
NUM_FAMILIES=$(find ${OUTPUT_DIR} -maxdepth 1 -type d | wc -l)
echo ""
echo "Number of family directories created: $((NUM_FAMILIES - 1))"