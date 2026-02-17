#!/bin/bash

CODE_DIR="../../code/gene_level_model/training"
DATA_DIR="../../data"
OUTPUT_DIR="gene_model_v4_pretrained_3conv"

PRETRAINED_CHECKPOINT="../../code/pairwise_binding_site_model/model_outputs/pairwise_onehot_model_20260105_200141.pt"
TRAIN_FILE="${DATA_DIR}/Agarwal2015_train_data/Agarwal2015_training_dataset_log2fold.pkl"


python "${CODE_DIR}/train.py" \
    --pretrained_checkpoint "${PRETRAINED_CHECKPOINT}" \
    --train_file "${TRAIN_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --gene_col utr3 \
    --mirna_col miRNA \
    --label_col log2fc \
    --num_heads 8 \
    --max_gene_length 3000 \
    --attention_hidden_dim 128 \
    --batch_size 128 \
    --learning_rate 0.001 \
    --num_epochs 300 \
    --patience 25 \
    --pretrained_lr_factor 0.01 \
    --use_layer_norm \
    --use_discriminative_lr \
    --freeze_encoder \
    --unfreeze_epoch 30 \
    --filter_sizes 128,64,32 \
    --kernel_sizes 6,5,4 \
    --early_stop_metric pearson \
    --dropout_rate 0.3
