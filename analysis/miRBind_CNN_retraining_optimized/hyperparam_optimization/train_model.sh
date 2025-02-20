#!/bin/bash

train_file="encoded_dataset/Manakov2022_flat/AGO2_eCLIP_Manakov2022_train"

train_file_size=2516195
model_dir="mirBind_001_long_training"
out_dir="evaluation_results/${model_dir}"


python train_model.py \
  --dataset-train "../${train_file}_dataset.npy" \
  --labels-train "../${train_file}_labels.npy" \
  --dataset-size $train_file_size \
  --cnn-num 4 \
  --kernel-size 9 \
  --pool-size 2 \
  --dropout-rate 0.3 \
  --dense-num 2 \
  --learning-rate 0.00008241877487855944 \
  --batch-size 32 \
  --epochs 20 \
  --patience 4 \
  --output-dir out_dir \
  --model-name mirBind_001_long_training \
  --seed 42