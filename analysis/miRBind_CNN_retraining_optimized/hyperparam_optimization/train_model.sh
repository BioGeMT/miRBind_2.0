#!/bin/bash

# set the model_name to a unique name for your run
timestamp=$(date +"%Y%m%d_%H%M%S")
model_name="mirBind_${timestamp}"
out_dir="evaluation_results/${model_name}"

train_file="encoded_dataset/Manakov2022_flat/AGO2_eCLIP_Manakov2022_train"
train_file_size=2516195

python train_model.py \
  --dataset-train "../${train_file}_dataset.npy" \
  --labels-train "../${train_file}_labels.npy" \
  --dataset-size $train_file_size \
  --cnn-num 2 \
  --kernel-size 6 \
  --pool-size 2 \
  --dropout-rate 0.3 \
  --dense-num 2 \
  --learning-rate 0.00008241877487855944 \
  --batch-size 32 \
  --epochs 100 \
  --patience 6 \
  --output-dir $out_dir \
  --model-name $model_name \
  --seed 42
  