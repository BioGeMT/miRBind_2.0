#!/bin/bash


train_file="encoded_dataset/Manakov2022_flat/AGO2_eCLIP_Manakov2022_train"

train_file_size=2516195
model_dir="mirBind_2002"
best_model_path="models/${model_dir}/best_model.keras"
evaluation_out_dir="evaluation_results/${model_dir}"


# run hyper parameter optimisation
python hyperparam_optimization.py \
  --dataset-train "../${train_file}_dataset.npy" \
  --labels-train "../${train_file}_labels.npy" \
  --dataset-size $train_file_size \
  --dataset-ratio 1 \
  --batch-size 32 \
  --validation-split 0.1 \
  --n-trials 30 \
  --best-model "$best_model_path" \
  --log-file "${evaluation_out_dir}/hyperparam_optimization.log"
  --seed 42
  --epochs 8