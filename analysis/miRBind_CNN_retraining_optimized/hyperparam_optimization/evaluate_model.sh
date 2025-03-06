#!/bin/bash

test_file_out="encoded_dataset/Manakov2022_flat/AGO2_eCLIP_Manakov2022_test"
leftout_file_out="encoded_dataset/Manakov2022_flat/AGO2_eCLIP_Manakov2022_leftout"

# set the model_name to how you named your run
timestamp=#TODO_SET_YOUR_TRAINED_MODEL'S_TIMESTAMP
model_name="mirBind_${timestamp}"
best_model_path="evaluation_results/${model_name}/${model_name}_final.keras"
evaluation_out_dir="evaluation_results/${model_name}"

python evaluate_model.py \
  --model-path $best_model_path \
  --dataset-test "../${test_file_out}_dataset.npy" \
  --labels-test "../${test_file_out}_labels.npy" \
  --batch-size 32 \
  --log-file "model_evaluation_test.log" \
  --save-plots \
  --output-dir $evaluation_out_dir

python evaluate_model.py \
  --model-path $best_model_path \
  --dataset-test "../${leftout_file_out}_dataset.npy" \
  --labels-test "../${leftout_file_out}_labels.npy" \
  --batch-size 32 \
  --log-file "model_evaluation_leftout.log" \
  --save-plots \
  --output-dir $evaluation_out_dir