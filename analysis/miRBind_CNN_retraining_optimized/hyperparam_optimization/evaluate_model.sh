#!/bin/bash

test_file_out="encoded_dataset/Manakov2022_flat/AGO2_eCLIP_Manakov2022_test"
leftout_file_out="encoded_dataset/Manakov2022_flat/AGO2_eCLIP_Manakov2022_leftout"

# best_model_path="models/best_model.keras"
best_model_path="evaluation_results/mirBind_2002/models_tmp/best_model.keras"
# evaluation_out_dir="evaluation_results/mirBind_1902"
evaluation_out_dir="evaluation_results/mirBind_2002/models_tmp"

# evaluate the best model
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


# hyperparam_optimization/evaluation_results/mirBind_2002/models_tmp/best_model.keras