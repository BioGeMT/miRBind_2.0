#!/bin/bash

timestamp=$(date +"%Y%m%d_%H%M%S")
model_name="mirBind_${timestamp}"
best_model_path="models/${model_name}.keras"
evaluation_out_dir="evaluation_results/${model_name}_hyperopt"

train_file_in="../../data/chimeric_datasets/Manakov2022_flat/AGO2_eCLIP_Manakov2022_train.tsv"
test_file_in="../../data/chimeric_datasets/Manakov2022_flat/AGO2_eCLIP_Manakov2022_test.tsv"
leftout_file_in="../../data/chimeric_datasets/Manakov2022_flat/AGO2_eCLIP_Manakov2022_leftout.tsv"

train_file_out="encoded_dataset/Manakov2022_flat/AGO2_eCLIP_Manakov2022_train"
test_file_out="encoded_dataset/Manakov2022_flat/AGO2_eCLIP_Manakov2022_test"
leftout_file_out="encoded_dataset/Manakov2022_flat/AGO2_eCLIP_Manakov2022_leftout"

train_file_size=2516195

CODE="../../code/machine_learning"

mkdir -p encoded_dataset/Manakov2022_flat

# encode datasets
python $CODE/encode/binding_2D_matrix_encoder.py --i_file $train_file_in --o_prefix $train_file_out
python $CODE/encode/binding_2D_matrix_encoder.py --i_file $test_file_in --o_prefix $test_file_out
python $CODE/encode/binding_2D_matrix_encoder.py --i_file $leftout_file_in --o_prefix $leftout_file_out

# run hyper parameter optimisation
python hyperparam_optimization/hyperparam_optimization.py \
  --dataset-train "../${train_file_out}_dataset.npy" \
  --labels-train "../${train_file_out}_labels.npy" \
  --dataset-size $train_file_size \
  --dataset-ratio 1 \
  --batch-size 32 \
  --validation-split 0.1 \
  --n-trials 20 \
  --best-model $best_model_path \
  --log-file "hyperparam_optimization.log"
  --seed 42
  --epochs 5
  
# evaluate the best model
python hyperparam_optimization/evaluate_model.py \
  --model-path $best_model_path \
  --dataset-test "../${test_file_out}_dataset.npy" \
  --labels-test "../${test_file_out}_labels.npy" \
  --batch-size 32 \
  --log-file "model_evaluation_test.log" \
  --save-plots \
  --output-dir $evaluation_out_dir

python hyperparam_optimization/evaluate_model.py \
  --model-path $best_model_path \
  --dataset-test "../${leftout_file_out}_dataset.npy" \
  --labels-test "../${leftout_file_out}_labels.npy" \
  --batch-size 32 \
  --log-file "model_evaluation_leftout.log" \
  --save-plots \
  --output-dir $evaluation_out_dir
