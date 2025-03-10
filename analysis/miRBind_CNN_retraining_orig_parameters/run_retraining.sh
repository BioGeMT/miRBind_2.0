#!/bin/bash

MODEL="../../models/miRBind_CNN_retrained_Manakov_1_orig_parameters.keras"
CODE="../../code/machine_learning"

# train model
python $CODE/train/CNN_miRBind_2022/miRBind_CNN_training_orig_parameters.py \
--data encoded_dataset/Manakov2022_flat/AGO2_eCLIP_Manakov2022_1_train_dataset.npy \
--labels encoded_dataset/Manakov2022_flat/AGO2_eCLIP_Manakov2022_1_train_labels.npy \
--dataset_size 2516195 \
--ratio 1 \
--model $MODEL