#!/bin/bash

DATASET="../../data/chimeric_datasets/Manakov2022/AGO_eCLIP_Manakov2022_1_train_dataset.tsv"
MODEL="../../models/miRBind_CNN_retrained_Manakov_1_orig_parameters.keras"
CODE="../../code/machine_learning"

mkdir -p encoded_dataset

# encode dataset
python $CODE/encode/binding_2D_matrix_encoder.py --i_file $DATASET --o_prefix encoded_dataset/AGO2_eCLIP_Manakov2022_1_train

# train model
python $CODE/train/CNN_miRBind_2022/miRBind_CNN_training_orig_parameters.py \
--data encoded_dataset/AGO2_eCLIP_Manakov2022_1_train_dataset.npy \
--labels encoded_dataset/AGO2_eCLIP_Manakov2022_1_train_labels.npy \
--dataset_size 2524246 \
--ratio 1 \
--model $MODEL