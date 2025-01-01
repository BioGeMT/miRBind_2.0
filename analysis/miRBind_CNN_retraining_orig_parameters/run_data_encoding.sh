#!/bin/bash


# train_file = "manakov/AGO2_eCLIP_Manakov2022_train.tsv"
# test_file_1 = "manakov/AGO2_eCLIP_Manakov2022_test.tsv"
# test_file_2 = "manakov/AGO2_eCLIP_Manakov2022_leftout.tsv"
DATASET="../../data/chimeric_datasets/Manakov2022_flat/AGO2_eCLIP_Manakov2022_test.tsv"
# DATASET="../../data/chimeric_datasets/Manakov2022_flat/AGO2_eCLIP_Manakov2022_train.tsv"
# DATASET="../../data/chimeric_datasets/Manakov2022/AGO2_eCLIP_Manakov2022_1_train_dataset.tsv"
# DATASET="../../data/chimeric_datasets/Manakov2022/AGO2_eCLIP_Manakov2022_1_test_dataset.tsv"


DATASET_OUT="encoded_dataset/Manakov2022_flat/AGO2_eCLIP_Manakov2022_1_test"
# DATASET_OUT="encoded_dataset/Manakov2022_flat/AGO2_eCLIP_Manakov2022_1_train"
# DATASET_OUT="encoded_dataset/AGO2_eCLIP_Manakov2022_1_test"


CODE="../../code/machine_learning"

mkdir -p encoded_dataset/Manakov2022_flat

# encode dataset
python $CODE/encode/binding_2D_matrix_encoder.py --i_file $DATASET --o_prefix $DATASET_OUT
