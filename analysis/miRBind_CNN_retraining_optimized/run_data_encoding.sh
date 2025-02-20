#!/bin/bash

DATASET="../../data/chimeric_datasets/Manakov2022_flat/AGO2_eCLIP_Manakov2022_leftout.tsv"
# DATASET="../../data/chimeric_datasets/Manakov2022_flat/AGO2_eCLIP_Manakov2022_test.tsv"
# DATASET="../../data/chimeric_datasets/Manakov2022_flat/AGO2_eCLIP_Manakov2022_train.tsv"


DATASET_OUT="encoded_dataset/Manakov2022_flat/AGO2_eCLIP_Manakov2022_leftout.tsv"
# DATASET_OUT="encoded_dataset/Manakov2022_flat/AGO2_eCLIP_Manakov2022_test"
# DATASET_OUT="encoded_dataset/Manakov2022_flat/AGO2_eCLIP_Manakov2022_train"


CODE="../../code/machine_learning"

mkdir -p encoded_dataset/Manakov2022_flat

# encode dataset
python $CODE/encode/binding_2D_matrix_encoder.py --i_file $DATASET --o_prefix $DATASET_OUT
