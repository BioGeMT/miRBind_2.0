#!/bin/bash


TEST_DATASET="../../data/chimeric_datasets/Manakov2022_flat/AGO2_eCLIP_Manakov2022_test.tsv"
LEFTOUT_DATASET="../../data/chimeric_datasets/Manakov2022_flat/AGO2_eCLIP_Manakov2022_leftout.tsv"
TRAIN_DATASET="../../data/chimeric_datasets/Manakov2022_flat/AGO2_eCLIP_Manakov2022_train.tsv"


TEST_DATASET_OUT="encoded_dataset/Manakov2022_flat/AGO2_eCLIP_Manakov2022_1_test"
LEFTOUT_DATASET_OUT="encoded_dataset/Manakov2022_flat/AGO2_eCLIP_Manakov2022_1_leftout"
TRAIN_DATASET_OUT="encoded_dataset/Manakov2022_flat/AGO2_eCLIP_Manakov2022_1_train"


CODE="../../code/machine_learning"

mkdir -p encoded_dataset/Manakov2022_flat

# encode datasets
python $CODE/encode/binding_2D_matrix_encoder.py --i_file $TEST_DATASET --o_prefix $TEST_DATASET_OUT
python $CODE/encode/binding_2D_matrix_encoder.py --i_file $LEFTOUT_DATASET --o_prefix $LEFTOUT_DATASET_OUT
python $CODE/encode/binding_2D_matrix_encoder.py --i_file $TRAIN_DATASET --o_prefix $TRAIN_DATASET_OUT
