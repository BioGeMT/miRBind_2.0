python cnn_pairwise_encoding.py --train_file "../../data/chimeric_datasets/Manakov2022_flat/AGO2_eCLIP_Manakov2022_train.tsv" --test_file_1 "../../data/chimeric_datasets/Manakov2022_flat/AGO2_eCLIP_Manakov2022_test.tsv" --test_file_2 "../../data/chimeric_datasets/Manakov2022_flat/AGO2_eCLIP_Manakov2022_leftout.tsv" --batch_size 32 --num_epochs 5 --embedding_dim 1 --train_fraction 1


# python cnn_pairwise_encoding_optuna.py --train_file "../../data/chimeric_datasets/Manakov2022_flat/AGO2_eCLIP_Manakov2022_train.tsv" --n_trials 100 --study_name your_study_name

# python train.py --model_type 'mirbind' --train_data '../../data/chimeric_datasets/Manakov2022_flat/AGO2_eCLIP_Manakov2022_1_train_dataset.npy' --train_labels '../../data/chimeric_datasets/Manakov2022_flat/AGO2_eCLIP_Manakov2022_1_train_labels.npy' --dataset_size 2516195

