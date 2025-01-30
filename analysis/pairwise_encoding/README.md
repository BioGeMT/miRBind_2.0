# Pairwise encoding for CNN
A CNN model for miRNA binding site prediction

1) Sequence input of miRNA and target RNA 
2) Encodes nucleotide pairs into indices (16 possible pairs + N padding)
3) Processes these through:
- Embedding layer (configurable dimension)
- Three convolutional layers with batch normalization, max pooling, and dropout
- Two fully connected layers
4) Outputs binding probability (0-1)

Training involves:

- Random train/validation split
- Monitors train/validation/test metrics
- Logs configuration (JSON) and training metrics (TSV) with timestamp identifiers

CLI arguments allow configuration of model parameters, training settings, and data paths.

Example of how to run (see train_cnn.sh):
python cnn_pairwise_encoding.py --train_file "../../data/chimeric_datasets/Manakov2022_flat/AGO2_eCLIP_Manakov2022_train.tsv" --test_file_1 "../../data/chimeric_datasets/Manakov2022_flat/AGO2_eCLIP_Manakov2022_test.tsv" --test_file_2 "../../data/chimeric_datasets/Manakov2022_flat/AGO2_eCLIP_Manakov2022_leftout.tsv" --batch_size 32 --num_epochs 5 --embedding_dim 1 --train_fraction 1

python cnn_pairwise_encoding_optuna.py --train_file "../../data/chimeric_datasets/Manakov2022_flat/AGO2_eCLIP_Manakov2022_train.tsv" --n_trials 100 --study_name your_study_name