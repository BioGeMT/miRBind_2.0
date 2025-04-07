# Pairwise encoding CNN with Conservation Scores

A CNN model for miRNA binding site prediction incorporating evolutionary conservation data

## Model Architecture

1) Sequence input of miRNA and target RNA 
2) Encodes nucleotide pairs into indices (16 possible pairs + N padding)
3) Incorporates phylogenetic conservation scores (PhyloP and PhastCons) as additional input channels
4) Processes these through:
   - Embedding layer (configurable dimension)
   - Three convolutional layers with batch normalization, max pooling, and dropout
   - Two fully connected layers
5) Outputs binding probability (0-1)

## Key Features

- **Pairwise encoding** of nucleotide interactions between miRNA and target RNA
- **Conservation score integration** (PhyloP and PhastCons) as additional channels
- **CNN architecture** optimized for sequence data with conservation information

## Training Process

Training involves:
- Random train/validation split
- Monitors train/validation/test metrics (loss and AUPRC)
- Early stopping based on validation performance
- Logs configuration (JSON) and training metrics
- 
## Usage

CLI arguments allow configuration of model parameters, training settings, and data paths.

### Example Commands

Training with standard parameters:
```bash
python train_pairwise_model.py \
  --train_file "../../data/chimeric_datasets/Manakov2022_flat/AGO2_eCLIP_Manakov2022_train.tsv" \
  --test_file1 "../../data/chimeric_datasets/Manakov2022_flat/AGO2_eCLIP_Manakov2022_test.tsv" \
  --test_file2 "../../data/chimeric_datasets/Manakov2022_flat/AGO2_eCLIP_Manakov2022_leftout.tsv" \
  --batch_size 32 \
  --num_epochs 5 \
  --embedding_dim 8
```

See `train_pairwise_model.sh` 
