# Pairwise Encoding CNN with Conservation Scores

CNN models for miRNA binding site prediction incorporating evolutionary conservation data. This directory contains both standard pairwise training and family-specific modeling approaches.

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

See `../../analysis/train_pairwise_model.sh`

## Directory Structure

```
pairwise_cnn_conservation/
├── models.py                              # Shared CNN model architecture
├── dataset.py                             # Shared data loading utilities
├── utils.py                               # Shared utility functions
├── train_pairwise_model.py                # Standard pairwise training
└── single_family_pairwise_cnn_conservation/ # Family-specific extension
    ├── train_family_models.py             # Family-specific training
    ├── single_family_eval.py              # Family-specific evaluation
    ├── single_family_plot.py              # Family-specific visualization
    └── single_family_datasets/            # Data processing pipeline
        ├── annotate_dataset.py            # Sequence annotation
        ├── count_fams.py                  # Family counting
        ├── split_fams.py                  # Family-based splitting
        ├── prepare_datasets.sh            # Pipeline orchestration
        └── reference files
```

## Two Approaches

### 1. Standard Pairwise Training
Trains a single model on all miRNA families combined:
```bash
python train_pairwise_model.py [options]
```

### 2. Family-Specific Training
Trains individual models for each miRNA family:
```bash
# First prepare family-specific datasets
cd single_family_pairwise_cnn_conservation/single_family_datasets/
./prepare_datasets.sh

# Then train family-specific models
python ../train_family_models.py --train_dir family_train --output_dir family_outputs
```

## Analysis Scripts
- `../../analysis/train_pairwise_model.sh` - Standard pairwise training
- `../../analysis/train_family_models.sh` - Family-specific training

## Shared Components
All approaches use the same core architecture:
- **models.py**: CNN model definitions
- **dataset.py**: Data loading and preprocessing
- **utils.py**: Logging and utility functions 
