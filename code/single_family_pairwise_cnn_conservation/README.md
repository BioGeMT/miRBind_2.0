# Pairwise encoding CNN with Conservation Scores

A CNN model for miRNA binding site prediction incorporating evolutionary conservation data and family-specific training.

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
- **Family-specific modeling** with individual models trained per miRNA family
- **Data annotation pipeline** for mapping sequences to miRNA families using MirGeneDB

## Pipeline Components

### Data Annotation (`annotate_dataset.py`)
Maps miRNA sequences to families using MirGeneDB references:
- Matches sequences against FASTA to identify miRNAs
- Adds mirgenedb annotations

### Dataset splitting (`count_fams.py` and `split_fams.py`)
- Counts family occurrences in annotated data
- Splits datasets by family for families meeting sample thresholds
- Creates family-specific training files

### Model Training (`train_family_models.py`)
Trains individual CNN models for each miRNA family:
- Automatic train/validation split
- Early stopping based on validation performance
- Saves model checkpoints and training history

### Evaluation (`eval.py`)
Evaluates family-specific models on test data:
- Family-specific performance metrics
- Overall performance aggregation
- Visualization plots

## Training Process

Training involves:
- Data annotation with miRNA family mapping
- Family-based data splitting for samples above threshold
- Individual model training per family with random train/validation split
- Monitors train/validation metrics (loss and AUPRC)
- Early stopping based on validation performance
- Logs configuration (JSON) and training metrics

## Usage

### Data Annotation
```bash
python annotate_dataset.py \
  --fasta mirgenedb_sequences.fa \
  --tsv input_data.tsv \
  --mirgenedb mirgenedb_families.tsv \
  --output annotated_data.tsv
```

### Family Counting and Splitting
```bash
python count_fams.py annotated_data.tsv family_counts.tsv
python split_fams.py annotated_data.tsv family_counts.tsv output_dir/ --threshold 1000
```

### Family Model Training
```bash
python train_family_models.py \
  --train_dir single_fam_train/ \
  --output_dir family_model_outputs/ \
  --batch_size 32 \
  --num_epochs 30 \
  --embedding_dim 8
```

### Model Evaluation
```bash
python eval.py \
  --test_file test_data.tsv \
  --models_dir family_model_outputs/ \
  --output_dir evaluation_outputs/
```

See `train_family_models.sh`