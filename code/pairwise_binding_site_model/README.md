# Pairwise Binding Site Model

Deep learning pipeline for predicting miRNA-mRNA binding sites using pairwise CNN models with SHAP-based explainability.

## Structure

```
pairwise_binding_site_model/
├── shared/              # Core components
│   ├── constants.py     # Nucleotide pairs, colors, device utilities
│   ├── encoding.py      # Sequence encoding functions
│   ├── dataset.py       # PyTorch datasets
│   └── models.py        # CNN architectures
├── training/            # Training & evaluation
│   ├── train.py         # Training with early stopping
│   └── evaluate.py      # Model evaluation
├── inference/           # Prediction & SHAP computation
│   ├── predict.py       # Basic inference
│   └── explainability.py # GradientShap computation
├── explainability/      # SHAP analysis pipeline
│   ├── shap_utils.py    # SHAP utilities
│   ├── clustering.py    # K-means/GMM clustering
│   ├── plotting.py      # Visualization functions
│   └── aggregate.py     # miRNA importance aggregation
└── scripts/             # Shell scripts
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python -m training.train \
    --train_file data/train.tsv \
    --test_file1 data/test.tsv \
    --test_file2 data/leftout.tsv \
    --model pairwise_onehot \
    --output_dir model_outputs
```

### Inference with SHAP

```bash
python -m inference.explainability \
    --model_path model_outputs/model.pt \
    --input_file data/test.tsv \
    --output_file results_shap.tsv \
    --model_type pairwise_onehot
```

### Clustering

```bash
python -m explainability.clustering \
    --input results_shap.tsv \
    --shap_col shap_values_2d \
    --mirna_seq_col noncodingRNA \
    --mirna_name_col noncodingRNA_name \
    --approach both \
    --output_dir clustering_results
```

### Importance Aggregation

```bash
python -m explainability.aggregate \
    --input_file results_shap.tsv \
    --output_dir importance_plots \
    --axis_mode mirna \
    --stratify_by all TP TN
```

## Models

- **PairwiseEncodingCNN**: Embedding layer for integer-encoded nucleotide pairs
- **PairwiseOneHotCNN**: Linear layer for one-hot encoded pairs (recommended for SHAP)

## Data Format

Input TSV with columns:
- Column 0: target/mRNA sequence
- Column 1: miRNA sequence  
- `label`: binary binding label (0/1)

## Architecture

The model encodes miRNA-mRNA interactions as 2D matrices where each cell represents a nucleotide pair. A CNN extracts features from these matrices to predict binding probability.

SHAP values are computed using GradientShap (Captum) and reduced from 3D (one-hot) to 2D (miRNA × mRNA positions) for interpretability.
