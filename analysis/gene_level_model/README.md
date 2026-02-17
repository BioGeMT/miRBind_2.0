# Gene-Level Repression Model

Predicts gene-level fold change from full 3'UTR sequences using transfer learning from a pretrained pairwise binding site CNN.

## Architecture (V4)

The model takes a miRNA sequence and a full gene 3'UTR sequence, constructs a 2D pairwise nucleotide encoding, and processes it through:

1. **3 pretrained convolutional blocks** — transferred from a binding-site-level `PairwiseOneHotCNN`, each consisting of Conv2d → BatchNorm → Pool → Dropout
2. **Multi-head spatial attention** — multiple attention heads learn different interaction patterns (e.g., seed matches, site clustering, positional preferences), combined via learned weights
3. **FC prediction head** — produces a single scalar fold change prediction

Optional features: Layer Normalization, discriminative learning rates (lower LR for pretrained layers), weighted MSE loss, and staged unfreezing of pretrained layers.

## Directory Structure

```
code/gene_level_model/
├── shared/                    # Core components
│   ├── __init__.py
│   ├── encoding.py            # Nucleotide one-hot encoding
│   ├── dataset.py             # GeneLevelDataset (TSV/pickle → PyTorch)
│   └── model.py               # GeneRepressionModelV4, MultiHeadSpatialAttention
└── training/                  # Training & evaluation scripts
    ├── __init__.py
    ├── train.py               # Training with early stopping
    └── evaluate.py            # Evaluation with competitor comparison

analysis/gene_level_model/
├── train.sh                   # Example training command
└── evaluate.sh                # Example evaluation command
```

## Dependencies

- The pretrained CNN (`PairwiseOneHotCNN`) is imported from `code/pairwise_binding_site_model/shared/models.py`. Path resolution is handled automatically via `sys.path` in the training scripts.
- Standard dependencies: `torch`, `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `scikit-learn`

## Training

Requires either a pretrained binding site model checkpoint or `--no_pretrain` for random initialization.

```bash
python code/gene_level_model/training/train.py \
    --pretrained_checkpoint path/to/binding_site_model.pt \
    --train_file path/to/training_data.pkl \
    --gene_col utr3 \
    --mirna_col miRNA \
    --label_col log2fc \
    --num_heads 8 \
    --max_gene_length 3000 \
    --attention_hidden_dim 128 \
    --batch_size 128 \
    --learning_rate 0.001 \
    --num_epochs 300 \
    --patience 25 \
    --use_layer_norm \
    --use_discriminative_lr \
    --pretrained_lr_factor 0.01 \
    --freeze_encoder \
    --unfreeze_epoch 30 \
    --dropout_rate 0.3 \
    --output_dir gene_model_outputs
```

### Key training options

| Flag | Description |
|---|---|
| `--freeze_encoder` / `--unfreeze_encoder` | Freeze pretrained conv layers (default: frozen) |
| `--unfreeze_epoch N` | Unfreeze encoder at epoch N for staged fine-tuning |
| `--use_discriminative_lr` | Lower LR for pretrained layers, full LR for new layers |
| `--pretrained_lr_factor` | LR multiplier for pretrained layers (e.g., 0.01 = 100× lower) |
| `--use_layer_norm` | Add Layer Normalization for training stability |
| `--use_weighted_loss` | Upweight samples with strong repression |
| `--no_pretrain` | Train from random initialization (no transfer learning) |
| `--gradient_accumulation_steps` | Accumulate gradients for larger effective batch size |
| `--early_stop_metric` | Early stopping on `pearson` or `spearman` correlation |

### Outputs

Training produces four files in `--output_dir`:
- `gene_level_model_<timestamp>.pt` — model checkpoint (includes architecture params for reproducible loading)
- `training_history_<timestamp>.json` — per-epoch metrics
- `config_<timestamp>.json` — full configuration and results
- `gene_level_training_<timestamp>.png` — training curves

## Evaluation

```bash
python code/gene_level_model/training/evaluate.py \
    --model_checkpoint path/to/gene_level_model.pt \
    --test_file path/to/test_data.pkl \
    --gene_col sequence \
    --mirna_col miRNA_seq \
    --label_col fold_change \
    --competitor_cols "weighted context++ score" "context++ score" \
    --competitor_names "TargetScan Weighted" "TargetScan Raw" \
    --model_name "Our Model" \
    --common_samples_only \
    --fill_empty_preds_with_zero \
    --repression_only all \
    --output_dir test_results
```

### Key evaluation options

| Flag | Description |
|---|---|
| `--competitor_cols` / `--competitor_names` | Compare against prediction columns in test data |
| `--repression_only all` | Zero out positive values before evaluation (focus on downregulation) |
| `--common_samples_only` | Evaluate only on samples where all methods have predictions |
| `--fill_empty_preds_with_zero` | Treat missing competitor predictions as no effect |
| `--calibration_file` | Apply linear calibration from a JSON file |

### Outputs

Evaluation produces in `--output_dir`:
- `test_results_<timestamp>.json` — metrics for all methods
- `predictions_<timestamp>.csv` — per-sample predictions
- Plots: scatter comparisons, metrics bar charts, residuals, error distributions, attention heatmaps, per-head attention patterns, and head weight visualizations

## Prerequisites

Set up the Python environment using either conda or pip:

```bash
conda env create -f environment.yaml
conda activate <env_name>
```

or

```bash
pip install -r requirements.txt
```

Then download the required datasets:

```bash
bash analysis/gene_level_model/download_data.sh
```

## Data Format

Input data should be a TSV or pickle file with at minimum:
- A column with gene/3'UTR sequences (can be thousands of nucleotides)
- A column with miRNA sequences (typically 20–28 nt)
- A column with fold change values

Model architecture parameters (mirna_length, filter sizes, etc.) are saved in the checkpoint and automatically restored during evaluation.