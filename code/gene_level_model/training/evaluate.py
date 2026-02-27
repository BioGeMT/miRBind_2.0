"""
Test script for gene-level repression model (V4).
Supports multiple competitor comparisons and repression-only evaluation.
"""
import argparse
import os
import sys

# Add gene_level_model/ to path (for `from shared import ...`)
_GENE_MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _GENE_MODEL_DIR)

# Add pairwise_binding_site_model/shared/ to path (for `from models import PairwiseOneHotCNN`)
_BINDING_SITE_DIR = os.path.join(os.path.dirname(_GENE_MODEL_DIR), 'pairwise_binding_site_model', 'shared')
sys.path.insert(0, _BINDING_SITE_DIR)

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
from datetime import datetime
from torch.utils.data import DataLoader

from models import PairwiseOneHotCNN
from shared import GeneLevelDataset, GeneRepressionModelV4


# Color palette for multiple competitors
COLORS = ['blue', 'green', 'orange', 'red', 'purple', 'brown', 'pink', 'gray']


def evaluate_predictions(y_true, y_pred, method_name="Model"):
    """Calculate comprehensive evaluation metrics"""
    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            'method': method_name,
            'pearson_r': float('nan'),
            'pearson_p': float('nan'),
            'spearman_r': float('nan'),
            'spearman_p': float('nan'),
            'mse': float('nan'),
            'rmse': float('nan'),
            'mae': float('nan'),
            'r2': float('nan'),
            'n_samples': 0
        }

    pearson_r, pearson_p = pearsonr(y_true, y_pred)
    spearman_r, spearman_p = spearmanr(y_true, y_pred)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        'method': method_name,
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'n_samples': int(len(y_true))
    }


def apply_repression_only_transform(y_true, y_pred):
    """
    Transform data to focus on repression only.
    Positive values (upregulation) are set to zero.
    """
    y_true_transformed = np.clip(y_true, None, 0)
    y_pred_transformed = np.clip(y_pred, None, 0)
    return y_true_transformed, y_pred_transformed


def infer_num_heads_from_state_dict(state_dict):
    """Infer number of attention heads from model state_dict."""
    head_indices = []
    for key in state_dict.keys():
        if 'multi_head_attention.attention_heads.' in key:
            parts = key.split('.')
            try:
                idx = int(parts[2])
                head_indices.append(idx)
            except (ValueError, IndexError):
                continue

    if head_indices:
        return max(head_indices) + 1
    return None


def infer_layer_norm_from_state_dict(state_dict):
    """Detect if model was trained with layer normalization by checking state_dict keys."""
    layer_norm_indicators = [
        'conv_layer_norm.weight',
        'conv_layer_norm.bias',
        'multi_head_attention.layer_norm.weight',
        'multi_head_attention.output_layer_norm.weight',
    ]

    for key in state_dict.keys():
        for indicator in layer_norm_indicators:
            if indicator in key:
                return True

    fc_keys = [k for k in state_dict.keys() if k.startswith('fc.')]
    if len(fc_keys) > 6:
        return True

    return False


def load_checkpoint(checkpoint_path, device):
    """Load model checkpoint with error handling."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)

    if 'model_state_dict' not in checkpoint:
        if isinstance(checkpoint, dict) and any('weight' in k for k in checkpoint.keys()):
            print("Warning: Checkpoint appears to be a raw state_dict, wrapping it")
            checkpoint = {'model_state_dict': checkpoint}
        else:
            print("Error: Checkpoint missing 'model_state_dict' key")
            print(f"Available keys: {list(checkpoint.keys())}")
            sys.exit(1)

    return checkpoint


def load_test_data(test_file):
    """Load test dataframe from file."""
    if test_file.endswith('.pkl') or test_file.endswith('.pickle'):
        import pickle
        with open(test_file, 'rb') as f:
            return pickle.load(f)
    elif test_file.endswith('.csv'):
        return pd.read_csv(test_file)
    else:
        return pd.read_csv(test_file, sep="\t")


def build_model(checkpoint, gene_params, pretrained_params, device):
    """
    Build V4 model from checkpoint and parameters.

    Args:
        checkpoint: loaded checkpoint dict
        gene_params: gene model parameters
        pretrained_params: pretrained CNN parameters
        device: torch device

    Returns:
        nn.Module: constructed model
    """
    # Determine num_heads
    num_heads_from_params = gene_params.get('num_heads')
    num_heads_from_state = infer_num_heads_from_state_dict(checkpoint['model_state_dict'])

    if num_heads_from_params:
        num_heads = num_heads_from_params
        print(f"Using num_heads from checkpoint params: {num_heads}")
    elif num_heads_from_state:
        num_heads = num_heads_from_state
        print(f"Inferred num_heads from model weights: {num_heads}")
    else:
        num_heads = 4
        print(f"Warning: Could not determine num_heads, defaulting to: {num_heads}")

    # Determine use_layer_norm
    use_layer_norm_from_params = gene_params.get('use_layer_norm')
    use_layer_norm_from_state = infer_layer_norm_from_state_dict(checkpoint['model_state_dict'])

    if use_layer_norm_from_params is not None:
        use_layer_norm = use_layer_norm_from_params
        print(f"Using use_layer_norm from checkpoint params: {use_layer_norm}")
    elif use_layer_norm_from_state:
        use_layer_norm = True
        print(f"Inferred use_layer_norm from model weights: {use_layer_norm}")
    else:
        use_layer_norm = False
        print(f"No layer norm detected, using: {use_layer_norm}")

    dropout_rate = gene_params.get('dropout_rate',
                                   pretrained_params.get('dropout_rate', 0.1))
    print(f"Using dropout_rate: {dropout_rate}")

    # Create pretrained CNN
    pretrained_cnn = PairwiseOneHotCNN(**pretrained_params)

    model = GeneRepressionModelV4(
        pretrained_cnn=pretrained_cnn,
        mirna_length=gene_params['mirna_length'],
        freeze_encoder=True,
        attention_hidden_dim=gene_params['attention_hidden_dim'],
        num_heads=num_heads,
        num_pairs=gene_params['num_pairs'],
        use_layer_norm=use_layer_norm,
        dropout_rate=dropout_rate
    )

    return model.to(device)


# ==================== PLOTTING FUNCTIONS ====================

def plot_predictions_vs_actual(y_true, y_pred, method_name, output_path, color='blue'):
    """Create scatter plot of predictions vs actual values"""
    plt.figure(figsize=(8, 8))

    plt.scatter(y_true, y_pred, alpha=0.5, s=20, c=color, edgecolors='none')

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')

    pearson_r, _ = pearsonr(y_true, y_pred)
    spearman_r, _ = spearmanr(y_true, y_pred)

    plt.xlabel('Actual Fold Change', fontsize=12)
    plt.ylabel('Predicted Fold Change', fontsize=12)
    plt.title(f'{method_name}\nPearson r = {pearson_r:.3f}, Spearman ρ = {spearman_r:.3f}', fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")


def plot_multi_comparison(y_true, predictions_dict, output_path, ncols=3, title_suffix=""):
    """Create comparison plots for model and multiple competitors."""
    n_methods = len(predictions_dict)
    nrows = (n_methods + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    axes = np.array(axes).flatten() if n_methods > 1 else [axes]

    for idx, (method_name, (y_pred, color)) in enumerate(predictions_dict.items()):
        ax = axes[idx]

        valid_mask = ~np.isnan(y_pred)
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]

        ax.scatter(y_true_valid, y_pred_valid, alpha=0.5, s=20, c=color, edgecolors='none')

        min_val = min(y_true_valid.min(), y_pred_valid.min())
        max_val = max(y_true_valid.max(), y_pred_valid.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

        pearson_r, _ = pearsonr(y_true_valid, y_pred_valid)
        spearman_r, _ = spearmanr(y_true_valid, y_pred_valid)

        ax.set_xlabel('Actual Fold Change', fontsize=11)
        ax.set_ylabel('Predicted', fontsize=11)
        ax.set_title(f'{method_name}\nPearson r = {pearson_r:.3f}, Spearman ρ = {spearman_r:.3f}', fontsize=11)
        ax.grid(True, alpha=0.3)

    for idx in range(n_methods, len(axes)):
        axes[idx].set_visible(False)

    if title_suffix:
        fig.suptitle(f'Predictions vs Actual {title_suffix}', fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved multi-comparison plot: {output_path}")


def plot_metrics_comparison_multi(metrics_list, output_path, title_suffix=""):
    """Create bar chart comparing metrics across multiple methods."""
    metrics_to_plot = ['pearson_r', 'spearman_r', 'r2']
    metric_names = ['Pearson r', 'Spearman ρ', 'R²']

    n_methods = len(metrics_list)
    n_metrics = len(metrics_to_plot)

    x = np.arange(n_metrics)
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(max(10, 2*n_methods), 6))

    for i, metrics in enumerate(metrics_list):
        values = [metrics[m] for m in metrics_to_plot]
        offset = (i - n_methods/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=metrics['method'],
                     color=COLORS[i % len(COLORS)], alpha=0.7)

        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8, rotation=45)

    ax.set_ylabel('Score', fontsize=12)
    title = 'Performance Comparison'
    if title_suffix:
        title += f' {title_suffix}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved metrics comparison: {output_path}")


def plot_residuals_multi(y_true, predictions_dict, output_path, ncols=3, title_suffix=""):
    """Plot residuals for multiple methods"""
    n_methods = len(predictions_dict)
    nrows = (n_methods + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    axes = np.array(axes).flatten() if n_methods > 1 else [axes]

    for idx, (method_name, (y_pred, color)) in enumerate(predictions_dict.items()):
        ax = axes[idx]

        valid_mask = ~np.isnan(y_pred)
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]

        residuals = y_true_valid - y_pred_valid

        ax.scatter(y_pred_valid, residuals, alpha=0.5, s=20, c=color, edgecolors='none')
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('Residuals', fontsize=11)
        ax.set_title(f'{method_name}\nMAE = {np.abs(residuals).mean():.4f}', fontsize=11)
        ax.grid(True, alpha=0.3)

    for idx in range(n_methods, len(axes)):
        axes[idx].set_visible(False)

    if title_suffix:
        fig.suptitle(f'Residuals {title_suffix}', fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved residuals plot: {output_path}")


def plot_error_distribution_multi(y_true, predictions_dict, output_path, title_suffix=""):
    """Plot error distribution for multiple methods"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    for method_name, (y_pred, color) in predictions_dict.items():
        valid_mask = ~np.isnan(y_pred)
        errors = np.abs(y_true[valid_mask] - y_pred[valid_mask])
        ax.hist(errors, bins=50, alpha=0.5, label=method_name, color=color, density=True)

    ax.set_xlabel('Absolute Error', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Error Distribution', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for method_name, (y_pred, color) in predictions_dict.items():
        valid_mask = ~np.isnan(y_pred)
        errors = np.abs(y_true[valid_mask] - y_pred[valid_mask])
        ax.hist(errors, bins=50, alpha=0.7, label=method_name, color=color,
                density=True, cumulative=True, histtype='step', linewidth=2)

    ax.set_xlabel('Absolute Error', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title('Cumulative Error Distribution', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    if title_suffix:
        fig.suptitle(f'Error Analysis {title_suffix}', fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved error distribution plot: {output_path}")


def plot_attention_heatmap(attention_weights, output_path, n_samples=20):
    """Visualize attention weights for sample genes"""
    attention_subset = attention_weights[:n_samples]

    plt.figure(figsize=(14, 8))
    sns.heatmap(attention_subset, cmap='YlOrRd', cbar_kws={'label': 'Attention Weight'})
    plt.xlabel('Window Position', fontsize=12)
    plt.ylabel('Gene Sample', fontsize=12)
    plt.title(f'Attention Weights Across Gene Windows (First {n_samples} samples)', fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved attention heatmap: {output_path}")


def plot_multihead_attention(head_attention, head_weights, output_path, n_samples=10):
    """Visualize attention from each head separately."""
    num_heads = head_attention.shape[1]
    n_samples = min(n_samples, head_attention.shape[0])

    fig, axes = plt.subplots(num_heads + 1, 1, figsize=(14, 3 * (num_heads + 1)))

    for h in range(num_heads):
        ax = axes[h]
        head_data = head_attention[:n_samples, h, :]

        sns.heatmap(head_data, cmap='YlOrRd', ax=ax,
                   cbar_kws={'label': 'Attention'})
        ax.set_xlabel('Position', fontsize=10)
        ax.set_ylabel('Sample', fontsize=10)
        ax.set_title(f'Head {h+1} (weight: {head_weights[h]:.3f})', fontsize=11)

    ax = axes[num_heads]
    combined = np.zeros((n_samples, head_attention.shape[2]))
    for h in range(num_heads):
        combined += head_attention[:n_samples, h, :] * head_weights[h]

    sns.heatmap(combined, cmap='YlOrRd', ax=ax,
               cbar_kws={'label': 'Attention'})
    ax.set_xlabel('Position', fontsize=10)
    ax.set_ylabel('Sample', fontsize=10)
    ax.set_title('Combined Attention (weighted sum of all heads)', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved multi-head attention heatmap: {output_path}")


def plot_head_weights_comparison(head_weights, output_path):
    """Bar plot showing learned importance of each attention head."""
    num_heads = len(head_weights)

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(range(num_heads), head_weights, color=plt.cm.Set2(np.linspace(0, 1, num_heads)))

    ax.set_xlabel('Attention Head', fontsize=12)
    ax.set_ylabel('Learned Weight', fontsize=12)
    ax.set_title('Learned Importance of Each Attention Head', fontsize=13)
    ax.set_xticks(range(num_heads))
    ax.set_xticklabels([f'Head {i+1}' for i in range(num_heads)])

    for bar, weight in zip(bars, head_weights):
        ax.annotate(f'{weight:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=10)

    ax.set_ylim(0, max(head_weights) * 1.15)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved head weights plot: {output_path}")


def plot_head_attention_distribution(head_attention, head_weights, output_path):
    """Show distribution of attention patterns for each head."""
    num_heads = head_attention.shape[1]

    fig, axes = plt.subplots(1, num_heads, figsize=(4 * num_heads, 4))
    if num_heads == 1:
        axes = [axes]

    for h in range(num_heads):
        ax = axes[h]

        avg_attention = head_attention[:, h, :].mean(axis=0)
        positions = np.arange(len(avg_attention))

        ax.fill_between(positions, avg_attention, alpha=0.5, color=f'C{h}')
        ax.plot(positions, avg_attention, color=f'C{h}', linewidth=1.5)

        ax.set_xlabel('Position', fontsize=10)
        ax.set_ylabel('Avg Attention', fontsize=10)
        ax.set_title(f'Head {h+1}\n(weight: {head_weights[h]:.3f})', fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Average Attention Pattern by Head', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved head attention distribution: {output_path}")


def print_metrics(metrics, indent=2):
    """Pretty print metrics"""
    prefix = " " * indent
    print(f"{prefix}Pearson correlation:  r = {metrics['pearson_r']:.4f} (p = {metrics['pearson_p']:.2e})")
    print(f"{prefix}Spearman correlation: ρ = {metrics['spearman_r']:.4f} (p = {metrics['spearman_p']:.2e})")
    print(f"{prefix}R² score:             {metrics['r2']:.4f}")
    print(f"{prefix}RMSE:                 {metrics['rmse']:.4f}")
    print(f"{prefix}MAE:                  {metrics['mae']:.4f}")
    print(f"{prefix}N samples:            {metrics['n_samples']}")


# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(description='Test gene-level repression model (V4) and compare with competitors')

    # Required arguments
    parser.add_argument('--model_checkpoint', type=str, required=True, help='Path to trained gene-level model')
    parser.add_argument('--test_file', type=str, required=True, help='Path to test dataset (.pkl)')

    # Column names
    parser.add_argument('--gene_col', type=str, default='gene', help='Column name for gene sequences')
    parser.add_argument('--mirna_col', type=str, default='noncodingRNA', help='Column name for miRNA sequences')
    parser.add_argument('--label_col', type=str, default='fold_change', help='Column name for fold change labels')

    # Multiple competitors support
    parser.add_argument('--competitor_cols', type=str, nargs='+',
                       default=['weighted context++ score'],
                       help='Column names for competitor predictions (space-separated)')
    parser.add_argument('--competitor_names', type=str, nargs='+',
                       default=None,
                       help='Display names for competitors (space-separated, same order as competitor_cols)')

    # Repression-only evaluation
    parser.add_argument('--repression_only', type=str, nargs='*', default=None,
                       help='Methods to evaluate in repression-only mode (positive values zeroed). '
                            'Use "all" to apply to all methods, or list specific method names. '
                            'If flag is present without arguments, applies to all.')

    # Sample filtering options
    parser.add_argument('--common_samples_only', action='store_true',
                       help='Evaluate all methods only on intersection of valid samples')
    parser.add_argument('--fill_empty_preds_with_zero', action='store_true',
                       help='Fill NaN predictions with zero (treat no-prediction as no-effect)')
    parser.add_argument('--calibration_file', type=str, default=None,
                       help='Path to calibration parameters JSON file')

    # Model parameters
    parser.add_argument('--max_gene_length', type=int, default=2000, help='Maximum gene length')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')

    # Output
    parser.add_argument('--output_dir', type=str, default='test_results', help='Output directory for results')
    parser.add_argument('--model_name', type=str, default='Our Model', help='Name for your model in plots')

    args = parser.parse_args()

    # Handle repression_only argument
    if args.repression_only is not None:
        if len(args.repression_only) == 0:
            args.repression_only = ['all']

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set competitor names if not provided
    if args.competitor_names is None:
        args.competitor_names = args.competitor_cols
    elif len(args.competitor_names) != len(args.competitor_cols):
        print("Warning: competitor_names length doesn't match competitor_cols. Using column names.")
        args.competitor_names = args.competitor_cols

    # Load model checkpoint
    print(f"\nLoading model from: {args.model_checkpoint}")
    checkpoint = load_checkpoint(args.model_checkpoint, device)

    # Get model parameters from checkpoint
    if 'gene_model_params' in checkpoint:
        gene_params = checkpoint['gene_model_params']
    else:
        print("Warning: gene_model_params not in checkpoint, using defaults")
        gene_params = {
            'mirna_length': 25,
            'target_window': 50,
            'attention_hidden_dim': 32,
            'num_pairs': 16
        }

    if 'pretrained_params' in checkpoint:
        pretrained_params = checkpoint['pretrained_params']
    else:
        print("Warning: pretrained_params not in checkpoint, using gene_params as fallback")
        pretrained_params = gene_params.copy()

    print("\nModel architecture parameters:")
    for key, value in gene_params.items():
        print(f"  {key}: {value}")

    # Print training info
    was_pretrained = gene_params.get('pretrained', True)
    init_method = gene_params.get('init_method', None)

    if was_pretrained:
        print("\n✓ Model trained WITH pretraining on binding site data")
    else:
        print(f"\n✗ Model trained WITHOUT pretraining (random {init_method} initialization)")

    use_weighted_loss = gene_params.get('use_weighted_loss', False)
    if use_weighted_loss:
        threshold = gene_params.get('weighted_loss_threshold', 0.1)
        weight = gene_params.get('weighted_loss_weight', 3.0)
        print(f"✓ Trained with Weighted MSE Loss (threshold={threshold}, weight={weight})")
    else:
        print("○ Trained with standard MSE Loss")

    use_layer_norm = gene_params.get('use_layer_norm', False)
    if use_layer_norm:
        print("✓ Trained with Layer Normalization")
    else:
        print("○ Trained without Layer Normalization")

    use_discriminative_lr = gene_params.get('use_discriminative_lr', False)
    if use_discriminative_lr:
        pretrained_lr_factor = gene_params.get('pretrained_lr_factor', 0.1)
        print(f"✓ Trained with Discriminative Learning Rates (pretrained factor={pretrained_lr_factor})")
    else:
        print("○ Trained with uniform learning rate")

    # Build and load model
    model = build_model(checkpoint, gene_params, pretrained_params, device)

    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        print(f"Error loading model state dict: {e}")
        print("\nThis may indicate a mismatch between the checkpoint and model architecture.")
        sys.exit(1)

    model.eval()
    print("Model loaded successfully")

    # Setup pair mapping
    nucleotide_pairs = [
        ('A', 'A'), ('A', 'T'), ('A', 'C'), ('A', 'G'),
        ('T', 'A'), ('T', 'T'), ('T', 'C'), ('T', 'G'),
        ('C', 'A'), ('C', 'T'), ('C', 'C'), ('C', 'G'),
        ('G', 'A'), ('G', 'T'), ('G', 'C'), ('G', 'G')
    ]
    pair_to_index = {pair: i for i, pair in enumerate(nucleotide_pairs)}
    pair_to_index[('N', 'N')] = len(pair_to_index)

    # Load test dataframe
    print(f"\nLoading test data from: {args.test_file}")
    test_df = load_test_data(args.test_file)

    # Create dataset
    test_dataset = GeneLevelDataset(
        file_path=args.test_file,
        gene_col=args.gene_col,
        mirna_col=args.mirna_col,
        label_col=args.label_col,
        max_gene_length=args.max_gene_length,
        mirna_length=gene_params['mirna_length'],
        target_window=gene_params.get('target_window', 50),
        pair_to_index=pair_to_index,
        num_pairs=gene_params['num_pairs']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=GeneLevelDataset.collate_fn
    )

    print(f"Test dataset size: {len(test_dataset)}")

    # Load competitor predictions
    print(f"\nLoading competitor predictions...")
    competitor_predictions = {}
    for col, name in zip(args.competitor_cols, args.competitor_names):
        if col not in test_df.columns:
            print(f"  Warning: Column '{col}' not found in dataset!")
            print(f"  Available columns: {list(test_df.columns)[:20]}...")
        else:
            preds = test_df[col].values.astype(float)
            competitor_predictions[name] = preds
            n_valid = (~np.isnan(preds)).sum()
            n_nan = np.isnan(preds).sum()
            print(f"  Loaded '{col}' as '{name}': {n_valid} valid, {n_nan} NaN")

    # Run inference
    print("\nRunning inference...")
    all_predictions = []
    all_labels = []
    all_attention = []
    all_head_attention = []
    head_weights = None

    with torch.no_grad():
        for batch in test_loader:
            gene_onehot = batch['gene_onehot'].to(device)
            mirna_onehot = batch['mirna_onehot'].to(device)
            gene_lengths = batch['gene_length'].to(device)
            labels = batch['label']

            outputs, attention = model(gene_onehot, mirna_onehot, gene_lengths)

            all_predictions.extend(outputs.squeeze().cpu().numpy())
            all_labels.extend(labels.numpy())
            all_attention.append(attention.cpu().numpy())

            # Get per-head attention
            if hasattr(model, 'get_attention_details'):
                _, attention_details = model.get_attention_details(gene_onehot, mirna_onehot, gene_lengths)
                all_head_attention.append(attention_details['head_attention'].cpu().numpy())
                if head_weights is None:
                    head_weights = attention_details['head_weights'].cpu().numpy()

    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred_model = np.array(all_predictions)
    attention_weights = np.vstack(all_attention)

    if len(all_head_attention) > 0:
        head_attention = np.vstack(all_head_attention)
        print(f"Captured attention from {head_attention.shape[1]} heads")
    else:
        head_attention = None

    print(f"Predictions completed: {len(y_pred_model)} samples")

    # ==================== HANDLE SAMPLE FILTERING OPTIONS ====================

    if args.fill_empty_preds_with_zero:
        print("\n" + "-"*70)
        print("Filling NaN predictions with zero (--fill_empty_preds_with_zero)")
        print("-"*70)
        for name, preds in competitor_predictions.items():
            n_nan = np.isnan(preds).sum()
            if n_nan > 0:
                competitor_predictions[name] = np.nan_to_num(preds, nan=0.0)
                print(f"  {name}: filled {n_nan} NaN values with 0")

    common_mask = None
    if args.common_samples_only:
        print("\n" + "-"*70)
        print("Filtering to common samples only (--common_samples_only)")
        print("-"*70)

        common_mask = np.ones(len(y_true), dtype=bool)

        model_valid = ~np.isnan(y_pred_model)
        common_mask &= model_valid
        print(f"  {args.model_name}: {model_valid.sum()} valid samples")

        for name, preds in competitor_predictions.items():
            valid = ~np.isnan(preds)
            common_mask &= valid
            print(f"  {name}: {valid.sum()} valid samples")

        print(f"  Intersection: {common_mask.sum()} samples ({common_mask.sum()/len(common_mask)*100:.1f}%)")

        y_true = y_true[common_mask]
        y_pred_model = y_pred_model[common_mask]
        attention_weights = attention_weights[common_mask]

        if head_attention is not None:
            head_attention = head_attention[common_mask]

        for name in competitor_predictions:
            competitor_predictions[name] = competitor_predictions[name][common_mask]

        print(f"  Filtered to {len(y_true)} common samples")

    # ==================== PREDICTION RANGE DIAGNOSTICS ====================
    print("\n" + "="*70)
    print("PREDICTION RANGE DIAGNOSTICS")
    print("="*70)

    print(f"\nActual fold change:")
    print(f"  min={y_true.min():.4f}, max={y_true.max():.4f}, mean={y_true.mean():.4f}, std={y_true.std():.4f}")

    print(f"\n{args.model_name} predictions:")
    print(f"  min={y_pred_model.min():.4f}, max={y_pred_model.max():.4f}, mean={y_pred_model.mean():.4f}, std={y_pred_model.std():.4f}")

    for name, preds in competitor_predictions.items():
        valid_preds = preds[~np.isnan(preds)]
        print(f"\n{name} predictions:")
        print(f"  min={valid_preds.min():.4f}, max={valid_preds.max():.4f}, mean={valid_preds.mean():.4f}, std={valid_preds.std():.4f}")

    # ==================== OPTIONAL CALIBRATION ====================
    calibration_applied = False
    if args.calibration_file:
        print("\n" + "="*70)
        print("APPLYING CALIBRATION FROM FILE")
        print("="*70)

        try:
            with open(args.calibration_file, 'r') as f:
                calib_params = json.load(f)

            slope = calib_params['calibration']['slope']
            intercept = calib_params['calibration']['intercept']

            print(f"\nLoaded from: {args.calibration_file}")
            print(f"Calibration source: {calib_params.get('calibration_source', 'unknown')}")
            print(f"Calibration parameters:")
            print(f"  slope (a):     {slope:.4f}")
            print(f"  intercept (b): {intercept:.4f}")

            y_pred_model = slope * y_pred_model + intercept
            calibration_applied = True

            print(f"\nAfter calibration:")
            print(f"  min={y_pred_model.min():.4f}, max={y_pred_model.max():.4f}, mean={y_pred_model.mean():.4f}, std={y_pred_model.std():.4f}")
        except FileNotFoundError:
            print(f"Warning: Calibration file not found: {args.calibration_file}")
            print("Proceeding without calibration.")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Error reading calibration file: {e}")
            print("Proceeding without calibration.")

    # ==================== STANDARD EVALUATION ====================
    print("\n" + "="*70)
    print("STANDARD EVALUATION RESULTS")
    print("="*70)

    metrics_model = evaluate_predictions(y_true, y_pred_model, args.model_name)
    print(f"\n{args.model_name}:")
    print_metrics(metrics_model)

    all_metrics = [metrics_model]
    competitor_metrics = {}

    for name, preds in competitor_predictions.items():
        if args.common_samples_only or args.fill_empty_preds_with_zero:
            metrics = evaluate_predictions(y_true, preds, name)
        else:
            valid_mask = ~np.isnan(preds)
            metrics = evaluate_predictions(y_true[valid_mask], preds[valid_mask], name)

        all_metrics.append(metrics)
        competitor_metrics[name] = metrics
        print(f"\n{name}:")
        print_metrics(metrics)

    if len(competitor_predictions) > 0:
        print(f"\n{'-'*70}")
        print("PERFORMANCE COMPARISON (vs each competitor):")
        print(f"{'-'*70}")

        for name, comp_metrics in competitor_metrics.items():
            if comp_metrics['pearson_r'] != 0:
                improvement_pearson = (metrics_model['pearson_r'] - comp_metrics['pearson_r']) / abs(comp_metrics['pearson_r']) * 100
            else:
                improvement_pearson = float('inf')

            if comp_metrics['spearman_r'] != 0:
                improvement_spearman = (metrics_model['spearman_r'] - comp_metrics['spearman_r']) / abs(comp_metrics['spearman_r']) * 100
            else:
                improvement_spearman = float('inf')

            print(f"\n  vs {name}:")
            print(f"    Pearson r improvement:  {improvement_pearson:+.2f}%")
            print(f"    Spearman ρ improvement: {improvement_spearman:+.2f}%")

    # ==================== REPRESSION-ONLY EVALUATION ====================
    repression_metrics = {}
    all_metrics_repression = []
    methods_to_transform = set()

    if args.repression_only is not None:
        print("\n" + "="*70)
        print("REPRESSION-ONLY EVALUATION (positive values zeroed)")
        print("="*70)

        apply_to_all = 'all' in args.repression_only

        if apply_to_all:
            methods_to_transform.add(args.model_name)
            methods_to_transform.update(competitor_predictions.keys())
        else:
            methods_to_transform = set(args.repression_only)

        print(f"\nApplying repression-only transform to: {methods_to_transform}")

        if args.model_name in methods_to_transform:
            y_true_rep, y_pred_rep = apply_repression_only_transform(y_true, y_pred_model)
        else:
            y_true_rep, y_pred_rep = y_true, y_pred_model

        metrics_model_rep = evaluate_predictions(y_true_rep, y_pred_rep, f"{args.model_name} (repression-only)")
        repression_metrics[args.model_name] = metrics_model_rep
        all_metrics_repression.append(metrics_model_rep)
        print(f"\n{args.model_name} (repression-only):")
        print_metrics(metrics_model_rep)

        for name, preds in competitor_predictions.items():
            if args.common_samples_only or args.fill_empty_preds_with_zero:
                y_true_valid = y_true
                preds_valid = preds
            else:
                valid_mask = ~np.isnan(preds)
                y_true_valid = y_true[valid_mask]
                preds_valid = preds[valid_mask]

            if name in methods_to_transform:
                y_true_rep, preds_rep = apply_repression_only_transform(y_true_valid, preds_valid)
            else:
                y_true_rep, preds_rep = y_true_valid, preds_valid

            metrics = evaluate_predictions(y_true_rep, preds_rep, f"{name} (repression-only)")
            repression_metrics[name] = metrics
            all_metrics_repression.append(metrics)
            print(f"\n{name} (repression-only):")
            print_metrics(metrics)

        if len(competitor_predictions) > 0:
            print(f"\n{'-'*70}")
            print("REPRESSION-ONLY COMPARISON:")
            print(f"{'-'*70}")

            for name in competitor_predictions.keys():
                comp_metrics = repression_metrics[name]
                model_metrics = repression_metrics[args.model_name]

                if comp_metrics['pearson_r'] != 0:
                    improvement_pearson = (model_metrics['pearson_r'] - comp_metrics['pearson_r']) / abs(comp_metrics['pearson_r']) * 100
                else:
                    improvement_pearson = float('inf')

                if comp_metrics['spearman_r'] != 0:
                    improvement_spearman = (model_metrics['spearman_r'] - comp_metrics['spearman_r']) / abs(comp_metrics['spearman_r']) * 100
                else:
                    improvement_spearman = float('inf')

                print(f"\n  vs {name}:")
                print(f"    Pearson r improvement:  {improvement_pearson:+.2f}%")
                print(f"    Spearman ρ improvement: {improvement_spearman:+.2f}%")

    # ==================== SAVE RESULTS ====================
    results = {
        'timestamp': timestamp,
        'test_file': args.test_file,
        'model_checkpoint': args.model_checkpoint,
        'n_test_samples': len(y_true),
        'model_name': args.model_name,
        'model_version': 'v4',
        'model_config': {
            'use_layer_norm': gene_params.get('use_layer_norm', False),
            'use_discriminative_lr': gene_params.get('use_discriminative_lr', False),
            'pretrained_lr_factor': gene_params.get('pretrained_lr_factor'),
            'was_pretrained': gene_params.get('pretrained', True)
        },
        'competitor_cols': args.competitor_cols,
        'competitor_names': args.competitor_names,
        'options': {
            'common_samples_only': args.common_samples_only,
            'fill_empty_preds_with_zero': args.fill_empty_preds_with_zero,
            'repression_only': args.repression_only,
            'calibration_file': args.calibration_file,
            'calibration_applied': calibration_applied
        },
        'standard_evaluation': {
            'model': metrics_model,
            'competitors': competitor_metrics
        }
    }

    if args.repression_only is not None:
        results['repression_only_evaluation'] = {
            'methods_transformed': list(methods_to_transform),
            'metrics': repression_metrics
        }

    # Build filename suffix based on flags
    suffix_parts = []
    title_parts = []
    if args.common_samples_only:
        suffix_parts.append("common")
        title_parts.append("Common Samples")
    if args.fill_empty_preds_with_zero:
        suffix_parts.append("fillzero")
        title_parts.append("NaN→0")
    if args.repression_only is not None:
        suffix_parts.append("repronly")
        title_parts.append("Repression Only")
    if calibration_applied:
        suffix_parts.append("calibrated")
        title_parts.append("Calibrated")

    flag_suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
    title_suffix = "(" + ", ".join(title_parts) + ")" if title_parts else ""

    results_file = os.path.join(args.output_dir, f"test_results{flag_suffix}_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # ==================== GENERATE PLOTS ====================
    print("\nGenerating plots...")

    predictions_dict = {args.model_name: (y_pred_model, COLORS[0])}
    for i, (name, preds) in enumerate(competitor_predictions.items()):
        predictions_dict[name] = (preds, COLORS[(i + 1) % len(COLORS)])

    plot_multi_comparison(
        y_true, predictions_dict,
        os.path.join(args.output_dir, f"comparison_all{flag_suffix}_{timestamp}.png"),
        title_suffix=title_suffix
    )

    plot_metrics_comparison_multi(
        all_metrics,
        os.path.join(args.output_dir, f"metrics_comparison{flag_suffix}_{timestamp}.png"),
        title_suffix=title_suffix
    )

    plot_residuals_multi(
        y_true, predictions_dict,
        os.path.join(args.output_dir, f"residuals_all{flag_suffix}_{timestamp}.png"),
        title_suffix=title_suffix
    )

    plot_error_distribution_multi(
        y_true, predictions_dict,
        os.path.join(args.output_dir, f"error_distribution{flag_suffix}_{timestamp}.png"),
        title_suffix=title_suffix
    )

    plot_attention_heatmap(
        attention_weights,
        os.path.join(args.output_dir, f"attention_heatmap{flag_suffix}_{timestamp}.png"),
        n_samples=min(20, len(attention_weights))
    )

    # Multi-head attention plots
    if head_attention is not None and head_weights is not None:
        print("\nGenerating multi-head attention plots...")

        plot_multihead_attention(
            head_attention, head_weights,
            os.path.join(args.output_dir, f"multihead_attention{flag_suffix}_{timestamp}.png"),
            n_samples=min(10, len(head_attention))
        )

        plot_head_weights_comparison(
            head_weights,
            os.path.join(args.output_dir, f"head_weights{flag_suffix}_{timestamp}.png")
        )

        plot_head_attention_distribution(
            head_attention, head_weights,
            os.path.join(args.output_dir, f"head_attention_distribution{flag_suffix}_{timestamp}.png")
        )

    # Repression-only plots
    if args.repression_only is not None:
        predictions_dict_rep = {}

        if args.model_name in methods_to_transform:
            y_true_rep, y_pred_rep = apply_repression_only_transform(y_true, y_pred_model)
        else:
            y_true_rep, y_pred_rep = y_true, y_pred_model
        predictions_dict_rep[f"{args.model_name}"] = (y_pred_rep, COLORS[0])

        for i, (name, preds) in enumerate(competitor_predictions.items()):
            valid_mask = ~np.isnan(preds)
            if name in methods_to_transform:
                _, preds_rep = apply_repression_only_transform(y_true[valid_mask], preds[valid_mask])
                preds_full = preds.copy()
                preds_full[valid_mask] = preds_rep
                preds_rep = preds_full
            else:
                preds_rep = preds
            predictions_dict_rep[f"{name}"] = (preds_rep, COLORS[(i + 1) % len(COLORS)])

        plot_multi_comparison(
            y_true_rep, predictions_dict_rep,
            os.path.join(args.output_dir, f"comparison{flag_suffix}_{timestamp}.png"),
            title_suffix=title_suffix
        )

        plot_metrics_comparison_multi(
            all_metrics_repression,
            os.path.join(args.output_dir, f"metrics{flag_suffix}_{timestamp}.png"),
            title_suffix=title_suffix
        )

    # Save predictions to CSV
    predictions_df = pd.DataFrame({
        'actual_fold_change': y_true,
        f'predicted_{args.model_name}': y_pred_model
    })

    for name, preds in competitor_predictions.items():
        predictions_df[f'predicted_{name}'] = preds

    predictions_file = os.path.join(args.output_dir, f"predictions{flag_suffix}_{timestamp}.csv")
    predictions_df.to_csv(predictions_file, index=False)
    print(f"Predictions saved to: {predictions_file}")

    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)
    print(f"\nAll results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()