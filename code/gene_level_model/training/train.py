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
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import json
from datetime import datetime

from models import PairwiseOneHotCNN
from shared import GeneLevelDataset, GeneRepressionModelV4


class WeightedMSELoss(nn.Module):
    """
    MSE Loss with higher weight for negative samples below a threshold.

    This emphasizes learning strong repression events while
    still considering all samples.
    """
    def __init__(self, threshold=-0.01, weight=3.0):
        super(WeightedMSELoss, self).__init__()
        self.threshold = threshold
        self.weight = weight

    def forward(self, predictions, targets):
        squared_errors = (predictions - targets) ** 2

        weights = torch.ones_like(targets)
        negative_mask = targets < self.threshold
        weights[negative_mask] = self.weight

        weighted_errors = squared_errors * weights
        return weighted_errors.mean()


def get_parameter_groups(model, base_lr, pretrained_lr_factor=0.1):
    """
    Create parameter groups with different learning rates for discriminative learning.

    Pretrained layers get a lower learning rate to preserve learned features,
    while new layers get the full learning rate for faster adaptation.
    """
    pretrained_params = []
    new_params = []
    pretrained_param_count = 0
    new_param_count = 0

    pretrained_layer_prefixes = [
        'pair_linear',
        'conv1', 'bn1', 'pool1', 'dropout1',
        'conv2', 'bn2', 'pool2', 'dropout2',
        'conv3', 'bn3', 'pool3', 'dropout3',
    ]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_pretrained = any(name.startswith(prefix) for prefix in pretrained_layer_prefixes)

        if is_pretrained:
            pretrained_params.append(param)
            pretrained_param_count += param.numel()
        else:
            new_params.append(param)
            new_param_count += param.numel()

    print(f"\nDiscriminative Learning Rates:")
    print(f"  Pretrained parameters: {pretrained_param_count:,} (lr={base_lr * pretrained_lr_factor:.6f})")
    print(f"  New parameters: {new_param_count:,} (lr={base_lr:.6f})")

    if pretrained_param_count == 0:
        print("  WARNING: No pretrained parameters found! Check layer naming.")
        print("  Trainable parameter names in model:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"    {name}: {param.numel():,} params")

    param_groups = []

    if pretrained_params:
        param_groups.append({
            'params': pretrained_params,
            'lr': base_lr * pretrained_lr_factor,
            'name': 'pretrained'
        })

    if new_params:
        param_groups.append({
            'params': new_params,
            'lr': base_lr,
            'name': 'new'
        })

    return param_groups


def train_epoch(model, train_loader, optimizer, criterion, device, accumulation_steps=1):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        gene_onehot = batch['gene_onehot'].to(device)
        mirna_onehot = batch['mirna_onehot'].to(device)
        gene_lengths = batch['gene_length'].to(device)
        labels = batch['label'].to(device)

        outputs, _ = model(gene_onehot, mirna_onehot, gene_lengths)

        outputs = outputs.squeeze()
        loss = criterion(outputs, labels)

        loss = loss / accumulation_steps
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    pearson_r, _ = pearsonr(all_preds, all_labels)
    spearman_r, _ = spearmanr(all_preds, all_labels)

    return total_loss / len(train_loader), pearson_r, spearman_r


def evaluate(model, val_loader, criterion, device):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_attentions = []

    with torch.no_grad():
        for batch in val_loader:
            gene_onehot = batch['gene_onehot'].to(device)
            mirna_onehot = batch['mirna_onehot'].to(device)
            gene_lengths = batch['gene_length'].to(device)
            labels = batch['label'].to(device)

            outputs, attention = model(gene_onehot, mirna_onehot, gene_lengths)

            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_attentions.append(attention.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    pearson_r, _ = pearsonr(all_preds, all_labels)
    spearman_r, _ = spearmanr(all_preds, all_labels)

    return total_loss / len(val_loader), pearson_r, spearman_r, all_preds, all_labels


def plot_training_history(history, output_dir, timestamp):
    """Plot training metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (MSE)')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()

    axes[0, 1].plot(history['train_pearson'], label='Train')
    axes[0, 1].plot(history['val_pearson'], label='Validation')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Pearson r')
    axes[0, 1].set_title('Pearson Correlation')
    axes[0, 1].legend()

    axes[1, 0].plot(history['train_spearman'], label='Train')
    axes[1, 0].plot(history['val_spearman'], label='Validation')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Spearman rho')
    axes[1, 0].set_title('Spearman Correlation')
    axes[1, 0].legend()

    if 'final_preds' in history and 'final_labels' in history:
        axes[1, 1].scatter(history['final_labels'], history['final_preds'], alpha=0.5, s=10)
        axes[1, 1].plot([min(history['final_labels']), max(history['final_labels'])],
                        [min(history['final_labels']), max(history['final_labels'])], 'r--')
        axes[1, 1].set_xlabel('Actual Fold Change')
        axes[1, 1].set_ylabel('Predicted Fold Change')
        axes[1, 1].set_title('Predictions vs Actual (Final)')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"gene_level_training_{timestamp}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return plot_path


def main():
    parser = argparse.ArgumentParser(description='Train gene-level repression model (V4) with transfer learning')

    # Data parameters
    parser.add_argument('--train_file', type=str, required=True, help='Path to gene-level training data')
    parser.add_argument('--test_file', type=str, default=None, help='Path to test data (optional)')
    parser.add_argument('--gene_col', type=str, default='gene', help='Column name for gene sequences')
    parser.add_argument('--mirna_col', type=str, default='noncodingRNA', help='Column name for miRNA sequences')
    parser.add_argument('--label_col', type=str, default='fold_change', help='Column name for fold change labels')

    # Pretrained model parameters
    parser.add_argument('--pretrained_checkpoint', type=str, default=None,
                       help='Path to pretrained binding site model (not required if --no_pretrain)')
    parser.add_argument('--freeze_encoder', action='store_true', dest='freeze_encoder',
                       help='Freeze pretrained encoder (default)')
    parser.add_argument('--unfreeze_encoder', action='store_false', dest='freeze_encoder',
                       help='Do not freeze pretrained encoder (allow fine-tuning)')
    parser.set_defaults(freeze_encoder=True)
    parser.add_argument('--unfreeze_epoch', type=int, default=-1, help='Epoch to unfreeze encoder (-1 = never)')

    # Model parameters (should match pretrained model or used for random init)
    parser.add_argument('--mirna_length', type=int, default=28, help='miRNA sequence length')
    parser.add_argument('--target_window', type=int, default=50, help='Target window size')
    parser.add_argument('--embedding_dim', type=int, default=8, help='Embedding dimension')
    parser.add_argument('--filter_sizes', type=str, default='128,64,32', help='CNN filter sizes')
    parser.add_argument('--kernel_sizes', type=str, default='6,5,4', help='CNN kernel sizes')

    # Gene-level model parameters
    parser.add_argument('--max_gene_length', type=int, default=2000, help='Maximum gene length')
    parser.add_argument('--attention_hidden_dim', type=int, default=32, help='Hidden dim for attention')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--use_layer_norm', action='store_true',
                       help='Use layer normalization in model')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Number of batches to accumulate gradients (effective_batch = batch_size * this)')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--val_fraction', type=float, default=0.1, help='Validation fraction')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--early_stop_metric', type=str, default='pearson', choices=['pearson', 'spearman'],
                       help='Metric to use for early stopping: pearson or spearman')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')

    # Discriminative Learning Rate parameters
    parser.add_argument('--use_discriminative_lr', action='store_true',
                       help='Use different learning rates for pretrained vs new layers')
    parser.add_argument('--pretrained_lr_factor', type=float, default=0.1,
                       help='LR multiplier for pretrained layers (e.g., 0.1 means 10x lower LR)')

    # Loss function parameters
    parser.add_argument('--use_weighted_loss', action='store_true',
                       help='Use weighted MSE loss (higher weight for large fold changes)')
    parser.add_argument('--weighted_loss_threshold', type=float, default=0.1,
                       help='Threshold for weighted loss (|fold_change| > threshold gets higher weight)')
    parser.add_argument('--weighted_loss_weight', type=float, default=3.0,
                       help='Weight multiplier for samples above threshold')

    parser.add_argument('--output_dir', type=str, default='gene_model_outputs', help='Output directory')

    # Option to train without pretraining
    parser.add_argument('--no_pretrain', action='store_true',
                       help='Train from random initialization (no pretraining)')
    parser.add_argument('--init_method', type=str, default='kaiming',
                       choices=['kaiming', 'xavier', 'normal', 'uniform'],
                       help='Weight initialization method when --no_pretrain is used')

    args = parser.parse_args()

    # Validate pretrained_checkpoint vs no_pretrain
    if not args.no_pretrain and args.pretrained_checkpoint is None:
        parser.error("--pretrained_checkpoint is required unless --no_pretrain is specified")

    if args.no_pretrain and args.pretrained_checkpoint is not None:
        print("Warning: --pretrained_checkpoint ignored when using --no_pretrain")

    # Discriminative LR makes no sense with frozen encoder or no pretraining
    if args.use_discriminative_lr:
        if args.no_pretrain:
            print("Warning: --use_discriminative_lr ignored when using --no_pretrain (no pretrained layers)")
            args.use_discriminative_lr = False
        elif args.freeze_encoder:
            print("Warning: --use_discriminative_lr has no effect when encoder is frozen")
            print("         Consider using --unfreeze_epoch to unfreeze later, or use --unfreeze_encoder")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Parse filter and kernel sizes
    filter_sizes = [int(x) for x in args.filter_sizes.split(',')]
    kernel_sizes = [int(x) for x in args.kernel_sizes.split(',')]

    # Setup pair mapping
    nucleotide_pairs = [
        ('A', 'A'), ('A', 'T'), ('A', 'C'), ('A', 'G'),
        ('T', 'A'), ('T', 'T'), ('T', 'C'), ('T', 'G'),
        ('C', 'A'), ('C', 'T'), ('C', 'C'), ('C', 'G'),
        ('G', 'A'), ('G', 'T'), ('G', 'C'), ('G', 'G')
    ]
    pair_to_index = {pair: i for i, pair in enumerate(nucleotide_pairs)}
    pair_to_index[('N', 'N')] = len(pair_to_index)
    num_pairs = len(nucleotide_pairs)

    # Load pretrained model or initialize randomly
    if args.no_pretrain:
        print(f"\n{'='*60}")
        print(f"Training WITHOUT pretraining (random initialization)")
        print(f"Initialization method: {args.init_method}")
        print(f"{'='*60}")

        pretrained_params = {
            'num_pairs': num_pairs,
            'mirna_length': args.mirna_length,
            'target_length': args.target_window,
            'embedding_dim': args.embedding_dim,
            'dropout_rate': args.dropout_rate,
            'filter_sizes': filter_sizes,
            'kernel_sizes': kernel_sizes,
        }

        print(f"  num_pairs: {pretrained_params['num_pairs']}")
        print(f"  mirna_length: {pretrained_params['mirna_length']}")
        print(f"  target_length: {pretrained_params['target_length']}")
        print(f"  embedding_dim: {pretrained_params['embedding_dim']}")
        print(f"  filter_sizes: {pretrained_params['filter_sizes']}")
        print(f"  kernel_sizes: {pretrained_params['kernel_sizes']}")

        pretrained_cnn = PairwiseOneHotCNN(**pretrained_params)

        def init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if args.init_method == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif args.init_method == 'xavier':
                    nn.init.xavier_normal_(m.weight)
                elif args.init_method == 'normal':
                    nn.init.normal_(m.weight, mean=0, std=0.02)
                elif args.init_method == 'uniform':
                    nn.init.uniform_(m.weight, a=-0.05, b=0.05)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        pretrained_cnn.apply(init_weights)
        print(f"Applied {args.init_method} initialization to all layers")

        if args.freeze_encoder:
            print("Warning: --freeze_encoder ignored when using --no_pretrain")
            args.freeze_encoder = False

    else:
        print(f"\nLoading pretrained model from: {args.pretrained_checkpoint}")

        checkpoint = torch.load(args.pretrained_checkpoint, map_location=device)

        if 'model_params' in checkpoint:
            pretrained_params = checkpoint['model_params']
            print("Using architecture parameters from checkpoint:")
        elif 'architecture_summary' in checkpoint and 'model_params' in checkpoint['architecture_summary']:
            pretrained_params = checkpoint['architecture_summary']['model_params']
            print("Using architecture parameters from checkpoint:")
        else:
            print("Warning: No model_params in checkpoint, using command line arguments")
            pretrained_params = {
                'num_pairs': num_pairs,
                'mirna_length': args.mirna_length,
                'target_length': args.target_window,
                'embedding_dim': args.embedding_dim,
                'dropout_rate': args.dropout_rate,
                'filter_sizes': filter_sizes,
                'kernel_sizes': kernel_sizes,
            }

        print(f"  num_pairs: {pretrained_params['num_pairs']}")
        print(f"  mirna_length: {pretrained_params['mirna_length']}")
        print(f"  target_length: {pretrained_params['target_length']}")
        print(f"  embedding_dim: {pretrained_params['embedding_dim']}")
        print(f"  filter_sizes: {pretrained_params['filter_sizes']}")
        print(f"  kernel_sizes: {pretrained_params['kernel_sizes']}")

        pretrained_cnn = PairwiseOneHotCNN(**pretrained_params)
        pretrained_cnn.load_state_dict(checkpoint['model_state_dict'])
        print("Pretrained model loaded successfully")

    # Update num_pairs from loaded model
    num_pairs = pretrained_params['num_pairs']

    # Create gene-level model
    print(f"\nCreating GeneRepressionModelV4...")
    if args.use_layer_norm:
        print(f"  Using Layer Normalization")

    model = GeneRepressionModelV4(
        pretrained_cnn=pretrained_cnn,
        mirna_length=args.mirna_length,
        freeze_encoder=args.freeze_encoder,
        attention_hidden_dim=args.attention_hidden_dim,
        num_heads=args.num_heads,
        num_pairs=num_pairs,
        use_layer_norm=args.use_layer_norm,
        dropout_rate=args.dropout_rate
    ).to(device)

    print(f"\nModel: {model.__class__.__name__}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    initial_trainable_params = trainable_params
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    if args.gradient_accumulation_steps > 1:
        effective_batch = args.batch_size * args.gradient_accumulation_steps
        print(f"\nGradient accumulation enabled:")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Accumulation steps: {args.gradient_accumulation_steps}")
        print(f"  Effective batch size: {effective_batch}")

    # Load dataset
    print(f"\nLoading dataset from: {args.train_file}")
    full_dataset = GeneLevelDataset(
        file_path=args.train_file,
        gene_col=args.gene_col,
        mirna_col=args.mirna_col,
        label_col=args.label_col,
        max_gene_length=args.max_gene_length,
        mirna_length=args.mirna_length,
        target_window=args.target_window,
        pair_to_index=pair_to_index,
        num_pairs=num_pairs
    )

    # Split dataset
    train_dataset, val_dataset = GeneLevelDataset.create_train_validation_split(
        full_dataset, validation_fraction=args.val_fraction
    )

    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=GeneLevelDataset.collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=GeneLevelDataset.collate_fn
    )

    # Setup optimizer with optional discriminative learning rates
    if args.use_discriminative_lr and not args.freeze_encoder:
        param_groups = get_parameter_groups(
            model,
            args.learning_rate,
            pretrained_lr_factor=args.pretrained_lr_factor
        )
        optimizer = Adam(param_groups)
    else:
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
        print(f"\nUsing uniform learning rate: {args.learning_rate}")

    # Setup loss function
    if args.use_weighted_loss:
        criterion = WeightedMSELoss(
            threshold=args.weighted_loss_threshold,
            weight=args.weighted_loss_weight
        )
        print(f"\nUsing Weighted MSE Loss (threshold={args.weighted_loss_threshold}, weight={args.weighted_loss_weight})")
    else:
        criterion = nn.MSELoss()
        print("\nUsing standard MSE Loss")

    # Training loop
    history = {
        'train_loss': [], 'val_loss': [],
        'train_pearson': [], 'val_pearson': [],
        'train_spearman': [], 'val_spearman': []
    }

    best_val_metric = -float('inf')
    patience_counter = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(args.output_dir, f"gene_level_model_{timestamp}.pt")

    early_stop_metric_name = f"val_{args.early_stop_metric}"
    print(f"\nEarly stopping metric: {early_stop_metric_name}")

    print("\nStarting training...")
    for epoch in range(args.num_epochs):
        # Unfreeze encoder if specified
        if epoch == args.unfreeze_epoch and args.freeze_encoder:
            print(f"\nUnfreezing encoder at epoch {epoch}")

            if hasattr(model, 'unfreeze'):
                model.unfreeze()
            else:
                print("Warning: Model has no unfreeze method")

            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Trainable parameters after unfreeze: {trainable_params:,}")

            if args.use_discriminative_lr:
                param_groups = get_parameter_groups(
                    model,
                    args.learning_rate,
                    pretrained_lr_factor=args.pretrained_lr_factor
                )
                optimizer = Adam(param_groups)
                print("Optimizer recreated with discriminative learning rates")
            else:
                optimizer = Adam(model.parameters(), lr=args.learning_rate)
                print("Optimizer recreated with uniform learning rate")

        # Train
        train_loss, train_pearson, train_spearman = train_epoch(
            model, train_loader, optimizer, criterion, device,
            accumulation_steps=args.gradient_accumulation_steps
        )

        # Validate
        val_loss, val_pearson, val_spearman, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device
        )

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_pearson'].append(train_pearson)
        history['val_pearson'].append(val_pearson)
        history['train_spearman'].append(train_spearman)
        history['val_spearman'].append(val_spearman)

        print(f"Epoch {epoch+1}/{args.num_epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, Pearson: {train_pearson:.4f}, Spearman: {train_spearman:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Pearson: {val_pearson:.4f}, Spearman: {val_spearman:.4f}")

        if args.early_stop_metric == 'pearson':
            current_metric = val_pearson
        else:
            current_metric = val_spearman

        # Check for improvement
        if current_metric > best_val_metric:
            best_val_metric = current_metric
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_pearson': val_pearson,
                'val_spearman': val_spearman,
                'val_loss': val_loss,
                'pretrained_params': pretrained_params,
                'gene_model_params': {
                    'model_version': 'v4',
                    'mirna_length': args.mirna_length,
                    'target_window': args.target_window,
                    'max_gene_length': args.max_gene_length,
                    'attention_hidden_dim': args.attention_hidden_dim,
                    'num_pairs': num_pairs,
                    'num_heads': args.num_heads,
                    'dropout_rate': args.dropout_rate,
                    'use_layer_norm': args.use_layer_norm,
                    'pretrained': not args.no_pretrain,
                    'init_method': args.init_method if args.no_pretrain else None,
                    'freeze_encoder': args.freeze_encoder,
                    'unfreeze_epoch': args.unfreeze_epoch,
                    'learning_rate': args.learning_rate,
                    'use_discriminative_lr': args.use_discriminative_lr,
                    'pretrained_lr_factor': args.pretrained_lr_factor if args.use_discriminative_lr else None,
                    'use_weighted_loss': args.use_weighted_loss,
                    'weighted_loss_threshold': args.weighted_loss_threshold if args.use_weighted_loss else None,
                    'weighted_loss_weight': args.weighted_loss_weight if args.use_weighted_loss else None,
                    'batch_size': args.batch_size,
                    'gradient_accumulation_steps': args.gradient_accumulation_steps,
                    'early_stop_metric': args.early_stop_metric,
                    'patience': args.patience,
                    'val_fraction': args.val_fraction,
                }
            }, model_save_path)
            print(f"  Model saved! (best {early_stop_metric_name}: {best_val_metric:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs (best {early_stop_metric_name}: {best_val_metric:.4f})")

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Load best model for final evaluation
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    val_loss, val_pearson, val_spearman, final_preds, final_labels = evaluate(
        model, val_loader, criterion, device
    )

    history['final_preds'] = final_preds.tolist()
    history['final_labels'] = final_labels.tolist()

    print(f"\nFinal evaluation (best model from epoch {checkpoint['epoch']+1}):")
    print(f"  Validation - Loss: {val_loss:.4f}, Pearson: {val_pearson:.4f}, Spearman: {val_spearman:.4f}")
    print(f"  Early stopping was based on: {early_stop_metric_name}")

    # Save training history
    history_path = os.path.join(args.output_dir, f"training_history_{timestamp}.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Plot training history
    plot_path = plot_training_history(history, args.output_dir, timestamp)

    # Save comprehensive config
    config = {
        'timestamp': timestamp,
        'output_files': {
            'model': model_save_path,
            'history': history_path,
            'config': os.path.join(args.output_dir, f"config_{timestamp}.json"),
            'plot': plot_path
        },
        'environment': {
            'device': str(device),
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        },
        'data': {
            'train_file': args.train_file,
            'test_file': args.test_file,
            'gene_col': args.gene_col,
            'mirna_col': args.mirna_col,
            'label_col': args.label_col,
            'max_gene_length': args.max_gene_length,
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'val_fraction': args.val_fraction
        },
        'model': {
            'version': 'v4',
            'class_name': model.__class__.__name__,
            'mirna_length': args.mirna_length,
            'target_window': args.target_window,
            'attention_hidden_dim': args.attention_hidden_dim,
            'num_pairs': num_pairs,
            'num_heads': args.num_heads,
            'dropout_rate': args.dropout_rate,
            'use_layer_norm': args.use_layer_norm,
            'total_parameters': total_params,
            'initial_trainable_parameters': initial_trainable_params,
            'final_trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'frozen_parameters_initial': total_params - initial_trainable_params
        },
        'pretraining': {
            'pretrained': not args.no_pretrain,
            'pretrained_checkpoint': args.pretrained_checkpoint if not args.no_pretrain else None,
            'init_method': args.init_method if args.no_pretrain else None,
            'freeze_encoder_initially': args.freeze_encoder,
            'unfreeze_epoch': args.unfreeze_epoch,
            'encoder_was_unfrozen': (args.unfreeze_epoch > 0 and checkpoint['epoch'] >= args.unfreeze_epoch) or not args.freeze_encoder,
            'pretrained_params': pretrained_params
        },
        'training': {
            'num_epochs': args.num_epochs,
            'actual_epochs': checkpoint['epoch'] + 1,
            'batch_size': args.batch_size,
            'gradient_accumulation_steps': args.gradient_accumulation_steps,
            'effective_batch_size': args.batch_size * args.gradient_accumulation_steps,
            'learning_rate': args.learning_rate,
            'optimizer': 'Adam',
            'use_discriminative_lr': args.use_discriminative_lr,
            'pretrained_lr_factor': args.pretrained_lr_factor if args.use_discriminative_lr else None,
            'pretrained_lr': args.learning_rate * args.pretrained_lr_factor if args.use_discriminative_lr else args.learning_rate,
            'new_layers_lr': args.learning_rate,
            'patience': args.patience,
            'early_stop_metric': args.early_stop_metric,
            'early_stopped': (checkpoint['epoch'] + 1) < args.num_epochs
        },
        'loss': {
            'type': 'WeightedMSELoss' if args.use_weighted_loss else 'MSELoss',
            'use_weighted_loss': args.use_weighted_loss,
            'weighted_loss_threshold': args.weighted_loss_threshold if args.use_weighted_loss else None,
            'weighted_loss_weight': args.weighted_loss_weight if args.use_weighted_loss else None
        },
        'results': {
            'best_epoch': checkpoint['epoch'] + 1,
            'best_val_metric': float(best_val_metric),
            'best_val_metric_name': early_stop_metric_name,
            'final_val_loss': float(val_loss),
            'final_val_pearson': float(val_pearson),
            'final_val_spearman': float(val_spearman),
            'best_val_pearson': float(checkpoint['val_pearson']),
            'best_val_spearman': float(checkpoint['val_spearman']),
            'best_val_loss': float(checkpoint.get('val_loss', val_loss))
        },
        'raw_args': vars(args)
    }

    config_path = config['output_files']['config']
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nTraining complete!")
    print(f"  Model saved: {model_save_path}")
    print(f"  History saved: {history_path}")
    print(f"  Config saved: {config_path}")
    print(f"  Plot saved: {plot_path}")


if __name__ == "__main__":
    main()