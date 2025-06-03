#!/usr/bin/env python3
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
import os
import json
import glob
import pandas as pd
from datetime import datetime

from models import get_model
from dataset import MiRNAConservationDataset

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_train_preds = []
    all_train_labels = []
    
    for batch_X, batch_phylop, batch_phastcons, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_phylop = batch_phylop.to(device)
        batch_phastcons = batch_phastcons.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X, batch_phylop, batch_phastcons)
        
        # FIX: Handle shape mismatch for BCELoss
        if outputs.dim() == 2 and outputs.size(1) == 1:
            outputs = outputs.squeeze(1)  # [batch, 1] -> [batch]
        if batch_y.dim() == 2 and batch_y.size(1) == 1:
            batch_y = batch_y.squeeze(1)  # [batch, 1] -> [batch]
        
        loss = criterion(outputs, batch_y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_train_preds.extend(outputs.detach().cpu().numpy())
        all_train_labels.extend(batch_y.cpu().numpy())
    
    precision, recall, _ = precision_recall_curve(all_train_labels, all_train_preds)
    train_auprc = auc(recall, precision)
    
    return total_loss / len(train_loader), train_auprc


def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_phylop, batch_phastcons, batch_y in val_loader:
            batch_X = batch_X.to(device)
            batch_phylop = batch_phylop.to(device)
            batch_phastcons = batch_phastcons.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_X, batch_phylop, batch_phastcons)
            
            # FIX: Handle shape mismatch for BCELoss
            if outputs.dim() == 2 and outputs.size(1) == 1:
                outputs = outputs.squeeze(1)  # [batch, 1] -> [batch]
            if batch_y.dim() == 2 and batch_y.size(1) == 1:
                batch_y = batch_y.squeeze(1)  # [batch, 1] -> [batch]
            
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    precision, recall, _ = precision_recall_curve(all_labels, all_preds)
    auprc = auc(recall, precision)
    return total_loss / len(val_loader), auprc


def plot_training_history(training_history, output_dir, timestamp, family_name):
    """
    Plot and save the training and validation metrics history.
    """
    plt.figure(figsize=(12, 8))
    
    # Loss plots
    plt.subplot(2, 1, 1)
    plt.plot(training_history['train_loss'], label='Train Loss')
    plt.plot(training_history['val_loss'], label='Val Loss')
    plt.title(f'Training and Validation Loss - {family_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # AUPRC plots
    plt.subplot(2, 1, 2)
    plt.plot(training_history['train_auprc'], label='Train AUPRC')
    plt.plot(training_history['val_auprc'], label='Val AUPRC')
    plt.title(f'Training and Validation AUPRC - {family_name}')
    plt.xlabel('Epoch')
    plt.ylabel('AUPRC')
    plt.legend()
    
    plt.tight_layout()
    family_dir = os.path.join(output_dir, family_name)
    os.makedirs(family_dir, exist_ok=True)
    plot_path = os.path.join(family_dir, f"training_plots_{timestamp}.png")
    plt.savefig(plot_path)
    plt.close()  # Close the figure to free memory
    return plot_path


def save_training_history(training_history, output_dir, timestamp, family_name):
    """
    Save the training history to a JSON file.
    """
    family_dir = os.path.join(output_dir, family_name)
    os.makedirs(family_dir, exist_ok=True)
    history_file = os.path.join(family_dir, f"training_history_{timestamp}.json")
    with open(history_file, 'w') as f:
        json.dump(training_history, f, indent=2)
    return history_file


def train_family_model(train_file, family_name, args, device, pair_to_index):
    """Train a model for a specific miRNA family"""
    print(f"\n{'='*80}")
    print(f"Training model for family: {family_name}")
    print(f"{'='*80}")
    
    # Create family-specific output directory
    family_dir = os.path.join(args.output_dir, family_name)
    os.makedirs(family_dir, exist_ok=True)
    
    # Get timestamp for this model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize dataset with conservation scores
    try:
        full_dataset = MiRNAConservationDataset(
            train_file,
            args.target_length,
            args.mirna_length,
            pair_to_index,
            num_pairs=len(pair_to_index)
        )
        
        # Skip if dataset is too small
        if len(full_dataset) < 10:
            print(f"Skipping {family_name}: too few samples ({len(full_dataset)})")
            return None
            
        # Create train/val split
        train_dataset, val_dataset = MiRNAConservationDataset.create_train_validation_split(
            full_dataset, validation_fraction=args.val_fraction
        )
        
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        
        # Create data loaders with drop_last=True to avoid BatchNorm issues
        train_loader = DataLoader(
            train_dataset, 
            batch_size=min(args.batch_size, len(train_dataset)), 
            shuffle=True,
            drop_last=True  # FIX: Avoid single-sample batches
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=min(args.batch_size, len(val_dataset)),
            shuffle=False,
            drop_last=False  # Keep all validation data
        )
        
        # Parse filter sizes and kernel sizes from strings to lists
        filter_sizes = [int(x) for x in args.filter_sizes.split(',')]
        kernel_sizes = [int(x) for x in args.kernel_sizes.split(',')]
        
        # Model architecture parameters
        model_params = {
            'num_pairs': len(pair_to_index),
            'mirna_length': args.mirna_length,
            'target_length': args.target_length,
            'embedding_dim': args.embedding_dim,
            'dropout_rate': args.dropout_rate,
            'filter_sizes': filter_sizes,
            'kernel_sizes': kernel_sizes
        }
        
        # Initialize model with conservation
        model = get_model("pairwise_conservation", **model_params).to(device)
        
        # Initialize optimizer and loss function
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
        criterion = nn.BCELoss()
        
        # Training loop with early stopping
        best_val_auprc = 0
        patience_counter = 0
        training_history = {
            'train_loss': [],
            'train_auprc': [],
            'val_loss': [],
            'val_auprc': []
        }
        
        model_save_path = os.path.join(family_dir, f"{family_name}_model_{timestamp}.pt")
        
        # Architecture summary
        architecture_summary = {
            'model_type': "pairwise_conservation",
            'family_name': family_name,
            'num_pairs': len(pair_to_index),
            'mirna_length': args.mirna_length,
            'target_length': args.target_length,
            'embedding_dim': args.embedding_dim,
            'dropout_rate': args.dropout_rate,
            'filter_sizes': filter_sizes,
            'kernel_sizes': kernel_sizes,
            'total_params': sum(p.numel() for p in model.parameters()),
            'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'model_summary': str(model),
        }
        
        # Save architecture summary
        architecture_file = os.path.join(family_dir, f"architecture_summary_{timestamp}.json")
        with open(architecture_file, 'w') as f:
            json.dump(architecture_summary, f, indent=2)
        
        print("Starting training...")
        for epoch in range(args.num_epochs):
            train_loss, train_auprc = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_auprc = evaluate(model, val_loader, criterion, device)
            
            # Save training history
            training_history['train_loss'].append(train_loss)
            training_history['train_auprc'].append(train_auprc)
            training_history['val_loss'].append(val_loss)
            training_history['val_auprc'].append(val_auprc)
            
            print(f"Epoch {epoch+1}/{args.num_epochs}")
            print(f"  Train - Loss: {train_loss:.4f}, AUPRC: {train_auprc:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, AUPRC: {val_auprc:.4f}")
            
            # Check for improvement
            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_auprc': val_auprc,
                    'model_params': model_params,
                    'architecture_summary': architecture_summary,
                    'family_name': family_name
                }, model_save_path)
                print(f"Model saved at {model_save_path}")
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter} epochs")
            
            # Early stopping
            if patience_counter >= args.patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save training history and plot
        history_file = save_training_history(training_history, args.output_dir, timestamp, family_name)
        plot_path = plot_training_history(training_history, args.output_dir, timestamp, family_name)
        
        # Final results
        final_results = {
            'family_name': family_name,
            'best_epoch': epoch + 1 - patience_counter,
            'best_val_auprc': best_val_auprc,
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset)
        }
        
        # Save final results
        results_file = os.path.join(family_dir, f"results_summary_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"Training complete for {family_name}. Best validation AUPRC: {best_val_auprc:.4f}")
        
        return {
            "family_name": family_name,
            "model_path": model_save_path,
            "best_val_auprc": best_val_auprc,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset)
        }
        
    except Exception as e:
        print(f"Error training model for {family_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Train family-specific miRNA binding prediction models')
    
    # Input/output parameters
    parser.add_argument('--train_dir', type=str, required=True, 
                        help='Directory containing training data files (one per family)')
    parser.add_argument('--output_dir', type=str, default='family_model_outputs', 
                        help='Directory to save model outputs')
    
    # Model parameters
    parser.add_argument('--target_length', type=int, default=50, 
                        help='Length of target sequence')
    parser.add_argument('--mirna_length', type=int, default=25, 
                        help='Length of miRNA sequence')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=30, 
                        help='Maximum number of epochs to train')
    parser.add_argument('--val_fraction', type=float, default=0.1, 
                        help='Fraction of data to use for validation')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                        help='Learning rate')
    parser.add_argument('--embedding_dim', type=int, default=8, 
                        help='Embedding dimension')
    parser.add_argument('--dropout_rate', type=float, default=0.2, 
                        help='Dropout rate')
    parser.add_argument('--filter_sizes', type=str, default='128,64,32', 
                        help='Comma-separated list of filter sizes')
    parser.add_argument('--kernel_sizes', type=str, default='6,5,5', 
                        help='Comma-separated list of kernel sizes')
    parser.add_argument('--patience', type=int, default=5, 
                        help='Patience for early stopping')
    parser.add_argument('--max_families', type=int, default=None, 
                        help='Maximum number of families to train (for testing)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define nucleotide pairs and mapping for pairwise model
    nucleotide_pairs = [
        ('A', 'A'), ('A', 'T'), ('A', 'C'), ('A', 'G'),
        ('T', 'A'), ('T', 'T'), ('T', 'C'), ('T', 'G'),
        ('C', 'A'), ('C', 'T'), ('C', 'C'), ('C', 'G'),
        ('G', 'A'), ('G', 'T'), ('G', 'C'), ('G', 'G')
    ]
    pair_to_index = {pair: i for i, pair in enumerate(nucleotide_pairs)}
    
    # Add a padding token
    pair_to_index[('N', 'N')] = len(pair_to_index)
    
    # Get all family training files
    train_files = glob.glob(os.path.join(args.train_dir, "*.tsv"))
    print(f"Found {len(train_files)} family training files")
    
    # Limit number of families if specified
    if args.max_families and len(train_files) > args.max_families:
        train_files = train_files[:args.max_families]
        print(f"Limiting to {args.max_families} families for training")
    
    # Train a model for each family
    family_models = {}
    
    for train_file in sorted(train_files):
        # Extract family name from filename
        family_name = os.path.basename(train_file).replace('.tsv', '')
        
        # Train the model
        model_info = train_family_model(train_file, family_name, args, device, pair_to_index)
        
        if model_info:
            family_models[family_name] = model_info
    
    # Print training summary
    print(f"\n{'='*80}")
    print(f"Training Summary")
    print(f"{'='*80}")
    print(f"Total families trained: {len(family_models)}/{len(train_files)}")
    
    if family_models:
        # Sort by validation AUPRC
        sorted_models = sorted(family_models.values(), key=lambda x: x["best_val_auprc"], reverse=True)
        
        print("\nTop 5 families by validation performance:")
        for i, model in enumerate(sorted_models[:5]):
            print(f"  {i+1}. {model['family_name']}: Val AUPRC = {model['best_val_auprc']:.4f} (samples: {model['train_samples']})")
        
        # Save training summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = os.path.join(args.output_dir, f"training_summary_{timestamp}.json")
        with open(summary_file, 'w') as f:
            json.dump({
                'total_families': len(family_models),
                'family_models': family_models,
                'training_parameters': vars(args)
            }, f, indent=2)
        print(f"\nTraining summary saved to: {summary_file}")


if __name__ == "__main__":
    main()