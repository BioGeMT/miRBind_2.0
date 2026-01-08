import argparse
import os
import json
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import get_model, get_dataset_class, get_pair_to_index, get_device


def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X).squeeze()
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())
    
    precision, recall, _ = precision_recall_curve(all_labels, all_preds)
    auprc = auc(recall, precision)
    aps = average_precision_score(all_labels, all_preds)
    return total_loss / len(train_loader), auprc, aps


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    precision, recall, _ = precision_recall_curve(all_labels, all_preds)
    auprc = auc(recall, precision)
    aps = average_precision_score(all_labels, all_preds)
    return total_loss / len(data_loader), auprc, aps


def plot_training_history(history, output_dir, timestamp):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    
    axes[1].plot(history['train_auprc'], label='Train')
    axes[1].plot(history['val_auprc'], label='Val')
    axes[1].set_title('AUPRC')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"training_plots_{timestamp}.png")
    plt.savefig(plot_path)
    plt.close()
    return plot_path


def get_architecture_summary(model, model_type, model_params):
    """Generate architecture summary dictionary."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "model_type": model_type,
        "num_pairs": model_params['num_pairs'],
        "mirna_length": model_params['mirna_length'],
        "target_length": model_params['target_length'],
        "embedding_dim": model_params['embedding_dim'],
        "dropout_rate": model_params['dropout_rate'],
        "filter_sizes": model_params['filter_sizes'],
        "kernel_sizes": model_params['kernel_sizes'],
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_summary": str(model)
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Train miRNA binding model')
    
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--test_file1', type=str, required=True)
    parser.add_argument('--test_file2', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='model_outputs')
    
    parser.add_argument('--model', type=str, default='pairwise_onehot',
                       choices=['pairwise', 'pairwise_onehot'])
    parser.add_argument('--target_length', type=int, default=50)
    parser.add_argument('--mirna_length', type=int, default=28)
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--val_fraction', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=5)
    
    parser.add_argument('--embedding_dim', type=int, default=8)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--filter_sizes', type=str, default='128,64,32')
    parser.add_argument('--kernel_sizes', type=str, default='6,3,3')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")
    
    pair_to_index = get_pair_to_index()
    num_pairs = len(pair_to_index)
    
    filter_sizes = [int(x) for x in args.filter_sizes.split(',')]
    kernel_sizes = [int(x) for x in args.kernel_sizes.split(',')]
    
    DatasetClass = get_dataset_class(args.model)
    
    full_dataset = DatasetClass(
        args.train_file, args.target_length, args.mirna_length,
        pair_to_index, num_pairs
    )
    train_dataset, val_dataset = DatasetClass.create_train_validation_split(
        full_dataset, args.val_fraction
    )
    test_dataset1 = DatasetClass(
        args.test_file1, args.target_length, args.mirna_length,
        pair_to_index, num_pairs
    )
    test_dataset2 = DatasetClass(
        args.test_file2, args.target_length, args.mirna_length,
        pair_to_index, num_pairs
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, "
          f"Test1: {len(test_dataset1)}, Test2: {len(test_dataset2)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader1 = DataLoader(test_dataset1, batch_size=args.batch_size)
    test_loader2 = DataLoader(test_dataset2, batch_size=args.batch_size)
    
    model_params = {
        'num_pairs': num_pairs,
        'mirna_length': args.mirna_length,
        'target_length': args.target_length,
        'embedding_dim': args.embedding_dim,
        'dropout_rate': args.dropout_rate,
        'filter_sizes': filter_sizes,
        'kernel_sizes': kernel_sizes,
    }
    
    model = get_model(args.model, **model_params).to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCELoss()
    
    print(f"Model: {args.model}, Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(args.output_dir, f"{args.model}_model_{timestamp}.pt")
    
    history = {k: [] for k in ['train_loss', 'train_auprc', 'train_aps', 
                               'val_loss', 'val_auprc', 'val_aps']}
    
    best_val_auprc = 0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(args.num_epochs):
        train_loss, train_auprc, train_aps = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auprc, val_aps = evaluate(model, val_loader, criterion, device)
        
        for key, value in [('train_loss', train_loss), ('train_auprc', train_auprc), ('train_aps', train_aps),
                           ('val_loss', val_loss), ('val_auprc', val_auprc), ('val_aps', val_aps)]:
            history[key].append(value)
        
        print(f"Epoch {epoch+1}/{args.num_epochs} | "
              f"Train: {train_auprc:.4f}/{train_aps:.4f} | Val: {val_auprc:.4f}/{val_aps:.4f}")
        
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auprc': val_auprc,
                'model_params': model_params,
            }, model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model and evaluate on test sets
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    _, val_auprc, val_aps = evaluate(model, val_loader, criterion, device)
    _, test1_auprc, test1_aps = evaluate(model, test_loader1, criterion, device)
    _, test2_auprc, test2_aps = evaluate(model, test_loader2, criterion, device)
    
    # Add best model info and architecture summary to history
    history['best_epoch'] = best_epoch
    history['best_val_auprc'] = best_val_auprc
    history['final_test1_auprc'] = test1_auprc
    history['final_test2_auprc'] = test2_auprc
    history['final_test1_aps'] = test1_aps
    history['final_test2_aps'] = test2_aps
    history['architecture_summary'] = get_architecture_summary(model, args.model, model_params)
    
    with open(os.path.join(args.output_dir, f"history_{timestamp}.json"), 'w') as f:
        json.dump(history, f, indent=2)
    
    plot_training_history(history, args.output_dir, timestamp)
    
    print(f"\nBest model (epoch {checkpoint['epoch']+1}):")
    print(f"  Val:   AUPRC={val_auprc:.4f}, APS={val_aps:.4f}")
    print(f"  Test1: AUPRC={test1_auprc:.4f}, APS={test1_aps:.4f}")
    print(f"  Test2: AUPRC={test2_auprc:.4f}, APS={test2_aps:.4f}")
    print(f"  Saved to: {model_save_path}")


if __name__ == "__main__":
    main()