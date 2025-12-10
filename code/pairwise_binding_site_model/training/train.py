import argparse
import os
import json
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
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
    return total_loss / len(train_loader), auc(recall, precision)


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
    return total_loss / len(data_loader), auc(recall, precision)


def plot_training_history(history, output_dir, timestamp):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Val')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    
    axes[0, 1].plot(history['test1_loss'], label='Test1')
    axes[0, 1].plot(history['test2_loss'], label='Test2')
    axes[0, 1].set_title('Test Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    
    axes[1, 0].plot(history['train_auprc'], label='Train')
    axes[1, 0].plot(history['val_auprc'], label='Val')
    axes[1, 0].set_title('AUPRC')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()
    
    axes[1, 1].plot(history['test1_auprc'], label='Test1')
    axes[1, 1].plot(history['test2_auprc'], label='Test2')
    axes[1, 1].set_title('Test AUPRC')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"training_plots_{timestamp}.png")
    plt.savefig(plot_path)
    plt.close()
    return plot_path


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
    
    history = {k: [] for k in ['train_loss', 'train_auprc', 'val_loss', 'val_auprc',
                               'test1_loss', 'test1_auprc', 'test2_loss', 'test2_auprc']}
    
    best_val_auprc = 0
    patience_counter = 0
    
    for epoch in range(args.num_epochs):
        train_loss, train_auprc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auprc = evaluate(model, val_loader, criterion, device)
        test1_loss, test1_auprc = evaluate(model, test_loader1, criterion, device)
        test2_loss, test2_auprc = evaluate(model, test_loader2, criterion, device)
        
        for key, value in [('train_loss', train_loss), ('train_auprc', train_auprc),
                           ('val_loss', val_loss), ('val_auprc', val_auprc),
                           ('test1_loss', test1_loss), ('test1_auprc', test1_auprc),
                           ('test2_loss', test2_loss), ('test2_auprc', test2_auprc)]:
            history[key].append(value)
        
        print(f"Epoch {epoch+1}/{args.num_epochs} | "
              f"Train: {train_auprc:.4f} | Val: {val_auprc:.4f} | "
              f"Test1: {test1_auprc:.4f} | Test2: {test2_auprc:.4f}")
        
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
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
    
    with open(os.path.join(args.output_dir, f"history_{timestamp}.json"), 'w') as f:
        json.dump(history, f, indent=2)
    
    plot_training_history(history, args.output_dir, timestamp)
    
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    _, val_auprc = evaluate(model, val_loader, criterion, device)
    _, test1_auprc = evaluate(model, test_loader1, criterion, device)
    _, test2_auprc = evaluate(model, test_loader2, criterion, device)
    
    print(f"\nBest model (epoch {checkpoint['epoch']+1}):")
    print(f"  Val: {val_auprc:.4f}, Test1: {test1_auprc:.4f}, Test2: {test2_auprc:.4f}")
    print(f"  Saved to: {model_save_path}")


if __name__ == "__main__":
    main()
