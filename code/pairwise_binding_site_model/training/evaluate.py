import argparse
import os
from datetime import datetime

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_recall_curve, auc, average_precision_score,
    accuracy_score, confusion_matrix
)

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import load_model, get_dataset_class, get_pair_to_index, get_device


def generate_predictions(model, dataset, batch_size, device):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size)
    all_preds = []
    
    with torch.no_grad():
        for batch_X, _ in loader:
            outputs = model(batch_X.to(device)).squeeze()
            all_preds.extend(outputs.cpu().numpy())
    
    return np.array(all_preds)


def compute_metrics(labels, predictions):
    """Compute classification metrics."""
    binary_preds = (predictions >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels.astype(int), binary_preds).ravel()
    
    precision, recall, _ = precision_recall_curve(labels, predictions)
    
    return {
        'auprc': auc(recall, precision),
        'avg_precision': average_precision_score(labels, predictions),
        'accuracy': accuracy_score(labels.astype(int), binary_preds),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        'precision': tp / max(1, tp + fp),
        'recall': tp / max(1, tp + fn),
        'specificity': tn / max(1, tn + fp),
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--model_type', type=str, default='pairwise',
                       choices=['pairwise', 'pairwise_onehot'])
    parser.add_argument('--target_length', type=int, default=50)
    parser.add_argument('--mirna_length', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=64)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = get_device()
    model, checkpoint = load_model(args.model_path, args.model_type, device)
    print(f"Loaded model from epoch {checkpoint.get('epoch', '?')}, "
          f"val_auprc={checkpoint.get('val_auprc', '?'):.4f}")
    
    pair_to_index = get_pair_to_index()
    df = pd.read_csv(args.input_file, sep='\t')
    
    DatasetClass = get_dataset_class(args.model_type)
    dataset = DatasetClass(
        args.input_file, args.target_length, args.mirna_length,
        pair_to_index, len(pair_to_index)
    )
    
    predictions = generate_predictions(model, dataset, args.batch_size, device)
    
    df['prediction_score'] = predictions
    df['predicted_class'] = (predictions >= 0.5).astype(int)
    
    if args.output_file is None:
        base = os.path.splitext(os.path.basename(args.input_file))[0]
        args.output_file = f"{base}_predicted.tsv"
    
    df.to_csv(args.output_file, sep='\t', index=False)
    print(f"Saved predictions to: {args.output_file}")
    
    if 'label' in df.columns:
        metrics = compute_metrics(df['label'].values, predictions)
        print(f"\nMetrics: AUPRC={metrics['auprc']:.4f}, Acc={metrics['accuracy']:.4f}")
        print(f"  TP={metrics['tp']}, FP={metrics['fp']}, TN={metrics['tn']}, FN={metrics['fn']}")


if __name__ == "__main__":
    main()
