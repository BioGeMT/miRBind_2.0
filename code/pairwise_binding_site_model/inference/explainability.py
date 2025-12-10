import argparse
import os
import json

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from captum.attr import GradientShap

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import load_model, get_dataset_class, get_pair_to_index, get_device


def reduce_shap_3d_to_2d(shap_3d):
    """Reduce 3D SHAP (one-hot) to 2D by summing along pair dimension."""
    return np.sum(shap_3d, axis=2)


def shap_to_json(shap_array):
    """Convert SHAP array to JSON string."""
    return json.dumps({'shape': list(shap_array.shape), 'values': shap_array.tolist()})


def run_inference_with_shap(model, data_loader, device):
    """Run inference and compute SHAP values (reduced to 2D)."""
    model.eval()
    explainer = GradientShap(model)
    
    all_preds, all_shap_2d = [], []
    
    for i, (batch_X, _) in enumerate(data_loader):
        batch_X = batch_X.to(device)
        
        with torch.no_grad():
            outputs = model(batch_X).squeeze()
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            all_preds.extend(outputs.cpu().numpy())
        
        baseline = torch.zeros_like(batch_X)
        attributions = explainer.attribute(batch_X, baseline, target=0)
        
        for j in range(batch_X.shape[0]):
            shap_3d = attributions[j].cpu().numpy()
            all_shap_2d.append(reduce_shap_3d_to_2d(shap_3d))
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {len(all_preds)} samples...")
    
    return all_preds, all_shap_2d


def parse_args():
    parser = argparse.ArgumentParser(description='Inference with SHAP explainability')
    
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['pairwise', 'pairwise_onehot'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--target_length', type=int, default=None)
    parser.add_argument('--mirna_length', type=int, default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug_samples', type=int, default=100)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = get_device()
    print(f"Using device: {device}")
    
    model, checkpoint = load_model(args.model_path, args.model_type, device)
    model_params = checkpoint['model_params']
    
    target_length = args.target_length or model_params['target_length']
    mirna_length = args.mirna_length or model_params['mirna_length']
    
    pair_to_index = get_pair_to_index()
    DatasetClass = get_dataset_class(args.model_type)
    
    df = pd.read_csv(args.input_file, sep='\t')
    dataset = DatasetClass(
        args.input_file, target_length, mirna_length,
        pair_to_index, len(pair_to_index)
    )
    
    if args.debug:
        n = min(args.debug_samples, len(dataset))
        dataset = Subset(dataset, list(range(n)))
        df = df.iloc[:n]
        print(f"DEBUG: Using {n} samples")
    
    print(f"Processing {len(dataset)} samples...")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    predictions, shap_values_2d = run_inference_with_shap(model, loader, device)
    
    if len(predictions) != len(df):
        min_len = min(len(predictions), len(df))
        predictions, shap_values_2d = predictions[:min_len], shap_values_2d[:min_len]
        df = df.iloc[:min_len]
    
    df['prediction_score'] = predictions
    df['predicted_class'] = [1 if p > 0.5 else 0 for p in predictions]
    df['shap_values_2d'] = [shap_to_json(arr) for arr in shap_values_2d]
    
    df.to_csv(args.output_file, sep='\t', index=False)
    print(f"Saved to: {args.output_file}")
    
    if shap_values_2d:
        print(f"SHAP shape: {shap_values_2d[0].shape} (miRNA x mRNA)")


if __name__ == "__main__":
    main()
