import argparse
import os

import torch
import pandas as pd
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import load_model, get_dataset_class, get_pair_to_index, get_device


def run_inference(model, data_loader, device):
    """Run inference on a dataset."""
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for i, (batch_X, _) in enumerate(data_loader):
            outputs = model(batch_X.to(device)).squeeze()
            all_preds.extend(outputs.cpu().numpy())
            if (i + 1) % 100 == 0:
                print(f"  Processed {len(all_preds)} samples...")
    
    return all_preds


def parse_args():
    parser = argparse.ArgumentParser(description='Run inference')
    
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['pairwise', 'pairwise_onehot'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--target_length', type=int, default=None)
    parser.add_argument('--mirna_length', type=int, default=None)
    
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
    
    dataset = DatasetClass(
        args.input_file, target_length, mirna_length,
        pair_to_index, len(pair_to_index)
    )
    print(f"Loaded {len(dataset)} samples")
    
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    predictions = run_inference(model, loader, device)
    
    df = pd.read_csv(args.input_file, sep='\t')
    if len(predictions) != len(df):
        min_len = min(len(predictions), len(df))
        predictions = predictions[:min_len]
        df = df.iloc[:min_len]
    
    df['prediction_score'] = predictions
    df['predicted_class'] = [1 if p > 0.5 else 0 for p in predictions]
    
    df.to_csv(args.output_file, sep='\t', index=False)
    print(f"Saved {len(predictions)} predictions to: {args.output_file}")


if __name__ == "__main__":
    main()
