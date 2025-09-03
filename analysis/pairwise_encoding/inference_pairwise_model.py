import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os

from models import get_model
from dataset import MiRNADataset, MiRNAOneHotDataset


def load_model_checkpoint(checkpoint_path, device, args):
    """
    Load a trained model from checkpoint.
    
    Parameters:
    checkpoint_path (str): Path to the saved model checkpoint
    device: PyTorch device
    
    Returns:
    model: Loaded model
    metadata: Model metadata including architecture info
    """
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model parameters and architecture info
    model_params = checkpoint['model_params']
    architecture_summary = checkpoint.get('architecture_summary', {})

    # Initialize model with saved parameters
    model = get_model(args.model_type, **model_params)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint


def run_inference(model, data_loader, device):
    """
    Run inference on a dataset.
    
    Parameters:
    model: Trained PyTorch model
    data_loader: DataLoader for the dataset
    device: PyTorch device
    
    Returns:
    predictions: List of prediction scores
    """
    model.eval()
    all_preds = []
    
    print("Running inference...")
    with torch.no_grad():
        for i, (batch_X, batch_y) in enumerate(data_loader):
            batch_X = batch_X.to(device)
            
            outputs = model(batch_X).squeeze()
            all_preds.extend(outputs.cpu().numpy())
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {(i + 1) * len(batch_X)} samples...")
    
    return all_preds


def main():
    parser = argparse.ArgumentParser(description='Run inference and append predictions to dataset')
    
    # Model and data parameters
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the trained model checkpoint (.pt file)')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to input data file for inference')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to save dataset with predictions appended')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    
    # Optional parameters (will be overridden by model checkpoint if not specified)
    parser.add_argument('--target_length', type=int, default=None,
                        help='Length of target sequence (will use model default if not specified)')
    parser.add_argument('--mirna_length', type=int, default=None,
                        help='Length of miRNA sequence (will use model default if not specified)')
    parser.add_argument('--model_type', type=str, default=None,
                        help='Model type: pairwise or pairwise_onehot (will auto-detect from checkpoint)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    try:
        model, checkpoint = load_model_checkpoint(args.model_path, device, args)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Get model parameters from checkpoint
    model_params = checkpoint['model_params']
    target_length = args.target_length if args.target_length is not None else model_params['target_length']
    mirna_length = args.mirna_length if args.mirna_length is not None else model_params['mirna_length']
    
    print(f"Using parameters: target_length={target_length}, mirna_length={mirna_length}, model_type={args.model_type}")
    
    # Define nucleotide pairs and mapping (same as in training)
    nucleotide_pairs = [
        ('A', 'A'), ('A', 'T'), ('A', 'C'), ('A', 'G'),
        ('T', 'A'), ('T', 'T'), ('T', 'C'), ('T', 'G'),
        ('C', 'A'), ('C', 'T'), ('C', 'C'), ('C', 'G'),
        ('G', 'A'), ('G', 'T'), ('G', 'C'), ('G', 'G')
    ]
    pair_to_index = {pair: i for i, pair in enumerate(nucleotide_pairs)}
    pair_to_index[('N', 'N')] = len(pair_to_index)  # Add padding token
    
    # Choose appropriate dataset class
    if args.model_type == "pairwise":
        Dataset_obj = MiRNADataset
    elif args.model_type == "pairwise_onehot":
        Dataset_obj = MiRNAOneHotDataset
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Load inference dataset
    try:
        inference_dataset = Dataset_obj(
            args.input_file,
            target_length,
            mirna_length,
            pair_to_index,
            num_pairs=len(pair_to_index)
        )
        print(f"Loaded inference dataset with {len(inference_dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Create data loader
    inference_loader = DataLoader(inference_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Run inference
    try:
        predictions = run_inference(model, inference_loader, device)
        print(f"Inference complete. Generated {len(predictions)} predictions.")
    except Exception as e:
        print(f"Error during inference: {e}")
        return
    
    # Load original dataset and append predictions
    try:
        # Read the original dataset
        original_data = pd.read_csv(args.input_file, sep='\t')
        print(f"Original dataset shape: {original_data.shape}")
        
        # Ensure we have the same number of predictions as rows in the original data
        if len(predictions) != len(original_data):
            print(f"Warning: Number of predictions ({len(predictions)}) doesn't match dataset rows ({len(original_data)})")
            print("Using minimum length to avoid index errors")
            min_len = min(len(predictions), len(original_data))
            predictions = predictions[:min_len]
            original_data = original_data.iloc[:min_len]
        
        # Add prediction columns
        original_data['prediction_score'] = predictions
        original_data['predicted_class'] = [1 if p > 0.5 else 0 for p in predictions]
        
        # Save the dataset with predictions
        original_data.to_csv(args.output_file, sep='\t', index=False)
        print(f"Dataset with predictions saved to: {args.output_file}")
        print(f"Added columns: prediction_score, predicted_class")
        
    except Exception as e:
        print(f"Error processing and saving results: {e}")
        return
    
    print("Inference pipeline completed successfully!")


if __name__ == "__main__":
    main()