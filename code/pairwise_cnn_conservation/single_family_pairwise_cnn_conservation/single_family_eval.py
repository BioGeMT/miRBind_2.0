#!/usr/bin/env python3
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import os
import json
import glob
import pandas as pd
from datetime import datetime
import traceback
from collections import defaultdict

import sys
import os
# Add parent directory to path to import shared components
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from models import get_model
from dataset import MiRNAConservationDataset


def load_model_for_family(family_name, models_dir, device, config, pair_to_index):
    # load a model for a specific family
    # find model file for this family
    family_dir = os.path.join(models_dir, family_name)
    if not os.path.exists(family_dir):
        return None
        
    # look for model files
    model_files = glob.glob(os.path.join(family_dir, f"{family_name}_model_*.pt"))
    if not model_files:
        return None
        
    # use most recent model if multiple exist
    model_path = max(model_files, key=os.path.getmtime)
    
    try:
        # load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Parse filter sizes and kernel sizes
        filter_sizes = [int(x) for x in config['filter_sizes'].split(',')]
        kernel_sizes = [int(x) for x in config['kernel_sizes'].split(',')]
        
        # Model parameters
        model_params = {
            'num_pairs': len(pair_to_index),
            'mirna_length': config['mirna_length'],
            'target_length': config['target_length'],
            'embedding_dim': config['embedding_dim'],
            'dropout_rate': config['dropout_rate'],
            'filter_sizes': filter_sizes,
            'kernel_sizes': kernel_sizes
        }
        
        # Initialize and load model
        model = get_model("pairwise_conservation", **model_params).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model
    except Exception as e:
        print(f"Error loading model for {family_name}: {e}")
        return None


def process_single_sample(row, model, pair_to_index, config, device):
    # process a single sample through the model
    # Extract sequences and conservation scores
    target_seq = row['gene']
    mirna_seq = row['noncodingRNA']
    
    # Parse conservation scores
    try:
        if isinstance(row['gene_phyloP'], str):
            phylop_scores = json.loads(row['gene_phyloP'].replace("'", "\""))
        else:
            phylop_scores = row['gene_phyloP']
            
        if isinstance(row['gene_phastCons'], str):
            phastcons_scores = json.loads(row['gene_phastCons'].replace("'", "\""))
        else:
            phastcons_scores = row['gene_phastCons']
    except:
        # If parsing fails, use default values
        phylop_scores = [0.0] * len(target_seq)
        phastcons_scores = [0.5] * len(target_seq)
    
    # Ensure sequences are the right length
    target_seq = target_seq[:config['target_length']].ljust(config['target_length'], 'N')
    mirna_seq = mirna_seq[:config['mirna_length']].ljust(config['mirna_length'], 'N')
    
    # Create 2D pairwise encoding (target_length x mirna_length)
    encoding = np.zeros((config['target_length'], config['mirna_length']), dtype=np.int32)
    
    for i in range(config['target_length']):
        for j in range(config['mirna_length']):
            target_base = target_seq[i].upper()
            mirna_base = mirna_seq[j].upper()
            
            # Handle non-standard nucleotides
            if target_base not in 'ATCG':
                target_base = 'N'
            if mirna_base not in 'ATCG':
                mirna_base = 'N'
            
            # Get pair encoding
            pair = (target_base, mirna_base)
            encoding[i, j] = pair_to_index.get(pair, pair_to_index[('N', 'N')])
    
    # Create conservation score matrices (target_length x mirna_length)
    phylop_matrix = np.zeros((config['target_length'], config['mirna_length']), dtype=np.float32)
    phastcons_matrix = np.zeros((config['target_length'], config['mirna_length']), dtype=np.float32)
    
    # Fill conservation scores (broadcast across mirna dimension)
    for i in range(min(len(phylop_scores), config['target_length'])):
        phylop_matrix[i, :] = phylop_scores[i]
        phastcons_matrix[i, :] = phastcons_scores[i]
    
    # Convert to tensors and add batch dimension
    encoding_tensor = torch.from_numpy(encoding).long().unsqueeze(0).to(device)
    phylop_tensor = torch.from_numpy(phylop_matrix).float().unsqueeze(0).to(device)
    phastcons_tensor = torch.from_numpy(phastcons_matrix).float().unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(encoding_tensor, phylop_tensor, phastcons_tensor)
        prediction = output.squeeze().cpu().numpy().item()
    
    return prediction


def evaluate_test_data(test_file, models_dir, output_dir, device, config):
    # evaluate test data using family-specific models with single-pass approach
    print(f"\n{'='*80}")
    print(f"Evaluating test data with family-specific models")
    print(f"{'='*80}")
    
    try:
        # Read test data
        test_df = pd.read_csv(test_file, sep='\t')
        print(f"Test file contains {len(test_df)} samples")
        
        # Define nucleotide pairs
        nucleotide_pairs = [
            ('A', 'A'), ('A', 'T'), ('A', 'C'), ('A', 'G'),
            ('T', 'A'), ('T', 'T'), ('T', 'C'), ('T', 'G'),
            ('C', 'A'), ('C', 'T'), ('C', 'C'), ('C', 'G'),
            ('G', 'A'), ('G', 'T'), ('G', 'C'), ('G', 'G')
        ]
        pair_to_index = {pair: i for i, pair in enumerate(nucleotide_pairs)}
        pair_to_index[('N', 'N')] = len(pair_to_index)
        
        # Initialize tracking variables
        model_cache = {}
        current_model = None
        current_family = None
        
        # Storage for predictions
        all_predictions = []
        family_predictions = defaultdict(list)
        family_labels = defaultdict(list)
        families_without_models = set()
        
        # Process test data row by row
        print("\nProcessing test samples...")
        for idx, row in test_df.iterrows():
            family = row['mirgenedb_fam']
            true_label = row['label']
            
            # Check if we need to load a new model
            if family != current_family:
                if family in model_cache:
                    current_model = model_cache[family]
                else:
                    # Try to load model
                    model = load_model_for_family(family, models_dir, device, config, pair_to_index)
                    if model is None:
                        families_without_models.add(family)
                        current_model = None
                    else:
                        model_cache[family] = model
                        current_model = model
                        print(f"Loaded model for family: {family}")
                current_family = family
            
            # Get prediction if model is available
            if current_model is not None:
                try:
                    prediction = process_single_sample(row, current_model, pair_to_index, config, device)
                    predicted_class = 1 if prediction > 0.5 else 0
                    
                    # Store prediction with all metadata
                    pred_data = {
                        'index': idx,
                        'family': family,
                        'prediction': prediction,
                        'true_label': true_label,
                        'predicted_class': predicted_class
                    }
                    
                    # Add optional columns if they exist
                    for col in ['miRNA_ID', 'species', 'target_seq', 'mirna_seq']:
                        if col in row:
                            pred_data[col] = row[col]
                    
                    all_predictions.append(pred_data)
                    
                    # Store for family-specific metrics
                    family_predictions[family].append(prediction)
                    family_labels[family].append(true_label)
                    
                except Exception as e:
                    print(f"Error processing sample {idx} for family {family}: {e}")
                    traceback.print_exc()
                    
            # Progress indicator
            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx + 1}/{len(test_df)} samples...")
        
        print(f"\nProcessing complete. Evaluated {len(all_predictions)} samples.")
        
        # Calculate metrics
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save all predictions
        if all_predictions:
            all_predictions_df = pd.DataFrame(all_predictions)
            all_pred_file = os.path.join(output_dir, f"all_predictions_{timestamp}.csv")
            all_predictions_df.to_csv(all_pred_file, index=False)
            print(f"\nSaved all predictions to: {all_pred_file}")
            
            # Calculate overall AUPRC
            all_preds = [p['prediction'] for p in all_predictions]
            all_labels = [p['true_label'] for p in all_predictions]
            precision, recall, _ = precision_recall_curve(all_labels, all_preds)
            overall_auprc = auc(recall, precision)
            
            print(f"\nOverall AUPRC: {overall_auprc:.4f}")
            
            # Calculate per-family metrics
            family_results = []
            for family in family_predictions:
                if len(family_predictions[family]) > 0:
                    # Calculate family AUPRC
                    precision, recall, _ = precision_recall_curve(
                        family_labels[family], 
                        family_predictions[family]
                    )
                    family_auprc = auc(recall, precision)
                    
                    family_results.append({
                        'family_name': family,
                        'test_samples': len(family_predictions[family]),
                        'test_auprc': family_auprc
                    })
            
            # Sort by test samples
            family_results.sort(key=lambda x: x['test_samples'], reverse=True)
            
            # Save family-specific results
            family_df = pd.DataFrame(family_results)
            family_csv = os.path.join(output_dir, f"family_results_{timestamp}.csv")
            family_df.to_csv(family_csv, index=False)
            print(f"Saved family results to: {family_csv}")
            
            # Create summary JSON
            results_summary = {
                'evaluation_timestamp': timestamp,
                'test_file': test_file,
                'models_dir': models_dir,
                'total_test_samples': len(test_df),
                'total_samples_evaluated': len(all_predictions),
                'samples_without_models': len(test_df) - len(all_predictions),
                'families_without_models': list(families_without_models),
                'coverage_percentage': 100 * len(all_predictions) / len(test_df),
                'overall_auprc': overall_auprc,
                'family_results': family_results,
                'output_files': {
                    'predictions': all_pred_file,
                    'family_results': family_csv
                }
            }
            
            results_file = os.path.join(output_dir, f"test_results_summary_{timestamp}.json")
            with open(results_file, 'w') as f:
                json.dump(results_summary, f, indent=2)
            print(f"Saved results summary to: {results_file}")
            
            # Print summary
            print(f"\n{'='*80}")
            print(f"EVALUATION SUMMARY")
            print(f"{'='*80}")
            print(f"Total test samples: {len(test_df)}")
            print(f"Samples evaluated: {len(all_predictions)}")
            print(f"Coverage: {100 * len(all_predictions) / len(test_df):.1f}%")
            print(f"Overall AUPRC: {overall_auprc:.4f}")
            print(f"Number of families evaluated: {len(family_results)}")
            
            if families_without_models:
                print(f"\nFamilies without models ({len(families_without_models)}):")
                for fam in sorted(families_without_models):
                    count = len(test_df[test_df['mirgenedb_fam'] == fam])
                    print(f"  - {fam}: {count} samples")
            
            # Top performing families
            if family_results:
                print(f"\nTop 5 performing families:")
                for i, fam in enumerate(family_results[:5]):
                    print(f"  {i+1}. {fam['family_name']}: AUPRC = {fam['test_auprc']:.4f} ({fam['test_samples']} samples)")
            
            return results_summary
            
        else:
            print("\nNo predictions could be made - no models found for any test samples.")
            return None
            
    except Exception as e:
        print(f"Error during evaluation: {e}")
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Evaluate family-specific miRNA binding prediction models')
    
    # Input/output parameters
    parser.add_argument('--test_file', type=str, required=True, 
                        help='Path to test data file')
    parser.add_argument('--models_dir', type=str, required=True, 
                        help='Directory containing trained family models')
    parser.add_argument('--output_dir', type=str, default='evaluation_outputs', 
                        help='Directory to save evaluation outputs')
    
    # Model parameters - should match training parameters
    parser.add_argument('--target_length', type=int, default=50, 
                        help='Length of target sequence')
    parser.add_argument('--mirna_length', type=int, default=25, 
                        help='Length of miRNA sequence')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for evaluation')
    parser.add_argument('--embedding_dim', type=int, default=8, 
                        help='Embedding dimension')
    parser.add_argument('--filter_sizes', type=str, default='128,64,32', 
                        help='Comma-separated list of filter sizes')
    parser.add_argument('--kernel_sizes', type=str, default='6,3,3', 
                        help='Comma-separated list of kernel sizes')
    parser.add_argument('--dropout_rate', type=float, default=0.2, 
                        help='Dropout rate')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Convert args to dict for easier passing
    config = {
        'target_length': args.target_length,
        'mirna_length': args.mirna_length,
        'batch_size': args.batch_size,
        'embedding_dim': args.embedding_dim,
        'filter_sizes': args.filter_sizes,
        'kernel_sizes': args.kernel_sizes,
        'dropout_rate': args.dropout_rate
    }
    
    # Run evaluation
    results = evaluate_test_data(args.test_file, args.models_dir, args.output_dir, device, config)
    
    if results:
        print("\nEvaluation completed successfully!")
    else:
        print("\nEvaluation failed or no results generated.")


if __name__ == "__main__":
    main()