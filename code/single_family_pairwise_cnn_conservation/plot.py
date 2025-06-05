
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import glob
import argparse
from datetime import datetime

def load_evaluation_results(results_path):
    """
    Load evaluation results from JSON file or CSV files.
    
    Args:
        results_path: Path to results JSON file or directory containing CSV files
        
    Returns:
        dict: Results summary dictionary
    """
    if os.path.isfile(results_path) and results_path.endswith('.json'):
        # Load from JSON summary file
        with open(results_path, 'r') as f:
            return json.load(f)
    
    elif os.path.isdir(results_path):
        # Try to reconstruct from CSV files in directory
        print(f"Loading results from directory: {results_path}")
        
        # Find prediction and family results files
        pred_files = glob.glob(os.path.join(results_path, "all_predictions_*.csv"))
        family_files = glob.glob(os.path.join(results_path, "family_results_*.csv"))
        
        if not pred_files or not family_files:
            raise ValueError(f"Could not find prediction or family results files in {results_path}")
        
        # Use most recent files
        pred_file = max(pred_files, key=os.path.getmtime)
        family_file = max(family_files, key=os.path.getmtime)
        
        print(f"Using prediction file: {os.path.basename(pred_file)}")
        print(f"Using family results file: {os.path.basename(family_file)}")
        
        # Load data
        pred_df = pd.read_csv(pred_file)
        family_df = pd.read_csv(family_file)
        
        # Calculate overall AUPRC
        from sklearn.metrics import precision_recall_curve, auc
        precision, recall, _ = precision_recall_curve(pred_df['true_label'], pred_df['prediction'])
        overall_auprc = auc(recall, precision)
        
        # Convert family results to expected format
        family_results = []
        for _, row in family_df.iterrows():
            family_results.append({
                'family_name': row['family_name'],
                'test_samples': row['test_samples'],
                'test_auprc': row['test_auprc'],
                # Note: We don't have training samples info from evaluation results
                # This would need to be added separately if needed
                'training_samples': None  # Placeholder
            })
        
        # Create summary structure
        return {
            'overall_auprc': overall_auprc,
            'total_samples_evaluated': len(pred_df),
            'coverage_percentage': 100.0,  # Assume 100% since we only have evaluated samples
            'family_results': family_results
        }
    
    else:
        raise ValueError(f"Invalid results path: {results_path}")

def create_evaluation_plot(results_summary, output_dir, output_prefix="evaluation_plot", training_data_dir=None):
    """
    Create a single plot with:
    - Top: Line plot for family-specific AUPRC values + horizontal line for overall AUPRC
    - Bottom: Inverted bar plot for training dataset sizes
    
    Args:
        results_summary: Dictionary containing evaluation results
        output_dir: Directory to save the plot
        output_prefix: Prefix for output filename
        training_data_dir: Directory containing training data files to count training samples
    """
    # Extract data for plotting
    plot_data = []
    for family_result in results_summary['family_results']:
        family_name = family_result['family_name']
        
        # Get training sample count from training data directory
        training_samples = None
        if training_data_dir:
            # Look for file with same name as family
            training_file = os.path.join(training_data_dir, f"{family_name}.tsv")
            if os.path.exists(training_file):
                try:
                    with open(training_file, 'r') as f:
                        training_samples = sum(1 for _ in f) - 1  # Count all lines minus header
                    print(f"Found {family_name}: {training_samples} training samples")
                except Exception as e:
                    print(f"Error reading {training_file}: {e}")
            else:
                print(f"Warning: Training file not found for {family_name} at {training_file}")
        
        # If no training data found, this is an error since it's required
        if training_samples is None:
            raise ValueError(f"Training data not found for family {family_name}")
            
        plot_data.append({
            'family': family_name,
            'test_auprc': family_result['test_auprc'],
            'test_samples': family_result['test_samples'],
            'training_samples': training_samples
        })
    
    # Convert to DataFrame and sort by training samples (descending)
    plot_df = pd.DataFrame(plot_data)
    plot_df = plot_df.sort_values('training_samples', ascending=False)
    
    # Debug: Check for any training samples < 1000
    small_datasets = plot_df[plot_df['training_samples'] < 1000]
    if len(small_datasets) > 0:
        print(f"WARNING: Found {len(small_datasets)} families with <1000 training samples:")
        for _, row in small_datasets.iterrows():
            print(f"  {row['family']}: {row['training_samples']} samples")
    
    # Create log-transformed training size values
    plot_df['Log_Training_Size'] = np.log10(plot_df['training_samples'])
    
    # Get overall AUPRC
    overall_auprc = results_summary['overall_auprc']
    
    # Setup plot
    x = range(len(plot_df))
    
    # Create figure with subplots - unified x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[2, 1], 
                                   gridspec_kw={'hspace': 0}, sharex=True)
    
    # Top plot: Family AUPRCs (line plot) + Overall AUPRC (horizontal line) + Fully Trained line
    ax1.plot(x, plot_df['test_auprc'], marker='o', linewidth=2, markersize=6, 
             color='#000000', label='Family')
    ax1.axhline(y=overall_auprc, color='#FF3333', linestyle='--', linewidth=3,
                label='Overall')
    ax1.axhline(y=0.8593, color='#00AA00', linestyle='--', linewidth=3,
                label='Fully Trained')
    ax1.set_ylabel('PR-AUC', fontsize=20, fontweight='bold')
    ax1.set_ylim(0.7, 1.0)  # Start from 0.7 like in the example
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.legend(loc='upper right', fontsize=16)
    ax1.set_xticks([])  # Remove x-axis labels for top plot
    
    # Make y-axis tick labels bigger and bold
    ax1.tick_params(axis='y', labelsize=18)
    for label in ax1.get_yticklabels():
        label.set_fontweight('bold')
    
    # Set x-axis limits to ensure proper alignment
    ax1.set_xlim(-0.5, len(plot_df) - 0.5)
    
    # Remove 0 from y-axis ticks and also remove 0.70 (with tolerance for floating point)
    y_ticks = ax1.get_yticks()
    y_ticks = [tick for tick in y_ticks if tick != 0.0 and abs(tick - 0.70) > 1e-10]
    ax1.set_yticks(y_ticks)
    
    # Bottom plot: Inverted bar plot for training dataset sizes
    bars = ax2.bar(x, plot_df['Log_Training_Size'], color='#0066CC', alpha=0.7, width=0.8)
    ax2.set_ylabel('Training Dataset Size (log10)', fontsize=20, fontweight='bold')
    
    # Manually position the y-axis label to align with top plot
    ax2.yaxis.set_label_coords(-0.05, 0.5)  # Move it left to align with top
    
    ax2.set_xlabel('miRNA Family (sorted by training set size)', fontsize=18)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.invert_yaxis()  # Invert y-axis so zero is at the top
    
    # Set y-axis limits starting from 0 (inverted, so 0 at top)
    min_log_val = 0.0  # Start from 0
    max_log_val = np.ceil(max(plot_df['Log_Training_Size']))
    ax2.set_ylim(max_log_val, min_log_val)  # Reversed because axis is inverted
    
    # Create appropriate log-scale y-ticks starting from 0
    min_log = 0  # Start from 0
    max_log = np.ceil(max(plot_df['Log_Training_Size']))
    
    # Create reasonable tick spacing including 0
    if max_log - min_log > 20:
        num_ticks = 6
        log_ticks = np.linspace(min_log, max_log, num_ticks)
    else:
        log_ticks = np.arange(min_log, max_log + 1)
    
    ax2.set_yticks(log_ticks)
    
    # Show log values directly instead of converting to original scale
    tick_labels = [f"{y:.0f}" for y in log_ticks]
    ax2.set_yticklabels(tick_labels)
    
    # Make y-axis tick labels bigger and bold
    ax2.tick_params(axis='y', labelsize=18)
    for label in ax2.get_yticklabels():
        label.set_fontweight('bold')
    
    # Set x-axis to match exactly with top plot
    ax2.set_xticks(range(len(plot_df)))
    ax2.set_xticklabels(plot_df['family'], rotation=90, ha='center', fontsize=14)  # Vertical rotation
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"{output_prefix}_performance_plot.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved: {plot_path}")
    return plot_path

def create_summary_table(results_summary, output_dir, output_prefix="evaluation_table"):
    """
    Create a summary table of results.
    
    Args:
        results_summary: Dictionary containing evaluation results
        output_dir: Directory to save the table
        output_prefix: Prefix for output filename
        
    Returns:
        str: Path to saved table file
    """
    # Create summary table
    table_data = []
    for family_result in results_summary['family_results']:
        table_data.append({
            'Family': family_result['family_name'],
            'Test Samples': family_result['test_samples'],
            'Test AUPRC': f"{family_result['test_auprc']:.4f}"
        })
    
    # Convert to DataFrame and sort by AUPRC (descending)
    table_df = pd.DataFrame(table_data)
    table_df['Test AUPRC (numeric)'] = [family_result['test_auprc'] for family_result in results_summary['family_results']]
    table_df = table_df.sort_values('Test AUPRC (numeric)', ascending=False)
    table_df = table_df.drop('Test AUPRC (numeric)', axis=1)
    
    # Add summary row
    total_samples = results_summary['total_samples_evaluated']
    overall_auprc = results_summary['overall_auprc']
    
    summary_row = pd.DataFrame({
        'Family': ['OVERALL'],
        'Test Samples': [total_samples],
        'Test AUPRC': [f"{overall_auprc:.4f}"]
    })
    
    final_table = pd.concat([summary_row, table_df], ignore_index=True)
    
    # Save table
    table_filename = f"{output_prefix}_summary_table.csv"
    table_path = os.path.join(output_dir, table_filename)
    final_table.to_csv(table_path, index=False)
    
    print(f"Summary table saved: {table_path}")
    return table_path

def main():
    parser = argparse.ArgumentParser(description='Create evaluation plots and tables')
    parser.add_argument('--results', type=str, required=True,
                        help='Path to results JSON file or directory containing CSV files')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Directory to save output files (default: current directory)')
    parser.add_argument('--prefix', type=str, default='evaluation',
                        help='Prefix for output filenames (default: evaluation)')
    parser.add_argument('--training-data-dir', type=str, required=True,
                        help='Directory containing training data files to count training samples')
    parser.add_argument('--no-table', action='store_true',
                        help='Skip creating the summary table')
    
    args = parser.parse_args()
    
    try:
        # Load results
        print(f"Loading evaluation results from: {args.results}")
        results_summary = load_evaluation_results(args.results)
        
        print(f"Found {len(results_summary['family_results'])} families")
        print(f"Overall AUPRC: {results_summary['overall_auprc']:.4f}")
        print(f"Total samples evaluated: {results_summary['total_samples_evaluated']:,}")
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Create plot
        print(f"\nCreating evaluation plot...")
        plot_file = create_evaluation_plot(results_summary, args.output_dir, args.prefix, args.training_data_dir)
        
        # Create summary table (unless disabled)
        if not args.no_table:
            print(f"\nCreating summary table...")
            table_file = create_summary_table(results_summary, args.output_dir, args.prefix)
        
        print(f"\nCompleted successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())