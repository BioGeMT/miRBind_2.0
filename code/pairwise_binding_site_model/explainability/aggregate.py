import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import NUCLEOTIDE_COLORS
from .shap_utils import load_shap_from_json, reduce_2d_to_1d, stratify_samples


def compute_consensus_sequence(sequences):
    """Compute consensus nucleotide at each position (>70% frequency) or show distribution."""
    if not sequences:
        return []
    
    max_len = max(len(seq) for seq in sequences)
    consensus = []
    
    for pos in range(max_len):
        nucleotides = [seq[pos] for seq in sequences if pos < len(seq)]
        if not nucleotides:
            consensus.append('N')
            continue
        
        counts = Counter(nucleotides)
        total = len(nucleotides)
        most_common, count = counts.most_common(1)[0]
        
        if count / total > 0.7:
            consensus.append(most_common)
        else:
            consensus.append({nt: c/total for nt, c in counts.items()})
    
    return consensus


def aggregate_importances_by_mirna(df, shap_column='shap_values_2d', mirna_column='miRNA_sequence',
                                   target_column='target_sequence', reduction_method='sum',
                                   aggregation_method='mean', axis_mode='mirna'):
    """Aggregate nucleotide importances for each unique miRNA."""
    mirna_importances = {}
    unique_mirnas = df[mirna_column].unique()
    print(f"Found {len(unique_mirnas)} unique miRNAs")
    
    use_both = (reduction_method == 'max_both')
    
    for mirna_seq in unique_mirnas:
        samples = df[df[mirna_column] == mirna_seq]
        vectors, vectors_pos, vectors_neg = [], [], []
        target_sequences = []
        
        for _, row in samples.iterrows():
            try:
                shap_2d = load_shap_from_json(row[shap_column])
                shap_1d = reduce_2d_to_1d(shap_2d, reduction_method, axis_mode)
                
                if axis_mode == 'mrna' and target_column in df.columns:
                    target_sequences.append(row[target_column])
                
                if use_both:
                    vectors_pos.append(shap_1d['positive'])
                    vectors_neg.append(shap_1d['negative'])
                else:
                    vectors.append(shap_1d)
            except Exception as e:
                print(f"Warning: {mirna_seq}: {e}")
                continue
        
        if (use_both and vectors_pos) or (not use_both and vectors):
            result = _aggregate_vectors(
                vectors_pos if use_both else vectors,
                vectors_neg if use_both else None,
                aggregation_method, use_both
            )
            result['n_samples'] = len(vectors_pos if use_both else vectors)
            result['use_both'] = use_both
            result['axis_mode'] = axis_mode
            
            if axis_mode == 'mrna' and target_sequences:
                result['target_consensus'] = compute_consensus_sequence(target_sequences)
            
            mirna_importances[mirna_seq] = result
            print(f"  {mirna_seq}: {result['n_samples']} samples")
    
    return mirna_importances


def _aggregate_vectors(vectors, neg_vectors, method, use_both):
    """Aggregate importance vectors using specified method."""
    result = {}
    
    if use_both:
        pos_matrix = np.array(vectors)
        neg_matrix = np.array(neg_vectors)
        
        if method == 'mean':
            result['aggregated'] = {
                'positive': np.mean(pos_matrix, axis=0),
                'negative': np.mean(neg_matrix, axis=0),
            }
        elif method == 'median':
            result['aggregated'] = {
                'positive': np.median(pos_matrix, axis=0),
                'negative': np.median(neg_matrix, axis=0),
            }
        elif method == 'max':
            result['aggregated'] = {
                'positive': np.max(pos_matrix, axis=0),
                'negative': np.min(neg_matrix, axis=0),
            }
        
        result['all_values_pos'] = pos_matrix
        result['all_values_neg'] = neg_matrix
        result['importance_length'] = pos_matrix.shape[1]
    else:
        matrix = np.array(vectors)
        
        if method == 'mean':
            result['aggregated'] = np.mean(matrix, axis=0)
        elif method == 'median':
            result['aggregated'] = np.median(matrix, axis=0)
        elif method == 'max':
            max_idx = np.argmax(np.abs(matrix), axis=0)
            result['aggregated'] = np.array([matrix[max_idx[i], i] for i in range(matrix.shape[1])])
        
        result['all_values'] = matrix
        result['importance_length'] = matrix.shape[1]
    
    return result


def plot_nucleotide_importance(mirna_seq, importance_data, output_dir, plot_type='bar',
                               axis_mode='mirna', stratify_label=''):
    """Plot nucleotide importance for a single miRNA."""
    os.makedirs(output_dir, exist_ok=True)
    
    n_samples = importance_data['n_samples']
    aggregated = importance_data['aggregated']
    use_both = importance_data.get('use_both', False)
    importance_length = importance_data['importance_length']
    
    positions = np.arange(importance_length)
    
    if axis_mode == 'mirna':
        nucleotides = list(mirna_seq[:importance_length])
        nucleotides += ['N'] * (importance_length - len(nucleotides))
        colors = [NUCLEOTIDE_COLORS.get(n, 'gray') for n in nucleotides]
    else:
        nucleotides = [str(i+1) for i in positions]
        colors = ['steelblue'] * importance_length
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if use_both:
        ax.bar(positions, aggregated['positive'], color=colors, alpha=0.7, label='Positive')
        ax.bar(positions, aggregated['negative'], color=colors, alpha=0.5, hatch='//', label='Negative')
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.legend()
    else:
        values = aggregated if isinstance(aggregated, np.ndarray) else aggregated.get('mean', aggregated)
        ax.bar(positions, values, color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_xticks(positions)
    ax.set_xticklabels(nucleotides)
    ax.set_xlabel('Position' if axis_mode == 'mrna' else 'Nucleotide')
    ax.set_ylabel('Importance Score')
    
    title = f'{mirna_seq}'
    if stratify_label:
        title += f' ({stratify_label})'
    title += f'\n(n={n_samples})'
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    
    if importance_length > 20:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    
    plt.tight_layout()
    
    safe_name = mirna_seq.replace('/', '_')[:50]
    suffix = f'_{stratify_label.lower()}' if stratify_label else ''
    filename = f"importance_{safe_name}_{plot_type}_{axis_mode}{suffix}.png"
    output_path = os.path.join(output_dir, filename)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_summary_statistics(mirna_importances, output_file, axis_mode='mirna'):
    """Create summary CSV with statistics for each miRNA."""
    summary = []
    
    for mirna_seq, data in mirna_importances.items():
        aggregated = data['aggregated']
        use_both = data.get('use_both', False)
        
        if use_both:
            pos_vals = aggregated['positive']
            neg_vals = aggregated['negative']
            combined = np.abs(pos_vals) + np.abs(neg_vals)
            max_idx = np.argmax(combined)
            
            row = {
                'miRNA_sequence': mirna_seq,
                'n_samples': data['n_samples'],
                'mean_pos': float(np.mean(pos_vals)),
                'mean_neg': float(np.mean(neg_vals)),
                'max_pos': float(np.max(pos_vals)),
                'max_neg': float(np.min(neg_vals)),
                'total_abs': float(np.sum(combined)),
            }
        else:
            values = aggregated if isinstance(aggregated, np.ndarray) else aggregated.get('mean', aggregated)
            max_idx = np.argmax(np.abs(values))
            
            row = {
                'miRNA_sequence': mirna_seq,
                'n_samples': data['n_samples'],
                'mean': float(np.mean(values)),
                'max': float(np.max(values)),
                'min': float(np.min(values)),
                'total_abs': float(np.sum(np.abs(values))),
            }
        
        if axis_mode == 'mirna':
            row['most_important_pos'] = max_idx + 1
            row['most_important_nt'] = mirna_seq[max_idx] if max_idx < len(mirna_seq) else 'N'
        else:
            row['most_important_target_pos'] = max_idx + 1
        
        summary.append(row)
    
    df = pd.DataFrame(summary).sort_values('total_abs', ascending=False)
    df.to_csv(output_file, index=False)
    return df


def parse_args():
    parser = argparse.ArgumentParser(description='Aggregate miRNA nucleotide importances')
    
    parser.add_argument('--input_file', required=True)
    parser.add_argument('--output_dir', default='importance_plots')
    
    parser.add_argument('--axis_mode', choices=['mirna', 'mrna'], default='mirna')
    parser.add_argument('--stratify_by', nargs='+', default=['all'],
                       choices=['all', 'TP', 'TN', 'FP', 'FN', 'correct', 'incorrect'])
    parser.add_argument('--reduction_method', default='sum',
                       choices=['sum', 'mean', 'max', 'abs_sum', 'max_both', 'max_pos', 'max_neg'])
    parser.add_argument('--aggregation_method', default='mean',
                       choices=['mean', 'median', 'max'])
    parser.add_argument('--plot_type', default='bar', choices=['bar', 'violin', 'box'])
    parser.add_argument('--plot_top_n', type=int, default=20)
    parser.add_argument('--min_samples', type=int, default=1)
    
    parser.add_argument('--shap_column', default='shap_values_2d')
    parser.add_argument('--mirna_column', default='miRNA_sequence')
    parser.add_argument('--target_column', default='target_sequence')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"Loading: {args.input_file}")
    df = pd.read_csv(args.input_file, sep='\t')
    print(f"Loaded {len(df)} samples")
    
    mirna_counts = df[args.mirna_column].value_counts()
    top_mirnas = mirna_counts.head(args.plot_top_n).index.tolist()
    print(f"Top {len(top_mirnas)} miRNAs by sample count")
    
    for stratify in args.stratify_by:
        print(f"\n{'='*60}\nStratification: {stratify}\n{'='*60}")
        
        df_filtered = stratify_samples(df.copy(), stratify)
        if len(df_filtered) == 0:
            print(f"No samples for {stratify}")
            continue
        
        importances = aggregate_importances_by_mirna(
            df_filtered, args.shap_column, args.mirna_column, args.target_column,
            args.reduction_method, args.aggregation_method, args.axis_mode
        )
        
        if args.min_samples > 1:
            importances = {k: v for k, v in importances.items() if v['n_samples'] >= args.min_samples}
        
        stratify_dir = os.path.join(args.output_dir, stratify.lower())
        os.makedirs(stratify_dir, exist_ok=True)
        
        summary_file = os.path.join(stratify_dir, f'summary_{args.axis_mode}.csv')
        create_summary_statistics(importances, summary_file, args.axis_mode)
        print(f"Summary: {summary_file}")
        
        for mirna_seq in top_mirnas:
            if mirna_seq in importances:
                path = plot_nucleotide_importance(
                    mirna_seq, importances[mirna_seq], stratify_dir,
                    args.plot_type, args.axis_mode, stratify
                )
                print(f"  {mirna_seq}: {os.path.basename(path)}")
    
    print(f"\nComplete. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
