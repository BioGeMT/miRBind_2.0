import os
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import NUCLEOTIDE_COLORS
from .shap_utils import load_shap_from_json, compute_global_shap_range, normalize_shap_values


def plot_shap_heatmap(shap_2d, mirna_seq, target_seq, output_path,
                      prediction_score=None, predicted_class=None, true_label=None,
                      colormap='seismic', vmin=-1, vmax=1, sample_id=None):
    """Plot SHAP heatmap for a single sample."""
    fig_width = max(12, len(target_seq) * 0.3)
    fig_height = max(8, len(mirna_seq) * 0.3)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    im = ax.imshow(shap_2d, cmap=colormap, aspect='auto', norm=norm)
    
    plt.colorbar(im, label='SHAP Value')
    
    ax.set_xticks(range(len(target_seq)))
    ax.set_yticks(range(len(mirna_seq)))
    ax.set_xticklabels(list(target_seq))
    ax.set_yticklabels(list(mirna_seq))
    ax.set_xlabel('mRNA Sequence')
    ax.set_ylabel('miRNA Sequence')
    
    title_parts = [f'Sample {sample_id}' if sample_id else 'SHAP Heatmap']
    if prediction_score is not None:
        title_parts.append(f'Score: {prediction_score:.3f}')
    if predicted_class is not None:
        title_parts.append(f'Pred: {predicted_class}')
    if true_label is not None:
        title_parts.append(f'True: {true_label}')
    ax.set_title(' | '.join(title_parts))
    
    if len(target_seq) > 20:
        plt.setp(ax.get_xticklabels(), rotation=90, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_cluster_centers(centers, output_path, title='Cluster Centers',
                        colormap='RdBu_r', indexing='5prime'):
    """Plot heatmap of cluster centers."""
    plt.figure(figsize=(14, 6))
    
    if indexing == '3prime':
        centers = centers[:, ::-1]
    
    sns.heatmap(centers, cmap=colormap, center=0, cbar_kws={'label': 'SHAP value'},
                annot=centers.shape[1] <= 30, fmt='.2f')
    
    xlabel = "Position from 3' end" if indexing == '3prime' else "Position from 5' end"
    plt.xlabel(xlabel)
    plt.ylabel('Cluster')
    plt.title(f"{title}\n(Indexed: {'5 -> 3' if indexing == '5prime' else '3 -> 5'})")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_nucleotide_importance(mirna_seq, importance_data, output_path,
                               plot_type='bar', axis_mode='mirna'):
    """Plot nucleotide importance for a miRNA."""
    n_samples = importance_data['n_samples']
    aggregated = importance_data['aggregated']
    use_both = importance_data.get('use_both', False)
    
    if use_both:
        pos_values = aggregated['positive']
        neg_values = aggregated['negative']
        importance_length = len(pos_values)
    else:
        values = aggregated if isinstance(aggregated, np.ndarray) else aggregated.get('mean', aggregated)
        importance_length = len(values)
    
    positions = np.arange(importance_length)
    
    if axis_mode == 'mirna':
        nucleotides = list(mirna_seq[:importance_length])
        if len(nucleotides) < importance_length:
            nucleotides += ['N'] * (importance_length - len(nucleotides))
        colors = [NUCLEOTIDE_COLORS.get(n, 'gray') for n in nucleotides]
    else:
        nucleotides = [str(i+1) for i in positions]
        colors = ['steelblue'] * importance_length
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if plot_type == 'bar':
        if use_both:
            ax.bar(positions, pos_values, color=colors, alpha=0.7, label='Positive')
            ax.bar(positions, neg_values, color=colors, alpha=0.5, hatch='//', label='Negative')
            ax.axhline(y=0, color='black', linewidth=0.5)
            ax.legend()
        else:
            ax.bar(positions, values, color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_xticks(positions)
    ax.set_xticklabels(nucleotides)
    ax.set_xlabel('Position' if axis_mode == 'mrna' else 'Nucleotide')
    ax.set_ylabel('Importance Score')
    ax.set_title(f'{mirna_seq}\n(n={n_samples} samples)')
    ax.grid(True, alpha=0.3, axis='y')
    
    if importance_length > 20:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def batch_plot_shap_heatmaps(df, output_dir, shap_column='shap_values_2d',
                             mirna_column='noncodingRNA', target_column='gene',
                             num_samples=10, selection_method='first',
                             class_filter=None, normalize=True):
    """Generate SHAP heatmaps for multiple samples."""
    os.makedirs(output_dir, exist_ok=True)
    
    if class_filter is not None:
        df = df[df['predicted_class'] == class_filter]
    
    if selection_method == 'random':
        indices = df.sample(min(num_samples, len(df)), random_state=42).index.tolist()
    elif selection_method == 'top_scores':
        indices = df.nlargest(num_samples, 'prediction_score').index.tolist()
    elif selection_method == 'bottom_scores':
        indices = df.nsmallest(num_samples, 'prediction_score').index.tolist()
    else:
        indices = df.head(num_samples).index.tolist()
    
    if normalize:
        _, _, global_abs_max = compute_global_shap_range(df, shap_column)
        vmin, vmax = -1, 1
    else:
        vmin, vmax = None, None
        global_abs_max = None
    
    for i, idx in enumerate(indices):
        row = df.loc[idx]
        shap_2d = load_shap_from_json(row[shap_column])
        
        if normalize and global_abs_max:
            shap_2d = normalize_shap_values(shap_2d, global_abs_max)
        elif vmin is None:
            abs_max = np.max(np.abs(shap_2d))
            vmin, vmax = -abs_max, abs_max
        
        output_path = os.path.join(output_dir, f"shap_sample_{idx}.png")
        plot_shap_heatmap(
            shap_2d, row[mirna_column], row[target_column], output_path,
            prediction_score=row.get('prediction_score'),
            predicted_class=row.get('predicted_class'),
            true_label=row.get('label'),
            vmin=vmin, vmax=vmax, sample_id=idx
        )
        print(f"  [{i+1}/{len(indices)}] Saved: {output_path}")


def load_clustering_data(data_dir):
    """Load saved clustering data for plotting."""
    data_dir = Path(data_dir)
    
    with open(data_dir / 'clustering_params.json', 'r') as f:
        params = json.load(f)
    
    global_centers = None
    centers_file = data_dir / 'global_cluster_centers.npy'
    if centers_file.exists():
        global_centers = np.load(centers_file)
    
    mirna_data = {}
    mirna_file = data_dir / 'mirna_cluster_data.pkl'
    if mirna_file.exists():
        with open(mirna_file, 'rb') as f:
            mirna_data = pickle.load(f)
    
    return params, global_centers, mirna_data


def plot_clustering_results(data_dir, output_dir=None, indexing='both'):
    """Generate plots from saved clustering data."""
    if output_dir is None:
        output_dir = data_dir
    output_dir = Path(output_dir)
    
    params, global_centers, mirna_data = load_clustering_data(data_dir)
    
    indexings = ['5prime', '3prime'] if indexing == 'both' else [indexing]
    
    for idx_mode in indexings:
        idx_dir = output_dir / f'plots_{idx_mode}'
        idx_dir.mkdir(exist_ok=True, parents=True)
        
        if global_centers is not None:
            plot_cluster_centers(
                global_centers, idx_dir / 'global_centers.png',
                title='Global Cluster Centers', indexing=idx_mode
            )
        
        if mirna_data:
            mirna_dir = idx_dir / 'mirna_clusters'
            mirna_dir.mkdir(exist_ok=True)
            
            for mirna_seq, data in mirna_data.items():
                safe_name = mirna_seq.replace('/', '_')[:50]
                plot_cluster_centers(
                    data['centers'], mirna_dir / f'{safe_name}_centers.png',
                    title=f"{data.get('mirna_name', mirna_seq)} (n={data['n_samples']})",
                    indexing=idx_mode
                )
    
    print(f"Plots saved to: {output_dir}")
