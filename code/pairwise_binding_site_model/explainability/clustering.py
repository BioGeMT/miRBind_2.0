import argparse
import os
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .shap_utils import parse_shap_matrix, reduce_2d_to_1d


def load_data(input_file, shap_col, mirna_seq_col, mirna_name_col,
              prediction_filter=None, true_label_col=None, pred_label_col=None):
    """Load and filter data."""
    df = pd.read_csv(input_file, sep='\t')
    print(f"Loaded {len(df)} samples")
    
    df[shap_col] = df[shap_col].apply(parse_shap_matrix)
    df = df.rename(columns={
        shap_col: 'shap_matrix',
        mirna_seq_col: 'mirna_seq',
        mirna_name_col: 'mirna_name'
    })
    
    if prediction_filter and prediction_filter != ['all']:
        masks = {
            'TP': (df[true_label_col] == 1) & (df[pred_label_col] == 1),
            'TN': (df[true_label_col] == 0) & (df[pred_label_col] == 0),
            'FP': (df[true_label_col] == 0) & (df[pred_label_col] == 1),
            'FN': (df[true_label_col] == 1) & (df[pred_label_col] == 0),
        }
        combined = pd.Series([False] * len(df))
        for f in prediction_filter:
            if f in masks:
                combined |= masks[f]
        df = df[combined].copy()
        print(f"Filtered to {len(df)} samples ({'+'.join(prediction_filter)})")
    
    return df


def select_optimal_gmm_components(X, max_components=15):
    """Select optimal GMM components using BIC."""
    max_components = min(max_components, len(X) // 10, 15)
    if max_components < 2:
        return 2
    
    scores = []
    for n in range(2, max_components + 1):
        gmm = GaussianMixture(n_components=n, random_state=42, n_init=3)
        gmm.fit(X)
        scores.append(gmm.bic(X))
    
    return np.argmin(scores) + 2


def global_clustering(df, reduction_method, k, normalize=True, shap_filter='all',
                     clustering_method='kmeans', gmm_components='auto'):
    """Cluster all samples globally."""
    vectors = [reduce_2d_to_1d(m, reduction_method, shap_filter=shap_filter)
               for m in df['shap_matrix']]
    X = np.array(vectors)
    X_original = X.copy()
    
    if normalize:
        X = StandardScaler().fit_transform(X)
    
    if clustering_method == 'gmm':
        n_comp = select_optimal_gmm_components(X) if gmm_components == 'auto' else gmm_components
        model = GaussianMixture(n_components=n_comp, random_state=42, n_init=10)
        clusters = model.fit_predict(X)
        probs = model.predict_proba(X)
    else:
        model = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        clusters = model.fit_predict(X)
        probs = None
    
    # Compute centers from original data
    n_clusters = len(np.unique(clusters))
    centers = np.zeros((n_clusters, X_original.shape[1]))
    for i in range(n_clusters):
        centers[i] = np.mean(X_original[clusters == i], axis=0)
    
    if shap_filter == 'positive':
        centers = np.maximum(centers, 0)
    elif shap_filter == 'negative':
        centers = np.minimum(centers, 0)
    
    df_out = df.copy()
    df_out['cluster'] = clusters
    if probs is not None:
        df_out['cluster_confidence'] = np.max(probs, axis=1)
        for i in range(probs.shape[1]):
            df_out[f'prob_cluster_{i}'] = probs[:, i]
    
    return df_out, centers


def hierarchical_clustering(df, reduction_method, k, min_samples=10, normalize=True,
                           shap_filter='all', clustering_method='kmeans', gmm_components='auto'):
    """Cluster samples within each miRNA group."""
    results = []
    mirna_data = {}
    
    groups = [(seq, grp) for seq, grp in df.groupby('mirna_seq') if len(grp) >= min_samples]
    print(f"Processing {len(groups)} miRNAs with >= {min_samples} samples")
    
    for mirna_seq, group in groups:
        vectors = [reduce_2d_to_1d(m, reduction_method, shap_filter=shap_filter)
                   for m in group['shap_matrix']]
        X = np.array(vectors)
        X_original = X.copy()
        
        if normalize:
            X = StandardScaler().fit_transform(X)
        
        actual_k = min(k, len(group))
        
        if clustering_method == 'gmm':
            n_comp = min(select_optimal_gmm_components(X, max_components=10), actual_k)
            model = GaussianMixture(n_components=n_comp, random_state=42, n_init=10)
        else:
            model = KMeans(n_clusters=actual_k, init='k-means++', n_init=10, random_state=42)
        
        clusters = model.fit_predict(X)
        
        centers = np.zeros((len(np.unique(clusters)), X_original.shape[1]))
        for i in range(len(centers)):
            centers[i] = np.mean(X_original[clusters == i], axis=0)
        
        group_out = group.copy()
        group_out['cluster'] = clusters
        results.append(group_out)
        
        mirna_data[mirna_seq] = {
            'centers': centers,
            'n_samples': len(group),
            'mirna_name': group['mirna_name'].iloc[0],
        }
    
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame(), mirna_data


def save_results(output_dir, global_centers, mirna_data, params, approach):
    """Save clustering results."""
    output_dir = Path(output_dir)
    
    if global_centers is not None:
        np.save(output_dir / 'global_cluster_centers.npy', global_centers)
    
    if mirna_data:
        with open(output_dir / 'mirna_cluster_data.pkl', 'wb') as f:
            pickle.dump(mirna_data, f)
    
    with open(output_dir / 'clustering_params.json', 'w') as f:
        json.dump(params, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description='Cluster SHAP values')
    
    parser.add_argument('--input', required=True)
    parser.add_argument('--shap_col', required=True)
    parser.add_argument('--mirna_seq_col', required=True)
    parser.add_argument('--mirna_name_col', required=True)
    parser.add_argument('--output_dir', default='clustering_results')
    
    parser.add_argument('--approach', choices=['global', 'hierarchical', 'both'], default='both')
    parser.add_argument('--reduction_method', default='max',
                       choices=['sum', 'mean', 'max', 'abs_sum', 'max_pos', 'max_neg'])
    parser.add_argument('--clustering_method', choices=['kmeans', 'gmm'], default='kmeans')
    parser.add_argument('--k_global', type=int, default=5)
    parser.add_argument('--k_mirna', type=int, default=3)
    parser.add_argument('--gmm_components', default='auto')
    parser.add_argument('--min_samples_per_mirna', type=int, default=10)
    parser.add_argument('--normalize', type=bool, default=True)
    
    parser.add_argument('--prediction_filter', nargs='+', default=['all'],
                       choices=['TP', 'TN', 'FP', 'FN', 'all'])
    parser.add_argument('--true_label_col')
    parser.add_argument('--pred_label_col')
    parser.add_argument('--shap_filter', choices=['all', 'positive', 'negative'], default='all')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.gmm_components != 'auto':
        args.gmm_components = int(args.gmm_components)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    df = load_data(args.input, args.shap_col, args.mirna_seq_col, args.mirna_name_col,
                   args.prediction_filter, args.true_label_col, args.pred_label_col)
    
    params = {
        'reduction_method': args.reduction_method,
        'approach': args.approach,
        'clustering_method': args.clustering_method,
        'k_global': args.k_global,
        'k_mirna': args.k_mirna,
        'shap_filter': args.shap_filter,
        'prediction_filter': args.prediction_filter,
    }
    
    global_centers, mirna_data = None, {}
    
    if args.approach in ['global', 'both']:
        print("\nGlobal clustering...")
        df_global, global_centers = global_clustering(
            df, args.reduction_method, args.k_global, args.normalize,
            args.shap_filter, args.clustering_method, args.gmm_components
        )
        df_global.drop('shap_matrix', axis=1).to_csv(
            output_dir / 'global_clusters.tsv', sep='\t', index=False
        )
        print(f"Cluster sizes: {df_global['cluster'].value_counts().sort_index().to_dict()}")
    
    if args.approach in ['hierarchical', 'both']:
        print("\nHierarchical clustering...")
        df_hier, mirna_data = hierarchical_clustering(
            df, args.reduction_method, args.k_mirna, args.min_samples_per_mirna,
            args.normalize, args.shap_filter, args.clustering_method, args.gmm_components
        )
        if not df_hier.empty:
            df_hier.drop('shap_matrix', axis=1).to_csv(
                output_dir / 'hierarchical_clusters.tsv', sep='\t', index=False
            )
            print(f"Processed {len(mirna_data)} miRNAs")
    
    save_results(output_dir, global_centers, mirna_data, params, args.approach)
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
