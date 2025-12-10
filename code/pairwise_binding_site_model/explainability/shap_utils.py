import json
import ast
import numpy as np


def load_shap_from_json(shap_json_str):
    """Load SHAP values from JSON string."""
    return np.array(json.loads(shap_json_str)['values'])


def parse_shap_matrix(shap_value):
    """Parse SHAP matrix from various formats."""
    if isinstance(shap_value, str):
        shap_value = ast.literal_eval(shap_value)
    
    if isinstance(shap_value, (list, tuple)):
        return np.array(shap_value)
    elif isinstance(shap_value, dict) and 'values' in shap_value:
        return np.array(shap_value['values'])
    elif isinstance(shap_value, np.ndarray):
        return shap_value
    else:
        raise ValueError(f"Unsupported format: {type(shap_value)}")


def reduce_2d_to_1d(shap_2d, method='sum', axis='mirna', shap_filter='all'):
    """
    Reduce 2D SHAP matrix to 1D.
    
    Args:
        shap_2d: Array of shape (mirna_length, target_length)
        method: 'sum', 'mean', 'max', 'abs_sum', 'max_both', 'max_pos', 'max_neg'
        axis: 'mirna' (reduce along target) or 'mrna' (reduce along miRNA)
        shap_filter: 'all', 'positive', 'negative'
    """
    shap_2d = shap_2d.copy()
    
    if shap_filter == 'positive':
        shap_2d[shap_2d < 0] = 0
    elif shap_filter == 'negative':
        shap_2d[shap_2d > 0] = 0
    
    reduction_axis = 1 if axis == 'mirna' else 0
    
    if method == 'max_both':
        return {
            'positive': np.maximum(np.max(shap_2d, axis=reduction_axis), 0),
            'negative': np.minimum(np.min(shap_2d, axis=reduction_axis), 0)
        }
    
    reductions = {
        'sum': lambda: np.sum(shap_2d, axis=reduction_axis),
        'mean': lambda: np.mean(shap_2d, axis=reduction_axis),
        'max': lambda: np.max(np.abs(shap_2d), axis=reduction_axis),
        'abs_sum': lambda: np.sum(np.abs(shap_2d), axis=reduction_axis),
        'max_pos': lambda: np.maximum(np.max(shap_2d, axis=reduction_axis), 0),
        'max_neg': lambda: np.minimum(np.min(shap_2d, axis=reduction_axis), 0),
    }
    
    return reductions[method]()


def compute_shap_statistics(shap_2d):
    """Compute statistics for a SHAP matrix."""
    return {
        'mean': float(np.mean(shap_2d)),
        'std': float(np.std(shap_2d)),
        'min': float(np.min(shap_2d)),
        'max': float(np.max(shap_2d)),
        'abs_mean': float(np.mean(np.abs(shap_2d))),
        'abs_max': float(np.max(np.abs(shap_2d))),
    }


def compute_global_shap_range(df, shap_column):
    """Compute global min/max SHAP values across all samples."""
    global_min, global_max = float('inf'), float('-inf')
    
    for _, row in df.iterrows():
        try:
            shap_2d = load_shap_from_json(row[shap_column])
            global_min = min(global_min, np.min(shap_2d))
            global_max = max(global_max, np.max(shap_2d))
        except Exception:
            continue
    
    return global_min, global_max, max(abs(global_min), abs(global_max))


def normalize_shap_values(shap_2d, global_abs_max):
    """Normalize SHAP values to [-1, 1]."""
    return shap_2d / global_abs_max if global_abs_max != 0 else shap_2d


def stratify_samples(df, stratify_by='all'):
    """Filter dataframe by prediction outcome."""
    if stratify_by == 'all' or 'label' not in df.columns:
        return df
    
    pred, true = df['predicted_class'], df['label']
    
    filters = {
        'TP': (pred == 1) & (true == 1),
        'TN': (pred == 0) & (true == 0),
        'FP': (pred == 1) & (true == 0),
        'FN': (pred == 0) & (true == 1),
        'correct': pred == true,
        'incorrect': pred != true,
    }
    
    return df[filters.get(stratify_by, slice(None))]
