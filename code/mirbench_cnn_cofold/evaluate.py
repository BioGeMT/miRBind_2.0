"""
Evaluates predictions against labels, computing average precision and AUC-PR, and saves results (with metadata) as a JSON file.

Usage:
    python evaluate.py --preds_path <PREDS_NPY> --labels_path <LABELS_NPY> --output_path <OUTPUT_JSON> --model_name <MODEL_NAME> --test_set_name <TEST_SET>

Arguments:
    --preds_path     Path to predictions file (.npy)
    --labels_path    Path to labels file (.npy)
    --output_path    Output path for evaluation metrics (.json)
    --model_name     Name of the evaluated model
    --test_set_name  Name of the test set used
"""

import numpy as np
import argparse
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import json  

def main():
    parser = argparse.ArgumentParser(description='Evaluate performance metrics for predictions and labels.')
    parser.add_argument('--preds_path', type=str, required=True, help='Path to the predictions (.npy) file')
    parser.add_argument('--labels_path', type=str, required=True, help='Path to the labels (.npy) file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the evaluation metrics')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model being evaluated')
    parser.add_argument('--test_set_name', type=str, required=True, help='Name of the test set used for evaluation')

    args = parser.parse_args()

    preds = np.load(args.preds_path)
    labels = np.memmap(args.labels_path, dtype='float32', mode='r', shape=(len(preds),))

    aps = average_precision_score(labels, preds)
    precision, recall, _ = precision_recall_curve(labels, preds)
    auc_pr = auc(recall, precision)

    metrics = {
        "Average Precision Score": aps,
        "AUC-PR": auc_pr
    }

    metadata = {
        "model_name": args.model_name,
        "test_set_name": args.test_set_name,
    }

    result = {
        "metrics": metrics,
        "metadata": metadata
    }

    with open(args.output_path, 'w') as f:
        json.dump(result, f, indent=4)

    print(f"Evaluation metrics saved to: {args.output_path}")

if __name__ == "__main__":
    main()
