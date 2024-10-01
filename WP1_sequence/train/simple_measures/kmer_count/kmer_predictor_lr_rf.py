#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import random
import joblib  # For saving models

# Constants
K_MIN = 2
K_MAX = 12
TRAIN_SAMPLE_SIZE = 500000  # Number of training samples to use

# Desired miRNA length
MI_RNA_LENGTH = 20

# Complement map for reverse complement calculation
COMPLEMENT_MAP = str.maketrans('ATGC', 'TACG')


def reverse_complement(seq: str) -> str:
    """
    Return the reverse complement of a nucleotide sequence.

    Args:
        seq (str): Nucleotide sequence.

    Returns:
        str: Reverse complement of the sequence.
    """
    return seq.translate(COMPLEMENT_MAP)[::-1]


def preprocess_miRNA(noncodingRNA: str, desired_length: int = MI_RNA_LENGTH) -> str:
    """
    Truncate or pad the miRNA sequence to the desired length.

    Args:
        noncodingRNA (str): Original miRNA sequence.
        desired_length (int): Desired length of miRNA sequence.

    Returns:
        str: Processed miRNA sequence of exact desired_length.
    """
    noncodingRNA = ''.join([c for c in noncodingRNA.upper() if c in ['A', 'T', 'G', 'C', 'N']])
    if len(noncodingRNA) > desired_length:
        return noncodingRNA[:desired_length]
    else:
        # Pad with 'N' if shorter than desired_length
        return noncodingRNA.ljust(desired_length, 'N')


def count_kmers(noncodingRNA: str, gene: str, k: int) -> Dict[str, int]:
    """
    Count occurrences of k-mer reverse complements from noncodingRNA in the gene sequence.

    Args:
        noncodingRNA (str): miRNA sequence (processed to desired length).
        gene (str): Target gene sequence.
        k (int): Length of k-mer.

    Returns:
        Dict[str, int]: Dictionary with keys as 'pos{position}_k{k}' and values as counts.
    """
    kmer_counts = {}
    # Iterate through each possible k-mer position in noncodingRNA
    for i in range(len(noncodingRNA) - k + 1):
        kmer = noncodingRNA[i:i + k]
        rev_complement = reverse_complement(kmer)
        # Count occurrences of the reverse complement in the gene sequence
        count = gene.count(rev_complement)
        feature_name = f'pos{i + 1}_k{k}'
        kmer_counts[feature_name] = count
    return kmer_counts


def extract_features(train_file: str, k_min: int = K_MIN, k_max: int = K_MAX, sample_size: int = TRAIN_SAMPLE_SIZE):
    """
    Extract k-mer features and labels from the training data.

    Args:
        train_file (str): Path to the training data TSV file with 'noncodingRNA', 'gene', 'label' columns.
        k_min (int): Minimum k-mer length.
        k_max (int): Maximum k-mer length.
        sample_size (int): Number of samples to randomly select for training.

    Returns:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Labels.
    """
    # Read the first few lines to determine total rows
    print("Determining total number of training samples...")
    with open(train_file, 'r') as f:
        total_rows = sum(1 for _ in f) - 1  # Subtract header

    if sample_size > total_rows:
        print(f"Warning: Sample size {sample_size} is greater than total available samples {total_rows}. Using all samples.")
        sample_size = total_rows

    # Randomly select sample indices
    print(f"Selecting {sample_size} random samples from {total_rows} training samples...")
    random.seed(42)  # For reproducibility
    sampled_indices = set(random.sample(range(total_rows), sample_size))

    # Initialize lists to hold features and labels
    X_list = []
    y_list = []

    # Read the training file in chunks and extract sampled data
    chunk_size = 10000
    current_index = 0
    print("Extracting features from training data...")
    train_iter = pd.read_csv(train_file, sep='\t', chunksize=chunk_size)

    for chunk in tqdm(train_iter, total=np.ceil(total_rows / chunk_size), desc="Training Data Chunks"):
        # Determine which rows in the chunk are sampled
        # Calculate the global indices for the current chunk
        chunk_indices = range(current_index, current_index + len(chunk))
        sampled_chunk = chunk.iloc[[i - current_index for i in range(current_index, current_index + len(chunk)) if i in sampled_indices]]

        for _, row in sampled_chunk.iterrows():
            noncodingRNA = preprocess_miRNA(str(row['noncodingRNA']))
            gene = str(row['gene']).upper()
            label = row['label']

            kmer_counts = {}
            for k in range(k_min, k_max + 1):
                kmers = count_kmers(noncodingRNA, gene, k)
                for feature, count in kmers.items():
                    if feature in kmer_counts:
                        kmer_counts[feature] += count
                    else:
                        kmer_counts[feature] = count

            X_list.append(kmer_counts)
            y_list.append(label)

        current_index += chunk_size

    # Convert list of dicts to DataFrame
    X = pd.DataFrame(X_list).fillna(0)
    y = pd.Series(y_list)

    print("Feature extraction completed.")
    return X, y


def train_models(X: pd.DataFrame, y: pd.Series):
    """
    Train Logistic Regression and Random Forest models with the given features and labels.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Labels.

    Returns:
        lr_model (LogisticRegression): Trained Logistic Regression model.
        rf_model (RandomForestClassifier): Trained Random Forest model.
        scaler (StandardScaler): Fitted scaler object for Logistic Regression.
        metrics_lr (dict): Evaluation metrics for Logistic Regression.
        metrics_rf (dict): Evaluation metrics for Random Forest.
    """
    # Split into training and validation sets
    print("Splitting data into training and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature scaling for Logistic Regression
    print("Scaling features for Logistic Regression...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Initialize Logistic Regression model with regularization and class balancing
    print("Training Logistic Regression model...")
    lr_model = LogisticRegression(
        class_weight='balanced',
        penalty='l2',
        solver='liblinear',
        max_iter=1000
    )
    lr_model.fit(X_train_scaled, y_train)

    print("Logistic Regression training completed.")

    # Initialize Random Forest model with class balancing
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(
        class_weight='balanced',
        n_estimators=100,
        random_state=42,
        n_jobs=-1  # Utilize all available cores
    )
    rf_model.fit(X_train, y_train)

    print("Random Forest training completed.")

    # Evaluate Logistic Regression on validation set
    print("Evaluating Logistic Regression model on validation set...")
    y_val_pred_lr = lr_model.predict_proba(X_val_scaled)[:, 1]
    metrics_lr = evaluate_model(y_val, y_val_pred_lr)

    # Evaluate Random Forest on validation set
    print("Evaluating Random Forest model on validation set...")
    y_val_pred_rf = rf_model.predict_proba(X_val)[:, 1]
    metrics_rf = evaluate_model(y_val, y_val_pred_rf)

    print("\nLogistic Regression Validation Metrics:")
    for metric, value in metrics_lr.items():
        print(f"{metric}: {value:.4f}")

    print("\nRandom Forest Validation Metrics:")
    for metric, value in metrics_rf.items():
        print(f"{metric}: {value:.4f}")

    return lr_model, rf_model, scaler, metrics_lr, metrics_rf


def evaluate_model(true_labels: np.ndarray, predicted_scores: np.ndarray, threshold: float = 0.5):
    """
    Evaluate the model using common classification metrics, including ROC-AUC and PR AUC.

    Args:
        true_labels (np.ndarray): True binary labels (0 or 1).
        predicted_scores (np.ndarray): Predicted probability scores from the model (floats between 0 and 1).
        threshold (float): Threshold to convert probabilities to binary labels for precision, recall, and F1-Score.

    Returns:
        metrics (dict): Dictionary containing ROC-AUC, PR-AUC, Precision, Recall, and F1-Score.
    """
    # Convert predicted probabilities to binary labels based on the threshold
    predicted_labels = (predicted_scores >= threshold).astype(int)

    # Calculate ROC-AUC
    roc_auc = roc_auc_score(true_labels, predicted_scores)

    # Calculate PR AUC (Average Precision Score)
    pr_auc = average_precision_score(true_labels, predicted_scores)

    # Calculate Precision
    precision = precision_score(true_labels, predicted_labels, zero_division=0)

    # Calculate Recall
    recall = recall_score(true_labels, predicted_labels, zero_division=0)

    # Calculate F1-Score
    f1 = f1_score(true_labels, predicted_labels, zero_division=0)

    # Compile all metrics into a dictionary
    metrics = {
        'ROC-AUC': roc_auc,
        'PR-AUC': pr_auc,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }

    return metrics


def predict_with_models(inference_file: str, lr_model: LogisticRegression, rf_model: RandomForestClassifier, scaler: StandardScaler, k_min: int = K_MIN, k_max: int = K_MAX, batch_size: int = 10000):
    """
    Predict probability scores on inference data using the trained Logistic Regression and Random Forest models.

    Args:
        inference_file (str): Path to the inference data TSV file with 'noncodingRNA' and 'gene' columns.
        lr_model (LogisticRegression): Trained Logistic Regression model.
        rf_model (RandomForestClassifier): Trained Random Forest model.
        scaler (StandardScaler): Fitted scaler object for Logistic Regression.
        k_min (int): Minimum k-mer length.
        k_max (int): Maximum k-mer length.
        batch_size (int): Number of samples per batch.

    Returns:
        inference_df (pd.DataFrame): DataFrame with added 'score_lr' and 'score_rf' columns.
    """
    X_list = []
    scored_scores_lr = []
    scored_scores_rf = []

    # Determine total number of rows for progress bar
    with open(inference_file, 'r') as f:
        total_rows = sum(1 for _ in f) - 1  # Subtract header

    print("Extracting features from inference data...")
    inference_iter = pd.read_csv(inference_file, sep='\t', chunksize=batch_size)

    for chunk in tqdm(inference_iter, total=np.ceil(total_rows / batch_size), desc="Inference Batches"):
        for _, row in chunk.iterrows():
            # Preprocess miRNA sequence
            noncodingRNA = preprocess_miRNA(str(row['noncodingRNA']))
            gene = str(row['gene']).upper()

            kmer_counts = {}
            for k in range(k_min, k_max + 1):
                kmers = count_kmers(noncodingRNA, gene, k)
                for feature, count in kmers.items():
                    if feature in kmer_counts:
                        kmer_counts[feature] += count
                    else:
                        kmer_counts[feature] = count

            X_list.append(kmer_counts)

        # Convert list of dicts to DataFrame
        X = pd.DataFrame(X_list).fillna(0)

        # Align the inference features with training features
        # Ensuring that all training features are present in inference data
        # If any training features are missing in inference, add them with 0 counts
        training_features = lr_model.feature_names_in_ if hasattr(lr_model, 'feature_names_in_') else lr_model.coef_.shape[1]
        if hasattr(lr_model, 'feature_names_in_'):
            missing_features = set(lr_model.feature_names_in_) - set(X.columns)
            for feature in missing_features:
                X[feature] = 0
            X = X[lr_model.feature_names_in_]
        else:
            # If feature_names_in_ is not available, assume all features are present
            pass  # Alternatively, you can handle feature alignment differently

        # Scale features for Logistic Regression
        X_scaled = scaler.transform(X)

        # Predict with Logistic Regression
        scores_lr = lr_model.predict_proba(X_scaled)[:, 1]
        scored_scores_lr.extend(scores_lr)

        # Predict with Random Forest
        scores_rf = rf_model.predict_proba(X)[:, 1]
        scored_scores_rf.extend(scores_rf)

        # Reset for next batch
        X_list = []

    # Load entire inference data to append scores
    inference_df = pd.read_csv(inference_file, sep='\t')
    inference_df['score_lr'] = scored_scores_lr
    inference_df['score_rf'] = scored_scores_rf

    print("Inference completed. Scores computed.")
    return inference_df


def report_extreme_features(model, X, y, top_n=5, model_type='LogisticRegression'):
    """
    Report the top_n highest and lowest coefficient features along with the percentage of positive and negative samples that had each feature.

    Args:
        model (LogisticRegression or RandomForestClassifier): Trained model.
        X (pd.DataFrame): Feature matrix used for training.
        y (pd.Series): Labels corresponding to the feature matrix.
        top_n (int): Number of top features to display for both highest and lowest coefficients.
        model_type (str): Type of the model ('LogisticRegression' or 'RandomForest').

    Returns:
        None
    """
    # Ensure that X and y have the same number of samples
    if len(X) != len(y):
        raise ValueError("The length of X and y must be the same.")

    if model_type == 'LogisticRegression':
        # Get feature names
        feature_names = X.columns

        # Get coefficients
        coefficients = model.coef_[0]

        # Create a DataFrame for features and their coefficients
        coef_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients
        })

        # Sort features by coefficient values
        top_positive = coef_df.sort_values(by='coefficient', ascending=False).head(top_n)
        top_negative = coef_df.sort_values(by='coefficient').head(top_n)

        # Calculate percentages for top positive features
        print(f"\nTop Features with Highest Coefficients ({model_type}):")
        print("{:<15} {:<15} {:<25} {:<25}".format('Feature', 'Coefficient', '% Positive Samples', '% Negative Samples'))
        for _, row in top_positive.iterrows():
            feature = row['feature']
            coef = row['coefficient']
            pos_count = (X[feature] > 0)[y == 1].sum()
            neg_count = (X[feature] > 0)[y == 0].sum()
            pos_percent = (pos_count / (y == 1).sum()) * 100 if (y == 1).sum() > 0 else 0
            neg_percent = (neg_count / (y == 0).sum()) * 100 if (y == 0).sum() > 0 else 0
            print("{:<15} {:<15.4f} {:<25.2f} {:<25.2f}".format(feature, coef, pos_percent, neg_percent))

        # Calculate percentages for top negative features
        print(f"\nTop Features with Lowest Coefficients ({model_type}):")
        print("{:<15} {:<15} {:<25} {:<25}".format('Feature', 'Coefficient', '% Positive Samples', '% Negative Samples'))
        for _, row in top_negative.iterrows():
            feature = row['feature']
            coef = row['coefficient']
            pos_count = (X[feature] > 0)[y == 1].sum()
            neg_count = (X[feature] > 0)[y == 0].sum()
            pos_percent = (pos_count / (y == 1).sum()) * 100 if (y == 1).sum() > 0 else 0
            neg_percent = (neg_count / (y == 0).sum()) * 100 if (y == 0).sum() > 0 else 0
            print("{:<15} {:<15.4f} {:<25.2f} {:<25.2f}".format(feature, coef, pos_percent, neg_percent))

    elif model_type == 'RandomForest':
        # Get feature names
        feature_names = X.columns

        # Get feature importances
        importances = model.feature_importances_

        # Create a DataFrame for features and their importances
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })

        # Sort features by importance
        top_features = importance_df.sort_values(by='importance', ascending=False).head(top_n)
        bottom_features = importance_df.sort_values(by='importance').head(top_n)

        # Calculate percentages for top important features
        print(f"\nTop Features with Highest Importances ({model_type}):")
        print("{:<15} {:<15} {:<25} {:<25}".format('Feature', 'Importance', '% Positive Samples', '% Negative Samples'))
        for _, row in top_features.iterrows():
            feature = row['feature']
            importance = row['importance']
            pos_count = (X[feature] > 0)[y == 1].sum()
            neg_count = (X[feature] > 0)[y == 0].sum()
            pos_percent = (pos_count / (y == 1).sum()) * 100 if (y == 1).sum() > 0 else 0
            neg_percent = (neg_count / (y == 0).sum()) * 100 if (y == 0).sum() > 0 else 0
            print("{:<15} {:<15.4f} {:<25.2f} {:<25.2f}".format(feature, importance, pos_percent, neg_percent))

        # Calculate percentages for bottom important features
        print(f"\nTop Features with Lowest Importances ({model_type}):")
        print("{:<15} {:<15} {:<25} {:<25}".format('Feature', 'Importance', '% Positive Samples', '% Negative Samples'))
        for _, row in bottom_features.iterrows():
            feature = row['feature']
            importance = row['importance']
            pos_count = (X[feature] > 0)[y == 1].sum()
            neg_count = (X[feature] > 0)[y == 0].sum()
            pos_percent = (pos_count / (y == 1).sum()) * 100 if (y == 1).sum() > 0 else 0
            neg_percent = (neg_count / (y == 0).sum()) * 100 if (y == 0).sum() > 0 else 0
            print("{:<15} {:<15.4f} {:<25.2f} {:<25.2f}".format(feature, importance, pos_percent, neg_percent))
    else:
        raise ValueError("Unsupported model type. Use 'LogisticRegression' or 'RandomForest'.")


def main():
    parser = argparse.ArgumentParser(description="miRNA-Target Binding Predictor with Logistic Regression and Random Forest. Includes Training, Prediction, and Evaluation.")
    parser.add_argument('--train', type=str, required=True, help="Path to the training data TSV file with 'noncodingRNA', 'gene', 'label' columns.")
    parser.add_argument('--inference', type=str, required=True, help="Path to the inference data TSV file with 'noncodingRNA' and 'gene' columns.")
    parser.add_argument('--output', type=str, required=True, help="Path to the output TSV file with added 'score_lr' and 'score_rf' columns.")
    parser.add_argument('--evaluate', action='store_true', help="Include this flag to perform evaluation if labels are available in the inference data.")
    parser.add_argument('--threshold', type=float, default=0.5, help="Threshold for classification (default: 0.5).")
    parser.add_argument('--save_models', action='store_true', help="Include this flag to save the trained models and scaler.")
    parser.add_argument('--model_dir', type=str, default='models/', help="Directory to save the trained models and scaler.")

    args = parser.parse_args()

    # Validate input files
    for file_path in [args.train, args.inference]:
        try:
            with open(file_path, 'r') as f:
                pass
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            sys.exit(1)
        except Exception as e:
            print(f"Error accessing file '{file_path}': {e}")
            sys.exit(1)

    # Extract features from training data (100,000 samples)
    X, y = extract_features(args.train, sample_size=TRAIN_SAMPLE_SIZE)

    # Train Logistic Regression and Random Forest models
    lr_model, rf_model, scaler, metrics_lr, metrics_rf = train_models(X, y)

    # Save the models and scaler if requested
    if args.save_models:
        import os
        os.makedirs(args.model_dir, exist_ok=True)
        joblib.dump(lr_model, os.path.join(args.model_dir, 'logistic_regression_model.joblib'))
        joblib.dump(rf_model, os.path.join(args.model_dir, 'random_forest_model.joblib'))
        joblib.dump(scaler, os.path.join(args.model_dir, 'scaler.joblib'))
        print(f"\nModels and scaler saved to directory '{args.model_dir}'.")

    # Report extreme features for Logistic Regression
    report_extreme_features(lr_model, X, y, top_n=5, model_type='LogisticRegression')

    # Report extreme features for Random Forest
    report_extreme_features(rf_model, X, y, top_n=5, model_type='RandomForest')

    # Predict on inference data using both models
    scored_df = predict_with_models(args.inference, lr_model, rf_model, scaler)

    # Save the inference results with scores
    print(f"\nSaving inference results with scores to '{args.output}'...")
    try:
        scored_df.to_csv(args.output, sep='\t', index=False)
        print("Inference results saved successfully.")
    except Exception as e:
        print(f"Error saving output file '{args.output}': {e}")
        sys.exit(1)

    # Perform Evaluation if requested
    if args.evaluate:
        print("\nPerforming evaluation...")
        if 'label' not in scored_df.columns:
            print("Error: 'label' column not found in inference data. Cannot perform evaluation.")
            sys.exit(1)

        true_labels = scored_df['label'].values
        predicted_scores_lr = scored_df['score_lr'].values
        predicted_scores_rf = scored_df['score_rf'].values

        metrics_lr_eval = evaluate_model(true_labels, predicted_scores_lr, threshold=args.threshold)
        metrics_rf_eval = evaluate_model(true_labels, predicted_scores_rf, threshold=args.threshold)

        print("\nLogistic Regression Evaluation Metrics:")
        for metric, value in metrics_lr_eval.items():
            print(f"{metric}: {value:.4f}")

        print("\nRandom Forest Evaluation Metrics:")
        for metric, value in metrics_rf_eval.items():
            print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()
