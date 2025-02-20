import argparse
import os
import logging
import numpy as np
from tensorflow import keras as K
from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score, accuracy_score, average_precision_score

from data_generators import TestDataGenerator
from plots import plot_roc_curve, plot_pr_curve


def setup_logger(log_file):
    """Set up a logger to record evaluation results"""
    logger = logging.getLogger('model_evaluation')
    logger.setLevel(logging.INFO)
    
    # Create handlers
    file_handler = logging.FileHandler(log_file, 'w')
    console_handler = logging.StreamHandler()
    
    # Create formatters and add to handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
    

def evaluate_model(model, test_data, test_labels, logger, save_plots=True, output_dir='.', pred_threshold=0.5):
    """Evaluate model performance"""
    # Get predictions from prediction probabilities
    y_pred_proba = model.predict(test_data)
    y_pred = (y_pred_proba > pred_threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(test_labels, y_pred)
    
    fpr, tpr, _ = roc_curve(test_labels, y_pred_proba)
    roc_auc = roc_auc_score(test_labels, y_pred_proba)
    
    precision, recall, _ = precision_recall_curve(test_labels, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    avg_precision = average_precision_score(test_labels, y_pred_proba)
    
    logger.info(f"Model Evaluation Results:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"ROC AUC: {roc_auc:.4f}")
    logger.info(f"PR AUC: {pr_auc:.4f}")
    logger.info(f"Average Precision: {avg_precision:.4f}")
    
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        
        plot_roc_curve(fpr, tpr, roc_auc, output_dir, logger, fig_save_name='roc_curve.png')
        plot_pr_curve(recall, precision, pr_auc, avg_precision, output_dir, logger, fig_save_name='pr_curve.png')
    
    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'avg_precision': avg_precision,
        'fpr': fpr,
        'tpr': tpr,
        'precision': precision,
        'recall': recall
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained miRBind CNN model')
    parser.add_argument('--model-path', type=str, default='best_model.keras',
                      help='Path to the trained model file')
    parser.add_argument('--dataset-test', type=str, 
                      default='../encoded_dataset/Manakov2022_flat/AGO2_eCLIP_Manakov2022_test_dataset.npy',
                      help='Path to the test dataset')
    parser.add_argument('--labels-test', type=str,
                      default='../encoded_dataset/Manakov2022_flat/AGO2_eCLIP_Manakov2022_test_labels.npy',
                      help='Path to the test labels')
    parser.add_argument('--dataset-size', type=int, default=None,
                      help='Size of the test dataset (number of samples). If not provided, will attempt to determine automatically.')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for evaluation')
    parser.add_argument('--log-file', type=str, default='model_evaluation.log',
                      help='Path to the log file')
    parser.add_argument('--save-plots', action='store_true', default=True,
                      help='Save evaluation plots')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                      help='Directory to save evaluation results')
    args = parser.parse_args()

    # Set up logger
    logger = setup_logger(os.path.join(args.output_dir, args.log_file))
    logger.info("Starting model evaluation")
    
    try:
        logger.info(f"Loading model from {args.model_path}")
        model = K.models.load_model(args.model_path)
        logger.info(f"Model loaded successfully")
        
        logger.info(f"Loading test data from {args.dataset_test}")
        
        test_data_generator = TestDataGenerator(
            args.dataset_test,
            args.labels_test,
            batch_size=args.batch_size,
            dataset_size=args.dataset_size
        )
        test_data, test_labels = test_data_generator.get_data()
        logger.info(f"Dataset size: {len(test_data)} samples")

        logger.info("Evaluating model performance...")
        results = evaluate_model(
            model, 
            test_data, 
            test_labels, 
            logger,
            save_plots=args.save_plots,
            output_dir=args.output_dir
        )
        
        logger.info("Model evaluation completed successfully")
        
        # Save model summary
        with open(os.path.join(args.output_dir, 'model_summary.txt'), 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        logger.info(f"Model summary saved to {os.path.join(args.output_dir, 'model_summary.txt')}")
        
    except Exception as e:
        logger.error(f"An error occurred during model evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()