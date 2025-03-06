import argparse
import numpy as np
import logging
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
import random
import os
import sys
import matplotlib.pyplot as plt

from utils import set_seeds, setup_logger, compile_model
from plots import plot_training_history
from data_generators import TrainDataGenerator
sys.path.append("../../../code/machine_learning/train/CNN_miRBind_2022/") 
from miRBind_CNN_architecture import miRBind_CNN


def main():
    parser = argparse.ArgumentParser(description='Train miRBind CNN model with specified hyperparameters')
    
    # Data parameters
    parser.add_argument('--dataset-train', type=str, required=True,
                      help='Path to the training dataset (numpy array)')
    parser.add_argument('--labels-train', type=str, required=True,
                      help='Path to the training labels (numpy array)')
    parser.add_argument('--dataset-size', type=int, required=True,
                      help='Size of the dataset (number of samples)')
    parser.add_argument('--validation-split', type=float, default=0.1,
                      help='Validation split ratio (default: 0.1)')
    
    # Model architecture parameters
    parser.add_argument('--cnn-num', type=int, default=6,
                      help='Number of CNN layers (default: 6)')
    parser.add_argument('--kernel-size', type=int, default=5,
                      help='Kernel size for CNN layers (default: 5)')
    parser.add_argument('--pool-size', type=int, default=2,
                      help='Pool size for MaxPooling layers (default: 2)')
    parser.add_argument('--dropout-rate', type=float, default=0.3,
                      help='Dropout rate (default: 0.3)')
    parser.add_argument('--dense-num', type=int, default=2,
                      help='Number of dense layers (default: 2)')
    
    # Training parameters
    parser.add_argument('--learning-rate', type=float, default=0.00001,
                      help='Learning rate (default: 0.00152)')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=30,
                      help='Number of epochs to train (default: 30)')
    parser.add_argument('--patience', type=int, default=5,
                      help='Patience for early stopping (default: 5)')
    parser.add_argument('--class-weight', type=float, default=1.0,
                      help='Weight for positive class (default: 1.0)')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='./model_output',
                      help='Directory to save model and logs (default: ./model_output)')
    parser.add_argument('--model-name', type=str, default='mirbind_cnn_model',
                      help='Name for the saved model (default: mirbind_cnn_model)')
    parser.add_argument('--log-file', type=str, default='training.log',
                      help='Path to the log file (default: training.log)')
    
    # Misc parameters
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(args.output_dir):
        raise RuntimeError(f"Failed to create output directory: {args.output_dir}")
    
    log_path = os.path.join(args.output_dir, args.log_file)
    logger = setup_logger(log_path, 'mirbind_train')
    
    set_seeds(args.seed)
    logger.info(f"Starting training with seed: {args.seed}")
    
    logger.info("Training with the following parameters:")
    for arg in vars(args):
        logger.info(f"  {arg}: {getattr(args, arg)}")
    
    logger.info("Preparing data generators...")
    train_data_gen = TrainDataGenerator(
        args.dataset_train,
        args.labels_train,
        dataset_size=args.dataset_size,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        is_validation=False
    )

    val_data_gen = TrainDataGenerator(
        args.dataset_train,
        args.labels_train,
        dataset_size=args.dataset_size,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        is_validation=True
    )
    
    logger.info("Building model...")
    
    model_instance = miRBind_CNN(
        cnn_num=args.cnn_num,
        kernel_size=args.kernel_size,
        pool_size=args.pool_size,
        dropout_rate=args.dropout_rate,
        dense_num=args.dense_num
    ).model
    
    model = compile_model(model_instance, lr=args.learning_rate)
    
    model.summary(print_fn=logger.info)
    
    logger.info("Setting up training callbacks...")
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(args.output_dir, f"{args.model_name}_best.keras"),
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_auc',
            patience=args.patience,
            restore_best_weights=True,
            verbose=1
        ),
        CSVLogger(
            os.path.join(args.output_dir, 'training_log.csv')
        )
    ]
    
    class_weights = {0: 1, 1: args.class_weight}
    
    logger.info("Starting model training...")
    history = model.fit(
        train_data_gen,
        validation_data=val_data_gen,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    final_model_path = os.path.join(args.output_dir, f"{args.model_name}_final.keras")
    try:
        model.save(final_model_path)
    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}")
        raise
    logger.info(f"Final model saved to {final_model_path}")
    
    logger.info("Plotting training history...")
    plot_training_history(history, args.output_dir)
    
    logger.info("Evaluating model on validation set...")
    val_metrics = model.evaluate(val_data_gen, verbose=1)
    metric_names = model.metrics_names
    
    for name, value in zip(metric_names, val_metrics):
        logger.info(f"Validation {name}: {value:.4f}")
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()