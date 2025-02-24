import argparse
import numpy as np
import logging
import optuna
import optuna.visualization as vis
from optuna.integration import TFKerasPruningCallback
from tensorflow import keras as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
import tensorflow as tf
import random

from utils import set_seeds, compile_model
from data_generators import TrainDataGenerator
import sys
sys.path.append("../../../code/machine_learning/train/CNN_miRBind_2022/") 
from miRBind_CNN_architecture import miRBind_CNN


def objective(trial, train_data_gen, val_data_gen, dataset_ratio, best_model_path, epochs):
    global best_model, best_val_auc

    K.backend.clear_session()

    cnn_num = trial.suggest_int('cnn_layers_num', 2, 10)
    kernel_size = trial.suggest_int('kernel_size', 3, 10)
    pool_size = trial.suggest_int('pool_size', 1, 8)
    dense_num = trial.suggest_int('dense_layers_num', 2, cnn_num)
    model = miRBind_CNN(cnn_num=cnn_num, kernel_size=kernel_size, pool_size=pool_size, dense_num=dense_num).model
    lr = trial.suggest_float('learning_rate', 0.00001, 0.0001)    
    model = compile_model(model, lr=lr)

    model_history = model.fit(
        train_data_gen,
        validation_data=val_data_gen,
        epochs=epochs,
        class_weight={0: 1, 1: dataset_ratio},
        callbacks=[
            TFKerasPruningCallback(trial, "val_auc"),
            K.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        ],
    )

    num_epochs_trained = np.argmax(model_history.history['val_auc'])
    val_auc = model_history.history['val_auc'][num_epochs_trained]

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_model = model
        model.save(best_model_path)
        logger.info(f"New best model found and saved with Validation AUC: {val_auc}")

    print(f"Validation AU PRC: {val_auc}")
    return val_auc


def setup_logger(log_file):
    logger = logging.getLogger('optuna')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file, 'w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(file_handler)
    return logger


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter optimization for miRBind CNN model')
    parser.add_argument('--dataset-train', type=str, 
                      default='../encoded_dataset/Manakov2022_flat/AGO2_eCLIP_Manakov2022_train_dataset.npy',
                      help='Path to the train dataset')
    parser.add_argument('--labels-train', type=str,
                      default='../encoded_dataset/Manakov2022_flat/AGO2_eCLIP_Manakov2022_train_labels.npy',
                      help='Path to the train labels')
    parser.add_argument('--dataset-size', type=int, default=2516195,
                      help='Size of the dataset')
    parser.add_argument('--dataset-ratio', type=float, default=1,
                      help='Dataset ratio for class weighting')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--validation-split', type=float, default=0.1,
                      help='Validation split ratio')
    parser.add_argument('--n-trials', type=int, default=20,
                      help='Number of optimization trials')
    parser.add_argument('--epochs', type=int, default=5,
                      help='Number of max epochs per model')
    parser.add_argument('--best-model', type=str, default='best_model.log',
                      help='Path to the model trained with optimised hyperparameters')
    parser.add_argument('--log-file', type=str, default='hyperparam_optimization.log',
                      help='Path to the log file')
    parser.add_argument('--save-plots', action='store_true',
                      help='Save optimization plots')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    args = parser.parse_args()

    # Set seeds for reproducibility
    set_seeds(args.seed)

    global logger, best_model, best_val_auc
    logger = setup_logger(args.log_file)
    best_model = None
    best_val_auc = 0

    logger.info(f"Starting optimization with seed: {args.seed}")

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

    # Set seed for Optuna
    optuna_sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        direction='maximize', 
        study_name='miRBind_CNN',
        sampler=optuna_sampler
    )
    
    study.optimize(
        lambda trial: objective(trial, train_data_gen, val_data_gen, args.dataset_ratio, args.best_model, args.epochs),
        n_trials=args.n_trials
    )

    logger.info("\n")
    logger.info(f"Best hyperparameters: {study.best_params}")
    logger.info(f"Best value (validation AU PRC): {study.best_value}")

    if args.save_plots:
        plots = {
            'optimization_history': vis.plot_optimization_history,
            'contour': vis.plot_contour,
            'param_importances': vis.plot_param_importances,
            'slice': vis.plot_slice
        }
        
        for name, plot_func in plots.items():
            try:
                fig = plot_func(study)
                fig.write_image(f"{name}.png")
            except Exception as e:
                logger.error(f"Failed to save {name} plot: {str(e)}")

                
if __name__ == "__main__":
    main()