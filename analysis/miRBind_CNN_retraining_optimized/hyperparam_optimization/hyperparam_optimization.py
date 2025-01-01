#!/usr/bin/env python3

import argparse
import numpy as np
import logging
from tensorflow import keras as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
import optuna
import optuna.visualization as vis
from optuna.integration import TFKerasPruningCallback
import sys
sys.path.append("../../../code/machine_learning/train/CNN_miRBind_2022/")
from miRBind_CNN_architecture import miRBind_CNN

class DataGenerator(Sequence):
    def __init__(self, data_path, labels_path, dataset_size, batch_size=32, validation_split=0.1, is_validation=False, shuffle=True):
        self.size = dataset_size
        self.data = np.memmap(data_path, dtype='float32', mode='r', shape=(self.size, 50, 20, 1))
        self.labels = np.memmap(labels_path, dtype='float32', mode='r', shape=(self.size,))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.validation_split = validation_split
        self.num_samples = len(self.data)
        self.num_validation_samples = int(self.num_samples * validation_split)
        self.num_train_samples = self.num_samples - self.num_validation_samples

        indices = np.arange(self.num_samples)
        if shuffle:
            np.random.shuffle(indices)

        if is_validation:
            self.indices = indices[self.num_train_samples:]
        else:
            self.indices = indices[:self.num_train_samples]

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indices) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_data = self.data[batch_indices]
        batch_labels = self.labels[batch_indices]
        return batch_data, batch_labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def compile_model(model, lr):
    opt = Adam(
        learning_rate=lr,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        name="Adam")

    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=['accuracy', K.metrics.AUC(curve='PR')]
    )
    return model

def objective(trial, train_data_gen, val_data_gen, dataset_ratio):
    global best_model, best_val_auc

    K.backend.clear_session()

    cnn_num = trial.suggest_int('cnn_layers_num', 2, 10)
    kernel_size = trial.suggest_int('kernel_size', 3, 10)
    pool_size = trial.suggest_int('pool_size', 1, 8)
    dense_num = trial.suggest_int('dense_layers_num', 2, cnn_num)
    model = miRBind_CNN(cnn_num=cnn_num, kernel_size=kernel_size, pool_size=pool_size, dense_num=dense_num).model
    lr = trial.suggest_float('learning_rate', 0.00001, 0.1)
    model = compile_model(model, lr=lr)

    model_history = model.fit(
        train_data_gen,
        validation_data=val_data_gen,
        epochs=50,
        class_weight={0: 1, 1: dataset_ratio},
        callbacks=[
            TFKerasPruningCallback(trial, "val_auc"),
            K.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ],
    )

    num_epochs_trained = np.argmax(model_history.history['val_auc'])
    val_auc = model_history.history['val_auc'][num_epochs_trained]

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_model = model
        model.save('best_model.keras')
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
    parser.add_argument('--dataset', type=str, 
                      default='../../miRBind_CNN_retraining_orig_parameters/encoded_dataset/Manakov2022_flat/AGO2_eCLIP_Manakov2022_1_train_dataset.npy',
                      help='Path to the training dataset')
    parser.add_argument('--labels', type=str,
                      default='../../miRBind_CNN_retraining_orig_parameters/encoded_dataset/Manakov2022_flat/AGO2_eCLIP_Manakov2022_1_train_labels.npy',
                      help='Path to the training labels')
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
    parser.add_argument('--log-file', type=str, default='hyperparam_optimization.log',
                      help='Path to the log file')
    parser.add_argument('--save-plots', action='store_true',
                      help='Save optimization plots')
    args = parser.parse_args()

    global logger, best_model, best_val_auc
    logger = setup_logger(args.log_file)
    best_model = None
    best_val_auc = 0

    # Initialize data generators
    train_data_gen = DataGenerator(
        args.dataset,
        args.labels,
        dataset_size=args.dataset_size,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        is_validation=False
    )

    val_data_gen = DataGenerator(
        args.dataset,
        args.labels,
        dataset_size=args.dataset_size,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        is_validation=True
    )

    # Create and run the study
    study = optuna.create_study(direction='maximize', study_name='miRBind_CNN')
    study.optimize(lambda trial: objective(trial, train_data_gen, val_data_gen, args.dataset_ratio),
                  n_trials=args.n_trials)

    # Log results
    logger.info("\n")
    logger.info(f"Best hyperparameters: {study.best_params}")
    logger.info(f"Best value (validation AU PRC): {study.best_value}")

    # Save plots if requested
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