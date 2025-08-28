"""
Trains a CNN model for miRNA-target site binding prediction using a pre-encoded binding matrix dataset.
Architecture and parameters are reproduced from Hejret et al. (2023) (https://doi.org/10.1038/s41598-023-49757-z).
Original implementation: https://github.com/ML-Bioinfo-CEITEC/HybriDetector/blob/main/ML/Additional_scripts/training.ipynb

Usage:
    python train.py --data <ENCODED_DATA_NPY> --labels <LABELS_NPY> --num_rows <NUM_ROWS_NPY> --ratio <NEG_PER_POS> [--model <MODEL_OUT>] [--debug <BOOL>] [--channels <N>]

Arguments:
    --data          Path to encoded dataset (.npy)
    --labels        Path to dataset labels (.npy)
    --num_rows      Path to file with dataset size (.npy)
    --ratio         Number of negatives per positive in the dataset
    --model         Output file for trained model (default: model.keras)
    --debug         Set to True to output training/validation history and plots (default: False)
    --channels      Number of input channels (default: 1)
"""

import random
import numpy as np
import argparse
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.layers import (
                                BatchNormalization, LeakyReLU,
                                Input, Dense, Conv2D,
                                MaxPooling2D, Flatten, Dropout)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
import os

def make_architecture(channels):
    """
    Build model architecture
    The parameters are based on the CNN model presented in Hejret et al. (2023) https://doi.org/10.1038/s41598-023-49757-z
    Link to the original implementation: https://github.com/ML-Bioinfo-CEITEC/HybriDetector/blob/main/ML/Additional_scripts/training.ipynb 
    """
    CNN_NUM = 6
    KERNEL_SIZE = 5
    POOL_SIZE = 2
    DROPOUT_RATE = 0.3
    DENSE_NUM = 2

    x = Input(shape=(50, 20, channels),
                       dtype='float32', name='main_input'
                       )
    main_input = x

    for cnn_i in range(CNN_NUM):
        x = Conv2D(
            filters=32 * (cnn_i + 1),
            kernel_size=(KERNEL_SIZE, KERNEL_SIZE),
            padding="same",
            data_format="channels_last",
            name="conv_" + str(cnn_i + 1))(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(POOL_SIZE, POOL_SIZE), padding='same', name='Max_' + str(cnn_i + 1))(x)
        x = Dropout(rate=DROPOUT_RATE)(x)

    x = Flatten(name='2d_matrix')(x)

    for dense_i in range(DENSE_NUM):
        neurons = 32 * (CNN_NUM - dense_i)
        x = Dense(neurons)(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(rate=DROPOUT_RATE)(x)

    main_output = Dense(1, activation='sigmoid', name='main_output')(x)

    model = tf.keras.Model(inputs=[main_input], outputs=[main_output], name='arch_00')
    
    return model


def compile_model(channels):
    tf.keras.backend.clear_session()
    model = make_architecture(channels)
    
    opt = Adam(
        learning_rate=0.00152,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        name="Adam")

    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=['accuracy']
        )
    return model

def plot_history(history, prefix):
    """
    Plot history of the model training,
    accuracy and loss of the training and validation set.
    
    Additionally, save the training metrics to a TSV file for any downstream analysis.
    All outputs are saved in the current working directory.
    """
    
    # Extract metrics from the history object
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = list(range(1, len(acc) + 1))

    # Save metrics to a TSV file
    metrics = {
        'epoch': epochs,
        'accuracy': acc,
        'val_accuracy': val_acc,
        'loss': loss,
        'val_loss': val_loss
    }
    df = pd.DataFrame(metrics)
    df.to_csv(f"{prefix}_training_history.tsv", index=False, sep='\t')
    
    # Plot Accuracy
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(epochs, acc, 'bo', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"{prefix}_training_accuracy.svg")
    
    # Plot Loss
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(epochs, loss, 'bo', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{prefix}_training_loss.svg")

class DataGenerator(Sequence):
    def __init__(self, data_path, labels_path, dataset_size, channels, batch_size, validation_split=0.1, is_validation=False, shuffle=True):
        # preload the encoded numpy data
        # the dataset size is needed to properly load the numpy files
        self.size = dataset_size
            
        self.data = np.memmap(data_path, dtype='float32', mode='r', shape=(self.size, 50, 20, channels))
        self.labels = np.memmap(labels_path, dtype='float32', mode='r', shape=(self.size,))
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Determine number of train and validation samples
        self.validation_split = validation_split
        self.num_samples = len(self.data)
        self.num_validation_samples = int(self.num_samples * validation_split)
        self.num_train_samples = self.num_samples - self.num_validation_samples
        
        # Determine indices for validation and training
        indices = np.arange(self.num_samples)
        if shuffle:
            np.random.shuffle(indices)
        
        if is_validation:
            self.indices = indices[self.num_train_samples:]
        else:
            self.indices = indices[:self.num_train_samples]
        
        # Shuffle the data initially
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.ceil(len(self.indices) / float(self.batch_size)))

    def __getitem__(self, idx):
        # Generate one batch of data
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_data = self.data[batch_indices]
        batch_labels = self.labels[batch_indices]
        return batch_data, batch_labels

    def on_epoch_end(self):
        # Updates indices after each epoch for shuffling
        if self.shuffle:
            np.random.shuffle(self.indices)


def train_model(data, labels, dataset_size, ratio, model_file, channels, debug=False):

    # set random state for reproducibility
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    train_data_gen = DataGenerator(data, labels, dataset_size, channels, batch_size=32, validation_split=0.1, is_validation=False)
    val_data_gen = DataGenerator(data, labels, dataset_size, channels, batch_size=32, validation_split=0.1, is_validation=True)

    model = compile_model(channels)
    model_history = model.fit(
        train_data_gen,
        validation_data=val_data_gen,
        epochs=10,
        class_weight={0: 1, 1: ratio}
    )

    if debug:
        model_dir = os.path.dirname(model_file)
        model_basename = os.path.splitext(os.path.basename(model_file))[0]
        prefix = os.path.join(model_dir, model_basename)
        plot_history(model_history, prefix)

    model.save(model_file)


def main():
    parser = argparse.ArgumentParser(description="Train CNN model on encoded miRNA x target binding matrix dataset")
    parser.add_argument('--data', type=str, required=True, help="File with the encoded dataset")
    parser.add_argument('--labels', type=str, required=True, help="File with the dataset labels")
    parser.add_argument('--num_rows', type=str, required=True, help="File containing dataset size. Needed to properly load the numpy files.")
    parser.add_argument('--ratio', type=int, required=True, help="Number of negatives per positive in the dataset.")
    parser.add_argument('--model', type=str, required=False, help="Filename to save the trained model")
    parser.add_argument('--debug', type=bool, default=False, help="Set to True to output history and some plots about training")
    parser.add_argument('--channels', type=int, default=1, help="Number of channels in the input data")
    args = parser.parse_args()
    # Load the dataset size from .npy file
    dataset_size = int(np.load(args.num_rows)[0])

    if args.model is None:
        args.model = f"model.keras"

    start = time.time()
    train_model(args.data, args.labels, dataset_size, args.ratio, args.model, args.channels, args.debug)
    end = time.time()
    
    print("Elapsed time: ", end - start, " s.")

if __name__ == "__main__":
    main()

