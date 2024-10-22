import numpy as np
import argparse
import time
import tensorflow as tf
from tensorflow import keras as K
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Sequence

from miRBind_CNN_architecture import miRBind_CNN


class DataGenerator(Sequence):
    # preload the encoded numpy data
    def __init__(self, data_path, labels_path, dataset_size, batch_size, validation_split=0.1,
                 is_validation=False, shuffle=True):
        # the dataset size is needed to properly load the numpy files
        self.size = dataset_size

        self.data = np.memmap(data_path, dtype='float32', mode='r', shape=(self.size, 50, 20, 1))
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


def plot_history(history, ratio):
    """
    Plot history of the model training,
    accuracy and loss of the training and validation set
    """

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(8, 6), dpi=80)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Accuracy')
    plt.legend()
    plt.savefig(f"training_acc_1_{ratio}.jpg")

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Loss')
    plt.legend()
    plt.savefig(f"training_loss_1_{ratio}.jpg")


def train_model(data, labels, dataset_size, ratio, model_file, debug=False):
    # set random state for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    K.utils.set_random_seed(42)
    # TODO still not fully reproducible? why?

    train_data_gen = DataGenerator(data, labels, dataset_size, batch_size=32, validation_split=0.1,
                                   is_validation=False)
    val_data_gen = DataGenerator(data, labels, dataset_size, batch_size=32, validation_split=0.1,
                                 is_validation=True)

    model = miRBind_CNN().compile_model()
    model_history = model.fit(
        train_data_gen,
        validation_data=val_data_gen,
        epochs=10,
        class_weight={0: 1, 1: ratio}
    )

    if debug:
        plot_history(model_history, ratio)

    model.save(model_file)


def main():
    parser = argparse.ArgumentParser(description="Train CNN model on encoded miRNA x target binding matrix dataset")
    parser.add_argument('--ratio', type=int, required=True, help="Ratio of pos:neg in the training dataset")
    parser.add_argument('--data', type=str, required=True, help="File with the encoded dataset")
    parser.add_argument('--labels', type=str, required=True, help="File with the dataset labels")
    parser.add_argument('--dataset_size', type=int, required=True,
                        help="Number of samples in the dataset. Needed to properly load the numpy files.")
    parser.add_argument('--model', type=str, required=False, help="Filename to save the trained model")
    parser.add_argument('--debug', type=bool, default=False, help="Set to True to output some plots about training")
    args = parser.parse_args()

    if args.model is None:
        args.model = f"model_1_{args.ratio}.keras"

    start = time.time()
    train_model(data=args.data, labels=args.labels, dataset_size=args.dataset_size, ratio=args.ratio,
                model_file=args.model, debug=args.debug)
    end = time.time()

    print("Elapsed time: ", end - start, " s.")


if __name__ == "__main__":
    main()
