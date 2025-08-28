"""
Generates predictions from a Keras model for a large encoded dataset using batch processing and saves outputs as a NumPy file.

Usage:
    python predict.py --model_path <MODEL_H5_OR_KERAS> --dataset <DATA_NPY> --num_rows <NUM_ROWS_NPY> --channels <N> --output_path <PRED_NPY> [--batch_size <BATCH>]

Arguments:
    --model_path    Path to the trained Keras model (.keras or .h5)
    --dataset       Path to encoded dataset (.npy)
    --num_rows      Path to file with dataset size (.npy)
    --channels      Number of channels in the input data
    --output_path   Output path for predictions (.npy)
    --batch_size    Batch size for prediction (default: 32)
"""

import numpy as np
import argparse
import tensorflow as tf
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, data_path, dataset_size, channels, batch_size):
        # Preload the encoded numpy data
        self.size = dataset_size
        self.channels = channels
        self.data = np.memmap(data_path, dtype='float32', mode='r', shape=(self.size, 50, 20, self.channels))
        self.batch_size = batch_size
        self.num_samples = len(self.data)

    def __len__(self):
        # Denotes the number of batches
        return int(np.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, idx):
        # Generate one batch of data
        start = idx * self.batch_size
        end = min(start + self.batch_size, self.num_samples)  # Avoid out-of-bounds indexing
        return self.data[start:end]

# Main function
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--num_rows", type=str, required=True)
    parser.add_argument("--channels", type=int, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    # Load the dataset size from .npy file
    dataset_size = int(np.load(args.num_rows)[0])

    # Load the model
    model = tf.keras.models.load_model(args.model_path)
    
    # Initialize the data generator for predictions
    data_generator = DataGenerator(
        data_path=args.dataset,
        dataset_size=dataset_size,
        channels=args.channels,
        batch_size=args.batch_size
    )
    
    # Generate predictions in batches
    predictions = model.predict(data_generator, verbose=1)

    # Save predictions to a .npy file
    np.save(args.output_path, predictions)

# Call the main function
if __name__ == "__main__":
    main()
