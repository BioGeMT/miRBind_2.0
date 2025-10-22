import os
import numpy as np
from tensorflow.keras.utils import Sequence


class TrainDataGenerator(Sequence):
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
            

class TestDataGenerator:
    def __init__(self, data_path, labels_path, batch_size=32, dataset_size=None):
        if dataset_size is None:
            # Try to determine the dataset size by checking file properties
            try:
                # try to load just the header to get shape and dtype
                with open(data_path, 'rb') as f:
                    if f.read(6) == b'\x93NUMPY':
                        # This is a standard numpy file, we can get info from header
                        f.seek(0)
                        version = np.lib.format.read_magic(f)
                        shape_dict = np.lib.format.read_array_header_1_0(f) if version == (1, 0) else np.lib.format.read_array_header_2_0(f)
                        shape = shape_dict[0]
                        dataset_size = shape[0]
                    else:
                        # Not a standard numpy file, we'll try other methods
                        raise ValueError("Not a standard numpy file")
            except:
                # try to infer from file size
                # This assumes the files are memory-mapped in a specific format
                # For dataset: shape=(n, 50, 20, 1), dtype=float32 (4 bytes)
                # For labels: shape=(n,), dtype=float32 (4 bytes)
                data_size_bytes = os.path.getsize(data_path)
                labels_size_bytes = os.path.getsize(labels_path)

                # Calculate dataset_size based on assumed structure
                dataset_size_from_data = data_size_bytes // (4 * 50 * 20)
                dataset_size_from_labels = labels_size_bytes // 4

                # Verify sizes match approximately
                if abs(dataset_size_from_data - dataset_size_from_labels) < 10:
                    dataset_size = dataset_size_from_data
                else:
                    raise ValueError(f"Inconsistent file sizes: data suggests {dataset_size_from_data} samples, labels suggests {dataset_size_from_labels}")

        # Create memory-mapped arrays with the determined size
        self.data = np.memmap(data_path, dtype='float32', mode='r', shape=(dataset_size, 50, 20, 1))
        self.labels = np.memmap(labels_path, dtype='float32', mode='r', shape=(dataset_size,))
        self.batch_size = batch_size
        self.num_samples = dataset_size
        
    def get_data(self):
        """Return all test data and labels"""
        return self.data, self.labels