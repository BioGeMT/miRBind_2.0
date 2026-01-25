import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, random_split

from .encoding import pad_or_trim, encode_complementarity


class MiRNADataset(Dataset):
    """Dataset producing integer-encoded complementarity matrices."""
    
    def __init__(self, file_path, target_length, mirna_length, pair_to_index,
                 num_pairs, fraction=1.0):
        df = pd.read_csv(file_path, sep="\t")
        if fraction < 1.0:
            df = df.sample(frac=fraction, random_state=42).reset_index(drop=True)
        
        self.target_seqs = df.iloc[:, 0].values
        self.mirna_seqs = df.iloc[:, 1].values
        self.labels = torch.FloatTensor(df['label'].values)
        self.target_length = target_length
        self.mirna_length = mirna_length
        self.pair_to_index = pair_to_index
        self.num_pairs = num_pairs

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        t_seq = pad_or_trim(self.target_seqs[idx], self.target_length)
        m_seq = pad_or_trim(self.mirna_seqs[idx], self.mirna_length)
        
        X = torch.tensor(
            encode_complementarity(t_seq, m_seq, self.target_length,
                                  self.mirna_length, self.pair_to_index,
                                  self.num_pairs),
            dtype=torch.long
        )
        return X, self.labels[idx]

    @staticmethod
    def create_train_validation_split(dataset, validation_fraction=0.1):
        val_size = int(len(dataset) * validation_fraction)
        train_size = len(dataset) - val_size
        return random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )


class MiRNAOneHotDataset(Dataset):
    """Dataset producing one-hot encoded complementarity matrices."""
    
    def __init__(self, file_path, target_length, mirna_length, pair_to_index,
                 num_pairs, fraction=1.0):
        df = pd.read_csv(file_path, sep="\t")
        if fraction < 1.0:
            df = df.sample(frac=fraction, random_state=42).reset_index(drop=True)
        
        self.target_seqs = df.iloc[:, 0].values
        self.mirna_seqs = df.iloc[:, 1].values
        self.labels = torch.FloatTensor(df['label'].values)
        self.target_length = target_length
        self.mirna_length = mirna_length
        self.pair_to_index = pair_to_index
        self.num_pairs = num_pairs

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        t_seq = pad_or_trim(self.target_seqs[idx], self.target_length)
        m_seq = pad_or_trim(self.mirna_seqs[idx], self.mirna_length)
        
        indices = torch.tensor(
            encode_complementarity(t_seq, m_seq, self.target_length,
                                  self.mirna_length, self.pair_to_index,
                                  self.num_pairs),
            dtype=torch.long
        )
        # Shape: [mirna_length, target_length, num_pairs + 1]
        X_onehot = F.one_hot(indices, num_classes=self.num_pairs + 1).float()
        return X_onehot, self.labels[idx]

    @staticmethod
    def create_train_validation_split(dataset, validation_fraction=0.1):
        val_size = int(len(dataset) * validation_fraction)
        train_size = len(dataset) - val_size
        return random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )


def get_dataset_class(model_type):
    """Returns appropriate dataset class for the model type."""
    datasets = {
        "pairwise": MiRNADataset,
        "pairwise_onehot": MiRNAOneHotDataset,
    }
    if model_type not in datasets:
        raise ValueError(f"Unknown model type: {model_type}")
    return datasets[model_type]
