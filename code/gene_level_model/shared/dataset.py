import torch
import pandas as pd
from torch.utils.data import Dataset, random_split

from .encoding import nucleotide_to_onehot


class GeneLevelDataset(Dataset):
    """
    Dataset for gene-level repression prediction.

    Expected TSV format:
    - Column with gene/mRNA sequence (can be thousands of nucleotides)
    - Column with miRNA sequence (20-28 nucleotides)
    - Column with fold change (continuous value, the target)
    """
    def __init__(
        self,
        file_path: str,
        gene_col: str = 'gene',
        mirna_col: str = 'noncodingRNA',
        label_col: str = 'fold_change',
        max_gene_length: int = 2000,
        mirna_length: int = 25,
        target_window: int = 50,
        pair_to_index: dict = None,
        num_pairs: int = 16,
        fraction: float = 1.0
    ):
        """
        Args:
            file_path: Path to TSV or pickle file
            gene_col: Name of column containing gene sequences
            mirna_col: Name of column containing miRNA sequences
            label_col: Name of column containing fold change values
            max_gene_length: Maximum gene length to consider
            mirna_length: Length to pad/trim miRNA sequences
            target_window: Window size for binding site (for compatibility)
            pair_to_index: Dictionary mapping nucleotide pairs to indices
            num_pairs: Number of nucleotide pair types
            fraction: Fraction of data to use
        """
        # Load from pickle or TSV
        if file_path.endswith('.pkl') or file_path.endswith('.pickle'):
            import pickle
            with open(file_path, 'rb') as f:
                self.df = pickle.load(f)
            print(f"Loaded pickle file. Columns: {list(self.df.columns)}")
        else:
            self.df = pd.read_csv(file_path, sep="\t")

        if fraction < 1.0:
            self.df = self.df.sample(frac=fraction, random_state=42).reset_index(drop=True)

        self.gene_seqs = self.df[gene_col].values
        self.mirna_seqs = self.df[mirna_col].values
        self.labels = torch.FloatTensor(self.df[label_col].values)

        self.max_gene_length = max_gene_length
        self.mirna_length = mirna_length
        self.target_window = target_window
        self.num_pairs = num_pairs

        # Default pair mapping
        if pair_to_index is None:
            nucleotide_pairs = [
                ('A', 'A'), ('A', 'T'), ('A', 'C'), ('A', 'G'),
                ('T', 'A'), ('T', 'T'), ('T', 'C'), ('T', 'G'),
                ('C', 'A'), ('C', 'T'), ('C', 'C'), ('C', 'G'),
                ('G', 'A'), ('G', 'T'), ('G', 'C'), ('G', 'G')
            ]
            self.pair_to_index = {pair: i for i, pair in enumerate(nucleotide_pairs)}
            self.pair_to_index[('N', 'N')] = len(self.pair_to_index)
        else:
            self.pair_to_index = pair_to_index

        # Calculate actual gene lengths for masking
        self.gene_lengths = torch.tensor([
            min(len(seq), max_gene_length) for seq in self.gene_seqs
        ])

        print(f"GeneLevelDataset loaded: {len(self)} samples")
        print(f"  Gene lengths: min={self.gene_lengths.min()}, max={self.gene_lengths.max()}, mean={self.gene_lengths.float().mean():.1f}")
        print(f"  Label range: min={self.labels.min():.3f}, max={self.labels.max():.3f}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        gene_seq = self.gene_seqs[idx]
        mirna_seq = self.mirna_seqs[idx]

        # Encode gene sequence
        gene_onehot = nucleotide_to_onehot(gene_seq, self.max_gene_length, self.num_pairs)

        # Encode miRNA sequence
        mirna_onehot = nucleotide_to_onehot(mirna_seq, self.mirna_length, self.num_pairs)

        return {
            'gene_onehot': gene_onehot,
            'mirna_onehot': mirna_onehot,
            'gene_length': self.gene_lengths[idx],
            'label': self.labels[idx]
        }

    @staticmethod
    def collate_fn(batch):
        """Custom collate function to handle variable length sequences"""
        return {
            'gene_onehot': torch.stack([item['gene_onehot'] for item in batch]),
            'mirna_onehot': torch.stack([item['mirna_onehot'] for item in batch]),
            'gene_length': torch.stack([item['gene_length'] for item in batch]),
            'label': torch.stack([item['label'] for item in batch])
        }

    @staticmethod
    def create_train_validation_split(dataset, validation_fraction=0.1):
        val_size = int(len(dataset) * validation_fraction)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        return train_dataset, val_dataset
