import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import json
from utils import pad_or_trim, encode_complementarity
import numpy as np


class MiRNAConservationDataset(Dataset):
    """PyTorch Dataset for miRNA data with conservation scores"""
    def __init__(self, data_file, target_length, mirna_length, pair_to_index, num_pairs):
        self.data = []
        self.labels = []
        self.phylop_scores = []
        self.phastcons_scores = []
        
        with open(data_file, 'r') as f:
            # Skip header
            next(f)
            for line in f:
                fields = line.strip().split('\t')
                if len(fields) < 3:  # Ensure we have minimum required fields
                    continue
                    
                gene = fields[0]
                ncrna = fields[1]
                
                # Handle different label positions based on file format
                if len(fields) >= 6 and fields[5].isdigit():
                    # Original format (label in column 6)
                    label = int(fields[5])
                    phylop_idx = 11
                    phastcons_idx = 12
                else:
                    # New format (label in column 3)
                    label = int(fields[2])
                    phylop_idx = 3
                    phastcons_idx = 4
                
                # Parse conservation scores if they exist
                try:
                    if len(fields) > phylop_idx and len(fields) > phastcons_idx:
                        phylop = json.loads(fields[phylop_idx].replace("'", "\""))
                        phastcons = json.loads(fields[phastcons_idx].replace("'", "\""))
                        
                        # Ensure scores match gene length
                        if len(phylop) != len(gene) or len(phastcons) != len(gene):
                            print(f"Warning: conservation scores length mismatch for gene {gene}")
                            # Use zeros instead
                            phylop = [0.0] * len(gene)
                            phastcons = [0.5] * len(gene)
                    else:
                        # If conservation scores don't exist, use zeros
                        print(f"Warning: No conservation scores found, using zeros")
                        phylop = [0.0] * len(gene)
                        phastcons = [0.5] * len(gene)
                except Exception as e:
                    print(f"Warning: could not parse conservation scores: {e}, using zeros")
                    phylop = [0.0] * len(gene)
                    phastcons = [0.5] * len(gene)
                
                # Convert sequences to pairwise encoding
                encoding = self._create_pairwise_encoding(gene, ncrna, target_length, mirna_length, pair_to_index)
                
                self.data.append(encoding)
                self.labels.append(label)
                
                # Reshape conservation scores to match expected dimensions
                phylop_array = np.zeros((target_length, mirna_length))
                phastcons_array = np.zeros((target_length, mirna_length))
                
                # Fill in the actual scores for positions with gene data
                for i in range(min(len(phylop), target_length)):
                    phylop_array[i, :] = phylop[i]  # Broadcast to all mirna positions
                    phastcons_array[i, :] = phastcons[i]
                
                self.phylop_scores.append(phylop_array)
                self.phastcons_scores.append(phastcons_array)
        
        print(f"Loaded {len(self.data)} samples from {data_file}")
    
    def _create_pairwise_encoding(self, gene, ncrna, target_length, mirna_length, pair_to_index):
        """Create pairwise encoding for gene and miRNA sequences"""
        # Pad or truncate sequences to desired length
        gene = gene[:target_length].ljust(target_length, 'N')
        ncrna = ncrna[:mirna_length].ljust(mirna_length, 'N')
        
        # Create a 2D grid for pairwise interactions
        encoding = np.zeros((target_length, mirna_length), dtype=np.int32)
        
        # Fill in pairwise encodings
        for i in range(target_length):
            for j in range(mirna_length):
                gene_base = gene[i].upper()
                ncrna_base = ncrna[j].upper()
                
                # Handle non-standard nucleotides
                if gene_base not in 'ATCG':
                    gene_base = 'N'
                if ncrna_base not in 'ATCG':
                    ncrna_base = 'N'
                
                # Get pair encoding
                pair = (gene_base, ncrna_base)
                encoding[i, j] = pair_to_index.get(pair, pair_to_index[('N', 'N')])
        
        return encoding
    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        data = torch.from_numpy(self.data[idx]).long()
        label = torch.tensor(self.labels[idx]).float()
        phylop = torch.from_numpy(self.phylop_scores[idx]).float()
        phastcons = torch.from_numpy(self.phastcons_scores[idx]).float()
        
        return data, phylop, phastcons, label
    
    @staticmethod
    def create_train_validation_split(dataset, validation_fraction=0.1, random_seed=42):
        """Split a dataset into training and validation sets"""
        from torch.utils.data import Subset
        import numpy as np
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        # Get dataset size
        dataset_size = len(dataset)
        
        # Create indices
        indices = list(range(dataset_size))
        np.random.shuffle(indices)
        
        # Calculate split point
        split = int(np.floor(validation_fraction * dataset_size))
        
        # Create training and validation subsets
        val_indices = indices[:split]
        train_indices = indices[split:]
        
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        
        return train_dataset, val_dataset