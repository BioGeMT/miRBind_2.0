import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

class PairwiseEncodingCNNWithConservation(nn.Module):
    def __init__(
        self, num_pairs, mirna_length, target_length, 
        embedding_dim=4, dropout_rate=0.2, kernel_sizes=[6,5,5], 
        filter_sizes=[128,64,32],
    ):
        super(PairwiseEncodingCNNWithConservation, self).__init__()
        
        self.pair_linear = nn.Linear(num_pairs, embedding_dim)
        self.mirna_length = mirna_length
        self.target_length = target_length
        
        # First conv layer takes embedded pairs + 2 conservation channels
        self.conv1 = nn.Conv2d(embedding_dim + 2, filter_sizes[0], kernel_size=kernel_sizes[0], padding=2)
        self.bn1 = nn.BatchNorm2d(filter_sizes[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Rest of the architecture remains the same
        self.conv2 = nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_size=kernel_sizes[1], padding=1)
        self.bn2 = nn.BatchNorm2d(filter_sizes[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.conv3 = nn.Conv2d(filter_sizes[1], filter_sizes[2], kernel_size=kernel_sizes[2], padding=1)
        self.bn3 = nn.BatchNorm2d(filter_sizes[2])
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.flat_features = self._get_flat_features()
        
        self.fc1 = nn.Linear(self.flat_features, 30)
        self.bn4 = nn.BatchNorm1d(30)
        self.dropout4 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(30, 1)
        
    def _get_flat_features(self):
        # Create dummy one-hot encoded data to compute the output shape
        num_pairs = self.pair_linear.in_features
        x = torch.zeros(1, self.target_length, self.mirna_length, num_pairs)
        phylop = torch.zeros(1, self.target_length, self.mirna_length)
        phastcons = torch.zeros(1, self.target_length, self.mirna_length)
        
        # Process through the network
        seq_emb = self.pair_linear(x)
        seq_emb = seq_emb.permute(0, 3, 1, 2)
        
        # Normalize and reshape conservation scores
        phylop_norm = torch.clamp(phylop / 10.0, -1.0, 1.0).unsqueeze(1)
        phastcons = phastcons.unsqueeze(1)
        
        # Concatenate channels
        x = torch.cat([seq_emb, phylop_norm, phastcons], dim=1)
        
        # Forward pass through convolutional layers
        x = self.pool1(F.leaky_relu(self.bn1(self.conv1(x)), 0.1))
        x = self.pool2(F.leaky_relu(self.bn2(self.conv2(x)), 0.1))
        x = self.pool3(F.leaky_relu(self.bn3(self.conv3(x)), 0.1))
        
        return x.numel()
    
    def forward(self, x, phylop, phastcons):
        # Store shapes for logging
        shapes = {"input": x.shape}
        
        x = self.pair_linear(x)
        shapes["after_linear"] = x.shape
        
        x = x.permute(0, 3, 1, 2)
        shapes["after_permute"] = x.shape
        
        # Normalize and reshape conservation scores
        phylop_norm = torch.clamp(phylop / 10.0, -1.0, 1.0).unsqueeze(1)
        phastcons = phastcons.unsqueeze(1)
        
        # Concatenate channels
        x = torch.cat([x, phylop_norm, phastcons], dim=1)
        shapes["with_conservation"] = x.shape
        
        # Forward pass through convolutional layers
        x = self.dropout1(self.pool1(F.leaky_relu(self.bn1(self.conv1(x)), 0.1)))
        shapes["after_conv1"] = x.shape
        
        x = self.dropout2(self.pool2(F.leaky_relu(self.bn2(self.conv2(x)), 0.1)))
        shapes["after_conv2"] = x.shape
        
        x = self.dropout3(self.pool3(F.leaky_relu(self.bn3(self.conv3(x)), 0.1)))
        shapes["after_conv3"] = x.shape
        
        x = x.contiguous().view(x.size(0), -1)
        shapes["after_flatten"] = x.shape
        
        x = self.dropout4(F.leaky_relu(self.bn4(self.fc1(x)), 0.1))
        shapes["after_fc1"] = x.shape
        
        x = torch.sigmoid(self.fc2(x))
        shapes["output"] = x.shape
        
        # Print shapes for first forward pass
        if not hasattr(self, 'shapes_printed'):
            print("\nIntermediate tensor shapes (first batch):")
            for name, shape in shapes.items():
                print(f"  {name}: {shape}")
            self.shapes_printed = True
        
        return x



def get_model(model_type="pairwise_conservation", **kwargs):
    # factory function to create the specified model type
    if model_type == "pairwise_conservation":
        return PairwiseEncodingCNNWithConservation(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Available: 'pairwise_conservation'")