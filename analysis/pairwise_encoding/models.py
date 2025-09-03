import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os


class MiRBindCNN(nn.Module):
    """PyTorch implementation of miRBind CNN architecture"""
    def __init__(self, cnn_num=6, kernel_size=5, pool_size=2, dropout_rate=0.3, dense_num=2):
        super(MiRBindCNN, self).__init__()
        self.cnn_num = cnn_num
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dropout_rate = dropout_rate
        
        self.cnn_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        # CNN layers
        in_channels = 1
        for i in range(cnn_num):
            out_channels = 32 * (i + 1)
            self.cnn_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                         padding='same')
            )
            self.bn_layers.append(nn.BatchNorm2d(out_channels))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
            in_channels = out_channels
        
        # Calculate size after CNN layers for FC layer input
        input_height, input_width = 50, 20
        out_height, out_width = input_height, input_width
        
        # Calculate output dimensions after pooling layers
        final_channels = 32 * cnn_num
        for _ in range(cnn_num):
            out_height = (out_height + 2*1 - pool_size) // pool_size + 1  # account for padding=1
            out_width = (out_width + 2*1 - pool_size) // pool_size + 1
            
        self.flatten_size = final_channels * out_height * out_width
        
        # Dense layers
        self.dense_layers = nn.ModuleList()
        self.dense_bn_layers = nn.ModuleList()
        self.dense_dropout_layers = nn.ModuleList()
        
        in_features = self.flatten_size
        for i in range(dense_num):
            out_features = 32 * (cnn_num - i)
            self.dense_layers.append(nn.Linear(in_features, out_features))
            self.dense_bn_layers.append(nn.BatchNorm1d(out_features))
            self.dense_dropout_layers.append(nn.Dropout(dropout_rate))
            in_features = out_features
            
        self.final_dense = nn.Linear(in_features, 1)
        # self.pool = nn.MaxPool2d(pool_size, padding=1)
        
    def forward(self, x):
        # CNN layers
        for conv, bn, dropout in zip(self.cnn_layers, self.bn_layers, self.dropout_layers):
            x = conv(x)
            x = F.leaky_relu(x)
            x = bn(x)
            # x = self.pool(x)
            x = dropout(x)
            
        x = torch.flatten(x, 1)
        
        # Dense layers
        for dense, bn, dropout in zip(self.dense_layers, self.dense_bn_layers, self.dense_dropout_layers):
            x = dense(x)
            x = F.leaky_relu(x)
            x = bn(x)
            x = dropout(x)
            
        x = torch.sigmoid(self.final_dense(x))
        return x
    

class MiRNAMemMapDataset(Dataset):
    """PyTorch Dataset for memory-mapped miRNA data"""
    def __init__(self, data_path, labels_path, dataset_size, validation_split=0.1, is_validation=False):
        # Load data with correct shape for the miRBind model
        self.data = np.memmap(data_path, dtype='float32', mode='r', 
                            shape=(dataset_size, 50, 20, 1))
        self.labels = np.memmap(labels_path, dtype='float32', mode='r', 
                              shape=(dataset_size,))
        
        # Calculate split indices
        self.num_samples = len(self.data)
        self.num_validation = int(self.num_samples * validation_split)
        self.num_train = self.num_samples - self.num_validation
        
        # Determine whether this is training or validation set
        if is_validation:
            self.start_idx = self.num_train
            self.end_idx = self.num_samples
        else:
            self.start_idx = 0
            self.end_idx = self.num_train
        
        print(f"Dataset initialized with {self.end_idx - self.start_idx} samples")
        print(f"Data shape: {self.data.shape}")
            
    def __len__(self):
        return self.end_idx - self.start_idx
        
    def __getitem__(self, idx):
        # Adjust index based on split
        idx = idx + self.start_idx
        
        # Get data and convert to PyTorch tensor
        # The TF model expects shape (batch, height, width, channels)
        # PyTorch expects (batch, channels, height, width)
        data = torch.from_numpy(np.array(self.data[idx])).float()  # Make a copy to avoid warning
        data = data.permute(2, 0, 1)  # Change from (H,W,C) to (C,H,W) format
        label = torch.tensor(self.labels[idx]).float()
        
        return data, label
    

class PairwiseEncodingCNN(nn.Module):
    def __init__(
        self, num_pairs, mirna_length, target_length, 
        embedding_dim=4, dropout_rate=0.2, kernel_sizes=[6,5,5], 
        filter_sizes=[128,64,32],
    ):
        super(PairwiseEncodingCNN, self).__init__()
        
        # TODO DELETE this later, the reason for this was just easier playing with hyperparams
        # filter_sizes=[32,64,128,256]
        # kernel_sizes=[3,3,6,13]
        
        self.pair_embeddings = nn.Embedding(num_pairs + 1, embedding_dim)
        self.mirna_length = mirna_length
        self.target_length = target_length
        
        # self.conv1 = nn.Conv2d(embedding_dim, filter_sizes[0], kernel_size=kernel_sizes[0])
        self.conv1 = nn.Conv2d(embedding_dim, filter_sizes[0], kernel_size=kernel_sizes[0], padding=2)
        self.bn1 = nn.BatchNorm2d(filter_sizes[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # self.conv2 = nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_size=kernel_sizes[1])
        self.conv2 = nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_size=kernel_sizes[1], padding=1)
        self.bn2 = nn.BatchNorm2d(filter_sizes[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # self.conv3 = nn.Conv2d(filter_sizes[1], filter_sizes[2], kernel_size=kernel_sizes[2])
        # self.conv3 = nn.Conv2d(filter_sizes[1], filter_sizes[2], kernel_size=kernel_sizes[2], padding=1)
        # self.bn3 = nn.BatchNorm2d(filter_sizes[2])
        # self.pool3 = nn.MaxPool2d(kernel_size=2)
        # self.dropout3 = nn.Dropout(dropout_rate)
        
        # self.conv4 = nn.Conv2d(filter_sizes[2], filter_sizes[3], kernel_size=kernel_sizes[3])
        
        self.flat_features = self._get_flat_features()
        
        self.fc1 = nn.Linear(self.flat_features, 30)
        # self.fc1 = nn.Linear(self.flat_features, 60)
        self.bn4 = nn.BatchNorm1d(30)
        self.dropout4 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(30, 1)
        # self.fc2 = nn.Linear(60, 1)
        
        # Print layer dimensions for reference
        print("\nLayer dimensions:")
        print(f"  Input shape: [batch_size, {mirna_length}, {target_length}]")
        print(f"  Embedding shape: [batch_size, {mirna_length}, {target_length}, {embedding_dim}]")
        print(f"  After permute: [batch_size, {embedding_dim}, {mirna_length}, {target_length}]")
        print(f"  After conv1: [batch_size, {filter_sizes[0]}, {mirna_length}, {target_length}]")
        print(f"  After conv2: [batch_size, {filter_sizes[1]}, {mirna_length}, {target_length}]")
        # print(f"  After conv3: [batch_size, {filter_sizes[2]}, {mirna_length}, {target_length}]")
        print(f"  Flattened features: {self.flat_features}")
        print(f"  After fc1: [batch_size, 30]")
        print(f"  Output: [batch_size, 1]")
        
    def _get_flat_features(self):
        x = torch.zeros(1, self.mirna_length, self.target_length)
        x = self.pair_embeddings(x.long())
        x = x.permute(0, 3, 1, 2)
        x = self.pool1(F.leaky_relu(self.bn1(self.conv1(x)), 0.1))
        x = self.pool2(F.leaky_relu(self.bn2(self.conv2(x)), 0.1))
        # x = self.pool3(F.leaky_relu(self.bn3(self.conv3(x)), 0.1))

        # x = F.leaky_relu(self.conv1(x), 0.1)
        # x = F.leaky_relu(self.conv2(x), 0.1)
        # x = F.leaky_relu(self.conv3(x), 0.1)
        # x = F.leaky_relu(self.conv4(x), 0.1)
        return x.numel()
    
    def forward(self, x):
        # Store shapes for logging
        shapes = {"input": x.shape}
        
        x = self.pair_embeddings(x)
        shapes["after_embedding"] = x.shape
        
        x = x.permute(0, 3, 1, 2)
        shapes["after_permute"] = x.shape
        
        x = self.dropout1(self.pool1(F.leaky_relu(self.bn1(self.conv1(x)), 0.1)))
        # x = F.leaky_relu(self.conv1(x), 0.1)
        shapes["after_conv1"] = x.shape
        
        x = self.dropout2(self.pool2(F.leaky_relu(self.bn2(self.conv2(x)), 0.1)))
        # x = F.leaky_relu(self.conv2(x), 0.1)
        shapes["after_conv2"] = x.shape
        
        # x = self.dropout3(self.pool3(F.leaky_relu(self.bn3(self.conv3(x)), 0.1)))
        # x = F.leaky_relu(self.conv3(x), 0.1)
        # shapes["after_conv3"] = x.shape
        
        # x = F.leaky_relu(self.conv4(x), 0.1)
        # shapes["after_conv4"] = x.shape
        
        x = x.contiguous().view(x.size(0), -1)
        shapes["after_flatten"] = x.shape
        
        x = self.dropout4(F.leaky_relu(self.bn4(self.fc1(x)), 0.1))
        # x = F.leaky_relu(self.fc1(x), 0.1)
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
    
class PairwiseOneHotCNN(nn.Module):
    """Same as PairwiseEncodingCNN but uses linear layer instead of embedding for one-hot encoded input"""
    def __init__(
        self, num_pairs, mirna_length, target_length, 
        embedding_dim=4, dropout_rate=0.2, kernel_sizes=[6,5,5], 
        filter_sizes=[128,64,32],
    ):
        super(PairwiseOneHotCNN, self).__init__()
        
        # Linear layer instead of embedding - maps from one-hot to embedding dimension
        self.pair_linear = nn.Linear(num_pairs + 1, embedding_dim)
        self.mirna_length = mirna_length
        self.target_length = target_length
        
        self.conv1 = nn.Conv2d(embedding_dim, filter_sizes[0], kernel_size=kernel_sizes[0], padding=2)
        self.bn1 = nn.BatchNorm2d(filter_sizes[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.conv2 = nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_size=kernel_sizes[1], padding=1)
        self.bn2 = nn.BatchNorm2d(filter_sizes[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.flat_features = self._get_flat_features()
        
        self.fc1 = nn.Linear(self.flat_features, 30)
        self.bn4 = nn.BatchNorm1d(30)
        self.dropout4 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(30, 1)
        
        # Print layer dimensions for reference
        print("\nPairwiseOneHotCNN Layer dimensions:")
        print(f"  Input shape: [batch_size, {mirna_length}, {target_length}, {num_pairs + 1}] (one-hot)")
        print(f"  After linear: [batch_size, {mirna_length}, {target_length}, {embedding_dim}]")
        print(f"  After permute: [batch_size, {embedding_dim}, {mirna_length}, {target_length}]")
        print(f"  After conv1: [batch_size, {filter_sizes[0]}, {mirna_length}, {target_length}]")
        print(f"  After conv2: [batch_size, {filter_sizes[1]}, {mirna_length}, {target_length}]")
        print(f"  Flattened features: {self.flat_features}")
        print(f"  After fc1: [batch_size, 30]")
        print(f"  Output: [batch_size, 1]")
        
    def _get_flat_features(self):
        # Create dummy one-hot input for size calculation
        x = torch.zeros(1, self.mirna_length, self.target_length, self.pair_linear.in_features)
        x = self.pair_linear(x)
        x = x.permute(0, 3, 1, 2)
        x = self.pool1(F.leaky_relu(self.bn1(self.conv1(x)), 0.1))
        x = self.pool2(F.leaky_relu(self.bn2(self.conv2(x)), 0.1))
        return x.numel()
    
    def forward(self, x):
        # Store shapes for logging
        shapes = {"input": x.shape}
        
        # Input is expected to be one-hot encoded: [batch, mirna_length, target_length, num_pairs+1]
        x = self.pair_linear(x)
        shapes["after_linear"] = x.shape
        
        x = x.permute(0, 3, 1, 2)  # Change to [batch, embedding_dim, mirna_length, target_length]
        shapes["after_permute"] = x.shape
        
        x = self.dropout1(self.pool1(F.leaky_relu(self.bn1(self.conv1(x)), 0.1)))
        shapes["after_conv1"] = x.shape
        
        x = self.dropout2(self.pool2(F.leaky_relu(self.bn2(self.conv2(x)), 0.1)))
        shapes["after_conv2"] = x.shape
        
        x = x.contiguous().view(x.size(0), -1)
        shapes["after_flatten"] = x.shape
        
        x = self.dropout4(F.leaky_relu(self.bn4(self.fc1(x)), 0.1))
        shapes["after_fc1"] = x.shape
        
        x = torch.sigmoid(self.fc2(x))
        shapes["output"] = x.shape
        
        # Print shapes for first forward pass
        if not hasattr(self, 'shapes_printed'):
            print("\nPairwiseOneHotCNN intermediate tensor shapes (first batch):")
            for name, shape in shapes.items():
                print(f"  {name}: {shape}")
            self.shapes_printed = True
        
        return x
    

def pairwise_to_onehot(pairwise_input, num_pairs):
    """
    Convert pairwise integer indices to one-hot encoding format.
    
    Args:
        pairwise_input: Tensor of shape [batch_size, mirna_length, target_length] 
                       containing integer indices (0 to num_pairs)
        num_pairs: Number of pair types (excluding 0 for padding/no-pair)
    
    Returns:
        one_hot: Tensor of shape [batch_size, mirna_length, target_length, num_pairs + 1]
                containing one-hot encoded vectors
    """
    # Ensure input is long tensor for one_hot function
    pairwise_input = pairwise_input.long()
    
    # Create one-hot encoding
    # F.one_hot expects the last dimension to be the class dimension
    # and will create one-hot vectors of size num_classes
    one_hot = F.one_hot(pairwise_input, num_classes=num_pairs + 1)
    
    # Convert to float for compatibility with linear layer
    one_hot = one_hot.float()
    
    print(f"Original shape: {pairwise_input.shape}")
    print(f"One-hot shape: {one_hot.shape}")
    print(f"Value range in original: {pairwise_input.min().item()} to {pairwise_input.max().item()}")
    
    return one_hot


def get_model(model_type="mirbind", **kwargs):
    """Factory function to create the specified model type"""
    if model_type == "pairwise":
        return PairwiseEncodingCNN(**kwargs)
    elif model_type == "pairwise_onehot":
        return PairwiseOneHotCNN(**kwargs)
    elif model_type == "mirbind":
        return MiRBindCNN(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")