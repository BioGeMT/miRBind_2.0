import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwiseEncodingCNN(nn.Module):
    """CNN using embedding layer for integer-encoded nucleotide pairs."""
    
    def __init__(self, num_pairs, mirna_length, target_length,
                 embedding_dim=4, dropout_rate=0.2,
                 kernel_sizes=[6, 5, 5], filter_sizes=[128, 64, 32]):
        super().__init__()
        
        self.pair_embeddings = nn.Embedding(num_pairs + 1, embedding_dim)
        self.mirna_length = mirna_length
        self.target_length = target_length
        
        self.conv1 = nn.Conv2d(embedding_dim, filter_sizes[0],
                               kernel_size=kernel_sizes[0], padding=2)
        self.bn1 = nn.BatchNorm2d(filter_sizes[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.conv2 = nn.Conv2d(filter_sizes[0], filter_sizes[1],
                               kernel_size=kernel_sizes[1], padding=1)
        self.bn2 = nn.BatchNorm2d(filter_sizes[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.flat_features = self._compute_flat_features()
        
        self.fc1 = nn.Linear(self.flat_features, 30)
        self.bn3 = nn.BatchNorm1d(30)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(30, 1)

    def _compute_flat_features(self):
        x = torch.zeros(1, self.mirna_length, self.target_length)
        x = self.pair_embeddings(x.long())
        x = x.permute(0, 3, 1, 2)
        x = self.pool1(F.leaky_relu(self.bn1(self.conv1(x)), 0.1))
        x = self.pool2(F.leaky_relu(self.bn2(self.conv2(x)), 0.1))
        return x.numel()

    def forward(self, x):
        x = self.pair_embeddings(x)
        x = x.permute(0, 3, 1, 2)
        
        x = self.dropout1(self.pool1(F.leaky_relu(self.bn1(self.conv1(x)), 0.1)))
        x = self.dropout2(self.pool2(F.leaky_relu(self.bn2(self.conv2(x)), 0.1)))
        
        x = x.contiguous().view(x.size(0), -1)
        x = self.dropout3(F.leaky_relu(self.bn3(self.fc1(x)), 0.1))
        return torch.sigmoid(self.fc2(x))


# class PairwiseOneHotCNN(nn.Module):
#     """CNN using linear layer for one-hot encoded nucleotide pairs. Using a linear layer as an embedding to enable explainability."""
    
#     def __init__(self, num_pairs, mirna_length, target_length,
#                  embedding_dim=4, dropout_rate=0.2,
#                  kernel_sizes=[6, 5, 5], filter_sizes=[128, 64, 32]):
#         super().__init__()
        
#         self.pair_linear = nn.Linear(num_pairs + 1, embedding_dim)
#         self.mirna_length = mirna_length
#         self.target_length = target_length
        
#         self.conv1 = nn.Conv2d(embedding_dim, filter_sizes[0],
#                                kernel_size=kernel_sizes[0], padding=2)
#         self.bn1 = nn.BatchNorm2d(filter_sizes[0])
#         self.pool1 = nn.MaxPool2d(kernel_size=2)
#         self.dropout1 = nn.Dropout(dropout_rate)
        
#         self.conv2 = nn.Conv2d(filter_sizes[0], filter_sizes[1],
#                                kernel_size=kernel_sizes[1], padding=1)
#         self.bn2 = nn.BatchNorm2d(filter_sizes[1])
#         self.pool2 = nn.MaxPool2d(kernel_size=2)
#         self.dropout2 = nn.Dropout(dropout_rate)
        
#         self.flat_features = self._compute_flat_features()
        
#         self.fc1 = nn.Linear(self.flat_features, 30)
#         self.bn3 = nn.BatchNorm1d(30)
#         self.dropout3 = nn.Dropout(dropout_rate)
#         self.fc2 = nn.Linear(30, 1)

#     def _compute_flat_features(self):
#         x = torch.zeros(1, self.mirna_length, self.target_length,
#                        self.pair_linear.in_features)
#         x = self.pair_linear(x)
#         x = x.permute(0, 3, 1, 2)
#         x = self.pool1(F.leaky_relu(self.bn1(self.conv1(x)), 0.1))
#         x = self.pool2(F.leaky_relu(self.bn2(self.conv2(x)), 0.1))
#         return x.numel()

#     def forward(self, x):
#         x = self.pair_linear(x)
#         x = x.permute(0, 3, 1, 2)
        
#         x = self.dropout1(self.pool1(F.leaky_relu(self.bn1(self.conv1(x)), 0.1)))
#         x = self.dropout2(self.pool2(F.leaky_relu(self.bn2(self.conv2(x)), 0.1)))
        
#         x = x.contiguous().view(x.size(0), -1)
#         x = self.dropout3(F.leaky_relu(self.bn3(self.fc1(x)), 0.1))
#         return torch.sigmoid(self.fc2(x))
        

class PairwiseOneHotCNN(nn.Module):
    """CNN using linear layer for one-hot encoded nucleotide pairs. Using a linear layer as an embedding to enable explainability."""
    
    def __init__(self, num_pairs, mirna_length, target_length,
                 embedding_dim=4, dropout_rate=0.2,
                 kernel_sizes=[6, 5, 5], filter_sizes=[128, 64, 32]):
        super().__init__()
        
        assert len(kernel_sizes) == len(filter_sizes), \
            "kernel_sizes and filter_sizes must have the same length"
        
        self.pair_linear = nn.Linear(num_pairs + 1, embedding_dim)
        self.mirna_length = mirna_length
        self.target_length = target_length
        self.num_conv_layers = len(kernel_sizes)
        
        # Build conv layers dynamically
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        in_channels = embedding_dim
        for i, (kernel_size, num_filters) in enumerate(zip(kernel_sizes, filter_sizes)):
            padding = (kernel_size - 1) // 2
            self.conv_layers.append(
                nn.Conv2d(in_channels, num_filters, kernel_size=kernel_size, padding=padding)
            )
            self.bn_layers.append(nn.BatchNorm2d(num_filters))
            self.pool_layers.append(nn.MaxPool2d(kernel_size=2))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
            in_channels = num_filters
        
        self.flat_features = self._compute_flat_features()
        
        self.fc1 = nn.Linear(self.flat_features, 30)
        self.bn_fc = nn.BatchNorm1d(30)
        self.dropout_fc = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(30, 1)

    def _compute_flat_features(self):
        x = torch.zeros(1, self.mirna_length, self.target_length,
                        self.pair_linear.in_features)
        x = self.pair_linear(x)
        x = x.permute(0, 3, 1, 2)
        
        for conv, bn, pool in zip(self.conv_layers, self.bn_layers, self.pool_layers):
            x = pool(F.leaky_relu(bn(conv(x)), 0.1))
        
        return x.numel()

    def forward(self, x):
        x = self.pair_linear(x)
        x = x.permute(0, 3, 1, 2)
        
        for conv, bn, pool, dropout in zip(
            self.conv_layers, self.bn_layers, self.pool_layers, self.dropout_layers
        ):
            x = dropout(pool(F.leaky_relu(bn(conv(x)), 0.1)))
        
        x = x.contiguous().view(x.size(0), -1)
        x = self.dropout_fc(F.leaky_relu(self.bn_fc(self.fc1(x)), 0.1))
        return torch.sigmoid(self.fc2(x))


def get_model(model_type, **kwargs):
    """Factory function to create the specified model type."""
    models = {
        "pairwise": PairwiseEncodingCNN,
        "pairwise_onehot": PairwiseOneHotCNN,
    }
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    return models[model_type](**kwargs)


def load_model(checkpoint_path, model_type, device):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_params = checkpoint['model_params']
    
    model = get_model(model_type, **model_params)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint
