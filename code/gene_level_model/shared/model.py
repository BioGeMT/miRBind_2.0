import torch
import torch.nn as nn
import torch.nn.functional as F


def get_conv_layer(pretrained_model, layer_idx):
    """
    Get conv layer by index, compatible with both old and new PairwiseOneHotCNN structures.

    Old structure: conv1, conv2, conv3 attributes
    New structure: conv_layers[0], conv_layers[1], conv_layers[2] ModuleList
    """
    if hasattr(pretrained_model, 'conv_layers'):
        return pretrained_model.conv_layers[layer_idx]
    else:
        layer_names = ['conv1', 'conv2', 'conv3']
        return getattr(pretrained_model, layer_names[layer_idx])


def get_bn_layer(pretrained_model, layer_idx):
    """Get batch norm layer by index"""
    if hasattr(pretrained_model, 'bn_layers'):
        return pretrained_model.bn_layers[layer_idx]
    else:
        layer_names = ['bn1', 'bn2', 'bn3']
        return getattr(pretrained_model, layer_names[layer_idx])


def get_pool_layer(pretrained_model, layer_idx):
    """Get pooling layer by index"""
    if hasattr(pretrained_model, 'pool_layers'):
        return pretrained_model.pool_layers[layer_idx]
    else:
        layer_names = ['pool1', 'pool2', 'pool3']
        return getattr(pretrained_model, layer_names[layer_idx])


def get_dropout_layer(pretrained_model, layer_idx):
    """Get dropout layer by index"""
    if hasattr(pretrained_model, 'dropout_layers'):
        return pretrained_model.dropout_layers[layer_idx]
    else:
        layer_names = ['dropout1', 'dropout2', 'dropout3']
        return getattr(pretrained_model, layer_names[layer_idx])


class MultiHeadSpatialAttention(nn.Module):
    """
    Multi-head spatial attention module.
    Each head learns a different attention pattern over spatial positions.

    Supports optional Layer Normalization for improved training stability.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_heads: int = 4,
        use_layer_norm: bool = False,
        dropout_rate: float = 0.1
    ):
        super(MultiHeadSpatialAttention, self).__init__()

        self.num_heads = num_heads
        self.in_channels = in_channels
        self.use_layer_norm = use_layer_norm

        # Each head has its own attention network
        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                nn.Tanh(),
                nn.Conv2d(hidden_dim, 1, kernel_size=1)
            )
            for _ in range(num_heads)
        ])

        # Learnable head combination weights
        self.head_weights = nn.Parameter(torch.ones(num_heads) / num_heads)

        # Optional layer normalization (applied to channel dimension)
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(in_channels)
            self.output_layer_norm = nn.LayerNorm(in_channels)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Args:
            x: [batch, channels, H, W]

        Returns:
            weighted_features: [batch, channels] - pooled features
            attention_weights: [batch, num_heads, H*W] - attention maps per head
            combined_attention: [batch, H*W] - combined attention map
        """
        batch_size = x.size(0)

        # Optional: Apply layer norm before attention (pre-norm style)
        if self.use_layer_norm:
            x_normed = x.permute(0, 2, 3, 1)
            x_normed = self.layer_norm(x_normed)
            x_normed = x_normed.permute(0, 3, 1, 2)
        else:
            x_normed = x

        # Compute attention for each head
        all_attention_logits = []
        for head in self.attention_heads:
            logits = head(x_normed)  # [batch, 1, H, W]
            logits = logits.view(batch_size, -1)  # [batch, H*W]
            all_attention_logits.append(logits)

        # Stack: [batch, num_heads, H*W]
        attention_logits = torch.stack(all_attention_logits, dim=1)

        # Softmax per head
        attention_weights = F.softmax(attention_logits, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Normalize head combination weights
        head_weights_normalized = F.softmax(self.head_weights, dim=0)

        # Combine attention from all heads
        combined_attention = (attention_weights * head_weights_normalized.view(1, -1, 1)).sum(dim=1)

        # Apply combined attention to features
        x_flat = x.view(batch_size, self.in_channels, -1)  # [batch, channels, H*W]
        weighted_features = (x_flat * combined_attention.unsqueeze(1)).sum(dim=-1)  # [batch, channels]

        # Optional: Apply layer norm to output
        if self.use_layer_norm:
            weighted_features = self.output_layer_norm(weighted_features)

        return weighted_features, attention_weights, combined_attention


class GeneRepressionModelV4(nn.Module):
    """
    Multi-head spatial attention with 3 pretrained convolutional layers
    for gene-level repression prediction.

    Architecture:
    1. Pairwise one-hot encoding of miRNA-gene nucleotide pairs
    2. 3 pretrained convolutional blocks (pair_linear -> conv1 -> conv2 -> conv3)
    3. Multi-head spatial attention for weighted pooling
    4. FC layers for final repression prediction

    Features:
    - Uses all 3 conv layers from pretrained model for deeper feature extraction
    - Multi-head attention heads can learn different biological patterns
    - Optional Layer Normalization for training stability during fine-tuning
    - Parameter grouping for discriminative learning rates
    """
    def __init__(
        self,
        pretrained_cnn: nn.Module,
        mirna_length: int = 25,
        freeze_encoder: bool = True,
        attention_hidden_dim: int = 32,
        num_heads: int = 4,
        num_pairs: int = 17,
        use_layer_norm: bool = False,
        dropout_rate: float = 0.1
    ):
        super(GeneRepressionModelV4, self).__init__()

        self.mirna_length = mirna_length
        self.num_pairs = num_pairs
        self.num_heads = num_heads
        self.use_layer_norm = use_layer_norm
        self.freeze_encoder = freeze_encoder

        # Copy all 3 convolutional layers from pretrained model
        self.pair_linear = pretrained_cnn.pair_linear

        self.conv1 = get_conv_layer(pretrained_cnn, 0)
        self.bn1 = get_bn_layer(pretrained_cnn, 0)
        self.pool1 = get_pool_layer(pretrained_cnn, 0)
        self.dropout1 = get_dropout_layer(pretrained_cnn, 0)

        self.conv2 = get_conv_layer(pretrained_cnn, 1)
        self.bn2 = get_bn_layer(pretrained_cnn, 1)
        self.pool2 = get_pool_layer(pretrained_cnn, 1)
        self.dropout2 = get_dropout_layer(pretrained_cnn, 1)

        self.conv3 = get_conv_layer(pretrained_cnn, 2)
        self.bn3 = get_bn_layer(pretrained_cnn, 2)
        self.pool3 = get_pool_layer(pretrained_cnn, 2)
        self.dropout3 = get_dropout_layer(pretrained_cnn, 2)

        if freeze_encoder:
            self._freeze_conv_layers()

        conv_out_channels = self.conv3.out_channels
        self.conv_out_channels = conv_out_channels

        # Optional layer norm after conv layers
        if use_layer_norm:
            self.conv_layer_norm = nn.LayerNorm(conv_out_channels)

        # Multi-head spatial attention
        self.multi_head_attention = MultiHeadSpatialAttention(
            in_channels=conv_out_channels,
            hidden_dim=attention_hidden_dim,
            num_heads=num_heads,
            use_layer_norm=use_layer_norm,
            dropout_rate=dropout_rate
        )

        # Final prediction layers
        if use_layer_norm:
            self.fc = nn.Sequential(
                nn.Linear(conv_out_channels, attention_hidden_dim * 2),
                nn.LayerNorm(attention_hidden_dim * 2),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout_rate),
                nn.Linear(attention_hidden_dim * 2, attention_hidden_dim),
                nn.LayerNorm(attention_hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Linear(attention_hidden_dim, 1)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(conv_out_channels, attention_hidden_dim * 2),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout_rate),
                nn.Linear(attention_hidden_dim * 2, attention_hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Linear(attention_hidden_dim, 1)
            )

        self._setup_pair_encoding()

    def _setup_pair_encoding(self):
        """Setup the nucleotide pair to index mapping"""
        nucleotides = ['A', 'T', 'C', 'G']
        self.nuc_to_idx = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4}

        self.register_buffer('pair_encoding_matrix', torch.zeros(5, 5, dtype=torch.long))

        pair_idx = 0
        for m_nuc in nucleotides:
            for t_nuc in nucleotides:
                m_idx = self.nuc_to_idx[m_nuc]
                t_idx = self.nuc_to_idx[t_nuc]
                self.pair_encoding_matrix[m_idx, t_idx] = pair_idx
                pair_idx += 1

        self.pair_encoding_matrix[4, :] = self.num_pairs
        self.pair_encoding_matrix[:, 4] = self.num_pairs

    def _freeze_conv_layers(self):
        """Freeze all 3 convolutional blocks"""
        for param in self.pair_linear.parameters():
            param.requires_grad = False
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.bn1.parameters():
            param.requires_grad = False
        for param in self.conv2.parameters():
            param.requires_grad = False
        for param in self.bn2.parameters():
            param.requires_grad = False
        for param in self.conv3.parameters():
            param.requires_grad = False
        for param in self.bn3.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all 3 conv blocks for fine-tuning"""
        for param in self.pair_linear.parameters():
            param.requires_grad = True
        for param in self.conv1.parameters():
            param.requires_grad = True
        for param in self.bn1.parameters():
            param.requires_grad = True
        for param in self.conv2.parameters():
            param.requires_grad = True
        for param in self.bn2.parameters():
            param.requires_grad = True
        for param in self.conv3.parameters():
            param.requires_grad = True
        for param in self.bn3.parameters():
            param.requires_grad = True

    def get_pretrained_parameters(self):
        """Get parameters from pretrained layers (for discriminative learning rates)."""
        pretrained_params = []
        pretrained_params.extend(self.pair_linear.parameters())
        pretrained_params.extend(self.conv1.parameters())
        pretrained_params.extend(self.bn1.parameters())
        pretrained_params.extend(self.conv2.parameters())
        pretrained_params.extend(self.bn2.parameters())
        pretrained_params.extend(self.conv3.parameters())
        pretrained_params.extend(self.bn3.parameters())
        return pretrained_params

    def get_new_parameters(self):
        """Get parameters from newly added layers (for discriminative learning rates)."""
        new_params = []
        if self.use_layer_norm and hasattr(self, 'conv_layer_norm'):
            new_params.extend(self.conv_layer_norm.parameters())
        new_params.extend(self.multi_head_attention.parameters())
        new_params.extend(self.fc.parameters())
        return new_params

    def _create_pairwise_onehot(self, mirna_onehot, gene_onehot):
        """
        Create proper pairwise one-hot encoding from nucleotide one-hot encodings.

        Args:
            mirna_onehot: [batch, mirna_length, num_pairs+1] - first 5 dims are nucleotide one-hot
            gene_onehot: [batch, gene_length, num_pairs+1] - first 5 dims are nucleotide one-hot

        Returns:
            pairwise: [batch, mirna_length, gene_length, num_pairs+1] - proper pairwise one-hot
        """
        batch_size = mirna_onehot.size(0)
        mirna_len = mirna_onehot.size(1)
        gene_len = gene_onehot.size(1)

        mirna_nuc_idx = mirna_onehot[:, :, :5].argmax(dim=-1)
        gene_nuc_idx = gene_onehot[:, :, :5].argmax(dim=-1)

        mirna_expanded = mirna_nuc_idx.unsqueeze(2)
        gene_expanded = gene_nuc_idx.unsqueeze(1)

        mirna_flat = mirna_expanded.expand(-1, -1, gene_len).reshape(-1)
        gene_flat = gene_expanded.expand(-1, mirna_len, -1).reshape(-1)

        pair_indices = self.pair_encoding_matrix[mirna_flat, gene_flat]
        pair_indices = pair_indices.view(batch_size, mirna_len, gene_len)

        pairwise_onehot = F.one_hot(pair_indices, num_classes=self.num_pairs + 1).float()

        return pairwise_onehot

    def forward(self, gene_onehot, mirna_onehot, gene_lengths=None):
        """
        Args:
            gene_onehot: [batch, gene_length, num_pairs+1]
            mirna_onehot: [batch, mirna_length, num_pairs+1]
            gene_lengths: [batch] - optional

        Returns:
            repression: [batch, 1]
            combined_attention: [batch, H*W] - attention weights
        """
        # Create pairwise encoding
        pairwise = self._create_pairwise_onehot(mirna_onehot, gene_onehot)

        # Apply CNN layers - 3 conv blocks
        x = self.pair_linear(pairwise)
        x = x.permute(0, 3, 1, 2)

        x = self.dropout1(self.pool1(F.leaky_relu(self.bn1(self.conv1(x)), 0.1)))
        x = self.dropout2(self.pool2(F.leaky_relu(self.bn2(self.conv2(x)), 0.1)))
        x = self.dropout3(self.pool3(F.leaky_relu(self.bn3(self.conv3(x)), 0.1)))

        # Optional layer norm after conv layers
        if self.use_layer_norm and hasattr(self, 'conv_layer_norm'):
            x = x.permute(0, 2, 3, 1)
            x = self.conv_layer_norm(x)
            x = x.permute(0, 3, 1, 2)

        # Multi-head attention
        weighted_features, head_attention, combined_attention = self.multi_head_attention(x)

        # Final prediction
        repression = self.fc(weighted_features)

        return repression, combined_attention

    def get_attention_details(self, gene_onehot, mirna_onehot, gene_lengths=None):
        """
        Get detailed attention information for analysis.

        Returns:
            repression: [batch, 1]
            attention_details: dict with per-head attention maps and weights
        """
        pairwise = self._create_pairwise_onehot(mirna_onehot, gene_onehot)

        x = self.pair_linear(pairwise)
        x = x.permute(0, 3, 1, 2)

        x = self.dropout1(self.pool1(F.leaky_relu(self.bn1(self.conv1(x)), 0.1)))
        x = self.dropout2(self.pool2(F.leaky_relu(self.bn2(self.conv2(x)), 0.1)))
        x = self.dropout3(self.pool3(F.leaky_relu(self.bn3(self.conv3(x)), 0.1)))

        if self.use_layer_norm and hasattr(self, 'conv_layer_norm'):
            x = x.permute(0, 2, 3, 1)
            x = self.conv_layer_norm(x)
            x = x.permute(0, 3, 1, 2)

        weighted_features, head_attention, combined_attention = self.multi_head_attention(x)

        repression = self.fc(weighted_features)

        attention_details = {
            'combined_attention': combined_attention.detach(),
            'head_attention': head_attention.detach(),
            'head_weights': F.softmax(self.multi_head_attention.head_weights, dim=0).detach(),
            'spatial_shape': (x.size(2), x.size(3))
        }

        return repression, attention_details
