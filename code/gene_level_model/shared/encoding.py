import torch
import torch.nn.functional as F


def nucleotide_to_onehot(seq: str, max_length: int, num_pairs: int = 17) -> torch.Tensor:
    """
    Convert a nucleotide sequence to one-hot encoding.

    For gene-level prediction, we encode single nucleotides (not pairs).
    This creates a representation that can be combined with miRNA for pairwise interaction.

    Args:
        seq: Nucleotide sequence (A, T, C, G, N)
        max_length: Maximum sequence length (will pad or trim)
        num_pairs: Number of pair types (for compatibility with binding site model)

    Returns:
        Tensor of shape [max_length, num_pairs+1] where each position is one-hot encoded
    """
    nuc_to_idx = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4}

    # Pad or trim sequence
    if len(seq) > max_length:
        seq = seq[:max_length]
    elif len(seq) < max_length:
        seq = seq + 'N' * (max_length - len(seq))

    # Convert to indices
    indices = torch.tensor([nuc_to_idx.get(n.upper(), 4) for n in seq], dtype=torch.long)

    # One-hot encode (5 nucleotides)
    onehot = F.one_hot(indices, num_classes=5).float()

    # Expand to num_pairs+1 dimensions for compatibility
    if num_pairs + 1 > 5:
        padding = torch.zeros(max_length, num_pairs + 1 - 5)
        onehot = torch.cat([onehot, padding], dim=-1)

    return onehot
