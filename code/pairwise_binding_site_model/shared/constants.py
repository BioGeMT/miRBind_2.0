import torch

NUCLEOTIDE_PAIRS = [
    ('A', 'A'), ('A', 'T'), ('A', 'C'), ('A', 'G'),
    ('T', 'A'), ('T', 'T'), ('T', 'C'), ('T', 'G'),
    ('C', 'A'), ('C', 'T'), ('C', 'C'), ('C', 'G'),
    ('G', 'A'), ('G', 'T'), ('G', 'C'), ('G', 'G')
]

PADDING_PAIR = ('N', 'N')

NUCLEOTIDE_COLORS = {
    'A': 'green', 'U': 'blue', 'T': 'blue',
    'G': 'orange', 'C': 'red', 'N': 'gray', '-': 'gray'
}


def get_pair_to_index():
    """Returns mapping from nucleotide pairs to indices, including padding."""
    pair_to_index = {pair: i for i, pair in enumerate(NUCLEOTIDE_PAIRS)}
    pair_to_index[PADDING_PAIR] = len(pair_to_index)
    return pair_to_index


def get_num_pairs():
    return len(NUCLEOTIDE_PAIRS) + 1


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
