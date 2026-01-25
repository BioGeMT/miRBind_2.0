import numpy as np


def pad_or_trim(seq, desired_length):
    """Pad sequence with 'N' or trim to desired length."""
    if len(seq) > desired_length:
        return seq[:desired_length]
    return seq + 'N' * (desired_length - len(seq))


def encode_complementarity(target_seq, mirna_seq, target_length, mirna_length,
                          pair_to_index, num_pairs):
    """Encode target-miRNA pair as 2D matrix of pair indices."""
    arr = np.zeros((mirna_length, target_length), dtype=np.int32)
    
    for i in range(mirna_length):
        for j in range(target_length):
            if i < len(mirna_seq) and j < len(target_seq):
                if mirna_seq[i] == 'N' or target_seq[j] == 'N':
                    arr[i, j] = num_pairs
                else:
                    pair = (mirna_seq[i], target_seq[j])
                    arr[i, j] = pair_to_index.get(pair, num_pairs)
    return arr
