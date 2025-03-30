import time
import argparse
import numpy as np
import pandas as pd

# * functions taken from Dr. Alexiou's cnn.ipynb implementation ---
# nucleotide pairs and their corresponding indices
nucleotide_pairs = [
    ('A', 'A'), ('A', 'T'), ('A', 'C'), ('A', 'G'),
    ('T', 'A'), ('T', 'T'), ('T', 'C'), ('T', 'G'),
    ('C', 'A'), ('C', 'T'), ('C', 'C'), ('C', 'G'),
    ('G', 'A'), ('G', 'T'), ('G', 'C'), ('G', 'G')
]

# create a dictionary to map nucleotide pairs to indices
pair_to_index = {pair: i for i, pair in enumerate(nucleotide_pairs)}
num_pairs = len(nucleotide_pairs)

# Helper function to pad or trim sequences to the desired length
def pad_or_trim(seq, desired_length):
    # if sequence length is greater than the desired length
    if len(seq) > desired_length:
        # trim the sequence to the desired length
        return seq[:desired_length]
    else:
        # pad the sequence with 'N' to the desired length
        return seq + 'N' * (desired_length - len(seq))

# Encoding function to create a complementarity matrix with progress logging
def encode_complementarity(target_seq, mirna_seq, miRNA_length=25, target_length=50):
    """
    Encodes the complementarity between target and miRNA sequences into a matrix.
    Logs progress to a file.

    Parameters:
    - target_seq: Target RNA sequence (string).
    - mirna_seq: miRNA sequence (string).
    - miRNA_length: Desired length of the miRNA sequence.
    - target_length: Desired length of the target RNA sequence.
    - log_file: Path to the file where progress will be logged.

    Returns:
    - A 2D matrix (miRNA_length x target_length) with encoded nucleotide pair indices.
    """
    # initialize the encoding matrix
    arr = np.zeros((miRNA_length, target_length), dtype=np.int32)

    for i in range(miRNA_length):
        for j in range(target_length):
            if i < len(mirna_seq) and j < len(target_seq):
                if mirna_seq[i] == 'N' or target_seq[j] == 'N':
                    arr[i, j] = num_pairs  # Special index for 'N'
                else:
                    pair = (mirna_seq[i], target_seq[j])
                    arr[i, j] = pair_to_index.get(pair, num_pairs)

    return arr
# * ---

# * main function taken from binding_2D_matrix_encoder.py ---
# New function to encode the dataset and save as .npy files
def encode_tsv_to_npy(tsv_file_path, data_output_path, labels_output_path, miRNA_column, chunk_size=10000):
    encoded_matrices = []
    encoded_labels = []
    
    # Process the TSV file in chunks
    for chunk in pd.read_csv(tsv_file_path, sep='\t', chunksize=chunk_size):
        for _, row in chunk.iterrows():
            # Get the sequences and label; expected columns: miRNA_column, 'gene', and 'label'
            mirna_seq = str(row[miRNA_column])
            target_seq = str(row['gene'])
            label = row['label']
            
            # Pad or trim sequences to fixed lengths
            mirna_seq = pad_or_trim(mirna_seq, desired_length=25)
            target_seq = pad_or_trim(target_seq, desired_length=50)
            
            # Encode into a complementarity matrix
            matrix = encode_complementarity(target_seq, mirna_seq)
            encoded_matrices.append(matrix)
            encoded_labels.append(label)
    
    # Convert lists to numpy arrays
    data_array = np.array(encoded_matrices, dtype='float32')
    label_array = np.array(encoded_labels, dtype='float32')
    
    # Save to .npy files
    np.save(data_output_path, data_array)
    np.save(labels_output_path, label_array)

# main 
def main():
    parser = argparse.ArgumentParser(description="Encode TSV dataset to miRNA x target binding matrices and labels in .npy format. "
                                    "Expected TSV columns: miRNA column (user-specified), 'gene', and 'label'.")
    parser.add_argument('-i', '--i_file', type=str, required=True, help="Input TSV file name")
    parser.add_argument('-o', '--o_prefix', type=str, required=True, help="Output file name prefix")
    parser.add_argument('-col', '--column_name', type=str, required=True, help="Input miRNA column name")
    args = parser.parse_args()

    start = time.time()
    encode_tsv_to_npy(args.i_file, args.o_prefix + '_dataset.npy', args.o_prefix + '_labels.npy', args.column_name)
    end = time.time()
    
    print("Elapsed time:", end - start, "seconds.")


if __name__ == "__main__":
    main()
# * ---