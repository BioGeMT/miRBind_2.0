"""
Encodes miRNA-target pairs into a multi-channel 2D binding matrix:
    - Channel 1: Watson-Crick base-pairing.
    - Channel 2: Intermolecular binding from RNACofold dot-bracket structure.
Outputs NumPy arrays for input data, labels and total number of rows.

Usage:
    python encode_50_20_2.py --i_file <INPUT_TSV> --o_prefix <OUTPUT_PREFIX> [--ncRNA_column <COL>] [--gene_column <COL>] [--label_column <COL>] [--dotbracket_column <COL>]

Arguments:
    --i_file                  Path to input dataset (TSV)
    --o_prefix                Output file prefix for .npy arrays
    --ncRNA_column            Name of the noncoding RNA column (default: noncodingRNA)
    --gene_column             Name of the gene column (default: gene)
    --label_column            Name of the label column (default: label)
    --dotbracket_column       Name of the dot-bracket structure column (70 characters long, with the first 20 corresponding to miRNA and the next 50 to the gene) (default: RNACofold_structure)
"""

import pandas as pd
import numpy as np
import argparse
import time
from encode_50_20_1 import watsoncrick_encoding, labels_encoding

def dotbracket_encoding(df, 
                        dotbracket_col="RNACofold_structure",
                        tensor_dim=(50, 20, 1)):
    """
    Transform each dot-bracket structure in the dataframe into an intermolecular
    binding matrix.

    The dot-bracket string is assumed to be (miRNA_length + gene_length) characters long,
    with the first miRNA_length characters corresponding to the miRNA and the next gene_length
    corresponding to the gene.

    Parameters:
      - df: Pandas DataFrame with a column for the dot-bracket structure.
      - dotbracket_col: Column name for the dot-bracket structure.
      - tensor_dim: Desired output tensor dimensions (gene_length, miRNA_length, channels).
                    For example, (50, 20, 1) implies gene_length=50, miRNA_length=20.
                    
    Returns:
      - A numpy array of intermolecular binding matrices of shape 
        (N, gene_length, miRNA_length, 1), where N is the number of rows in df.
    """
    df = df.reset_index(drop=True)

    def encode_row(row):
        dot_bracket = row[dotbracket_col]
        miRNA_length = tensor_dim[1]
        gene_length = tensor_dim[0]

        # Initialize the binding matrix with the desired tensor dimensions.
        dotbracket_matrix = np.zeros(tensor_dim, dtype="float32")
        stack = []  # Stack to store positions of '('

        for pos, char in enumerate(dot_bracket):
            if char == '(':
                stack.append(pos)
            elif char == ')':
                if stack:
                    paired_pos = stack.pop()
                    # Check if one index is in the miRNA region (0 to miRNA_length-1)
                    # and the other in the gene region (miRNA_length to miRNA_length+gene_length-1)
                    if paired_pos < miRNA_length and pos >= miRNA_length:
                        gene_index = pos - miRNA_length  # Convert gene index (miRNA_length becomes 0, etc.)
                        miRNA_index = paired_pos
                        if gene_index < gene_length and miRNA_index < miRNA_length:
                            dotbracket_matrix[gene_index, miRNA_index, 0] = 1.0
                    elif pos < miRNA_length and paired_pos >= miRNA_length:
                        # alert, checking if this ever happens because it might be extra 
                        print("Alert: close bracket in miRNA region with open bracket in gene region")
                        gene_index = paired_pos - miRNA_length
                        miRNA_index = pos
                        if gene_index < gene_length and miRNA_index < miRNA_length:
                            dotbracket_matrix[gene_index, miRNA_index, 0] = 1.0
        return dotbracket_matrix

    dotbracket_matrix_2d = np.array(df.apply(encode_row, axis=1).tolist())
    return dotbracket_matrix_2d

def prepare_model_input(df, 
                        tensor_dim=(50, 20, 1), 
                        dotbracket_col="RNACofold_structure", 
                        ncRNA_col="noncodingRNA", 
                        gene_col="gene"):
    """
    Prepare model input by concatenating the Watson-Crick binding matrix and
    the dotbracket intermolecular binding matrix to produce a tensor of shape (N, 50, 20, 2).
    
    Parameters:
      - df: Pandas DataFrame containing the required columns.
      - tensor_dim: Desired dimensions for each individual channel 
                    (gene_length, miRNA_length, 1). For example, (50, 20, 1).
      - dotbracket_col: Column name for the dot-bracket structure.
      - ncRNA_col: Column name for the noncoding RNA sequence.
      - gene_col: Column name for the gene sequence.
      
    Returns:
      - A numpy array of shape (N, 50, 20, 2), where the last dimension contains:
          [complementary binding channel, intermolecular binding channel]
    """
    # watsoncrick_encoding returns a numpy array of shape (N, 50, 20, 1); defined in encode_50_20_1.py
    wc_encoding = watsoncrick_encoding(df, 
                                    tensor_dim=tensor_dim, 
                                    ncRNA_col=ncRNA_col, 
                                    gene_col=gene_col)
    # dotbracket_encoding returns a numpy array of shape (N, 50, 20, 1); defined earlier in this file
    db_encoding = dotbracket_encoding(df, 
                                      dotbracket_col=dotbracket_col, 
                                      tensor_dim=tensor_dim)
    
    # Concatenate along the channel dimension (last axis)
    input_tensor = np.concatenate([wc_encoding, db_encoding], axis=-1)
    return input_tensor

def encode_large_tsv_to_numpy(tsv_file_path, 
                              data_output_path, 
                              labels_output_path, 
                              num_rows_path, 
                              chunk_size=10000,
                              ncRNA_col="noncodingRNA", 
                              gene_col="gene", 
                              label_col="label",
                              dotbracket_col="RNACofold_structure"):
    """
    Encode a large TSV file into NumPy arrays using chunk processing,
    preparing the input tensor and extracting labels separately.

    This version uses:
      - `prepare_model_input` to produce a tensor of shape (N, 50, 20, 2)
      - `extract_labels` to produce a label vector of shape (N,)

    Parameters:
      - tsv_file_path: Path to the TSV file with the dataset.
      - data_output_path: Path to the output data .npy file.
      - labels_output_path: Path to the output labels .npy file.
      - num_rows_path: Path to the output file with the total number of rows.
      - chunk_size: Number of rows to process at a time.
      - ncRNA_col: Column name for the noncoding RNA sequences.
      - gene_col: Column name for the gene sequences.
      - label_col: Column name for the labels.
      - dotbracket_col: Column name for the dot-bracket structure (RNACofold structure).

    The function writes the encoded data and labels to the specified output files.
    """
    # Each channel is of shape (50, 20, 1); since we have two channels, the final input shape per sample will be (50, 20, 2)
    tensor_dim = (50, 20, 1)

    # Count total number of rows in the dataset and save it
    num_rows = sum(len(chunk) for chunk in pd.read_csv(tsv_file_path, sep='\t', usecols=[0], chunksize=chunk_size))

    np.save(num_rows_path, np.array([num_rows]))

    # Define output shapes:
    data_shape = (num_rows, 50, 20, 2)   # (N, 50, 20, 2)
    labels_shape = (num_rows, )           # (N,)

    try:
        # Create memory-mapped arrays for data and labels
        data_mmap = np.memmap(data_output_path, dtype='float32', mode='w+', shape=data_shape)
        labels_mmap = np.memmap(labels_output_path, dtype='float32', mode='w+', shape=labels_shape)

        row_offset = 0

        # Process the TSV file in chunks
        for chunk in pd.read_csv(tsv_file_path, sep='\t', chunksize=chunk_size):
            # Prepare input tensor (concatenation of Watson-Crick and dot-bracket channels)
            data_chunk = prepare_model_input(chunk, 
                                             tensor_dim=tensor_dim, 
                                             dotbracket_col=dotbracket_col, 
                                             ncRNA_col=ncRNA_col, 
                                             gene_col=gene_col)
            # Extract labels from the current chunk
            labels_chunk = labels_encoding(chunk, label_col=label_col)

            num_chunk = data_chunk.shape[0]
            data_mmap[row_offset:row_offset + num_chunk] = data_chunk
            labels_mmap[row_offset:row_offset + num_chunk] = labels_chunk
            row_offset += num_chunk

        # Flush changes to disk
        data_mmap.flush()
        labels_mmap.flush()

    except Exception as e:
        print(f"There was an error during encoding: {e}")

    finally:
        # Clean up the memory-mapped arrays
        del data_mmap
        del labels_mmap

def main():
    parser = argparse.ArgumentParser(
        description="Encode dataset to miRNA x target binding matrix. Outputs numpy file with matrices and and numpy file with corresponding labels. Expected columns of the dataset are 'noncodingRNA', 'gene' and 'label'")
    parser.add_argument('--i_file', type=str, required=True, help="Input dataset file name")
    parser.add_argument('--o_prefix', type=str, required=True, help="Output file name prefix")
    parser.add_argument('--ncRNA_column', type=str, default='noncodingRNA', help="Name of the column with noncoding RNA sequences")
    parser.add_argument('--gene_column', type=str, default='gene', help="Name of the column with gene sequences")
    parser.add_argument('--label_column', type=str, default='label', help="Name of the column with labels")
    parser.add_argument('--dotbracket_column', type=str, default='RNACofold_structure', help="Name of the column with dot-bracket structures")

    args = parser.parse_args()

    start = time.time()
    encode_large_tsv_to_numpy(args.i_file, args.o_prefix + '_dataset.npy', args.o_prefix + '_labels.npy', args.o_prefix + '_num_rows.npy', chunk_size=10000,
                              ncRNA_col=args.ncRNA_column, gene_col=args.gene_column, label_col=args.label_column, dotbracket_col=args.dotbracket_column)    
    end = time.time()

    print("Elapsed time: ", end - start, " s.")


if __name__ == "__main__":
    main()