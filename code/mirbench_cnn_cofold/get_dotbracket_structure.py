"""
Annotates each miRNA-target pair in a TSV with the dot-bracket structure from ViennaRNA cofold, saving results in a new TSV with only the relevant columns.

Usage:
    python get_dotbracket_structure.py --dataset_path <INPUT_TSV> --output_path <OUTPUT_TSV>

Arguments:
    --dataset_path   Path to input dataset (TSV)
    --output_path    Output path for annotated TSV file
"""

import pandas as pd
import argparse
import RNA

def get_dotbracket_structure(df):
    """
    Given a DataFrame with 'noncodingRNA' and 'gene' columns,
    returns a list of dot-bracket structures from RNA.cofold.
    """
    def merge_seq(row):
        # Truncate and concatenate the noncodingRNA and gene sequences with a separator
        return row['noncodingRNA'][:20] + "&" + row['gene'][:50]

    # Make a list of merged sequences
    seqs = df.apply(merge_seq, axis=1)
    
    # Use RNA.cofold to get dot-bracket structures
    # RNA.cofold returns a tuple (structure, mfe)
    return [RNA.cofold(seq)[0] for seq in seqs]

def process_in_chunks(dataset_path, output_path, chunk_size=10000):
    
    first_chunk = True

    with pd.read_csv(dataset_path, sep='\t', chunksize=chunk_size) as reader:
        for i, chunk in enumerate(reader):

            chunk = chunk.loc[:, ['noncodingRNA', 'gene', 'label']].copy()

            cofold_structure = get_dotbracket_structure(chunk)

            chunk["RNACofold_structure"] = cofold_structure
            
            mode = 'w' if first_chunk else 'a'
            header = first_chunk
            
            chunk.to_csv(output_path, sep='\t', index=False, mode=mode, header=header)
            
            first_chunk = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output file")
    args = parser.parse_args()
    
    process_in_chunks(args.dataset_path, args.output_path)

if __name__ == "__main__":
    main()
