from miRBench.encoder import get_encoder
from miRBench.predictor import get_predictor
from miRBench.dataset import get_dataset_df
import pandas as pd
import argparse
import os

def get_seeds(df):
    seed_types = ["Seed8mer", "Seed7mer", "Seed6mer", "Seed6merBulgeOrMismatch"]
    for tool in seed_types:       
        encoder = get_encoder(tool)
        predictor = get_predictor(tool)
        encoded_input = encoder(df)
        output = predictor(encoded_input)
        df[tool] = output
    return df
        
def main():
    parser = argparse.ArgumentParser(description="Add seed types via miRBench")
    parser.add_argument("--odir", type=str, default='.', help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.odir, exist_ok=True)

    dataset_name = "AGO2_eCLIP_Manakov2022"
    splits = ["train", "test", "leftout"]

    for split in splits:

        #Construct output filenames dynamically
        ofile_seedtypes = os.path.join(args.odir, f"{dataset_name}_{split}_seedtypes.tsv")

        # Get dataset 
        df = get_dataset_df(dataset_name, split=split)

        # Add seed types
        df_seedtypes = get_seeds(df)

        # Write seed types to file
        df_seedtypes.to_csv(ofile_seedtypes, sep='\t', index=False)

        print(f"Seed types for {dataset_name} dataset, {split} split, written to {ofile_seedtypes}")

if __name__ == "__main__":
    main()