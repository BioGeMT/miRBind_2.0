import pandas as pd
import argparse
import os

def define_interaction_groups(df):

    # Canonical seed: Seed6mer is 1
    df["CanonicalSeed"] = (df["Seed6mer"] == 1).astype(int)

    # Non-canonical seed: Seed6merBulgeOrMismatch is 1 AND Seed6mer is 0
    df["NonCanonicalSeed"] = ((df["Seed6merBulgeOrMismatch"] == 1) & (df["Seed6mer"] == 0)).astype(int)

    # No seed: Seed6merBulgeOrMismatch is 0
    df["NoSeed"] = (df["Seed6merBulgeOrMismatch"] == 0).astype(int)
    
    return df

def filter_interactions(df):
    df_canonical = df[df['CanonicalSeed'] == 1]
    df_noncanonical = df[df['NonCanonicalSeed'] == 1]
    df_noseed = df[df['NoSeed'] == 1]
    return df_canonical, df_noncanonical, df_noseed
        
def main():
    parser = argparse.ArgumentParser(description="Filter canonical/non-canonical/no-seed interactions, for all Manakov datasets")
    parser.add_argument("--ifile", type=str, help="Input file with seed types")
    parser.add_argument("--odir", type=str, default=".", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.odir, exist_ok=True)
        
    # Construct output filenames dynamically
    ifile_seed_types_basename = os.path.basename(args.ifile).replace(".tsv", "")
    ofile_canonical_interactions = os.path.join(args.odir, f"{ifile_seed_types_basename}_canonical6mer.tsv")
    ofile_noncanonical_interactions = os.path.join(args.odir, f"{ifile_seed_types_basename}_noncanonical.tsv")
    ofile_noseed_interactions = os.path.join(args.odir, f"{ifile_seed_types_basename}_noseed.tsv")

    # Read file with seed types
    df_seed_types = pd.read_csv(args.ifile, sep='\t')

    # Define interaction groups
    df_interaction_groups = define_interaction_groups(df_seed_types)
    
    # Filter canonical/non-canonical/no-seed interactions
    df_canonical, df_noncanonical, df_noseed = filter_interactions(df_interaction_groups)

    # Write interactions to file
    df_canonical.to_csv(ofile_canonical_interactions, sep='\t', index=False)
    print(f"Canonical interactions written to {ofile_canonical_interactions}")

    df_noncanonical.to_csv(ofile_noncanonical_interactions, sep='\t', index=False)
    print(f"Non-canonical interactions written to {ofile_noncanonical_interactions}")

    df_noseed.to_csv(ofile_noseed_interactions, sep='\t', index=False)
    print(f"Non-seed interactions written to {ofile_noseed_interactions}")

if __name__ == "__main__":
    main()