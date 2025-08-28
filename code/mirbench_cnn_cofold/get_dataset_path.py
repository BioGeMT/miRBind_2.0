"""
Prints the local path to a specific split of a miRBench dataset, downloading it if not already cached.

Usage:
    python get_dataset_path.py --dataset <DATASET_NAME> --split <SPLIT_NAME>

Arguments:
    --dataset   Name of the dataset to download
    --split     Name of the split to download
"""

from miRBench.dataset import get_dataset_path
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to download")
    parser.add_argument("--split", type=str, required=True, help="Name of the split to download")
    args = parser.parse_args()

    dataset_name = args.dataset
    split_name = args.split
    dataset_path = get_dataset_path(dataset_name, split_name)
    print(dataset_path)

if __name__ == "__main__":
    main()