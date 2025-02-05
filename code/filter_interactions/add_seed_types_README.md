# Seed Type Annotation

## Overview

This script fetches the **AGO2\_eCLIP\_Manakov2022** datasets (train, test, leftout splits) from miRBench and annotates seed types.

## Features

- Annotates seed types using **miRBench** seed encoders and predictors.
- Supports multiple dataset splits (**train, test, leftout**).
- Saves processed results in a specified output directory.

## Requirements

- Python 3.8
- Required Python packages:
  - `miRBench`
  - `pandas`
  - `argparse`
  - `os`

## Usage

Run the script with the following command:

```bash
python add_seed_types.py --odir <output_directory>
```

### Arguments

- `--odir` *(optional)*: Directory where output files will be stored. Defaults to the current directory (`.`).

## Output Files

For each dataset split (`train`, `test`, `leftout`), the script generates:

- `_<split>_seedtypes.tsv` → Contains the dataset split with added seed type annotations.

## Notes

- The dataset splits are hardcoded as `["train", "test", "leftout"]` intended for the `AGO2_eCLIP_Manakov2022` datasets. Modify as needed for different datasets.

