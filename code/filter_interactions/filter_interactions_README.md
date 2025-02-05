# Filter interactions

## Overview

This script filters the **AGO2\_eCLIP\_Manakov2022** datasets with annotated seed types into canonical, non-canonical, and non-seed interactions.

## Features

- Defines canonical, non-canonical, and non-seed interactions.
- Separates canonical, non-canonical, and non-seed interactions into 3 distinct files.
- Saves processed results in a specified output directory.

## Requirements

- Python 3.8
- Required Python packages:
  - `pandas`
  - `argparse`
  - `os`

## Usage

Run the script with the following command:

```bash
python filter_interactions.py --ifile <input_file_with_seed_types> --odir <output_directory>
```

### Arguments

- `--ifile`: Input file with seed type annotations (Output of add_seed_types.py script)
- `--odir` *(optional)*: Directory where output files will be stored. Defaults to the current directory (`.`).

## Output Files

For each input dataset, the script generates:

- `<input_file_basename>_canonical.tsv` → Filtered file with only canonical (6mer) seed interactions. 
- `<input_file_basename>_noncanonical.tsv` → Filtered file with only non-canonical (6merBulgeOrMismatch) seed interactions. 
- `<input_file_basename>_noseed.tsv` → Filtered file with only non-seed interactions.

