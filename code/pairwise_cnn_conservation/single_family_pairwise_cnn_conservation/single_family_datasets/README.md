# Single Family Datasets Pipeline

This directory contains all the data processing scripts needed to prepare family-specific datasets for miRNA binding site prediction.

## Pipeline Overview

The data processing pipeline transforms raw miRNA binding data into family-specific datasets suitable for training individual models:

1. **Annotation** - Map miRNA sequences to families using MirGeneDB
2. **Counting** - Count samples per family to identify viable families
3. **Splitting** - Split data into separate files per family above threshold

## Files

### Scripts
- `annotate_dataset.py` - Maps miRNA sequences to families using MirGeneDB reference
- `count_fams.py` - Counts occurrences of each miRNA family
- `split_fams.py` - Splits annotated data into family-specific files
- `prepare_datasets.sh` - Main pipeline script that orchestrates all steps

### Reference Files
- `hsa_mature.fas` - Human mature miRNA sequences for annotation
- `mirgenedb_family_mappings.tsv` - Family mapping reference data

## Usage

### Full Pipeline
```bash
# Edit prepare_datasets.sh to set your input/output paths
./prepare_datasets.sh
```

### Individual Steps

1. **Annotate dataset with family information:**
```bash
python annotate_dataset.py \
  --fasta hsa_mature.fas \
  --tsv input_data.tsv \
  --mirgenedb mirgenedb_family_mappings.tsv \
  --output annotated_data.tsv
```

2. **Count families:**
```bash
python count_fams.py annotated_data.tsv family_counts.tsv
```

3. **Split by families:**
```bash
python split_fams.py \
  annotated_data.tsv \
  family_counts.tsv \
  output_directory/ \
  --threshold 1000
```

## Output

The pipeline produces:
- `annotated_data.tsv` - Original data with added family columns
- `family_counts.tsv` - Count statistics per family
- `output_directory/` - Directory containing one `.tsv` file per family above threshold

## Requirements

- Input TSV file with miRNA sequences and conservation scores
- MirGeneDB reference files for family mapping
- Sufficient samples per family (default threshold: 1000)

## Next Steps

After data processing, use the family-specific files with:
- `../train_family_models.py` for training
- `../single_family_eval.py` for evaluation
- `../single_family_plot.py` for visualization

## Integration with Parent System

This data processing pipeline is part of the larger pairwise CNN conservation system:

```
pairwise_cnn_conservation/
├── models.py, dataset.py, utils.py       # Shared components used by family training
├── train_pairwise_model.py               # Standard pairwise training
└── single_family_pairwise_cnn_conservation/
    ├── train_family_models.py            # Uses shared components
    ├── single_family_eval.py             # Uses shared components  
    ├── single_family_plot.py             # Visualization
    └── single_family_datasets/           # This directory
        └── Data processing scripts
```

The family-specific scripts import shared components (models.py, dataset.py, utils.py) from the parent directory, ensuring consistency between pairwise and family-specific approaches.

## Analysis Scripts

For convenient execution, use the analysis scripts in the project root:
- `../../../../analysis/train_pairwise_model.sh` - Standard pairwise training
- `../../../../analysis/train_family_models.sh` - Family-specific training