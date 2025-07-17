#!/bin/bash

INPUT_TSV="cleaned_manakov_train.tsv"
FASTA_FILE="hsa_mature.fas"
MIRGENEDB_FILE="mirgenedb_family_mappings.tsv"
ANNOTATED_OUTPUT="cleaned_manakov_train_mirgenedb.tsv"
FAMILY_COUNTS="train_fams.tsv"
SPLIT_OUTPUT_DIR="family_train"
FAMILY_THRESHOLD=1000

python annotate_dataset.py --fasta "$FASTA_FILE" --tsv "$INPUT_TSV" --mirgenedb "$MIRGENEDB_FILE" --output "$ANNOTATED_OUTPUT"

python count_fams.py "$ANNOTATED_OUTPUT" "$FAMILY_COUNTS"

python split_fams.py "$ANNOTATED_OUTPUT" "$FAMILY_COUNTS" "$SPLIT_OUTPUT_DIR" --threshold "$FAMILY_THRESHOLD"