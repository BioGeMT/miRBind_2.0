#!/bin/bash

CODE_DIR="../../code/gene_level_model/training"
DATA_DIR="../../data"
OUTPUT_DIR="test_results_gene_level_V4_pretrained"

MODEL_CHECKPOINT="gene_model_v4_pretrained_3conv/gene_level_model_20260217_142557.pt"
TEST_FILE="${DATA_DIR}/3utr_hg19/3utr.sequences.refseq_id.mirna_fc.seed_cls.sequence.signal.conservation.seed_cls.ts_preds.full.pkl"


python "${CODE_DIR}/evaluate.py" \
    --model_checkpoint "${MODEL_CHECKPOINT}" \
    --test_file "${TEST_FILE}" \
    --gene_col sequence \
    --mirna_col miRNA_seq \
    --label_col fold_change \
    --competitor_cols "weighted context++ score" "context++ score" \
    --competitor_name "TargetScan Weighted Context++ Score" "TargetScan Context++ Score"\
    --output_dir "${OUTPUT_DIR}" \
    --model_name "V4 Model - pretrained, Discriminative LR + Layer Norm (unfrozen) - 20260113_191228" \
    --fill_empty_preds_with_zero
