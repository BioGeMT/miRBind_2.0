# # Gene Symbol mode
# python build_test_set.py \
#     --utrs "../../data/UCSC/hg19/3utr_sequences_hg19.txt" \
#     --fold-change ../../data/fold_change/mirna_fcs.csv \
#     --scores "../../data/TS8/Conserved_Site_Context_Scores.txt" "../../data/TS8/Nonconserved_Site_Context_Scores.txt" \
#     --ts-join gene_symbol \
#     --output ../../data/TS8/processed/test_set.gene_symbol_joined

# Ensembl IF mode
python build_test_set.py \
    --utrs "../../data/UCSC/hg19/3utr_sequences_hg19.txt" \
    --fold-change ../../data/fold_change/mirna_fcs.csv \
    --scores "../../data/TS8/Conserved_Site_Context_Scores.txt" "../../data/TS8/Nonconserved_Site_Context_Scores.txt" \
    --ts-join ensembl \
    --id-map "../../data/UCSC/hg19/id_map" \
    --gene-info "../../data/TS8/Gene_info.txt" \
    --output ../../data/TS8/processed/test_set.ensembl_id_joined
