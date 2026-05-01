# python build_evaluation_subsets.py \
#     --test-set ../../data/TS8/processed/test_set.gene_symbol_joined_long.tsv \
#     --ts-genes ../../data/TS8/processed/test_set.ensembl_id_joined_ts_all_genes.txt \
#     --output ../../data/TS8/processed/eval_gs

python build_evaluation_subsets.py \
    --test-set ../../data/TS8/processed/test_set.ensembl_id_joined_long.tsv \
    --ts-genes ../../data/TS8/processed/test_set.ensembl_id_joined_ts_all_genes.txt \
    --output ../../data/TS8/processed/eval_ens
