#!/bin/bash

python ../../code/shared/download_gdrive.py 1tGmuhABqkpDCIyrgcVAPJs8123dyIleH -o ../../code/pairwise_binding_site_model/model_outputs

python ../../code/shared/download_gdrive.py 1lbEzxHDs_LSqvdEizQII2lFVbKceH27S -o ../../data/Agarwal2015_train_data

python ../../code/shared/download_gdrive.py 1Nlmkd7g7ZZg5GlbEuAAVTzced_ctGAaS -o ../../data/3utr_hg19 -n 3utr.sequences.refseq_id.mirna_fc.seed_cls.sequence.signal.conservation.seed_cls.ts_preds.full.pkl
