# Retraining CNN on Larger, Unbiased Dataset enhanced with RNAcofold Dot-Bracket Intermolecular Structure 

## Description

This analysis retrains the convolutional neural network (CNN) architecture from [Hejret et al. (2023)](https://doi.org/10.1038/s41598-023-49757-z), on the larger, unbiased [AGO2_eCLIP_Manakov2022 train set](https://zenodo.org/records/14501607) using the same data representation enhanced with intermolecular cofolding information derived from the dot-bracket representation from RNAcofold. The script also runs inference and evaluation on the Manakov2022 evaluation datasets from the `miRBench` Python package. 

1. **Dataset Retrieval & Encoding**: Locates or downloads each required dataset split, adds dot-bracket RNA secondary structure information, and encodes each dataset into the `50_20_2` (Sequence & Co-folding) data representation.

2. **CNN Model Training**: Trains the Hejret CNN architecture on the `AGO2_eCLIP_Manakov2022_train` split using the `50_20_2` (Sequence & Co-folding) data representation.  

3. **Prediction & Evaluation**: Runs inference for the trained model on the `AGO2_eCLIP_Manakov2022` test/leftout splits available on `miRBench`, and evaluates predictions computing evaluation metrics for each split.

## Dependencies

- [miRBench](https://github.com/katarinagresova/miRBench) (version 1.0.1); for downloading the datasets (predictors and encoders are not required)
- [ViennaRNA package](https://www.tbi.univie.ac.at/RNA/ViennaRNA/doc/html/install.html#python-interface-only); for the `RNA` module
- `pandas`
- `numpy`
- `tensorflow`
- `scikit-learn`
- `matplotlib`

## Notes

All outputs are saved in corresponding `results/` subdirectories, as follows:
```
results/
├── encoding/           # encoded datasets and dot-bracket annotated files
├── training/           # trained CNN model file
├── predictions/        # model predictions for each test set
└── evaluation/         # evaluation metrics for each model/test set
```
Training history and plots are included in `results/training`. 
