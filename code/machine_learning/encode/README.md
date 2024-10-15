# Encoding the dataset into inner representation

### [Binding 2D matrix encoder](binding_2d_matrix_encoder.py)
 The encoder is based on the "miRBind: A deep learning method for miRNA binding classification." (2022) https://doi.org/10.3390/genes13122323
 with original python implementation here: https://github.com/ML-Bioinfo-CEITEC/miRBind

Encodes miRNA and gene sequences into 2D-binding matrix.
2D-binding matrix has shape (gene_max_len=50, miRNA_max_len=20, 1) and contains 1 for Watson-Crick interactions and 0 otherwise.

Outputs npy file with encoded matrices and npy file with corresponding labels.

#### Usage
Run the script from the command line with the following syntax:


```python binding_2d_matrix_encoder.py --i_file input_dataset_file.tsv --o_prefix output_prefix```

