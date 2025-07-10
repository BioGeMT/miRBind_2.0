import numpy as np


def one_hot_encoding(miRNA, gene, tensor_dim=(50, 20, 1)):
    """
    fun encodes miRNAs and mRNAs in df into binding matrices
    :param df: dataframe containing 'gene' and 'miRNA' columns
    :param tensor_dim: output shape of the matrix
    :return: numpy array of predictions
    """
    # alphabet for watson-crick interactions.
    alphabet = {"AT": 1., "TA": 1., "GC": 1., "CG": 1., "AU": 1., "UA": 1.}
    # create empty main 2d matrix array
    N = 1  # number of samples in df
    shape_matrix_2d = (N, *tensor_dim)  # 2d matrix shape
    # initialize dot matrix with zeros
    ohe_matrix_2d = np.zeros(shape_matrix_2d, dtype="float32")

    # compile matrix with watson-crick interactions.
    #for index, row in df.iterrows():
    for bind_index, bind_nt in enumerate(gene.upper()):
        for mirna_index, mirna_nt in enumerate(miRNA.upper()):
            base_pairs = bind_nt + mirna_nt
            ohe_matrix_2d[0, bind_index, mirna_index, 0] = alphabet.get(base_pairs, 0)

    return ohe_matrix_2d


def one_hot_encoding_batch(df, tensor_dim=(50, 20, 1), gene_column='gene', mirna_column='miRNA'):
    # alphabet for watson-crick interactions.
    alphabet = {"AT": 1., "TA": 1., "GC": 1., "CG": 1.}
    # labels to one hot encoding
    label = df["label"].to_numpy()
    # create empty main 2d matrix array
    N = df.shape[0]  # number of samples in df
    shape_matrix_2d = (N, *tensor_dim)  # 2d matrix shape
    # initialize dot matrix with zeros
    ohe_matrix_2d = np.zeros(shape_matrix_2d, dtype="float32")

    # compile matrix with watson-crick interactions.
    for index, row in df.iterrows():
        for bind_index, bind_nt in enumerate(row[gene_column].upper()):
            for mirna_index, mirna_nt in enumerate(row[mirna_column].upper()):
                base_pairs = bind_nt + mirna_nt
                ohe_matrix_2d[index, bind_index, mirna_index, 0] = alphabet.get(base_pairs, 0)

    return ohe_matrix_2d, label


def binding_encoding(df, alphabet={"AT": 1., "TA": 1., "GC": 1., "CG": 1.}, tensor_dim=(50, 20, 1),
                     ncRNA_col="noncodingRNA", gene_col="gene", label_col="label"):
    """
    Transform input sequence pairs to a binding matrix with corresponding labels.

    Parameters:
    - df: Pandas DataFrame with columns corresponding to ncRNA_col, gene_col, label_col
    - alphabet: dictionary with letter tuples as keys and 1s when they bind
    - tensor_dim: 2D binding matrix shape
    - ncRNA_col, gene_col, label_col: Column name for noncoding RNA sequences, gene sequences and label.

    Output:
    2D binding matrix, labels as np array
    """
    labels = df[label_col].to_numpy()

    df = df.reset_index(drop=True)

    def encode_row(row):
    # Helper function to encode a single row in dataframe
        ohe_matrix = np.zeros(tensor_dim, dtype="float32")
        for bind_index, bind_nt in enumerate(row[gene_col].upper()):
            if bind_index >= tensor_dim[0]:
                break
            for ncrna_index, ncrna_nt in enumerate(row[ncRNA_col].upper()):
                if ncrna_index >= tensor_dim[1]:
                    break
                base_pairs = bind_nt + ncrna_nt
                ohe_matrix[bind_index, ncrna_index, 0] = alphabet.get(base_pairs, 0)
        return ohe_matrix

    # Compile matrix with Watson-Crick interactions
    ohe_matrix_2d = np.array(
        df.apply(encode_row, axis=1).tolist())

    return ohe_matrix_2d, labels