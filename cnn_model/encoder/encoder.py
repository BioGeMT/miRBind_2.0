class miRBindEncoder():
    """
    Based on Klimentová, Eva, et al. "miRBind: A deep learning method for miRNA binding classification." Genes 13.12 (2022): 2323. https://doi.org/10.3390/genes13122323.
    Python implementation: https://github.com/ML-Bioinfo-CEITEC/miRBind

    Encodes miRNA and gene sequences into 2D-binding matrix.
    2D-binding matrix has shape (gene_max_len, miRNA_max_len, 1) and contains 1 for Watson-Crick interactions and 0 otherwise.
    Returns array with shape (num_of_samples, gene_max_len, miRNA_max_len, 1).
    """

    def __call__(self, df, miRNA_col="noncodingRNA", gene_col="gene", tensor_dim=(50, 20, 1)):
        return self.binding_encoding(df, miRNA_col, gene_col, tensor_dim)
    
    def binding_encoding(self, df, miRNA_col, gene_col, tensor_dim):
        """
        fun encodes miRNAs and mRNAs in df into binding matrices
        :param df: dataframe containing gene_col and miRNA_col columns
        :param tensor_dim: output shape of the matrix. If sequences are longer than tensor_dim, they will be truncated.
        :return: 2D binding matrix with shape (N, *tensor_dim)
        """

        # alphabet for watson-crick interactions.
        alphabet = {"AT": 1., "TA": 1., "GC": 1., "CG": 1., "AU": 1., "UA": 1.}
        # create empty main 2d matrix array
        N = df.shape[0]  # number of samples in df
        shape_matrix_2d = (N, *tensor_dim)  # 2d matrix shape
        # initialize dot matrix with zeros
        ohe_matrix_2d = np.zeros(shape_matrix_2d, dtype="float32")

        # compile matrix with watson-crick interactions.
        for index, row in df.iterrows():
            for bind_index, bind_nt in enumerate(row[gene_col][:tensor_dim[0]].upper()):
                for mirna_index, mirna_nt in enumerate(row[miRNA_col][:tensor_dim[1]].upper()):
                    base_pairs = bind_nt + mirna_nt
                    ohe_matrix_2d[index, bind_index, mirna_index, 0] = alphabet.get(base_pairs, 0)

        return ohe_matrix_2d