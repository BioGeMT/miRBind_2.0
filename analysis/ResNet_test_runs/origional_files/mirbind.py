#!/usr/bin/env python

import argparse
import numpy as np
import pandas as pd
from tensorflow import keras as k
import tensorflow_addons as tfa
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable


# defining a custom Keras layer which inturn implements a residual block
@register_keras_serializable()
class ResBlock(layers.Layer):
    """
    Defines a Residual block based on the original ResNet paper.
    The block either maintains the input dimensions or downsamples based on the specified parameters.
    """

    def __init__(self, downsample=False, filters=16, kernel_size=3):
        """
        Initializes the residual block with optional downsampling.
        
        Parameters:
        - downsample: Boolean, whether to downsample the input (using stride of 2)
        - filters: Number of filters for the Conv2D layers
        - kernel_size: Size of the convolution kernel
        """
        # calling the parent class constructor
        super(ResBlock, self).__init__()

        # parameters for the residual block
        self.downsample = downsample
        self.filters = filters
        self.kernel_size = kernel_size

        # initialize first convolution layer, with stride 1 or 2 depending on downsampling
        self.conv1 = layers.Conv2D(kernel_size=self.kernel_size,
                                   strides=(1 if not self.downsample else 2),
                                   filters=self.filters,
                                   padding="same")
        self.activation1 = layers.ReLU()  # activation function after first convolution
        self.batch_norm1 = layers.BatchNormalization()  # batch normalization after first convolution
        
        # initialize second convolution layer with stride 1 (no downsampling here)
        self.conv2 = layers.Conv2D(kernel_size=self.kernel_size,
                                   strides=1,
                                   filters=self.filters,
                                   padding="same")

        # third convolution if downsampling is needed to match input dimensions
        if self.downsample:
          self.conv3 = layers.Conv2D(kernel_size=1,
                                     strides=2,
                                     filters=self.filters,
                                     padding="same")

        self.activation2 = layers.ReLU()  # activation after second convolution
        self.batch_norm2 = layers.BatchNormalization()  # batch normalization after second convolution

    def call(self, inputs):
        """
        Forward pass for the residual block. Applies the convolutions, activation, and adds the skip connection.

        Parameters:
        - inputs: Input tensor

        Returns:
        - Tensor after applying the residual block transformation
        """
        # first convolution, activation, and batch normalization
        x = self.conv1(inputs)
        x = self.activation1(x)
        x = self.batch_norm1(x)
        
        # second convolution (no downsampling here)
        x = self.conv2(x)

        # adjust input dimensions if downsampling
        if self.downsample:
            inputs = self.conv3(inputs)

        # add the input (skip connection) to the output of the convolutions
        x = layers.Add()([inputs, x])

        # final activation and batch normalization
        x = self.activation2(x)
        x = self.batch_norm2(x)

        return x

    def get_config(self):
        """
        Returns the configuration of the residual block (required for saving and loading the model).
        """
        return {'filters': self.filters, 'downsample': self.downsample, 'kernel_size': self.kernel_size}


def parse_input():
    """
    Parses command line input arguments for the miRBind program.
    
    Expected parameters:
    - --input: Input TSV file with miRNA and gene pairs (default: "example.tsv")
    - --output: Output filename prefix for saving predictions (default: "example_scores")
    - --model: Path to the trained Keras model file (default: "Models/miRBind.h5")
    
    Returns:
    - Dictionary of input arguments
    """
    parser = argparse.ArgumentParser(description='miRBind: a method for prediction of potential miRNA:target site binding')
    parser.add_argument('--input', default="ResNet/AGO2_CLASH_Hejret2023.tsv", metavar='<input_tsv_filename>')
    parser.add_argument('--output', default="ResNet/example_scores", metavar='<output_filename_prefix>')
    parser.add_argument('--model', default="ResNet/miRBind.h5", metavar='<model_name>')
    
    args = parser.parse_args()
    
    return vars(args)


def one_hot_encoding(df, tensor_dim=(50, 20, 1)):
    """
    Encodes miRNA and mRNA sequences from the DataFrame into a one-hot encoded binding matrix.
    
    Parameters:
    - df: Pandas DataFrame containing 'gene' and 'miRNA' columns
    - tensor_dim: Shape of the output tensor (default: (50, 20, 1))
    
    Returns:
    - A 4D NumPy array representing the encoded miRNA-mRNA binding pairs
    """
    # define base pairing interactions (Watson-Crick pairing)
    alphabet = {"AT": 1., "TA": 1., "GC": 1., "CG": 1., "AU": 1., "UA": 1.}
    
    # create an empty matrix for storing the encoded sequences
    N = df.shape[0]  # number of samples
    shape_matrix_2d = (N, *tensor_dim)  # define the 2D matrix shape
    ohe_matrix_2d = np.zeros(shape_matrix_2d, dtype="float32")

    # populate the matrix with one-hot encoded base pair interactions
    for index, row in df.iterrows():
        for bind_index, bind_nt in enumerate(row.gene.upper()):  # iterate over gene sequence
            for noncodingRNA, mirna_nt in enumerate(row.noncodingRNA.upper()):  # iterate over noncodingRNA sequence
                base_pairs = bind_nt + mirna_nt  # create base pair
                ohe_matrix_2d[index, bind_index, noncodingRNA, 0] = alphabet.get(base_pairs, 0)  # encode if valid base pair

    return ohe_matrix_2d


def write_score(output_file, df, scores):
    """
    Writes the predicted miRNA-gene binding scores to an output file.
    
    Parameters:
    - output_file: The prefix for the output file to save the results
    - df: Input DataFrame with miRNA-gene pairs
    - scores: NumPy array of predicted binding scores
    """
    scores = scores.flatten()[::2]  # flattening the score array and taking every second score (assuming one-hot encoding symmetry)
    df["score"] = pd.Series(scores, index=df.index)  # adding scores to the DataFrame
    df.to_csv(output_file + '.tsv', sep='\t', index=False)  # save DataFrame to a TSV file


def predict_probs(df, model, output):
    """
    Predicts the probability of miRNA:target site binding based on input sequences.
    
    Parameters:
    - df: Input DataFrame containing 'miRNA' and 'gene' columns
    - model: Loaded Keras model for prediction
    - output: Output file to write the predicted probabilities
    """
    miRNA_length = 20  # expected length of miRNA sequences
    gene_length = 50  # expected length of gene sequences

    orig_len = len(df)
    # filter sequences to ensure they have the correct lengths
    mask = (df["noncodingRNA"].str.len() == miRNA_length) & (df["gene"].str.len() == gene_length)
    df = df[mask]
    processed_len = len(df)

    if orig_len != processed_len:
        print("Skipping " + str(orig_len - processed_len) + " pairs due to inappropriate length.")

    # encode the input sequences using one-hot encoding
    ohe = one_hot_encoding(df)
    
    # predict binding probabilities using ohe model
    prob = model.predict(ohe)
    
    # write predicted scores to output file
    write_score(output, df, prob)


def main():
    """
    Main function that runs the miRBind pipeline.
    Loads the model, processes the input file, predicts binding probabilities, and writes results.
    """
    # parse the input arguments
    arguments = parse_input()

    output = arguments["output"]

    # load the trained model
    try:
        model = k.models.load_model(arguments["model"])
    except (IOError, ImportError):
        print()
        print("Can't load the model", arguments["model"])
        return

    print("===========================================")

    # load the input miRNA-gene pairs from TSV file
    try:
        input_df = pd.read_csv(arguments["input"], names=['noncodingRNA', 'gene'], sep='\t')
    except IOError as e:
        print
        print("Can't load file", arguments["input"])
        print(e)
        return

    # predict binding probabilities and save output
    predict_probs(input_df, model, output)


main()