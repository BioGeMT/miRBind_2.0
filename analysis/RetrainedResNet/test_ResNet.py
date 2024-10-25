# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras as k
import tensorflow_addons as tfa
from tensorflow.keras import layers, models
from tensorflow.keras.utils import register_keras_serializable
from sklearn.model_selection import train_test_split
import code.machine_learning.encode.binding_2D_matrix_encoder as b2dme



# parameters
file_path = 'AGO2_CLASH_Hejret2023.tsv'
num_of_epochs = 10  # number of epochs - one epoch is one iteation of the entire dataset
alphabet = {"AT": 1., "TA": 1., "GC": 1., "CG": 1.}
input_shape = (50, 20, 1)  # shape of the input image



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




# define the ResNet model
def build_resnet(input_shape):
    """
    Builds a simple ResNet model using custom residual blocks.
    """
    inputs = layers.Input(shape=input_shape)

    # initial Conv Layer
    x = layers.Conv2D(64, kernel_size=(3, 3), padding='same')(inputs)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)

    # add ResBlocks
    x = ResBlock(filters=64, downsample=False)(x)
    x = ResBlock(filters=64, downsample=False)(x)

    # flatten and add dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(2, activation='softmax')(x)  # binary classification (0 or 1)

    # build model
    model = models.Model(inputs, x)
    
    # compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # output model summary
    model.summary()
    
    return model



def main():
    # load the dataset
    print("----- <Loading Dataset> -----")
    df = pd.read_csv(file_path, sep='\t')
    print("----- <Dataset loaded successfully> -----\n")
    
    # print the dataset shape and first few rows
    print(f"Dataset shape: {df.shape}")
    print(f"First few rows of the dataset:\n{df.head()}\n")
    
    

    # split into training and validation/testing sets
    training_data, validation_data = train_test_split(df, test_size=0.2, random_state=42)
    # print the size of the training and validation sets
    print(f"Size of training set: {len(training_data)}")
    print(f"Size of validation set: {len(validation_data)}\n")

    # encode the data using your binding_2D_matrix_encoder's binding_encoding function
    def encode_dataset(data):
        # use the function from the binding_2D_matrix_encoder module
        return b2dme.binding_encoding(data, alphabet)

    # encode the training data and validation data
    print("----- <Encoding> -----")
    encoded_training_data, training_labels = encode_dataset(training_data)
    encoded_validation_data, validation_labels = encode_dataset(validation_data)
    # print completion message example of encoded data
    print("----- <Encoding Completed> -----\n")
    # print(f"Encoded training data shape: {encoded_training_data.shape}")
    # print(f"Encoded validation data shape: {encoded_validation_data.shape}\n")
    # print(f"First encoded training example:\n{encoded_training_data[0]}")
    # print(f"First training label: {training_labels[0]}\n")
    
    

    # build the ResNet model
    input_shape = encoded_training_data.shape[1:]  # assuming the encoded data is 4D (samples, height, width, channels)
    print("----- <Building Model> -----")
    model = build_resnet(input_shape)
    print("----- <Model Built> -----\n")

    # train the model
    print("----- <Training Model> -----")
    model.fit(encoded_training_data, training_labels, epochs=10, validation_data=(encoded_validation_data, validation_labels))
    print("----- <Model Trained> -----\n")

    # save the model
    print("----- <Saving Model> -----")
    model.save("ResNet/miRBind_ResNet.keras")
    print("----- <Model Saved> -----\n")
 
    

main()