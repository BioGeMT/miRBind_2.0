# imports
import sys
sys.path.append('./')

import time
import os

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
from encoders.binding_2D_matrix_encoder import binding_encoding
# from code.machine_learning.encode.binding_2D_matrix_encoder import binding_encoding
# from miRBench.encoder import miRBindEncoder


# clean up resources to avoid OOM
import gc
import tensorflow as tf


# use non interactive backend for matplotlib
matplotlib.use('Agg')


# * PARAMS ---


# parameters
training_file_paths = [
    'datasets/training/Balanced_dataset.tsv',
    'datasets/training/train_set_1_20_CLASH2013_paper.tsv'
]

alphabet = {"AT": 1., "TA": 1., "GC": 1., "CG": 1., "AU": 1., "UA": 1.}
learning_rate = 0.001  # learning rate
epochs = 20  # number of epochs/dataset iterations
batch_size = 32  # batch size
split_used = '80-10-10'

results_file_path = 'autoTrain_modelResults.txt'

# define the directory where you want to save the model
save_dir = "SavedModels/ResNet"

# hyperparameter combinations
dropout_rates = [0.05, 0.09, 0.13, 0.17, 0.21, 0.25]


# * BUILDING RESNET ---


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
        self.conv3 = None  # Default to None
        if self.downsample:
          self.conv3 = layers.Conv2D(kernel_size=1,
                                     strides=2,
                                     filters=self.filters,
                                     padding="same")
          self.batch_norm3 = layers.BatchNormalization()  # batch normalization after third convolution

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
def build_resnet(input_shape, dropout_rate):
    # clear any previous models
    tf.keras.backend.clear_session()
    
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
    x = ResBlock(filters=128, downsample=True)(x)

    # flatten and add dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)  # dropout layer
    x = layers.Dense(1, activation='sigmoid')(x)  # binary classification (0 or 1)

    # build model
    model = models.Model(inputs, x)
    # compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy',metrics=['accuracy'])
    # output model summary
    model.summary()
    
    print()
    
    return model

# encode the data using your binding_2D_matrix_encoder's binding_encoding function
def encode_dataset(data, rna_type):
    # use the function from the binding_2D_matrix_encoder module
    return binding_encoding(data, rna_type, alphabet=alphabet)


# * PLOTTING ---


def plot_training(history, plot_names, split, count_models, count_plots):
    # plotting training and validation accuracy and loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.axis(ymin=0.4, ymax=1)
    plt.title('ResNet - Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Train_Accuracy', 'Validation_Accuracy'])
    plt.tight_layout()
    plt.grid()

    # plotting loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('ResNet - Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Train_Loss', 'Validation_Loss'])
    plt.tight_layout()
    plt.grid()

    plt.savefig(f'training_{plot_names}_{split}(NoReg)_{count_models}.{count_plots}.png')
    plt.close('all')

def plot_roc_curve(testing_labels, predictions, roc_auc, plot_names, split, count_models, count_plots):
    # plot ROC-AUC curve
    fpr, tpr, thresholds = roc_curve(testing_labels, predictions)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ResNet - Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    plt.savefig(f'ROC_{plot_names}_{split}(NoReg)_{count_models}.{count_plots}.png')
    plt.close('all')

def plot_pr_curve(testing_labels, predictions, plot_names, split, count_models, count_plots):
    precision, recall, thresholds = precision_recall_curve(testing_labels, predictions)
    pr_auc = auc(recall, precision)  # compute the AUC for Precision-Recall Curve
            
    # plot Precision-Recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"Precision-Recall Curve (AUC = {pr_auc:.4f})", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("ResNet - Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)

    plt.savefig(f'PR_{plot_names}_{split}(NoReg)_{count_models}.{count_plots}.png')
    plt.close('all')
    
    return pr_auc
    

# * MAIN ---


def main():
    # clear output .txt file
    with open(results_file_path, 'w') as results_file:
        pass

    # start main timer
    start_main_timer = time.time()


    # * TRAINING AND TESTING ---


    for training_file_path in training_file_paths:
        # change column name if the dataset is CLASH2013 and plot names
        if os.path.basename(training_file_path) == 'train_set_1_20_CLASH2013_paper.tsv':
            column_name = 'miRNA'
            dataset_name = 'CLASH_2013'
        else:
            column_name = 'noncodingRNA'
            dataset_name = 'Balanced_dataset'
        
        
        # * LOAD AND ENCODE DATA ---


        # load the training dataset
        print("\n\n----- <Loading Training Data> -----")
        df_train = pd.read_csv(training_file_path, sep='\t')

        # split df_train into actual training and validation sets
        training_data, testing_data = train_test_split(df_train, test_size=0.1, random_state=42)
        print("----- <Training Data Loaded Successfully> -----\n")
            
        # encode the training data
        print("----- <Encoding Training Data> -----")
        encoded_training_data, training_labels = encode_dataset(training_data, column_name)
        print("----- <Training Data Encoded Successfully> -----\n")

        # get model input shape from encoded data
        input_shape = encoded_training_data.shape[1:]
        print(f"Input shape: {input_shape}\n")
        
        # reset model count
        count_models = 1
            
        # loop through all hyperparameter combinations
        for dropout_rate in dropout_rates: 
            # reset plot count
            count_plots = 1
        
            print(f"Training model with {os.path.basename(training_file_path)}\n")

            with open(results_file_path, 'a') as results_file:
                results_file.write(f"Training model with {os.path.basename(training_file_path)}\n")
                results_file.write("=" * 100 + "\n\n")

            # start training timer
            start_training_timer = time.time()
            

            # build model
            model = build_resnet(input_shape, dropout_rate)


            # * TRAINING THE MODEL ---
            
            
            # train the model
            history = model.fit(encoded_training_data, 
                                training_labels, 
                                epochs=epochs,
                                batch_size=batch_size, 
                                validation_split=0.1,   # 0.1 for 80-10-10 split - 0.05 for 85-5-10 split
                                verbose=1)

            # end training timer
            end_training_timer = time.time()
            # calculate and print the main time taken
            elapsed_training_timer = end_training_timer - start_training_timer
            print(f"\nTime taken for training: {round(elapsed_training_timer / 60, 2)} minutes\n")

            # plot training and validation accuracy and loss
            plot_training(history, dataset_name, split_used, count_models, count_plots)
            count_plots += 1
                
                    
            # * TESTING THE MODEL ---

            # encode testing data
            print("----- <Encoding Testing Data> -----")
            encoded_testing_data, testing_labels = encode_dataset(testing_data, column_name)
            print("----- <Testing Data Encoded Successfully> -----\n")
            
            # validate input shape
            print(f"Expected input shape: {model.input_shape}")
            print(f"Encoded testing data shape: {encoded_testing_data.shape}\n")

            test_loss, test_accuracy = model.evaluate(encoded_testing_data, testing_labels, verbose=0)

            predictions = model.predict(encoded_testing_data, verbose=0)
            roc_auc = roc_auc_score(testing_labels, predictions)
                
            # plot ROC curve
            plot_roc_curve(testing_labels, predictions, roc_auc, dataset_name, split_used, count_models, count_plots)
            count_plots += 1
            
            # plot Precision-Recall curve
            pr_auc = plot_pr_curve(testing_labels, predictions, dataset_name, split_used, count_models, count_plots)
            count_plots += 1

            with open(results_file_path, 'a') as results_file:
                results_file.write(f"**Test loss:** {round(test_loss, 3)}\n")
                results_file.write(f"**Test accuracy:** {round(test_accuracy, 3)} - {round(test_accuracy * 100, 2)}%\n")
                results_file.write(f"**ROC-AUC:** {round(roc_auc, 3)}\n")
                results_file.write(f"**PR-AUC:** {round(pr_auc, 3)}\n\n")

            print(f"Results: Test_Loss={round(test_loss, 3)}, Test_Accuracy={round(test_accuracy, 3)}, ROC-AUC={round(roc_auc, 3)}, PR-AUC={round(pr_auc, 3)}")
            
            
            # * SAVE MODEL ---
            
            
            print("\n----- <Saving Model> -----")
            # ensure the directory exists
            os.makedirs(save_dir, exist_ok=True)
            # construct the full file path
            model_path = os.path.join(save_dir, f"ResNet_{split_used}(NoReg)-{dataset_name}_{count_models}.keras")
            
            model.save(model_path)
            print("----- <Model Saved Successfully> -----\n")
            
            
            count_models += 1
            
            
            # end main timer
            end_main_timer = time.time()
            # calculate main time taken
            elapsed_main_timer = end_main_timer - start_main_timer
            # print main time taken
            print(f"Time taken for training and testing: {round(elapsed_main_timer / 60, 2)} minutes\n\n")

            # write the time taken to the results file   
            with open(results_file_path, 'a') as results_file:
                results_file.write(f"**Time taken for training:** {round(elapsed_training_timer / 60, 2)} minutes\n")
                results_file.write(f"**Time taken for training and testing:** {round(elapsed_main_timer / 60, 2)} minutes\n\n")
                results_file.write("=" * 100 + "\n")
        
        
            # * CLEAN UP RESOURCES ---


            # delete objects
            del model, history
            del encoded_training_data, training_labels
            del encoded_testing_data, testing_labels, predictions
            del roc_auc, pr_auc

            # force garbage collection
            gc.collect()

            # reset TensorFlow graph
            tf.keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()


    print(f"\nResults saved to {results_file_path}. Graphs saved as '<plot_type>_<dataset_name>_{split_used}(NoReg)_<#>.png'")


# call main function
if __name__ == '__main__':
    main()
