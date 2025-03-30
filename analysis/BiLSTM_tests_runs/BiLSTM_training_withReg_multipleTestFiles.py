# imports
import sys
sys.path.append('./')

import time
import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import tensorflow.keras.layers as layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import L1, L2, L1L2
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
from encoders.sequence_encoder_16ntPairs import get_label_dict, get_encoded_matrix
# from code.machine_learning.encode.sequence_encoder_16ntPairs import get_label_dict, get_encoded_matrix


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

learning_rate = 0.001  # learning rate
epochs = 20  # number of epochs/dataset iterations
batch_size = 32  # batch size

results_file_path = 'autoTrain_modelResults.txt'

# define the directory where you want to save the model
save_dir = "SavedModels/BiLSTM"

# hyperparameter combinations
dropout_rates = [0.05, 0.09, 0.13, 0.17, 0.21, 0.25]

testing_file_paths = [
    'datasets/testing/test_set_1_1_CLASH2013_paper.tsv',
    'datasets/testing/test_set_1_10_CLASH2013_paper.tsv',
    'datasets/testing/test_set_1_100_CLASH2013_paper.tsv',
]

'''
Program stops running after completion of the first two regularisers. 
If there is not a fix, split the regularisers into seperate processes.
'''
regularizers = {
    "L1" : L1, 
    "L2" : L2, 
    # "L1L2" : L1L2
}


# * BUILDING BIDIRECTIONAL LSTM ---


# function to build the BiLSTM model with Attention layer
def BiLSTM(input_shape, dropout_rate, learning_rate, regularizer):
    # clear any previous models
    tf.keras.backend.clear_session()
    
    # define input layer with given input shape
    input_layer = layers.Input(shape=input_shape)
    
    # trainable weights for nucleotide pairs
    pair_embeddings = layers.Embedding(input_dim=16 + 1, output_dim=1)(input_layer)
    pair_embeddings = layers.Reshape((25,50))(pair_embeddings)

    # first BiLSTM layer (128 units) for bidirectional sequence processing with regularization  
    bilstm1 = layers.Bidirectional(layers.LSTM(units=128, 
                                               return_sequences=True, 
                                               kernel_regularizer=regularizer, 
                                               recurrent_regularizer=regularizer))(pair_embeddings)
    dropout1 = layers.Dropout(dropout_rate)(bilstm1)

    # second BiLSTM layer (64 units) for further feature extraction with regularization
    bilstm2 = layers.Bidirectional(layers.LSTM(units=64, 
                                               return_sequences=True, 
                                               kernel_regularizer=regularizer, 
                                               recurrent_regularizer=regularizer))(dropout1)
    dropout2 = layers.Dropout(dropout_rate)(bilstm2)

    # attention layer 
    attention = layers.Attention()([dropout2, dropout2])
    
    # global average pooling layer to reduce sequence dimension by averaging features across time steps
    pooled = layers.GlobalAveragePooling1D()(attention)

    # fully connected dense layer (512 neurons, ReLU) for high-level feature extraction with regularization
    dense = layers.Dense(units=512, activation='relu', kernel_regularizer=regularizer)(pooled)
    # batch normalization for stable and faster learning
    batch_norm = layers.BatchNormalization()(dense)
    dropout3 = layers.Dropout(dropout_rate)(batch_norm)

    # output layer for binary classification - 1 neuron (1 or 0)
    output = layers.Dense(units=1, activation='sigmoid')(dropout3)

    # build model
    model = Model(inputs=input_layer, outputs=output)
    # compile model with Adam optimizer and binary crossentropy loss and accuracy metric
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy',metrics=['accuracy'])
    # Summary
    model.summary()
    
    print()
    
    return model


# * PLOTTING ---


def plot_training(history, plot_names, regularizer_type, count_models, count_plots):
    # plotting training and validation accuracy and loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.axis(ymin=0.4, ymax=1)
    plt.title('BiLSTM - Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Train_Accuracy', 'Validation_Accuracy'])
    plt.tight_layout()
    plt.grid()

    # plotting loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('BiLSTM - Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Train_Loss', 'Validation_Loss'])
    plt.tight_layout()
    plt.grid()

    plt.savefig(f'training_{plot_names}_MultiTest(WithReg-{regularizer_type})_{count_models}.{count_plots}.png')
    plt.close('all')

def plot_roc_curve(testing_labels, predictions, roc_auc, plot_names, regularizer_type, count_models, count_plots):
    # plot ROC-AUC curve
    fpr, tpr, thresholds = roc_curve(testing_labels, predictions)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("BiLSTM - Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    plt.savefig(f'ROC_{plot_names}_MultiTest(WithReg-{regularizer_type})_{count_models}.{count_plots}.png')
    plt.close('all')

def plot_pr_curve(testing_labels, predictions, plot_names, regularizer_type, count_models, count_plots):
    precision, recall, thresholds = precision_recall_curve(testing_labels, predictions)
    pr_auc = auc(recall, precision)  # compute the AUC for Precision-Recall Curve
            
    # plot Precision-Recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"Precision-Recall Curve (AUC = {pr_auc:.4f})", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("BiLSTM - Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)

    plt.savefig(f'PR_{plot_names}_MultiTest(WithReg-{regularizer_type})_{count_models}.{count_plots}.png')
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
            
        # loop through all regularizer types
        for regularizer_type in regularizers.keys():
            # print regularizer type
            print(f"\n\nUsing Regularizer: {regularizer_type}")
            
            with open(results_file_path, 'a') as results_file:
                results_file.write(f"Using Regularizer: {regularizer_type}\n")
                results_file.write("=" * 100 + "\n")
        
        
            # * LOAD AND ENCODE DATA ---


            # load the training dataset
            print("\n\n----- <Loading Training Data> -----")
            df_train = pd.read_csv(training_file_path, sep='\t')
            print("----- <Training Data Loaded Successfully> -----\n")
            
            # encode the testing data
            print("----- <Encoding Training Data> -----")
            
            # preprocess labels to avoid repeated DataFrame filtering
            label_dict_training = get_label_dict(df_train, column_name)

            # initialize lists to store encoded matrices and labels
            encoded_training_data = []
            training_labels = []

            # encode training data
            for _, row in df_train.iterrows():
                target_sequence = row['gene']
                mirna_sequence = row[column_name]
                
                encoded_matrix, label = get_encoded_matrix(label_dict_training , target_sequence, mirna_sequence)
                
                encoded_training_data.append(encoded_matrix)
                training_labels.append(label)
                
            print("----- <Training Data Encoded Successfully> -----\n")
            
            # convert to numpy array
            encoded_training_data = np.array(encoded_training_data)
            training_labels = np.array(training_labels)
            
            print(f"Final shape of encoded_training_data: {encoded_training_data.shape}")  # Should be (N, 25, 50)
            print(f"Final shape of training_labels: {training_labels.shape}")  # Should be (N, 1)
                    
            # get model input shape from encoded data
            input_shape = encoded_training_data.shape[1:]
            print(f"Input shape: {input_shape}\n")
            
            # reset model count
            count_models = 1
            
            # loop through all hyperparameter combinations
            for dropout_rate in dropout_rates: 
                # reset plot count
                count_plots = 1
                
                print(f"Training BiLSTM with {os.path.basename(training_file_path)}, dropout_rate={dropout_rate}\n")
                
                with open(results_file_path, 'a') as results_file:
                    results_file.write(f"Training BiLSTM with {os.path.basename(training_file_path)}, dropout_rate={dropout_rate}\n")
                    results_file.write("=" * 100 + "\n\n")

                # start training timer
                start_training_timer = time.time()
                
                
                # build BiLSTM model
                model = BiLSTM(input_shape, dropout_rate, learning_rate, regularizer_type)
                
                print(f"Expected input shape for model: {model.input_shape}")
                print(f"Encoded training data shape: {encoded_training_data.shape}\n")
                
                
                # * TRAINING THE MODEL ---
                
                
                # train the model
                history = model.fit(encoded_training_data, 
                                    training_labels, 
                                    epochs=epochs,
                                    batch_size=batch_size, 
                                    validation_split=0.1,
                                    verbose=1)

                # end training timer
                end_training_timer = time.time()
                # calculate and print the main time taken
                elapsed_training_timer = end_training_timer - start_training_timer
                print(f"\nTime taken for training with dropout_rate={dropout_rate}: {round(elapsed_training_timer / 60, 2)} minutes\n")

                # plot training and validation accuracy and loss
                plot_training(history, dataset_name, regularizer_type, count_models, count_plots)
                count_plots += 1
                
                
                # * TESTING THE MODEL ---
                
                
                # evaluate the model on the testing datasets
                for i, testing_file_path in enumerate(testing_file_paths, start=1):
                    print(f"\n----- <Evaluating Dataset {i}: {os.path.basename(testing_file_path)}> -----\n")
                    
                    with open(results_file_path, 'a') as results_file:
                        results_file.write(f"{os.path.basename(testing_file_path)}\n")

                    # read testing dataset
                    df_test = pd.read_csv(testing_file_path, sep='\t')
                
                    # encode the testing data
                    print("----- <Encoding Testing Data> -----")
                    
                    # preprocess labels to avoid repeated DataFrame filtering
                    label_dict_testing = get_label_dict(df_test, column_name)


                    # initialize lists to store encoded matrices and labels
                    encoded_testing_data = []
                    testing_labels = []

                    # encode training data
                    for _, row in df_test.iterrows():
                        target_sequence = row['gene']
                        mirna_sequence = row[column_name]
                        
                        encoded_matrix, label = get_encoded_matrix(label_dict_testing, target_sequence, mirna_sequence)
                        
                        encoded_testing_data.append(encoded_matrix)
                        testing_labels.append(label)
                        
                    print("----- <Testing Data Encoded Successfully> -----\n")
                    
                    # convert to numpy array
                    encoded_testing_data = np.array(encoded_testing_data)
                    testing_labels = np.array(testing_labels)
                    
                    # validate input shape
                    print(f"Expected input shape: {model.input_shape}")
                    print(f"Encoded testing data shape: {encoded_testing_data.shape}\n")

                    test_loss, test_accuracy = model.evaluate(encoded_testing_data, testing_labels, verbose=0)

                    predictions = model.predict(encoded_testing_data, verbose=0)
                    roc_auc = roc_auc_score(testing_labels, predictions)
                    
                    # plot ROC curve
                    plot_roc_curve(testing_labels, predictions, roc_auc, dataset_name, regularizer_type, count_models, count_plots)
                    count_plots += 1
                    
                    # plot Precision-Recall curve
                    pr_auc = plot_pr_curve(testing_labels, predictions, dataset_name, regularizer_type, count_models, count_plots)
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
                model_path = os.path.join(save_dir, f"BiLSTM_multipleTestFiles(WithReg-{regularizer_type})-{dataset_name}_{count_models}.keras")
                
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
            
            
    print(f"\nResults saved to {results_file_path}. Graphs saved as '<plot_type>_<dataset_name>_MultiTest(WithReg-<regularizer_type>)_<#>.png'")


# call main function
if __name__ == '__main__':
    main()