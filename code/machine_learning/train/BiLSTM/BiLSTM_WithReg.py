# imports
import sys
sys.path.insert(1, 'code/')

import os
import argparse
import numpy as np
from tensorflow.keras.regularizers import L1, L2, L1L2
from sklearn.model_selection import KFold
from BiLSTM_Architectures import BiLSTM
from helper_functions.model_utils import (set_seed,
                                          #load_data, 
                                          make_files, 
                                          calculate_avg_std,
                                          subset_data, 
                                          train_model,
                                          save_model, 
                                          cleanup)

# * PARAMS ---

# parameters
epochs = 20  # number of epochs/dataset iterations
batch_size = 32  # batch size
learning_rate = 0.001  # learning rate
model_type = "BiLSTM"

list_of_large_datasets = ["AGO2_eCLIP_Manakov2022_train_dataset"]

# hyperparameter combinations
reg_factors = [0.01, 0.005, 0.005, 0.01, 0.003, 0.002]
dropout_rates = [0.05, 0.09, 0.13, 0.17, 0.21, 0.25]

n_splits = 5

# regularizer types
regularizers = {
    "L1" : L1, 
    "L2" : L2, 
    "L1L2" : L1L2
}

# when switching to binding_2D_matrix_encoder.py, change the function name to load_data from helper_functions folder
def load_data(data_file, label_file):
    # load data
    encoded_data = np.load(data_file, mmap_mode='r')
    encoded_labels = np.load(label_file, mmap_mode='r')
    
    return encoded_data, encoded_labels

# * MAIN - TRAINING FUNCTION ---

# main pipeline
def main():
    # argument parser for dataset path and learning rate
    parser = argparse.ArgumentParser(description="Train a BiLSTM model for miRNA-mRNA target site classification")
    parser.add_argument("-e_data", "--encoded_data", required=True, type=str, help="Path to the encoded training dataset (.npy file)")
    parser.add_argument("-e_labels", "--encoded_labels", required=True, type=str, help="Path to the encoded training labels (.npy file)")
    parser.add_argument("-reg", "--regularization", required=True, type=str, help="NoReg or WithReg")
    parser.add_argument("-plots", "--plot_plots", required=True, type=str, help="Whether to save the training plots or not (true/false)")
    parser.add_argument("-s", "--seed", required=True, type=int, help="Random seed for reproducibility")
    args = parser.parse_args()

    # seeding
    print(f"\nSeed set to: {args.seed}")
    set_seed(args.seed)
    
    # sorting the model and dataset paths and split them into lists
    training_data_files = sorted(args.encoded_data.split(','))
    training_labels_files = sorted(args.encoded_labels.split(','))
    
    # define the directory where you want to save the model and training logs
    results_file_path = f"Saves_{model_type}_{args.regregularization}/BiLSTM_WithReg_training_logs.txt"
    save_dir = f"Saves_{model_type}_{args.regregularization}/BiLSTM_Models"
    
    # create the save directory
    make_files(os.path.split(save_dir)[0], [os.path.split(save_dir)[1]])

    # clear the results file
    with open(results_file_path, 'w') as results_file:
        pass
    
    # loop through all datasets
    for training_data_file, training_label_file in zip(training_data_files, training_labels_files):
        # initialize a list to hold the configuration results
        config_results = []
        
        # extract dataset name
        dataset_name = os.path.splitext(os.path.basename(training_data_file))[0].replace('_train_dataset', '')

        # load the encoded training data and labels        
        full_training_data, full_training_labels = load_data(training_data_file, training_label_file)
        input_shape = full_training_data.shape[1:]
        
        # check if the dataset is large - subset if true
        if os.path.basename(os.path.splitext(training_data_file)[0]) in list_of_large_datasets:
            # subset the data for each configuration - 25000 samples
            encoded_data, encoded_labels = subset_data(full_training_data, full_training_labels)
            
            # replace input shape
            input_shape = encoded_data.shape[1:]
            print(f"\nSubsetted {dataset_name} dataset to {encoded_data.shape[0]} samples")
        else:
            # use the full training data
            encoded_data, encoded_labels = full_training_data, full_training_labels
        
        # loop through all regularizer types
        for regularizer_type in regularizers.keys():
            # print regularizer type
            print(f"\n\nUsing Regularizer: {regularizer_type}")
            
            with open(results_file_path, 'a') as results_file:
                results_file.write(f"Using Regularizer: {regularizer_type} ---\n")
                results_file.write("=" * 100 + "\n")
            
            # loop through all hyperparameter combinations
            for index, (reg_factor, dropout_rate) in enumerate(zip(reg_factors, dropout_rates), start=1):
                print(f"\nConfig {index} | Training model with {dataset_name}, reg_factor={reg_factor}, dropout_rate={dropout_rate}\n")
                
                with open(results_file_path, 'a') as results_file:
                    results_file.write(f"\nConfig {index} | Training model with {dataset_name}, reg_factor={reg_factor}, dropout_rate={dropout_rate}\n")

                # initialize lists to hold CV metrics for this config
                cv_accuracies, cv_losses, cv_f1s, cv_precisions, cv_recalls = [], [], [], [], []
                
                # create 5-fold cross validation splitter
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                
                for fold_count, (train_index, val_index) in enumerate(kf.split(encoded_data), start=1):
                    print(f"\n--- Fold {fold_count} of {n_splits} ---")
                    
                    # get training and validation data
                    X_train, y_train = encoded_data[train_index].copy(), encoded_labels[train_index].copy()
                    X_val, y_val = encoded_data[val_index].copy(), encoded_labels[val_index].copy()
                
                    # build model
                    model = BiLSTM(input_shape, dropout_rate, learning_rate, reg_factor, regularizers[regularizer_type])
                    
                    # train the model on folds
                    model, elapsed_training_time, metrics = train_model(model, epochs, batch_size, 
                                                                        X_train, y_train, model_type,
                                                                        dataset_name, regularizer_type,
                                                                        dropout_rate, save_dir, args, reg_factor, 
                                                                        val_data=X_val, val_labels=y_val)

                    with open(results_file_path, 'a') as results_file:
                        results_file.write(f"\nFold {fold_count} | Time taken for training with, regularizer: {regularizer_type}, reg_factor: {reg_factor}, dropout_rate: {dropout_rate} | {(elapsed_training_time):.3f} s")
                        results_file.write(f"\nFold {fold_count} Metrics | Accuracy: {metrics['accuracy']:.3f}, Loss: {metrics['loss']:.3f}, F1 Score: {metrics['f1']:.3f}, Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}\n")
                    
                    # save fold metrics
                    cv_accuracies.append(metrics['accuracy'])
                    cv_losses.append(metrics['loss'])
                    cv_f1s.append(metrics['f1'])
                    cv_precisions.append(metrics['precision'])
                    cv_recalls.append(metrics['recall'])
                
                    del elapsed_training_time, X_train, y_train, X_val, y_val, model, metrics
                    cleanup()    
                
                calculate_avg_std(cv_accuracies, cv_losses, cv_f1s, cv_precisions, cv_recalls, dropout_rate, regularizer_type, results_file_path, config_results, reg_factor)
                
        # check if any config_results exist
        if config_results:
            # initialise best_config as None and the best_f1 as negative infinity
            best_config = None
            best_f1 = -float('inf')  # negative infinity ensures any real F1 score will be higher
            
            # loop through each configuration in the list
            for config in config_results:
                # check if current configuration's average F1 score is greater than the best F1 score found so far
                if config['avg_f1'] > best_f1:
                    # update the best F1 score and configuration
                    best_f1 = config['avg_f1']
                    best_config = config
            
            print("\n----- Final Configuration Found -----")
            print(f"Regularizer: {best_config['regularizer_type']}, reg_factor: {best_config['reg_factor']}, dropout_rate: {best_config['dropout_rate']}, avg_f1: {best_config['avg_f1']:.3f}")
            with open(results_file_path, 'a') as results_file:
                results_file.write("\n----- Optimal Configuration Found -----\n")
                results_file.write(f"Regularizer: {best_config['regularizer_type']}, reg_factor: {best_config['reg_factor']}, dropout_rate: {best_config['dropout_rate']}, avg_f1: {best_config['avg_f1']:.3f}\n\n")
                results_file.write("=" * 100 + "\n")
            
            # train a final model on the full training set using the best configuration.
            print("\n----- Training Final Model on Full Training Data -----")
            
            # load the full training data
            encoded_data, encoded_labels = load_data(training_data_file, training_label_file)
            input_shape = encoded_data.shape[1:]
            
            final_model = BiLSTM(input_shape, best_config['dropout_rate'], learning_rate, best_config['reg_factor'], regularizers[best_config['regularizer_type']])
            
            # train final model (using validation_split here)
            final_model, _, _ = train_model(final_model, epochs, batch_size,
                                            encoded_data, encoded_labels, model_type,
                                            dataset_name, best_config['regularizer_type'], 
                                            best_config['dropout_rate'], save_dir, args, best_config['reg_factor'])
            
            # save the final model
            save_model(final_model, save_dir, model_type, best_config['regularizer_type'], dataset_name, best_config['dropout_rate'], best_config['reg_factor'])
            print("----- Final Model Trained and Saved -----\n")
        else:
            print("\n!!! No configurations found !!!")

        # write completion message to results file
        with open(results_file_path, 'a') as results_file:
            results_file.write("\n--- Done ---")
        
        print(f"Results saved to {results_file_path}\n")
        
        del encoded_data, encoded_labels, final_model, config_results, best_config
        cleanup()


if __name__ == "__main__":
    main()      
