# imports
import sys
sys.path.insert(1, 'code/')

import os
import argparse
from sklearn.model_selection import KFold
from ResNet_Architectures import build_resnet_small, build_resnet_medium, build_resnet_large
from helper_functions.model_utils import (load_data, 
                                          make_files, 
                                          calculate_avg_std, 
                                          train_model,
                                          save_model, 
                                          cleanup)

# * PARAMS ---

# parameters
epochs = 20  # number of epochs/dataset iterations
batch_size = 32  # batch size
learning_rate = 0.001  # learning rate
results_file_path = 'Saves/ResNet_WithReg_training_logs.txt'
# define the directory where you want to save the model
save_dir = "Saves/ResNet_Models"
regularizer_type = "NoReg"

# hyperparameter combinations
dropout_rates = [0.05, 0.09, 0.13, 0.17, 0.21, 0.25]

n_splits = 5

# * MAIN - TRAINING - NOREG ---

def main():
    # argument parser for dataset path and learning rate
    parser = argparse.ArgumentParser(description="Train a ResNet model for miRNA-mRNA target site classification")
    parser.add_argument("-rn_type", "--ResNet_type", required=True, default=None, type=str, help="Type of ResNet model to train (small [373,121], medium [1,360,001], large [16,691,073])")
    parser.add_argument("-e_data", "--encoded_data", required=True, default=None, type=str, help="Path to the encoded training dataset (.npy file)")
    parser.add_argument("-e_labels", "--encoded_labels", required=True, default=None, type=str, help="Path to the encoded training labels (.npy file)")
    parser.add_argument("-plots", "--plot_plots", required=True, default=None, type=str, help="Wheather to save the training plots or not (true/false)")
    args = parser.parse_args()

    training_data_files = sorted(args.encoded_data.split(','))
    training_labels_files = sorted(args.encoded_labels.split(','))
    
    # create the save directory
    make_files(os.path.split(save_dir)[0], [os.path.split(save_dir)[1]])

    # clear the results file
    with open(results_file_path, 'w') as results_file:
        pass
    
    model_type = (f"ResNet_{args.ResNet_type.lower()}")

    # loop through all datasets
    for training_data_file, training_label_file in zip(training_data_files, training_labels_files):
        # initialize a list to hold the configuration results
        config_results = []
        
        # extract dataset name
        dataset_name = os.path.splitext(os.path.basename(training_data_file))[0]
        dataset_name = dataset_name.replace('_train_dataset', '')

        # load the encoded training data and labels        
        encoded_training_data, training_labels = load_data(training_data_file, training_label_file)
        input_shape = encoded_training_data.shape[1:]

        # print regularizer type
        print(f"\n\nUsing Regularizer: {regularizer_type}")
        
        with open(results_file_path, 'a') as results_file:
            results_file.write(f"Using Regularizer: {regularizer_type} ---\n")
            results_file.write("=" * 100 + "\n")
        
        # loop through all hyperparameter combinations
        for index, dropout_rate in enumerate(dropout_rates, start=1):
            print(f"\nConfig {index} | Training model with {dataset_name}, dropout_rate={dropout_rate}\n")
            
            with open(results_file_path, 'a') as results_file:
                results_file.write(f"\nConfig {index} | Training model with {dataset_name}, dropout_rate={dropout_rate}\n")

            # initialize lists to hold CV metrics for this config
            cv_accuracies, cv_losses, cv_f1s, cv_precisions, cv_recalls = [], [], [], [], []
            
            # create 5-fold cross validation splitter
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            fold_count = 1
            
            for train_index, val_index in kf.split(encoded_training_data):
                print(f"\n--- Fold {fold_count} of {n_splits} ---")
                
                # get training and validation data
                X_train = encoded_training_data[train_index].copy()
                y_train = training_labels[train_index].copy()
                X_val = encoded_training_data[val_index].copy()
                y_val = training_labels[val_index].copy()
            
                # build model
                if args.ResNet_type.lower() == "small":
                    model = build_resnet_small(input_shape, dropout_rate, learning_rate)
                elif args.ResNet_type.lower() == "medium":
                    model = build_resnet_medium(input_shape, dropout_rate, learning_rate)
                elif args.ResNet_type.lower() == "large":
                    model = build_resnet_large(input_shape, dropout_rate, learning_rate)
                else:
                    raise ValueError("!!! Invalid ResNet type. Only 'small', 'medium', or 'large' are recognised !!!")
            
                # train the model on folds
                model, history, elapsed_training_time, metrics = train_model(model, epochs, batch_size,
                                                                             X_train, y_train, model_type,
                                                                             dataset_name, regularizer_type,
                                                                             dropout_rate, save_dir, args,
                                                                             val_data=X_val, val_labels=y_val)
                
                with open(results_file_path, 'a') as results_file:
                    results_file.write(f"\nFold {fold_count} | Time taken for training with regularizer: {regularizer_type}, dropout_rate: {dropout_rate} | {(elapsed_training_time):.3f} s")
                    results_file.write(f"\nFold {fold_count} Metrics | Accuracy: {metrics['accuracy']:.3f}, Loss: {metrics['loss']:.3f}, F1 Score: {metrics['f1']:.3f}, Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}\n")
                
                # Save fold metrics
                if metrics:
                    cv_accuracies.append(metrics['accuracy'])
                    cv_losses.append(metrics['loss'])
                    cv_f1s.append(metrics['f1'])
                    cv_precisions.append(metrics['precision'])
                    cv_recalls.append(metrics['recall'])
                
                fold_count += 1
            
                del elapsed_training_time, X_train, y_train, X_val, y_val, model, history
                cleanup()
                
            calculate_avg_std(cv_accuracies, cv_losses, cv_f1s, cv_precisions, cv_recalls, dropout_rate, regularizer_type, results_file_path, config_results)
            
            del metrics, cv_accuracies, cv_f1s, cv_precisions, cv_recalls
            cleanup() 
                
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
            print(f"Regularizer: {best_config['regularizer_type']}, dropout_rate: {best_config['dropout_rate']}")
            with open(results_file_path, 'a') as results_file:
                results_file.write("\n----- Final Configuration Found -----\n")
                results_file.write(f"Regularizer: {best_config['regularizer_type']}, dropout_rate: {best_config['dropout_rate']}\n\n")
                results_file.write("=" * 100 + "\n")
            
            # train a final model on the full training set using the best configuration.
            print("\n----- Training Final Model on Full Training Data -----")
            
            # build model
            if args.ResNet_type.lower() == "small":
                final_model = build_resnet_small(input_shape, dropout_rate, learning_rate)
            elif args.ResNet_type.lower() == "medium":
                final_model = build_resnet_medium(input_shape, dropout_rate, learning_rate)
            elif args.ResNet_type.lower() == "large":
                final_model = build_resnet_large(input_shape, dropout_rate, learning_rate)
            else:
                raise ValueError("!!! Invalid ResNet type. Only 'small', 'medium', or 'large' are recognised !!!")
            
            # train final model
            final_model, _, _, _ = train_model(final_model, epochs, batch_size, 
                                               encoded_training_data, training_labels, model_type,
                                               dataset_name, best_config['regularizer_type'], 
                                               best_config['dropout_rate'], save_dir, args)
            
            # save the final model
            save_model(final_model, save_dir, model_type, best_config['regularizer_type'], dataset_name, best_config['dropout_rate'])
            print("\n----- Final Model Trained and Saved -----\n")
        else:
            print("\n!!! No configurations found !!!")
            
        del encoded_training_data, training_labels, final_model, config_results, best_config, best_f1
        cleanup()

    # write completion message to results file
    with open(results_file_path, 'a') as results_file:
        results_file.write("\n--- Done ---")

    print(f"\nResults saved to {results_file_path}")


if __name__ == "__main__":
    main()      
