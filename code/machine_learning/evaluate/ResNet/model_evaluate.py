# imports
import sys
sys.path.insert(1, 'code/')

import os
import argparse
import pandas as pd
from helper_functions.model_utils import (load_data, 
                                          make_files, 
                                          plot_roc_crossval, 
                                          plot_pr_crossval,
                                          compute_metrics,  
                                          cleanup)

# * MAIN - EVALUATION FUNCTION ---

def main():
    # argument parser for dataset path and learning rate
    parser = argparse.ArgumentParser(description="Load trained model to elvauate the performance of miRNA-mRNA target site classification")
    parser.add_argument("-rn_type", "--ResNet_type", required=True, default=None, type=str, help="Type of ResNet model to train (small [373,121], medium [1,360,001], large [16,691,073])")
    parser.add_argument("-e_data", "--encoded_data", required=True, default=None, type=str, help="List of path to the encoded testing dataset (.npy files)")
    parser.add_argument("-e_labels", "--encoded_labels", required=True, default=None, type=str, help="Lists of path to the encoded testing labels (.npy files)")
    parser.add_argument("-preds", "--predictions", required=True, default=None, type=str, help="List of paths to prediction files (.tsv files)")
    parser.add_argument("-reg", "--regularization", required=True, default=None, type=str, help="NoReg or WithReg using in naming the .tsv file")
    parser.add_argument("-plots", "--plot_plots", required=True, default=None, type=str, help="Wheather to save the training plots or not (true/false)")    
    args = parser.parse_args()
    
    # split model and dataset paths into lists and sort them
    test_data_files = sorted(args.encoded_data.split(','))
    test_label_files = sorted(args.encoded_labels.split(','))
    prediction_files = sorted(args.predictions.split(','))
    
    # initialise save predictions path
    results_file_path = f'Saves/ResNet_{args.regularization}_evaluation_logs.txt'
    
    if args.plot_plots == 'true':
        # create directory for saving plots
        save_dir = "Saves/ResNet_Evaluation"
        make_files(os.path.split(save_dir)[0], [os.path.split(save_dir)[1]])
        
    # clear the results file
    with open(results_file_path, 'w') as results_file:
        pass
            
    count_preds = 1
    model_type = f"ResNet_{args.ResNet_type.lower()}"
        
    # iterate over all test data, test label and prediction files
    for test_data, test_label, prediction in zip(test_data_files, test_label_files, prediction_files):
        # initialise plot counter
        count_plots = 2
        
        # get test data name for easy identification
        test_data_name = os.path.basename(test_data)
        test_label_name = os.path.basename(test_label)
        prediction_name = os.path.basename(prediction)
        
        # load the data and labels
        print(f"\nLoading encoded data from: {test_data_name} and {test_label_name} ...")
        test_data, test_labels = load_data(test_data, test_label)
        
        # load the predictions
        print(f"Loading predictions from: {prediction_name} ...")
        predictions_df = pd.read_csv(prediction, sep='\t')
        
        # iterate over each model present in the prediction file header
        for model_name in predictions_df.columns:
            with open(results_file_path, 'a') as results_file:
                results_file.write(f"Evaluating {model_name}\n")
                results_file.write("=" * 100 + "\n")
            
            predictions = predictions_df[model_name].values

            # compute metrics
            metrics_dict = compute_metrics(val_labels=test_labels, predictions=predictions)
            
            if args.plot_plots == 'true':
                print("Plotting ROC and PR curves ...")
                roc_auc = plot_roc_crossval(test_labels, predictions, model_type, save_dir, model_name, count_plots, count_preds)
                count_plots += 1
                pr_ap = plot_pr_crossval(test_labels, predictions, model_type, save_dir, model_name, count_plots, count_preds)
            else:
                print("Skipping plotting ...")
            
            # write the results to the results file
            with open(results_file_path, 'a') as results_file:
                results_file.write(f"\nPlot ID: plot{count_plots}_pred{count_preds}\n")
                results_file.write(f"Dataset: {test_data_name}\n")
                results_file.write(f"**Test loss:** {metrics_dict['loss']:.3f}\n")
                results_file.write(f"**Test accuracy:** {metrics_dict['accuracy']:.3f} - {(metrics_dict['accuracy'] * 100):.3f}%\n")
                results_file.write(f"**Test precision:** {metrics_dict['precision']:.3f}\n")
                results_file.write(f"**Test recall:** {metrics_dict['recall']:.3f}\n")
                results_file.write(f"**PR-AP:** {pr_ap:.3f}\n")
                results_file.write(f"**ROC-AUC:** {roc_auc:.3f}\n")
                results_file.write(f"**F1-score:** {metrics_dict['f1']:.3f}\n\n")
                results_file.write("=" * 100 + "\n")
        
        count_preds += 1
        
        # clear the memory-mapped data
        del test_data, test_labels, predictions, predictions_df
        cleanup()
        
    # write completion message to results file
    with open(results_file_path, 'a') as results_file:
        results_file.write("\n--- Done ---")


if __name__ == "__main__":
    main()