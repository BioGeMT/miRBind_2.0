# imports
import os
import gc
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.metrics import (accuracy_score, 
                             f1_score, 
                             precision_score, 
                             recall_score, 
                             roc_curve, 
                             roc_auc_score, 
                             average_precision_score, 
                             precision_recall_curve)

# * SEEDING ---

def set_seed(seed):
    # set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # print the seed used
    print(f"Seed set to: {seed}")
    
# * PLOTTING ---

# plot training and validation accuracy and loss
def plot_training(save_dir, history, model_type, dataset_name, regularizer_type, dropout_rate, plot_count, reg_factor=None):
    plt.figure(figsize=(12, 6))
    
    # plotting accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.axis(ymin=0.4, ymax=1)
    plt.title(f'{model_type} - Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Train_Accuracy', 'Validation_Accuracy'])
    plt.grid()

    # plotting loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_type} - Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Train_Loss', 'Validation_Loss'])
    plt.grid()

    plt.tight_layout()
    
    if reg_factor is not None:
        plot_filename = f'{model_type}_{regularizer_type}_{dataset_name}_dr{dropout_rate}_rf{reg_factor}_plot{plot_count}_train.png'
    else:
        plot_filename = f'{model_type}_{regularizer_type}_{dataset_name}_dr{dropout_rate}_plot{plot_count}_train.png'
    
    plt.savefig(os.path.join(save_dir, plot_filename))
    plt.close('all')
    
# plot ROC curve for cross-validated model evaluation
def plot_roc_crossval(labels, predictions, model_type, save_dir, model_name, count_plots, count_preds):
    plt.figure(figsize=(9, 7))
    
    # compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(labels, predictions)
    roc_auc = roc_auc_score(labels, predictions)
    
    # plot individual fold ROC curve
    plt.plot(fpr, tpr, color="blue", label=f"ROC (AUC = {roc_auc:.3f})", linewidth=2)

    # plot chance level
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    
    # labels and legend
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f'{model_type} - ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plot_filename = f'{os.path.splitext(model_name)[0]}_plot{count_plots}_pred{count_preds}_ROC.png'
    plt.savefig(os.path.join(save_dir, plot_filename))
    plt.close('all')
    
    return roc_auc

# plot Precision-Recall curve for cross-validated model evaluation
def plot_pr_crossval(labels, predictions, model_type, save_dir, model_name, count_plots, count_preds):
    plt.figure(figsize=(9, 7))
    
    # compute Precision-Recall curve and AUC
    precision, recall, _ = precision_recall_curve(labels, predictions)
    pr_ap = average_precision_score(labels, predictions)
    
    # plot individual fold PR curve
    plt.plot(recall, precision, color="blue", label=f"PR (AP = {pr_ap:.3f})", linewidth=2)

    # labels and legend
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f'{model_type} - PR Curve - {model_name}')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plot_filename = f'{os.path.splitext(model_name)[0]}_plot{count_plots}_pred{count_preds}_PR.png'
    plt.savefig(os.path.join(save_dir, plot_filename))
    plt.close('all')
    
    return pr_ap

# * HANDELING MEMMAP ---   

# get the shape of the memory-mapped file (dataset) - helps in loading the data
def get_memmap_shape(file_path, element_shape, dtype=np.float32):
    """Infers the first dimension (dataset_size) for a memory-mapped file."""
    # size of one element in bytes
    item_size = np.prod(element_shape) * np.dtype(dtype).itemsize
    # file size in bytes
    total_size = os.path.getsize(file_path)
    # number of elements
    dataset_size = total_size // item_size
    
    # print(f"Total size: {dataset_size}")
    
    # return the shape tuple
    return (dataset_size, *element_shape)

# * LOADING DATA ---

def load_data(data_file=None, label_file=None):
    # check if at least one file is provided
    if data_file is None and label_file is None:
        raise RuntimeError("!!! Warning: No data or label file provided !!!")
    
    encoded_data, encoded_labels = None, None
    
    # load data if data is available
    if data_file is not None:
        # load encoded data
        data_element_shape = (50, 20, 1)
        encoded_data_shape = get_memmap_shape(data_file, data_element_shape)
        encoded_data = np.memmap(data_file, dtype='float32', mode='r', shape=encoded_data_shape)

    if label_file is not None:
        # load labels (1D array)
        label_element_shape = (1,)  # labels are typically scalar per row
        label_shape = get_memmap_shape(label_file, label_element_shape)
        encoded_labels = np.memmap(label_file, dtype='float32', mode='r', shape=label_shape)
    
    return encoded_data, encoded_labels

# * SORTING ---

def simple_sort_key(path):
    if "L1" in path and "L1L2" not in path:
        return 0  # L1
    elif "L1L2" in path:
        return 1  # L1L2
    elif "L2" in path:
        return 2  # L2
    else:
        return 3

# * CREATING DIRECTORY ---

# create directories for saving models and plots
def make_files(base_dir, sub_dirs):
    os.makedirs(base_dir, exist_ok=True)
    
    for sub_dir in sub_dirs:
        os.makedirs(os.path.join(base_dir, sub_dir), exist_ok=True)
        
# * COMPUTING METRICS ---

def compute_metrics(val_labels, model=None, val_data=None, predictions=None):
    # initialise metrics dictionary
    metrics_dict = {}
    
    # if predictions are not given compute them using the model and data.
    if predictions is None:
        # if model and val_data are not given, raise an error
        if model is None or val_data is None:
            raise ValueError("!!! Either predictions must be provided, or both model and val_data must be given !!!")
        
        predictions = model.predict(val_data)
        
    # apply threshold to for binary classification - 0.5
    pred_labels = (predictions > 0.5).astype(int)
    true_labels = val_labels.astype(int)
    
    bce = BinaryCrossentropy()
        
    acc = accuracy_score(true_labels, pred_labels)
    loss = bce(true_labels, predictions).numpy()
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    prec = precision_score(true_labels, pred_labels, zero_division=0)
    rec = recall_score(true_labels, pred_labels, zero_division=0)
        
    metrics_dict = {'accuracy': acc, 
                    'loss': loss, 
                    'f1': f1, 
                    'precision': prec, 
                    'recall': rec}
        
    print(f"Validation Metrics | Accuracy: {acc:.3f}, Loss: {loss:.3f}, F1 Score: {f1:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}")
        
    return metrics_dict

# * CALCULATE AVG AND STD ---

def calculate_avg_std(cv_accuracies, cv_losses, cv_f1s, cv_precisions, cv_recalls, dropout_rate, regularizer_type, results_file_path, config_results, reg_factor=None):
    # avergae CV metrics for this configuration
    avg_acc = np.mean(cv_accuracies) if cv_accuracies else 0
    avg_loss = np.mean(cv_losses) if cv_losses else 0
    avg_f1 = np.mean(cv_f1s) if cv_f1s else 0
    avg_prec = np.mean(cv_precisions) if cv_precisions else 0
    avg_rec = np.mean(cv_recalls) if cv_recalls else 0
    
    # standard deviation of CV metrics for this configuration
    std_acc = np.std(cv_accuracies) if cv_accuracies else 0
    std_loss = np.std(cv_losses) if cv_losses else 0
    std_f1 = np.std(cv_f1s) if cv_f1s else 0
    std_prec = np.std(cv_precisions) if cv_precisions else 0
    std_rec = np.std(cv_recalls) if cv_recalls else 0
    
    print(f"\nConfig (dropout_rate={dropout_rate}) - CV Metrics | Accuracy: {avg_acc:.3f}, F1 Score: {avg_f1:.3f}, Precision: {avg_prec:.3f}, Recall: {avg_rec:.3f}\n")
    with open(results_file_path, 'a') as results_file:
        results_file.write(f"\nAverage CV Metrics | Avg_Accuracy: {avg_acc:.3f}, Avg_Loss: {avg_loss:.3f}, Avg_F1 Score: {avg_f1:.3f}, Avg_Precision: {avg_prec:.3f}, Avg_Recall: {avg_rec:.3f}")
        results_file.write(f"\nStd Dev CV Metrics | Std_Accuracy: {std_acc:.3f}, Std_Loss: {std_loss:.3f}, Std_F1 Score: {std_f1:.3f}, Std_Precision: {std_prec:.3f}, Std_Recall: {std_rec:.3f}\n\n")
        results_file.write("=" * 100 + "\n")
    
    # Store configuration and its metrics
    config_results.append({
        'regularizer_type': regularizer_type,
        'reg_factor': reg_factor,
        'dropout_rate': dropout_rate,
        'avg_accuracy': avg_acc,
        'avg_loss': avg_loss,
        'avg_f1': avg_f1,
        'avg_precision': avg_prec,
        'avg_recall': avg_rec,
        'std_accuracy': std_acc,
        'std_loss': std_loss,
        'std_f1': std_f1,
        'std_precision': std_prec,
        'std_recall': std_rec
    })
    
# * TRAINING FUNCTION ---

def train_model(model, epochs, batch_size, training_data, training_labels, model_type, dataset_name, regularizer_type, dropout_rate, save_dir, args, reg_factor=None, val_data=None, val_labels=None):
    # initialise metrics dictionary
    metrics_dict = {}
    # start training timer
    start_training_timer = time.time()
    
    # train the model with or without validation data
    if val_data is not None and val_labels is not None:
        history = model.fit(training_data, training_labels,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(val_data, val_labels),
                            verbose=1)
        
        # compute metrics for cross-validation
        metrics_dict = compute_metrics(val_labels=val_labels, model=model, val_data=val_data)
        
    else:
        history = model.fit(training_data, training_labels,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_split=0.1,
                            verbose=1)

    # end training timer
    elapsed_training_timer = time.time() - start_training_timer
    
    # print the main time taken
    if reg_factor is not None:
        print(f"\nTime taken for training with regularizer: {regularizer_type}, reg_factor: {reg_factor}, dropout_rate: {dropout_rate} | {(elapsed_training_timer):.3f} s")
    else:
        print(f"\nTime taken for training with regularizer: {regularizer_type}, dropout_rate: {dropout_rate} | {(elapsed_training_timer):.3f} s")
    
    # ensure plot_flag is valid
    if args.plot_plots.lower() == "true" and (val_data is None or val_labels is None):
        plot_training(save_dir, history, model_type, dataset_name, regularizer_type, dropout_rate, plot_count=1, reg_factor=reg_factor)
    elif args.plot_plots.lower() not in ["true", "false"]:
        raise ValueError("!!! Invalid input for -plots. Only 'true' or 'false' are allowed !!!")
    
    # return model and history
    return model, history, elapsed_training_timer, metrics_dict

# * SUBSETTING TRAINING DATASETS ---

def subset_data(encoded_data, labels, subset_size=25000):
    # check if the subset size is larger than the available data
    if subset_size > len(encoded_data):
        raise ValueError(f"!!! Subset size {subset_size} is larger than the available data size {len(encoded_data)} !!!")
    
    # randomly select indices for the subset
    subset_indices = np.random.choice(len(encoded_data), subset_size, replace=False)
    
    # create the subset using the selected indices
    encoded_training_data = encoded_data[subset_indices]
    training_labels = labels[subset_indices]
    
    return encoded_training_data, training_labels

# * SAVING MODEL ---

def save_model(model, save_dir, model_type, regularizer_type, dataset_name, dropout_rate, reg_factor=None):
    # save the model
    print("\n----- <Saving Model> -----")
    # construct the full file path
    if reg_factor is not None:
        model_path = os.path.join(save_dir, f"{model_type}_{regularizer_type}_{dataset_name}_dr{dropout_rate}_rf{reg_factor}.keras")
    else:
        model_path = os.path.join(save_dir, f"{model_type}_{regularizer_type}_{dataset_name}_dr{dropout_rate}.keras")
        
    model.save(model_path)
    print("----- <Model Saved Successfully> -----\n\n")
    
# * CLEANUP ---

def cleanup():
    # invoke garbage collection
    gc.collect()
    # clear session backend 
    tf.keras.backend.clear_session()