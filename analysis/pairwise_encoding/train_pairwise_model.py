import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

from models import get_model, pairwise_to_onehot
from dataset import MiRNADataset, MiRNAOneHotDataset


def train_epoch(model, train_loader, optimizer, criterion, device, model_type):
    model.train()
    total_loss = 0
    all_train_preds = []
    all_train_labels = []
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X).squeeze()
        loss = criterion(outputs, batch_y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_train_preds.extend(outputs.detach().cpu().numpy())
        all_train_labels.extend(batch_y.cpu().numpy())
    
    precision, recall, _ = precision_recall_curve(all_train_labels, all_train_preds)
    train_auprc = auc(recall, precision)
    
    return total_loss / len(train_loader), train_auprc


def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    precision, recall, _ = precision_recall_curve(all_labels, all_preds)
    auprc = auc(recall, precision)
    return total_loss / len(val_loader), auprc


def plot_training_history(training_history, output_dir, timestamp):
    """
    Plot and save the training and validation metrics history.
    
    Parameters:
    training_history (dict): Dictionary containing training metrics history
    output_dir (str): Directory to save the plot
    timestamp (str): Timestamp to use in the filename
    """
    plt.figure(figsize=(16, 10))
    
    # Loss plots
    plt.subplot(2, 2, 1)
    plt.plot(training_history['train_loss'], label='Train Loss')
    plt.plot(training_history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(training_history['test1_loss'], label='Test1 Loss')
    plt.plot(training_history['test2_loss'], label='Test2 Loss')
    plt.title('Test Sets Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # AUPRC plots
    plt.subplot(2, 2, 3)
    plt.plot(training_history['train_auprc'], label='Train AUPRC')
    plt.plot(training_history['val_auprc'], label='Val AUPRC')
    plt.title('Training and Validation AUPRC')
    plt.xlabel('Epoch')
    plt.ylabel('AUPRC')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(training_history['test1_auprc'], label='Test1 AUPRC')
    plt.plot(training_history['test2_auprc'], label='Test2 AUPRC')
    plt.title('Test Sets AUPRC')
    plt.xlabel('Epoch')
    plt.ylabel('AUPRC')
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"training_plots_{timestamp}.png")
    plt.savefig(plot_path)
    return plot_path


def save_training_history(training_history, output_dir, timestamp):
    """
    Save the training history to a JSON file.
    
    Parameters:
    training_history (dict): Dictionary containing training metrics history
    output_dir (str): Directory to save the history
    timestamp (str): Timestamp to use in the filename
    
    Returns:
    str: Path to the saved history file
    """
    history_file = os.path.join(output_dir, f"training_history_{timestamp}.json")
    with open(history_file, 'w') as f:
        json.dump(training_history, f, indent=2)
    return history_file


def main():
    parser = argparse.ArgumentParser(description='Train miRNA pairwise binding site prediction model')
    
    # Training parameters
    parser.add_argument('--train_file', type=str, required=True, help='Path to training data file')
    parser.add_argument('--test_file1', type=str, 
                        default="../../data/chimeric_datasets/Manakov2022_flat/AGO2_eCLIP_Manakov2022_test.tsv", 
                        help='Path to first test data file')
    parser.add_argument('--test_file2', type=str, 
                        default="../../data/chimeric_datasets/Manakov2022_flat/AGO2_eCLIP_Manakov2022_leftout.tsv", 
                        help='Path to second test data file')
    parser.add_argument('--target_length', type=int, default=50, help='Length of target sequence')
    parser.add_argument('--mirna_length', type=int, default=25, help='Length of miRNA sequence')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=30, help='Maximum number of epochs to train')
    parser.add_argument('--val_fraction', type=float, default=0.1, help='Fraction of data to use for validation')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--embedding_dim', type=int, default=8, help='Embedding dimension')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--n_conv_layers', type=int, default=3, help='Number of convolutional layers')
    parser.add_argument('--filter_sizes', type=str, default='128,64,32', help='Comma-separated list of filter sizes')
    parser.add_argument('--kernel_sizes', type=str, default='6,3,3', help='Comma-separated list of kernel sizes')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--output_dir', type=str, default='model_outputs', help='Directory to save model outputs')
    parser.add_argument('--model', type=str, default='pairwise', help='Model architecture')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define nucleotide pairs and mapping for pairwise model
    nucleotide_pairs = [
        ('A', 'A'), ('A', 'T'), ('A', 'C'), ('A', 'G'),
        ('T', 'A'), ('T', 'T'), ('T', 'C'), ('T', 'G'),
        ('C', 'A'), ('C', 'T'), ('C', 'C'), ('C', 'G'),
        ('G', 'A'), ('G', 'T'), ('G', 'C'), ('G', 'G')
    ]
    pair_to_index = {pair: i for i, pair in enumerate(nucleotide_pairs)}
    
    # Add a padding token
    pair_to_index[('N', 'N')] = len(pair_to_index)

    if (args.model == "pairwise"):
        Dataset_obj = MiRNADataset
    elif (args.model == "pairwise_onehot"):
        Dataset_obj = MiRNAOneHotDataset
    else:
        raise ValueError("Invalid input of MODEL parameter")
        
    # Initialize dataset
    full_dataset = Dataset_obj(
        args.train_file,
        args.target_length,
        args.mirna_length,
        pair_to_index,
        num_pairs=len(pair_to_index)
    )
    
    # Create train/val split
    train_dataset, val_dataset = Dataset_obj.create_train_validation_split(
        full_dataset, validation_fraction=args.val_fraction
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Parse filter sizes and kernel sizes from strings to lists
    filter_sizes = [int(x) for x in args.filter_sizes.split(',')]
    kernel_sizes = [int(x) for x in args.kernel_sizes.split(',')]
    
    # Validate that the number of filters and kernels matches n_conv_layers
    if len(filter_sizes) != args.n_conv_layers or len(kernel_sizes) != args.n_conv_layers:
        print(f"Warning: Number of filter sizes ({len(filter_sizes)}) or kernel sizes ({len(kernel_sizes)}) "
              f"doesn't match the number of convolutional layers ({args.n_conv_layers})")
        print("Adjusting lists to match the specified number of layers...")
        
        # Adjust filter_sizes if needed
        if len(filter_sizes) < args.n_conv_layers:
            # Extend with last value repeated
            filter_sizes.extend([filter_sizes[-1]] * (args.n_conv_layers - len(filter_sizes)))
        elif len(filter_sizes) > args.n_conv_layers:
            # Truncate
            filter_sizes = filter_sizes[:args.n_conv_layers]
            
        # Adjust kernel_sizes if needed
        if len(kernel_sizes) < args.n_conv_layers:
            # Extend with last value repeated
            kernel_sizes.extend([kernel_sizes[-1]] * (args.n_conv_layers - len(kernel_sizes)))
        elif len(kernel_sizes) > args.n_conv_layers:
            # Truncate
            kernel_sizes = kernel_sizes[:args.n_conv_layers]
            
        print(f"Adjusted filter sizes: {filter_sizes}")
        print(f"Adjusted kernel sizes: {kernel_sizes}")
    
    # Model architecture parameters
    model_params = {
        'num_pairs': len(pair_to_index),
        'mirna_length': args.mirna_length,
        'target_length': args.target_length,
        'embedding_dim': args.embedding_dim,
        'dropout_rate': args.dropout_rate,
        # 'n_conv_layers': args.n_conv_layers,
        'filter_sizes': filter_sizes,
        'kernel_sizes': kernel_sizes,
    }
    
    # Initialize model
    if (args.model == "pairwise"):
        model = get_model("pairwise", **model_params).to(device)
    elif (args.model == "pairwise_onehot"):
        model = get_model("pairwise_onehot", **model_params).to(device)
    else:
        raise ValueError("Invalid input of MODEL parameter")
    
    # Log model architecture
    print("\nModel Architecture:")
    print(model)
    
    # Create architecture summary as a string to save in the metadata
    model_summary = str(model)
    architecture_summary = {
        'model_type': "pairwise",
        'num_pairs': len(pair_to_index),
        'mirna_length': args.mirna_length,
        'target_length': args.target_length,
        'embedding_dim': args.embedding_dim,
        'dropout_rate': args.dropout_rate,
        'filter_sizes': filter_sizes,
        'kernel_sizes': kernel_sizes,
        'total_params': sum(p.numel() for p in model.parameters()),
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'model_summary': model_summary,
    }
    
    print(f"\nArchitecture Summary:")
    print(f"  Model type: {architecture_summary['model_type']}")
    print(f"  Number of nucleotide pairs: {architecture_summary['num_pairs']}")
    print(f"  miRNA length: {architecture_summary['mirna_length']}")
    print(f"  Target length: {architecture_summary['target_length']}")
    print(f"  Embedding dimension: {architecture_summary['embedding_dim']}")
    print(f"  Dropout rate: {architecture_summary['dropout_rate']}")
    print(f"  Filter sizes: {architecture_summary['filter_sizes']}")
    print(f"  Kernel sizes: {architecture_summary['kernel_sizes']}")
    print(f"  Total parameters: {architecture_summary['total_params']:,}")
    print(f"  Trainable parameters: {architecture_summary['trainable_params']:,}")
    
    # Initialize optimizer and loss function
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCELoss()
    
    # Load test datasets
    test_dataset1 = Dataset_obj(
        args.test_file1,
        args.target_length,
        args.mirna_length,
        pair_to_index,
        num_pairs=len(pair_to_index)
    )
    
    test_dataset2 = Dataset_obj(
        args.test_file2,
        args.target_length,
        args.mirna_length,
        pair_to_index,
        num_pairs=len(pair_to_index)
    )
    
    print(f"Test dataset 1 size: {len(test_dataset1)}")
    print(f"Test dataset 2 size: {len(test_dataset2)}")
    
    # Create test data loaders
    test_loader1 = DataLoader(test_dataset1, batch_size=args.batch_size)
    test_loader2 = DataLoader(test_dataset2, batch_size=args.batch_size)
    
    # Training loop with early stopping
    best_val_auprc = 0
    patience_counter = 0
    training_history = {
        'train_loss': [],
        'train_auprc': [],
        'val_loss': [],
        'val_auprc': [],
        'test1_loss': [],
        'test1_auprc': [],
        'test2_loss': [],
        'test2_auprc': []
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(args.output_dir, f"{args.model}_model_{timestamp}.pt")
    
    # Save architecture summary
    architecture_file = os.path.join(args.output_dir, f"architecture_summary_{timestamp}.json")
    with open(architecture_file, 'w') as f:
        json.dump(architecture_summary, f, indent=2)
    print(f"Architecture summary saved at: {architecture_file}")
    
    print("Starting training...")
    for epoch in range(args.num_epochs):
        train_loss, train_auprc = train_epoch(model, train_loader, optimizer, criterion, device, args.model)
        val_loss, val_auprc = evaluate(model, val_loader, criterion, device)
        
        # Evaluate on test sets
        test1_loss, test1_auprc = evaluate(model, test_loader1, criterion, device)
        test2_loss, test2_auprc = evaluate(model, test_loader2, criterion, device)
        
        # Save training history
        training_history['train_loss'].append(train_loss)
        training_history['train_auprc'].append(train_auprc)
        training_history['val_loss'].append(val_loss)
        training_history['val_auprc'].append(val_auprc)
        training_history['test1_loss'].append(test1_loss)
        training_history['test1_auprc'].append(test1_auprc)
        training_history['test2_loss'].append(test2_loss)
        training_history['test2_auprc'].append(test2_auprc)
        
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, AUPRC: {train_auprc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, AUPRC: {val_auprc:.4f}")
        print(f"  Test1 - Loss: {test1_loss:.4f}, AUPRC: {test1_auprc:.4f}")
        print(f"  Test2 - Loss: {test2_loss:.4f}, AUPRC: {test2_auprc:.4f}")
        
        # Check for improvement
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auprc': val_auprc,
                'model_params': model_params,
                'architecture_summary': architecture_summary,
                'model_summary': model_summary
            }, model_save_path)
            print(f"Model saved at {model_save_path}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Save training history and plot
    history_file = save_training_history(training_history, args.output_dir, timestamp)
    plot_path = plot_training_history(training_history, args.output_dir, timestamp)
    
    # Final evaluation on all sets using the best model
    # Load the best model first
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    val_loss, val_auprc = evaluate(model, val_loader, criterion, device)
    test1_loss, test1_auprc = evaluate(model, test_loader1, criterion, device)
    test2_loss, test2_auprc = evaluate(model, test_loader2, criterion, device)
    
    print(f"\nFinal evaluation with best model (epoch {checkpoint['epoch']+1}):")
    print(f"Validation - Loss: {val_loss:.4f}, AUPRC: {val_auprc:.4f}")
    print(f"Test1     - Loss: {test1_loss:.4f}, AUPRC: {test1_auprc:.4f}")
    print(f"Test2     - Loss: {test2_loss:.4f}, AUPRC: {test2_auprc:.4f}")
    
    # Add final results to model metadata
    final_results = {
        'best_epoch': checkpoint['epoch'] + 1,
        'best_val_auprc': best_val_auprc,
        'final_test1_auprc': test1_auprc,
        'final_test2_auprc': test2_auprc,
        'architecture_summary': architecture_summary
    }
    
    # Save updated checkpoint with test results
    checkpoint['test1_auprc'] = test1_auprc
    checkpoint['test2_auprc'] = test2_auprc
    torch.save(checkpoint, model_save_path)
    
    # Save final results separately
    results_file = os.path.join(args.output_dir, f"results_summary_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nTraining complete. Best validation AUPRC: {best_val_auprc:.4f}")
    print(f"Model saved at: {model_save_path}")
    print(f"Training history saved at: {history_file}")
    print(f"Architecture summary saved at: {architecture_file}")
    print(f"Results summary saved at: {results_file}")
    print(f"Training plots saved at: {plot_path}")

if __name__ == "__main__":
    main()