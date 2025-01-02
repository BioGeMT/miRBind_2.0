import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from tqdm import tqdm
import gc
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
import os
import io
import datetime
import argparse

###################################
# Helper Functions
###################################
def get_log_dir():
    log_dir = "training_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"Created directory: {log_dir}")
    return log_dir

def get_log_filename(log_dir, auprc_test1=None):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if auprc_test1 is not None:
        filename = f"training_log_{timestamp}_auprc{auprc_test1:.3f}.tsv"
    else:
        filename = f"training_log_{timestamp}.tsv"
    return os.path.join(log_dir, filename)

def pad_or_trim(seq, desired_length):
    if len(seq) > desired_length:
        return seq[:desired_length]
    else:
        return seq + 'N' * (desired_length - len(seq))

def encode_complementarity(target_seq, mirna_seq, target_length, mirna_length, pair_to_index, num_pairs):
    arr = np.zeros((mirna_length, target_length), dtype=np.int32)
    for i in range(mirna_length):
        for j in range(target_length):
            if i < len(mirna_seq) and j < len(target_seq):
                if mirna_seq[i] == 'N' or target_seq[j] == 'N':
                    arr[i, j] = num_pairs
                else:
                    pair = (mirna_seq[i], target_seq[j])
                    arr[i, j] = pair_to_index.get(pair, num_pairs)
    return arr

def log_config(log_file, model, device, **kwargs):
    with open(log_file, "a") as log:
        log.write("\n=== Model Configuration ===\n")
        log.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"Device: {device}\n")
        if torch.cuda.is_available():
            log.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
        
        log.write("\nArchitecture:\n")
        log.write(f"miRNA length: {kwargs.get('mirna_length')}\n")
        log.write(f"Target length: {kwargs.get('target_length')}\n")
        log.write(f"Number of nucleotide pairs: {kwargs.get('num_pairs')}\n")
        
        log.write("\nHyperparameters:\n")
        for key, value in kwargs.items():
            log.write(f"{key}: {value}\n")
        
        log.write("\nModel Summary:\n")
        summary_io = io.StringIO()
        print(model, file=summary_io)
        log.write(summary_io.getvalue())
        log.write("\n" + "="*50 + "\n\n")

def log_learned_pair_values(model, log_dir, nucleotide_pairs):
    log_file = os.path.join(log_dir, "learned_pair_values_log.tsv")
    embedding_weights = model.pair_embeddings.weight.detach().cpu().numpy().flatten()
    pair_values = pd.DataFrame({
        "Pair": nucleotide_pairs + [('N', 'N')],
        "Learned Value": embedding_weights
    })
    pair_values.to_csv(log_file, sep="\t", index=False, mode='a', header=False)
    print(f"Learned nucleotide pair values logged to {log_file}")

###################################
# Dataset Class
###################################
class MiRNADataset(Dataset):
    def __init__(self, file_path, target_length, mirna_length, pair_to_index, num_pairs, fraction=1.0):
        df = pd.read_csv(file_path, sep="\t")
        if fraction < 1.0:
            df = df.sample(frac=fraction, random_state=42).reset_index(drop=True)
        
        self.target_seqs = df.iloc[:, 0].values
        self.mirna_seqs = df.iloc[:, 1].values
        self.labels = torch.FloatTensor(df['label'].values)
        self.target_length = target_length
        self.mirna_length = mirna_length
        self.pair_to_index = pair_to_index
        self.num_pairs = num_pairs
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        t_seq = pad_or_trim(self.target_seqs[idx], self.target_length)
        m_seq = pad_or_trim(self.mirna_seqs[idx], self.mirna_length)
        X = torch.tensor(encode_complementarity(t_seq, m_seq, self.target_length, 
                                             self.mirna_length, self.pair_to_index, 
                                             self.num_pairs), dtype=torch.long)
        return X, self.labels[idx]
        
    @staticmethod
    def create_train_validation_split(dataset, validation_fraction=0.1):
        val_size = int(len(dataset) * validation_fraction)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        return train_dataset, val_dataset

###################################
# Model Definition
###################################
class MiRNACNN(nn.Module):
    def __init__(self, num_pairs, mirna_length, target_length, embedding_dim=4):
        super(MiRNACNN, self).__init__()
        
        self.pair_embeddings = nn.Embedding(num_pairs + 1, embedding_dim)
        self.mirna_length = mirna_length
        self.target_length = target_length
        
        self.conv1 = nn.Conv2d(embedding_dim, 128, kernel_size=6, padding=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.2)
        
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.2)
        
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.dropout3 = nn.Dropout(0.2)
        
        self.flat_features = self._get_flat_features()
        
        self.fc1 = nn.Linear(self.flat_features, 30)
        self.bn4 = nn.BatchNorm1d(30)
        self.dropout4 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(30, 1)
        
    def _get_flat_features(self):
        x = torch.zeros(1, self.mirna_length, self.target_length)
        x = self.pair_embeddings(x.long())
        x = x.permute(0, 3, 1, 2)
        x = self.pool1(F.leaky_relu(self.bn1(self.conv1(x)), 0.1))
        x = self.pool2(F.leaky_relu(self.bn2(self.conv2(x)), 0.1))
        x = self.pool3(F.leaky_relu(self.bn3(self.conv3(x)), 0.1))
        return x.numel()
    
    def forward(self, x):
        x = self.pair_embeddings(x)
        x = x.permute(0, 3, 1, 2)
        
        x = self.dropout1(self.pool1(F.leaky_relu(self.bn1(self.conv1(x)), 0.1)))
        x = self.dropout2(self.pool2(F.leaky_relu(self.bn2(self.conv2(x)), 0.1)))
        x = self.dropout3(self.pool3(F.leaky_relu(self.bn3(self.conv3(x)), 0.1)))
        
        x = x.contiguous().view(x.size(0), -1)
        x = self.dropout4(F.leaky_relu(self.bn4(self.fc1(x)), 0.1))
        x = torch.sigmoid(self.fc2(x))
        
        return x

###################################
# Training Functions
###################################
def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_train_preds = []
    all_train_labels = []
    
    for batch_X, batch_y in tqdm(train_loader, desc="Training"):
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

def evaluate(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X).squeeze().cpu().numpy()
            all_preds.extend(outputs)
            all_labels.extend(batch_y.numpy())
    
    precision, recall, _ = precision_recall_curve(all_labels, all_preds)
    auprc = auc(recall, precision)
    return auprc

def train_model(args, model, train_dataset, test_dataset_1, test_dataset_2, device, log_dir, nucleotide_pairs):
    # Create train/val split
    train_dataset, val_dataset = MiRNADataset.create_train_validation_split(
        train_dataset, validation_fraction=args.val_fraction
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader_1 = DataLoader(test_dataset_1, batch_size=args.batch_size)
    test_loader_2 = DataLoader(test_dataset_2, batch_size=args.batch_size)
    
    print(f"Dataset sizes:")
    print(f"Training: {len(train_dataset)}")
    print(f"Validation: {len(val_dataset)}")
    print(f"Test 1: {len(test_dataset_1)}")
    print(f"Test 2: {len(test_dataset_2)}")
    
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCELoss()
    
    # Initialize log file
    log_file = get_log_filename(log_dir)
    
    # Log configuration
    config_params = {
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'validation_fraction': args.val_fraction,
        'learning_rate': args.learning_rate,
        'optimizer': optimizer.__class__.__name__,
        'embedding_dim': model.pair_embeddings.embedding_dim,
        'mirna_length': args.mirna_length,
        'target_length': args.target_length,
        'num_pairs': args.num_pairs
    }
    log_config(log_file, model, device, **config_params)
    
    # Training loop
    with open(log_file, "a") as log:
        log.write("Epoch\tTrain Loss\tTrain AUPRC\tVal Loss\tVal AUPRC\tAUPRC Test 1\tAUPRC Test 2\n")
        
        for epoch in range(args.num_epochs):
            # Train
            train_loss, train_auprc = train_epoch(model, train_loader, optimizer, criterion, device)
            
            # Evaluate on validation set
            model.eval()
            val_loss = 0
            all_val_preds = []
            all_val_labels = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X).squeeze()
                    val_loss += criterion(outputs, batch_y).item()
                    all_val_preds.extend(outputs.cpu().numpy())
                    all_val_labels.extend(batch_y.cpu().numpy())
            
            val_loss /= len(val_loader)
            precision, recall, _ = precision_recall_curve(all_val_labels, all_val_preds)
            val_auprc = auc(recall, precision)
            
            # Evaluate on test sets
            auprc_1 = evaluate(model, test_loader_1, device)
            auprc_2 = evaluate(model, test_loader_2, device)
            
            # Print progress
            print(f"\nEpoch {epoch+1}/{args.num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train AUPRC: {train_auprc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val AUPRC: {val_auprc:.4f}")
            print(f"AUPRC Test 1: {auprc_1:.4f}, AUPRC Test 2: {auprc_2:.4f}")
            
            # Log results
            log.write(f"{epoch+1}\t{train_loss:.4f}\t{train_auprc:.4f}\t{val_loss:.4f}\t{val_auprc:.4f}\t{auprc_1:.4f}\t{auprc_2:.4f}\n")
            log.flush()
            
            # Log learned nucleotide pair values
            log_learned_pair_values(model, log_dir, nucleotide_pairs)

            # Clear memory
            gc.collect()

    return model

def parse_args():
    parser = argparse.ArgumentParser(description='Train miRNA binding site prediction model')
    
    # Dataset parameters
    parser.add_argument('--train_file', type=str, default="AGO2_eCLIP_Manakov2022_train.tsv",
                        help='Path to training data file')
    parser.add_argument('--test_file_1', type=str, default="AGO2_eCLIP_Manakov2022_test.tsv",
                        help='Path to first test data file')
    parser.add_argument('--test_file_2', type=str, default="AGO2_eCLIP_Manakov2022_leftout.tsv",
                        help='Path to second test data file')
    
    # Model parameters
    parser.add_argument('--target_length', type=int, default=50,
                        help='Length of target sequence')
    parser.add_argument('--mirna_length', type=int, default=25,
                        help='Length of miRNA sequence')
    parser.add_argument('--embedding_dim', type=int, default=1,
                        help='Dimension of nucleotide pair embeddings')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--train_fraction', type=float, default=1,
                        help='Decrease the training data size for faster debugging or devices with less memory')
    parser.add_argument('--val_fraction', type=float, default=0.1,
                        help='Fraction of training data to use for validation')
    
    args = parser.parse_args()
    
    # Add derived parameters
    args.num_pairs = 16  # Number of possible nucleotide pairs
    args.alphabet = ['A', 'C', 'G', 'T', 'N']
    
    return args

def main():
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Define nucleotide pairs and mapping
    nucleotide_pairs = [
        ('A', 'A'), ('A', 'T'), ('A', 'C'), ('A', 'G'),
        ('T', 'A'), ('T', 'T'), ('T', 'C'), ('T', 'G'),
        ('C', 'A'), ('C', 'T'), ('C', 'C'), ('C', 'G'),
        ('G', 'A'), ('G', 'T'), ('G', 'C'), ('G', 'G')
    ]
    pair_to_index = {pair: i for i, pair in enumerate(nucleotide_pairs)}
    
    # Get log directory
    log_dir = get_log_dir()
    
    # Initialize datasets
    train_dataset = MiRNADataset(args.train_file, args.target_length, args.mirna_length, 
                                pair_to_index, args.num_pairs, args.train_fraction)
    test_dataset_1 = MiRNADataset(args.test_file_1, args.target_length, args.mirna_length, 
                                 pair_to_index, args.num_pairs)
    test_dataset_2 = MiRNADataset(args.test_file_2, args.target_length, args.mirna_length, 
                                 pair_to_index, args.num_pairs)
    
    # Initialize model
    model = MiRNACNN(args.num_pairs, args.mirna_length, args.target_length, 
                     args.embedding_dim).to(device)
    
    # Train model
    model = train_model(args, model, train_dataset, test_dataset_1, test_dataset_2, 
                       device, log_dir, nucleotide_pairs)
    
    print("\nModel training and evaluation complete.")

if __name__ == "__main__":
    main()