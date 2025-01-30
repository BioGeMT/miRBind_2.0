import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import gc
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
import os
import io
import datetime
import argparse
import optuna
from optuna.trial import TrialState

from utils import LogFileGenerator, pad_or_trim, encode_complementarity, log_config
from dataset import MiRNADataset

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

def train_epoch(model, train_loader, optimizer, criterion, device):
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

def evaluate(model, val_loader, device):
    model.eval()
    criterion = nn.BCELoss()
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

def objective(trial, train_dataset, device, args):
    # Suggest hyperparameters
    batch_size = trial.suggest_int('batch_size', 32, 64)
    # embedding_dim = trial.suggest_int('embedding_dim', 2, 16)
    embedding_dim = trial.suggest_categorical('embedding_dim', [2,4,8,16])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.3)
    
    # Create train/val split
    train_data, val_data = MiRNADataset.create_train_validation_split(
        train_dataset, validation_fraction=args.val_fraction
    )
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    
    # Initialize model with suggested hyperparameters
    model = MiRNACNN(
        num_pairs=args.num_pairs,
        mirna_length=args.mirna_length,
        target_length=args.target_length,
        embedding_dim=embedding_dim
    ).to(device)
    
    # Modify dropout rates
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = dropout_rate
    
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    
    best_val_auprc = 0
    patience = 5
    patience_counter = 0
    
    for epoch in range(args.num_epochs):
        train_loss, train_auprc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auprc = evaluate(model, val_loader, device)
        
        # Early stopping
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
            
        trial.report(val_auprc, epoch)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return best_val_auprc

def main():
    parser = argparse.ArgumentParser(description='Train miRNA binding site prediction model with Optuna')
    
    # Dataset parameters
    parser.add_argument('--train_file', type=str, default="AGO2_eCLIP_Manakov2022_train.tsv")
    parser.add_argument('--target_length', type=int, default=50)
    parser.add_argument('--mirna_length', type=int, default=25)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--val_fraction', type=float, default=0.1)
    parser.add_argument('--n_trials', type=int, default=100)
    parser.add_argument('--study_name', type=str, default='mirna_optimization')
    
    args = parser.parse_args()
    args.num_pairs = 16
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define nucleotide pairs and mapping
    nucleotide_pairs = [
        ('A', 'A'), ('A', 'T'), ('A', 'C'), ('A', 'G'),
        ('T', 'A'), ('T', 'T'), ('T', 'C'), ('T', 'G'),
        ('C', 'A'), ('C', 'T'), ('C', 'C'), ('C', 'G'),
        ('G', 'A'), ('G', 'T'), ('G', 'C'), ('G', 'G')
    ]
    pair_to_index = {pair: i for i, pair in enumerate(nucleotide_pairs)}
    
    # Initialize dataset
    train_dataset = MiRNADataset(
        args.train_file, 
        args.target_length, 
        args.mirna_length,
        pair_to_index, 
        args.num_pairs
    )
    
    # Create Optuna study
    study = optuna.create_study(
        study_name=args.study_name,
        direction='maximize',
        pruner=optuna.pruners.MedianPruner()
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, train_dataset, device, args),
        n_trials=args.n_trials
    )
    
    # Print results
    print("Best trial:")
    trial = study.best_trial
    
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Save study results
    study_results = {
        'best_value': trial.value,
        'best_params': trial.params,
        'all_trials': [{
            'number': t.number,
            'value': t.value,
            'params': t.params,
            'state': t.state
        } for t in study.trials]
    }
    
    import json
    with open(f"{args.study_name}_results.json", 'w') as f:
        json.dump(study_results, f, indent=2)
    
    # Plot optimization history
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_html(f"{args.study_name}_optimization_history.html")
    
    # Plot parameter importances
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_html(f"{args.study_name}_param_importances.html")

if __name__ == "__main__":
    main()