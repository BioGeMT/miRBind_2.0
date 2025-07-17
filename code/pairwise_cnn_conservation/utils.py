import os
import datetime
import torch
import io
import json
import numpy as np
import pandas as pd


class LogFileGenerator:
    def __init__(self, log_dir):
        self._timestamp = None
        self._log_dir = log_dir
        
        if not os.path.exists(self._log_dir):
            os.makedirs(self._log_dir)
            print(f"Created directory: {self._log_dir}")
    
    def _get_timestamp(self):
        if self._timestamp is None:
            self._timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return self._timestamp
    
    def get_train_log_filename(self):
        filename = f"{self._get_timestamp()}_training_log.tsv"
        return os.path.join(self._log_dir, filename)
    
    def get_hyperparameters_log_filename(self):
        filename = f"{self._get_timestamp()}_hyperparameters_log.json"
        return os.path.join(self._log_dir, filename)

    def log_learned_pair_values(self, model, nucleotide_pairs):
        log_file = os.path.join(self._log_dir, f"{self._get_timestamp()}_learned_pair_values_log.tsv")

        # Get the embedding weights and reshape them properly
        embedding_weights = model.pair_embeddings.weight.detach().cpu().numpy()
        embedding_dim = embedding_weights.shape[1]

        # Create the pairs list including the padding pair
        all_pairs = nucleotide_pairs + [('N', 'N')]

        # If embedding_dim is 1, flatten the weights
        if embedding_dim == 1:
            values = embedding_weights.flatten()
        else:
            # For multi-dimensional embeddings, we might want to keep all dimensions
            values = [weights for weights in embedding_weights]

        # Make sure we have the same number of pairs and values
        if len(all_pairs) != len(values):
            print(f"Warning: Mismatch in lengths - Pairs: {len(all_pairs)}, Values: {len(values)}")
            # Trim the longer one to match the shorter one
            min_len = min(len(all_pairs), len(values))
            all_pairs = all_pairs[:min_len]
            values = values[:min_len]

        # Create DataFrame
        pair_values = pd.DataFrame({
            "Pair": all_pairs,
            "Learned Value": values
        })

        # Save to file
        pair_values.to_csv(log_file, sep="\t", index=False, mode='a', header=False)
        print(f"Learned nucleotide pair values logged to {log_file}")
    
    def reset_timestamp(self):
        self._timestamp = None


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
    # Create the configuration dictionary
    config = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),  # Convert device to string
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        },
        "architecture": {
            "mirna_length": kwargs.get("mirna_length"),
            "target_length": kwargs.get("target_length"),
            "num_pairs": kwargs.get("num_pairs")
        },
        "hyperparameters": {},
        "model_summary": ""
    }
    
    # Add all hyperparameters, ensuring they're JSON serializable
    for key, value in kwargs.items():
        if key not in ["mirna_length", "target_length", "num_pairs"]:
            # Convert any non-serializable types to strings
            if isinstance(value, (torch.device, torch.dtype)):
                value = str(value)
            config["hyperparameters"][key] = value
    
    # Capture model summary
    summary_io = io.StringIO()
    print(model, file=summary_io)
    config["model_summary"] = summary_io.getvalue().strip()
    
    # Write to file
    with open(log_file, "a") as log:
        json.dump(config, log, indent=2)
        log.write("\n")  # Add newline for multiple entries
