import mirbench
import pandas as pd

# Define dataset details
dataset_name = "AGO2_eCLIP_Manakov2022"

# Load training data with ratio "1"
train_split = "train"
train_ratio = "1"
train_df = miRBench.dataset.get_dataset_df(dataset_name, split=train_split, ratio=train_ratio)
print("Training Data Sample:")
print(train_df.head())

# Save training data to TSV
train_df.to_csv('training_data.tsv', sep='\t', index=False)

# Load test data with ratio "1"
test_split = "test"
test_ratio = "1"
test_df = miRBench.dataset.get_dataset_df(dataset_name, split=test_split, ratio=test_ratio)
print("\nTest Data Sample:")
print(test_df.head())

# Save test data to TSV
test_df.to_csv('inference_data.tsv', sep='\t', index=False)
