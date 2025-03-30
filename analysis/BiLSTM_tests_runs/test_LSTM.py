# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Parameters
file_path = 'LSTM/AGO2_CLASH_Hejret2023.tsv'  # Path to the dataset
percentageOfDataForTraining = 80                     # 80% of the data for training
vocab_size = 4                                       # Number of unique nucleotides (A, C, G, T)
embedding_dim = 128                                  # Dimension of the dense embedding
max_sequence_length = 50                             # Maximum length of the input sequence
lstm_units = 128                                     # Number of LSTM units/nodes in the layer
num_classes = 2                                      # 1 or 0 - Number of classes for classification
learning_rate = 0.001                                # Learning rate for the optimizer
num_epochs = 20                                      # Number of epochs/iterations

# Load the dataset
df = pd.read_csv(file_path, sep='\t')

# Print DataFrame sample
print(f"\n{df.head()}\n")

# Extract columns of interest
noncodingRNA = df['noncodingRNA'].values    # ncRNA sequences
genes = df['gene'].values                   # Gene sequences
labels = df['label'].values                 # Labels for classification

# Function to encode sequences into integers
def encode_sequence(sequence):
    encoding = {'A': 0, 'C': 1, 'G': 2, 'T': 3}  # Nucleotide to integer mapping
    encoded = np.zeros(max_sequence_length, dtype=int)  # Pre-fill array with zeros
    for i, nucleotide in enumerate(sequence):
        if i < max_sequence_length:
            encoded[i] = encoding.get(nucleotide, -1)  # -1 for unknown chars
    return encoded

# Encode all ncRNA sequences
encoded_sequences = np.array([encode_sequence(seq) for seq in noncodingRNA])

# Encode all gene sequences
encoded_genes = np.array([encode_sequence(seq) for seq in genes])

# Check shapes of encoded data
print(f"\nEncoded Noncoding RNA Shape: {encoded_sequences.shape}\n")
print(f"\nEncoded Genes Shape: {encoded_genes.shape}\n")

# Combine ncRNA and gene features
combined_features = np.concatenate((encoded_sequences, encoded_genes), axis=1)

# Split data into train and validation sets
data_length = len(combined_features)
split_index = int(data_length * percentageOfDataForTraining / 100)  # 80% split

train_data = combined_features[:split_index]
train_labels = labels[:split_index]
validation_data = combined_features[split_index:]
validation_labels = labels[split_index:]

# Build Bidirectional LSTM model
input_dim = vocab_size * 2  # Adjust input size for combined data
model = Sequential()
model.add(Embedding(input_dim=input_dim, output_dim=embedding_dim, input_length=max_sequence_length * 2))  # Embedding layer
model.add(Bidirectional(LSTM(units=lstm_units, return_sequences=False)))  # Bidirectional LSTM
model.add(Dense(units=num_classes, activation='softmax'))  # Output layer

# Compile the model with optimizer and loss function
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])

# Display the model summary
print(f"\n{model.summary()}\n")

# Train the model
history = model.fit(train_data, train_labels, validation_data=(validation_data, validation_labels), epochs=num_epochs)

# Save the model to a file
'''
When using .h5 format

UserWarning: You are saving your model as an HDF5 file via model.save(). This file format is considered legacy. 
We recommend using instead the native Keras format, e.g. model.save('my_model.keras').
saving_api.save_model
'''
model.save('LSTM/Bidirectional_LSTM.keras')  # Save the model in Keras format