# imports
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, LSTM, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences



# Parameters
file_path = 'LSTM/AGO2_CLASH_Hejret2023.tsv'     # Path to the dataset
max_sequence_length = 100                               # Define the sequence length (how many time steps to consider)
validation_split = 0.2                                  # 80% training, 20% validation
vocab_size = 5                                          # A=1, C=2, G=3, T=4, padding
embedding_dim = 128                                     # Dimension of the dense embedding
lstm_units = 128                                        # Number of LSTM units in the layer
learning_rate = 0.001                                   # Learning rate for the optimizer
num_epochs = 20                                         # Number of epochs/iterations
batch_size = 32                                         # Batch size for training



# Load the dataset
df = pd.read_csv(file_path, sep='\t')
print("\nDataset loaded successfully.")
print(f"Dataset shape: {df.shape}")
print(f"First few rows of the dataset:\n{df.head()}\n")

# Extract ncRNA and gene sequences and the target labels
noncodingRNA = df['noncodingRNA'].values  # Assumed nucleotide sequences
genes = df['gene'].values  # Assumed nucleotide sequences
labels = df['label'].values  # Binary labels (binding or non-binding)

print("Extracted ncRNA, gene sequences, and labels.")
print(f"ncRNA example: {noncodingRNA[0]}")
print(f"Gene example: {genes[0]}")
print(f"Label example: {labels[0]}\n")



# Function to convert sequences to numbers (A=1, C=2, G=3, T=4)
nucleotide_map = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
def sequence_to_numeric(seq):
    return [nucleotide_map[n] for n in seq]

# Convert the sequences into numeric form
nonencodingRNA = [sequence_to_numeric(seq) for seq in noncodingRNA]
gene = [sequence_to_numeric(seq) for seq in genes]
print(f"ncRNA sequences converted to numeric: {nonencodingRNA[0]}")
print(f"Gene sequences converted to numeric: {gene[0]}\n")

# Combine ncRNA and gene sequences
combined_data = [ncRNA + gene for ncRNA, gene in zip(nonencodingRNA, gene)]
print(f"First combined sequence: {combined_data[0]}\n")

# Pad the sequences to ensure uniform length
padded_data = pad_sequences(combined_data, maxlen=max_sequence_length, padding='post')
print(f"Padded sequences shape: {padded_data.shape}")
print(f"Padded sequence example: {padded_data[0]}\n")

# Reshape X to have 3D shape (samples, time steps, features)
reshaped_data = np.expand_dims(padded_data, axis=-1)
print(f"Reshaped X shape (for LSTM): {reshaped_data.shape}\n")

# Convert X to float32 to match LSTM's expected input type
reshaped_data = reshaped_data.astype(np.float32)
print(f"Converted X shape to float32: {reshaped_data.shape}\n")

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(reshaped_data, labels, test_size=validation_split, random_state=42)
print(f"Training set shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Test labels shape: {y_test.shape}\n")


# Define the model architecture
model = Sequential()
# Add a Bidirectional LSTM layer
model.add(Bidirectional(LSTM(units=lstm_units, return_sequences=False, input_shape=(max_sequence_length, 1))))
# Add a Dense hidden layer
model.add(Dense(units=512, activation='relu'))
# Add a Dense output layer for binary classification (single neuron for binary output)
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model (binary crossentropy for binary classification)
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test, y_test))

# Model Summary
print(f"{model.summary()}\n")


# Save the model to a file
'''
When using .h5 format

UserWarning: You are saving your model as an HDF5 file via model.save(). This file format is considered legacy. 
We recommend using instead the native Keras format, e.g. model.save('my_model.keras').
saving_api.save_model
'''
model.save('LSTM/Bidirectional_LSTM.keras')
print("----- <Model Saved in Keras Format> -----\n")