# imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  # Import train_test_split for splitting the data
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam



# Parameters
file_path = 'LSTM/AGO2_CLASH_Hejret2023.tsv'  # Path to the dataset
sequence_length = 128       # Define the sequence length (how many time steps to consider)
sequences = []
target = []
validation_split = 0.2      # Define the validation split - 80% will be used for training, 20% for validation
vocab_size = 4              # Number of unique nucleotides (A, C, G, T)
embedding_dim = 128         # Dimension of the dense embedding
lstm_units = 128            # Number of LSTM units/nodes in the layer
num_classes = 2             # 1 or 0 - Number of classes for classification
learning_rate = 0.001       # Learning rate for the optimizer
num_epochs = 20             # Number of epochs/iterations
batch_size = 32             # Batch size for training

# Load the dataset and convert it to a numpy array
df = np.array(pd.read_csv(file_path, sep='\t'))

# Print DataFrame sample
print(f"\n{df[:5]}\n")



# Shaping the Data
'''
https://medium.com/@rebeen.jaff/what-is-lstm-introduction-to-long-short-term-memory-66bd3855b9ce
LSTM input - (samples, time steps, features)
'''

# Create sequences
for i in range(len(df) - sequence_length):
    # Append sequence of length sequence_length
    sequences.append(df[i:i + sequence_length])
    # Append the target value
    target.append(df[i + sequence_length])

# Reshape sequences and target to fit LSTM input format
X = np.array(sequences)
y = np.array(target)

# print the shapes
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}\n")

# Perform an 80/20 train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42)

# Print the shapes of the training and validation sets
print(f"Training set X_train shape: {X_train.shape}")
print(f"Training set y_train shape: {y_train.shape}")
print(f"Validation set X_val shape: {X_val.shape}")
print(f"Validation set y_val shape: {y_val.shape}")



# Defining the Model Architecture
model = Sequential()
# Add an embedding layer to convert input sequences to dense vectors
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=sequence_length))
# Add a Bidirectional LSTM layer
model.add(Bidirectional(LSTM(units=lstm_units, return_sequences=True)))
# Add a dense output layer
model.add(Dense(units=num_classes, activation='sigmoid'))
# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print the model summary
print(f"\n{model.summary()}\n")

# Compile the model
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_val, y_val))



# Save the model to a file
'''
When using .h5 format

UserWarning: You are saving your model as an HDF5 file via model.save(). This file format is considered legacy. 
We recommend using instead the native Keras format, e.g. model.save('my_model.keras').
saving_api.save_model
'''
model.save('LSTM/Bidirectional_LSTM.keras')  # Save the model in Keras format