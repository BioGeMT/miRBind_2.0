import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

# function to build the DeepRNN model with Attention layer and regularization
def DeepRNN(input_shape, dropout_rate, learning_rate, reg_factor=None, regularizer_type=None):
    # clear any previous models
    tf.keras.backend.clear_session()
    
    # get the regularizer if it is not None
    reg = regularizer_type(reg_factor) if (reg_factor is not None and regularizer_type is not None) else None
    
    # define input layer
    input_layer = layers.Input(shape=input_shape)
    
    # Trainable weights for nucleotide pairs
    pair_embeddings = layers.Embedding(input_dim=16 + 1, output_dim=1)(input_layer)
    pair_embeddings = layers.Reshape((25,50))(pair_embeddings)

    # first RNN layer
    rnn1 = layers.SimpleRNN(units=128, 
                            return_sequences=True, 
                            kernel_regularizer=reg, 
                            recurrent_regularizer=reg)(pair_embeddings)
    dropout1 = layers.Dropout(dropout_rate)(rnn1)

    # second RNN layer
    rnn2 = layers.SimpleRNN(units=128, 
                            return_sequences=True, 
                            kernel_regularizer=reg, 
                            recurrent_regularizer=reg)(dropout1)
    dropout2 = layers.Dropout(dropout_rate)(rnn2)
    
    # third RNN layer
    rnn3 = layers.SimpleRNN(units=64, 
                            return_sequences=True, 
                            kernel_regularizer=reg, 
                            recurrent_regularizer=reg)(dropout2)
    dropout3 = layers.Dropout(dropout_rate)(rnn3)

    # attention layer
    attention = layers.Attention()([dropout3, dropout3])
    
    pooled = layers.GlobalAveragePooling1D()(attention)

    # dense layer - fully connected hidden layer with 512 neurons
    dense = layers.Dense(units=512, activation='relu', kernel_regularizer=reg)(pooled)
    # batch normalization layer for stabilizing and accelerating the learning process
    batch_norm = layers.BatchNormalization()(dense)
    dropout4 = layers.Dropout(dropout_rate)(batch_norm)

    # output layer for binary classification - 1 neuron (1 or 0)
    output = layers.Dense(units=1, activation='sigmoid')(dropout4)

    # build model
    model = models.Model(inputs=input_layer, outputs=output)
    # compile model with Adam optimizer and binary crossentropy loss and accuracy metric
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy',metrics=['accuracy'])
    # Summary
    model.summary()
    
    print()
    
    return model