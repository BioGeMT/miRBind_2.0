import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

# function to build the DeepGRU model with Attention layer and regularization
def DeepGRU(input_shape, dropout_rate, learning_rate, reg_factor=None, regularizer_type=None):
    # clear any previous models
    tf.keras.backend.clear_session()
    
    # get the regularizer if it is not None
    reg = regularizer_type(reg_factor) if (reg_factor is not None and regularizer_type is not None) else None
    
    # define input layer
    input_layer = layers.Input(shape=input_shape)
    
    # Trainable weights for nucleotide pairs
    pair_embeddings = layers.Embedding(input_dim=17, output_dim=16)(input_layer)
    pair_embeddings = layers.Reshape((input_shape[0], input_shape[1] * 16))(pair_embeddings)

    # first GRU layer
    gru1 = layers.GRU(units=64, 
                      return_sequences=True, 
                      kernel_regularizer=reg,
                      recurrent_regularizer=reg)(pair_embeddings)
    dropout1 = layers.Dropout(dropout_rate)(gru1)


    # second GRU layer
    gru2 = layers.GRU(units=64, 
                      return_sequences=True, 
                      kernel_regularizer=reg, 
                      recurrent_regularizer=reg)(dropout1)
    dropout2 = layers.Dropout(dropout_rate)(gru2)

    # attention layer
    attention = layers.Attention()([dropout2, dropout2])
    
    pooled = layers.GlobalAveragePooling1D()(attention)

    # dense layer - fully connected hidden layer with 64 neurons
    dense = layers.Dense(units=64, activation='relu', kernel_regularizer=reg)(pooled)
    # batch normalization layer for stabilizing and accelerating the learning process
    batch_norm = layers.BatchNormalization()(dense)
    dropout3 = layers.Dropout(dropout_rate)(batch_norm)
    
    # output layer for binary classification - 1 neuron (1 or 0)
    output = layers.Dense(units=1, activation='sigmoid')(dropout3)

    # build model
    model = models.Model(inputs=input_layer, outputs=output)
    # compile model with Adam optimizer and binary crossentropy loss and accuracy metric
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy',metrics=['accuracy'])
    # Summary
    model.summary()
    
    print()
    
    return model