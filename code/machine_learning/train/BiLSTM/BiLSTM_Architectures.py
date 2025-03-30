import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

# function to build the BiLSTM model with Attention layer
def BiLSTM(input_shape, dropout_rate, learning_rate, reg_factor=None, regularizer_type=None):
    tf.keras.backend.clear_session()
    
    # get the regularizer if it is not None
    reg = regularizer_type(reg_factor) if (reg_factor is not None and regularizer_type is not None) else None
        
    # define input layer with given input shape
    input_layer = layers.Input(shape=input_shape)
    
    # trainable weights for nucleotide pairs
    pair_embeddings = layers.Embedding(input_dim=16 + 1, output_dim=1)(input_layer)
    pair_embeddings = layers.Reshape((25,50))(pair_embeddings)

    # first BiLSTM layer (128 units) for bidirectional sequence processing with regularization  
    bilstm1 = layers.Bidirectional(layers.LSTM(units=128, 
                                               return_sequences=True, 
                                               kernel_regularizer=reg, 
                                               recurrent_regularizer=reg))(pair_embeddings)
    dropout1 = layers.Dropout(dropout_rate)(bilstm1)

    # second BiLSTM layer (64 units) for further feature extraction with regularization
    bilstm2 = layers.Bidirectional(layers.LSTM(units=64, 
                                               return_sequences=True, 
                                               kernel_regularizer=reg, 
                                               recurrent_regularizer=reg))(dropout1)
    dropout2 = layers.Dropout(dropout_rate)(bilstm2)

    # attention layer (Self-attention over the BiLSTM output)
    attention = layers.Attention()([dropout2, dropout2])
    
    # global average pooling layer to reduce sequence dimensions
    pooled = layers.GlobalAveragePooling1D()(attention)

    # fully connected dense layer (512 neurons, ReLU) with batch normalization
    dense = layers.Dense(units=512, activation='relu', kernel_regularizer=reg)(pooled)
    # batch normalization for stable and faster learning
    batch_norm = layers.BatchNormalization()(dense)
    dropout3 = layers.Dropout(dropout_rate)(batch_norm)

    # output layer for binary classification)
    output = layers.Dense(units=1, activation='sigmoid')(dropout3)

    # build model
    model = models.Model(inputs=input_layer, outputs=output)
    # compile model with Adam optimizer and binary crossentropy loss and accuracy metric
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy',metrics=['accuracy'])
    # Summary
    model.summary()
    
    print()
    
    return model