from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Dropout, Attention, GlobalAveragePooling1D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model


# function to build the BiLSTM model with Attention layer
def BiLSTM(input_shape, dropout_rate, learning_rate):
    # define input layer
    input_layer = Input(shape=input_shape)

    # first BiLSTM layer
    bilstm1 = Bidirectional(LSTM(units=128, return_sequences=True))(input_layer)
    dropout1 = Dropout(dropout_rate)(bilstm1)

    # second BiLSTM layer
    bilstm2 = Bidirectional(LSTM(units=64, return_sequences=True))(dropout1)
    dropout2 = Dropout(dropout_rate)(bilstm2)

    # attention layer
    attention = Attention()([dropout2, dropout2])
    
    pooled = GlobalAveragePooling1D()(attention)

    # dense layer - fully connected hidden layer with 512 neurons
    dense = Dense(units=512, activation='relu')(pooled)
    # batch normalization layer for stabilizing and accelerating the learning process
    batch_norm = BatchNormalization()(dense)
    dropout3 = Dropout(dropout_rate)(batch_norm)

    # output layer for binary classification - 1 neuron (1 or 0)
    output = Dense(units=1, activation='sigmoid')(dropout3)

    # build model
    model = Model(inputs=input_layer, outputs=output)
    # compile model with Adam optimizer and binary crossentropy loss and accuracy metric
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy',metrics=['accuracy'])
    # Summary
    model.summary()
    
    print()
    
    return model


# * ---


def BiLSTM(input_shape, dropout_rate, learning_rate, regularizer):
    # define input layer
    input_layer = Input(shape=input_shape)

    # first BiLSTM layer
    bilstm1 = Bidirectional(LSTM(units=128, return_sequences=True, kernel_regularizer=regularizer, recurrent_regularizer=regularizer))(input_layer)
    dropout1 = Dropout(dropout_rate)(bilstm1)

    # second BiLSTM layer
    bilstm2 = Bidirectional(LSTM(units=64, return_sequences=True, kernel_regularizer=regularizer, recurrent_regularizer=regularizer))(dropout1)
    dropout2 = Dropout(dropout_rate)(bilstm2)

    # attention layer
    attention = Attention()([dropout2, dropout2])
    
    pooled = GlobalAveragePooling1D()(attention)

    # dense layer - fully connected hidden layer with 512 neurons
    dense = Dense(units=512, activation='relu', kernel_regularizer=regularizer)(pooled)
    # batch normalization layer for stabilizing and accelerating the learning process
    batch_norm = BatchNormalization()(dense)
    dropout3 = Dropout(dropout_rate)(batch_norm)

    # output layer for binary classification - 1 neuron (1 or 0)
    output = Dense(units=1, activation='sigmoid')(dropout3)

    # build model
    model = Model(inputs=input_layer, outputs=output)
    # compile model with Adam optimizer and binary crossentropy loss and accuracy metric
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy',metrics=['accuracy'])
    # Summary
    model.summary()
    
    print()
    
    return model