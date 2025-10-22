import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense


class miRBind_CNN():
    """
    Build model architecture based on the CNN model presented in miRBind paper (2022) https://doi.org/10.3390/genes13122323
    The default parameters are same as the ones used in the paper
    """
    def __init__(self, cnn_num = 6, kernel_size = 5, pool_size = 2, dropout_rate = 0.3, dense_num = 2):

        x = Input(shape=(50,20,1), dtype='float32')
        main_input = x

        for cnn_i in range(cnn_num):
            x = Conv2D(
                filters=32 * (cnn_i + 1),
                kernel_size=(kernel_size, kernel_size),
                padding="same",
                data_format="channels_last")(x)
            x = LeakyReLU()(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D(pool_size=(pool_size, pool_size), padding='same')(x)
            x = Dropout(rate=dropout_rate)(x)

        x = Flatten()(x)

        for dense_i in range(dense_num):
            neurons = 32 * (cnn_num - dense_i)
            x = Dense(neurons)(x)
            x = LeakyReLU()(x)
            x = BatchNormalization()(x)
            x = Dropout(rate=dropout_rate)(x)

        main_output = Dense(1, activation='sigmoid')(x)

        model = K.Model(inputs=[main_input], outputs=[main_output], name='miRBind_CNN')

        self.model = model

    def compile_model(self, lr=0.00152):
        K.backend.clear_session()
        model = self.model

        opt = Adam(
            learning_rate=lr,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=False,
            name="Adam")

        model.compile(
            optimizer=opt,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model