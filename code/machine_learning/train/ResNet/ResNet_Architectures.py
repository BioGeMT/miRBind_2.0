from tensorflow.keras import layers, models
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.optimizers import Adam
    
# defining a custom Keras layer which inturn implements a residual block for large ResNet
@register_keras_serializable()
class ResBlock(layers.Layer):
    """
    Defines a Residual block based on the original ResNet paper.
    The block either maintains the input dimensions or downsamples based on the specified parameters.
    """

    def __init__(self, downsample=False, filters=16, kernel_size=3, reg_factor=None, regularizer_type=None, **kwargs):
        """
        Initializes the residual block with optional downsampling.
        
        Parameters:
        - downsample: Boolean, whether to downsample the input (using stride of 2)
        - filters: Number of filters for the Conv2D layers
        - kernel_size: Size of the convolution kernel
        """
        # calling the parent class constructor
        super(ResBlock, self).__init__()

        # parameters for the residual block
        self.downsample = downsample
        self.filters = filters
        self.kernel_size = kernel_size
        self.reg_factor = reg_factor
        self.regularizer_type = regularizer_type
        
        # get the regularizer if it is not None
        reg = regularizer_type(reg_factor) if (reg_factor is not None and regularizer_type is not None) else None

        # initialize first convolution layer, with stride 1 or 2 depending on downsampling
        self.conv1 = layers.Conv2D(kernel_size=self.kernel_size,
                                   strides=(1 if not self.downsample else 2),
                                   filters=self.filters,
                                   padding="same",
                                   kernel_regularizer=reg)
        self.activation1 = layers.ReLU()  # activation function after first convolution
        self.batch_norm1 = layers.BatchNormalization()  # batch normalization after first convolution
        
        # initialize second convolution layer with stride 1 (no downsampling here)
        self.conv2 = layers.Conv2D(kernel_size=self.kernel_size,
                                   strides=1,
                                   filters=self.filters,
                                   padding="same",
                                   kernel_regularizer=reg)
        self.activation2 = layers.ReLU()
        self.batch_norm2 = layers.BatchNormalization()  # batch normalization after second convolution

        # third convolution if downsampling is needed to match input dimensions
        if self.downsample:
          self.shortcut_conv = layers.Conv2D(kernel_size=1,
                                     strides=2,
                                     filters=self.filters,
                                     padding="same",
                                     kernel_regularizer=reg)
          self.shortcut_bn = layers.BatchNormalization()  # batch normalization after third convolution
        else:
          self.shortcut_conv = None

        # final activation function
        self.final_relu = layers.ReLU()
        
    def build(self, input_shape):
        # tells keras that the layer is built
        super(ResBlock, self).build(input_shape)

    def call(self, inputs, training=False):
        """
        Forward pass for the residual block. Applies the convolutions, activation, and adds the skip connection.

        Parameters:
        - inputs: Input tensor

        Returns:
        - Tensor after applying the residual block transformation
        """
        # first convolution, activation, and batch normalization
        x = self.conv1(inputs)
        x = self.activation1(x)
        x = self.batch_norm1(x, training=training)
        
        # second convolution (no downsampling here)
        x = self.conv2(x)
        x = self.activation2(x)
        x = self.batch_norm2(x, training=training)

        # shortcut branch
        if self.downsample:
            shortcut = self.shortcut_conv(inputs)
            shortcut = self.shortcut_bn(shortcut, training=training)
        else:
            shortcut = inputs

        # merge the skip connection
        x = layers.Add()([x, shortcut])
        # final activation
        x = self.final_relu(x)

        return x

    def get_config(self):
        """
        Returns the configuration of the residual block (required for saving and loading the model).
        """
        base_config = super().get_config()
        base_config.update({
            'filters':      self.filters,
            'kernel_size':  self.kernel_size,
            'downsample':   self.downsample,
            'reg_factor':   self.reg_factor,
            'regularizer_type': (self.regularizer_type.__name__ if self.regularizer_type else None)
        })
        return base_config


# define the ResNet model
def build_resnet_small(input_shape, dropout_rate, learning_rate, reg_factor=None, regularizer_type=None):    
    """
    Builds a simple ResNet model using custom residual blocks.
    """
    inputs = layers.Input(shape=input_shape)

    # initial Conv Layer
    x = layers.Conv2D(64, kernel_size=(3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # add ResBlocks
    x = ResBlock(downsample=False, filters=64, reg_factor=reg_factor, regularizer_type=regularizer_type)(x)
    x = ResBlock(downsample=True, filters=128, reg_factor=reg_factor, regularizer_type=regularizer_type)(x)
    
    # use Global Average Pooling to reduce feature map dimensions
    x = layers.GlobalAveragePooling2D()(x)

    # add dense layers for classification
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)  # dropout layer
    x = layers.Dense(1, activation='sigmoid')(x)  # binary classification (0 or 1)

    # build model
    model = models.Model(inputs, x)
    # compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy',metrics=['accuracy'])
    # output model summary
    model.summary()
    
    print()
    
    return model

# define the ResNet model
def build_resnet_medium(input_shape, dropout_rate, learning_rate, reg_factor=None, regularizer_type=None):    
    """
    Builds a simple ResNet model using custom residual blocks.
    """
    inputs = layers.Input(shape=input_shape)

    # initial Conv Layer
    x = layers.Conv2D(64, kernel_size=(3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # add ResBlocks
    x = ResBlock(downsample=False, filters=64, reg_factor=reg_factor, regularizer_type=regularizer_type)(x)
    x = ResBlock(downsample=True, filters=128, reg_factor=reg_factor, regularizer_type=regularizer_type)(x)
    x = ResBlock(downsample=True, filters=128, reg_factor=reg_factor, regularizer_type=regularizer_type)(x)
    
    # use Global Average Pooling to reduce feature map dimensions
    x = layers.GlobalAveragePooling2D()(x)

    # add dense layers for classification
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)  # dropout layer
    x = layers.Dense(1, activation='sigmoid')(x)  # binary classification (0 or 1)

    # build model
    model = models.Model(inputs, x)
    # compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy',metrics=['accuracy'])
    # output model summary
    model.summary()
    
    print()
    
    return model

# define the ResNet model
def build_resnet_large(input_shape, dropout_rate, learning_rate, reg_factor=None, regularizer_type=None):    
    """
    Builds a simple ResNet model using custom residual blocks.
    """
    inputs = layers.Input(shape=input_shape)

    # initial Conv Layer
    x = layers.Conv2D(64, kernel_size=(3, 3), padding='same')(inputs)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)

    # add ResBlocks
    x = ResBlock(downsample=False, filters=64, reg_factor=reg_factor, regularizer_type=regularizer_type)(x)
    x = ResBlock(downsample=True, filters=128, reg_factor=reg_factor, regularizer_type=regularizer_type)(x)
    x = ResBlock(downsample=True, filters=128, reg_factor=reg_factor, regularizer_type=regularizer_type)(x)

    # flatten and add dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)  # dropout layer
    x = layers.Dense(1, activation='sigmoid')(x)  # binary classification (0 or 1)

    # build model
    model = models.Model(inputs, x)
    # compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy',metrics=['accuracy'])
    # output model summary
    model.summary()
    
    print()
    
    return model