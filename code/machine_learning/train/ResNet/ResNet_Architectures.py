from tensorflow.keras import layers, models
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.optimizers import Adam


# defining a custom Keras layer which inturn implements a residual block for large ResNet
@register_keras_serializable()
class ResBlock_Large(layers.Layer):
    """
    Defines a Residual block based on the original ResNet paper.
    The block either maintains the input dimensions or downsamples based on the specified parameters.
    """

    def __init__(self, downsample=False, filters=16, kernel_size=3, reg_factor=None, regularizer_type=None):
        """
        Initializes the residual block with optional downsampling.
        
        Parameters:
        - downsample: Boolean, whether to downsample the input (using stride of 2)
        - filters: Number of filters for the Conv2D layers
        - kernel_size: Size of the convolution kernel
        """
        # calling the parent class constructor
        super(ResBlock_Large, self).__init__()

        # parameters for the residual block
        self.downsample = downsample
        self.filters = filters
        self.kernel_size = kernel_size
        
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

        # third convolution if downsampling is needed to match input dimensions
        self.conv3 = None  # Default to None
        if self.downsample:
          self.conv3 = layers.Conv2D(kernel_size=1,
                                     strides=2,
                                     filters=self.filters,
                                     padding="same",
                                     kernel_regularizer=reg)
          self.batch_norm3 = layers.BatchNormalization()  # batch normalization after third convolution

        self.activation2 = layers.ReLU()  # activation after second convolution
        self.batch_norm2 = layers.BatchNormalization()  # batch normalization after second convolution
        
    def build(self, input_shape):
        # tells keras that the layer is built
        super(ResBlock_Large, self).build(input_shape)

    def call(self, inputs):
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
        x = self.batch_norm1(x)
        
        # second convolution (no downsampling here)
        x = self.conv2(x)

        # adjust input dimensions if downsampling
        if self.downsample:
            inputs = self.conv3(inputs)

        # add the input (skip connection) to the output of the convolutions
        x = layers.Add()([inputs, x])

        # final activation and batch normalization
        x = self.activation2(x)
        x = self.batch_norm2(x)

        return x

    def get_config(self):
        """
        Returns the configuration of the residual block (required for saving and loading the model).
        """
        return {'filters': self.filters, 'downsample': self.downsample, 'kernel_size': self.kernel_size}
    
# defining a custom Keras layer which inturn implements a residual block for small and medium ResNet
@register_keras_serializable()
class ResBlock_SmallAndMedium(layers.Layer):
    """
    Defines a Residual block based on the original ResNet paper.
    The block either maintains the input dimensions or downsamples based on the specified parameters.
    """

    def __init__(self, downsample=False, filters=16, kernel_size=3, reg_factor=None, regularizer_type=None):
        """
        Initializes the residual block with optional downsampling.
        
        Parameters:
        - downsample: Boolean, whether to downsample the input (using stride of 2)
        - filters: Number of filters for the Conv2D layers
        - kernel_size: Size of the convolution kernel
        """
        # calling the parent class constructor        
        super(ResBlock_SmallAndMedium, self).__init__()
        
        # parameters for the residual block
        self.downsample = downsample
        self.filters = filters
        self.kernel_size = kernel_size
        
        # get the regularizer if it is not None
        reg = regularizer_type(reg_factor) if (reg_factor is not None and regularizer_type is not None) else None
        
        # first convolution: Conv -> BN -> ReLU
        self.conv1 = layers.Conv2D(filters=self.filters, 
                                   kernel_size=self.kernel_size, 
                                   strides=(1 if not self.downsample else 2), 
                                   padding="same",
                                   kernel_regularizer=reg)
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.ReLU()
        
        # second convolution: Conv -> BN (activation applied after adding shortcut)
        self.conv2 = layers.Conv2D(filters=self.filters, 
                                   kernel_size=self.kernel_size, 
                                   strides=1, 
                                   padding="same",
                                   kernel_regularizer=reg)
        self.bn2 = layers.BatchNormalization()
        
        # if downsampling, adjust the shortcut branch with its own convolution and BN.
        if self.downsample:
            self.shortcut_conv = layers.Conv2D(filters=self.filters, 
                                               kernel_size=1, 
                                               strides=2, 
                                               padding="same",
                                               kernel_regularizer=reg)
            self.shortcut_bn = layers.BatchNormalization()
        else:
            self.shortcut_conv = None
            
    def build(self, input_shape):
        # tells Keras that the layer is built. 
        super(ResBlock_SmallAndMedium, self).build(input_shape)

    def call(self, inputs, training=False):
        """
        Forward pass for the residual block. Applies the convolutions, activation, and adds the skip connection.

        Parameters:
        - inputs: Input tensor

        Returns:
        - Tensor after applying the residual block transformation
        """
        # main branch: conv -> BN -> ReLU -> conv -> BN
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        # shortcut branch
        shortcut = inputs
        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(shortcut)
            shortcut = self.shortcut_bn(shortcut, training=training)
        
        # add the shortcut and apply final activation
        x = layers.add([x, shortcut])

        return x

    def get_config(self):
        """
        Returns the configuration of the residual block (required for saving and loading the model).
        """
        return {'filters': self.filters, 'downsample': self.downsample, 'kernel_size': self.kernel_size}

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
    x = ResBlock_SmallAndMedium(downsample=False, filters=64, reg_factor=reg_factor, regularizer_type=regularizer_type)(x)
    x = ResBlock_SmallAndMedium(downsample=True, filters=128, reg_factor=reg_factor, regularizer_type=regularizer_type)(x)
    
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
    x = ResBlock_SmallAndMedium(downsample=False, filters=64, reg_factor=reg_factor, regularizer_type=regularizer_type)(x)
    x = ResBlock_SmallAndMedium(downsample=True, filters=128, reg_factor=reg_factor, regularizer_type=regularizer_type)(x)
    x = ResBlock_SmallAndMedium(downsample=True, filters=128, reg_factor=reg_factor, regularizer_type=regularizer_type)(x)
    
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
    x = ResBlock_Large(downsample=False, filters=64, reg_factor=reg_factor, regularizer_type=regularizer_type)(x)
    x = ResBlock_Large(downsample=True, filters=128, reg_factor=reg_factor, regularizer_type=regularizer_type)(x)

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
