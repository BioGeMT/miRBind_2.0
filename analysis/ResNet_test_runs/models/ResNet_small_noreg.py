from tensorflow.keras import layers, models
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.optimizers import Adam

''' 
Total params: 373,121 (1.42 MB)
Trainable params: 371,969 (1.42 MB)
Non-trainable params: 1,152 (4.50 KB)
'''

# * --- NoReg

# defining a custom Keras layer which inturn implements a residual block
@register_keras_serializable()
class ResBlock(layers.Layer):
    """
    Defines a Residual block based on the original ResNet paper.
    The block either maintains the input dimensions or downsamples based on the specified parameters.
    """

    def __init__(self, downsample=False, filters=16, kernel_size=3):
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
        
        # first convolution: Conv -> BN -> ReLU
        self.conv1 = layers.Conv2D(filters=self.filters, 
                                   kernel_size=self.kernel_size, 
                                   strides=(1 if not self.downsample else 2), 
                                   padding="same")
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.Activation("relu")
        
        # second convolution: Conv -> BN (activation applied after adding shortcut)
        self.conv2 = layers.Conv2D(filters=self.filters, 
                                   kernel_size=self.kernel_size, 
                                   strides=1, 
                                   padding="same")
        self.bn2 = layers.BatchNormalization()
        
        # if downsampling, adjust the shortcut branch with its own convolution and BN.
        if self.downsample:
            self.shortcut_conv = layers.Conv2D(filters=self.filters, 
                                               kernel_size=1, 
                                               strides=2, 
                                               padding="same")
            self.shortcut_bn = layers.BatchNormalization()
        else:
            self.shortcut_conv = None

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
def build_resnet(input_shape, dropout_rate, learning_rate):    
    """
    Builds a simple ResNet model using custom residual blocks.
    """
    inputs = layers.Input(shape=input_shape)

    # initial Conv Layer
    x = layers.Conv2D(64, kernel_size=(3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # add ResBlocks
    x = ResBlock(filters=64, downsample=False)(x)
    x = ResBlock(filters=128, downsample=True)(x)
    #x = ResBlock(filters=256, downsample=True)(x)
    
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

build_resnet((50, 20, 1), 0.05, learning_rate=0.001)#

# Model: "functional"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ input_layer (InputLayer)             │ (None, 50, 20, 1)           │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ conv2d (Conv2D)                      │ (None, 50, 20, 64)          │             640 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ batch_normalization                  │ (None, 50, 20, 64)          │             256 │
# │ (BatchNormalization)                 │                             │                 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ activation (Activation)              │ (None, 50, 20, 64)          │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ res_block (ResBlock)                 │ (None, 50, 20, 64)          │          74,368 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ res_block_1 (ResBlock)               │ (None, 25, 10, 128)         │         231,296 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ global_average_pooling2d             │ (None, 128)                 │               0 │
# │ (GlobalAveragePooling2D)             │                             │                 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense (Dense)                        │ (None, 512)                 │          66,048 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dropout (Dropout)                    │ (None, 512)                 │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_1 (Dense)                      │ (None, 1)                   │             513 │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 373,121 (1.42 MB)
#  Trainable params: 371,969 (1.42 MB)
#  Non-trainable params: 1,152 (4.50 KB)