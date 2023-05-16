"""
Configuration file for setting hyperparameters.
"""

from keras.initializers.initializers import RandomNormal
from keras.layers import BatchNormalization, Activation
from keras.layers import Dense, Reshape, Flatten, Dropout
from keras.layers import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.optimizers import Adam
from MinibatchDiscrimination import MinibatchDiscrimination

GPU = "0"
RES_DIR = "results/"

# Data
IMG_SHAPE = (128, 128, 1)
"""
The shape of the input images.
"""

DATA_DIR = f"../data/{IMG_SHAPE[0]}"
"""
The directory path where the data is located.
"""

ROTATED_TRAIN = False
"""
Specifies whether to use rotated training images.
"""

RESCALE_PIXEL_VALUES = True
"""
Specifies whether to rescale the pixel values of the loaded images.
"""

# Training
EPOCHS = 3000
"""
The number of epochs for training.
"""

BATCH_SIZE = 30
"""
The batch size for training.
"""

EVALUATION_INTERVAL = 100
"""
The interval at which to perform evaluation during training.
"""

# Across models
INIT_WEIGHTS = RandomNormal(0.0, 0.02)
"""
The initializer for the weights of the model layers.
"""

BIAS = True
"""
Specifies whether to include bias terms in the model layers.
"""

BATCHNORM_MOMENTUM = 0.8
"""
The momentum parameter for batch normalization.
"""

DROPOUT_RATE = 0.25
"""
The dropout rate for regularization.
"""

LEAKY_RELU_ALPHA = 0.2
"""
The alpha parameter for LeakyReLU activation function.
"""

# Discriminator
MINIBATCH_DISCRIMINATION = False
"""
Specifies whether to use minibatch discrimination in the discriminator.
"""

DISC_LAYER_STACK = [
    Conv2D(
        32,
        kernel_size=3,
        strides=2,
        padding="same",
        kernel_initializer=INIT_WEIGHTS,
        use_bias=BIAS,
    ),
    Dropout(DROPOUT_RATE),
    LeakyReLU(LEAKY_RELU_ALPHA),
    Conv2D(
        64,
        kernel_size=3,
        strides=2,
        padding="same",
        kernel_initializer=INIT_WEIGHTS,
        use_bias=BIAS,
    ),
    BatchNormalization(momentum=BATCHNORM_MOMENTUM),
    Dropout(DROPOUT_RATE),
    LeakyReLU(LEAKY_RELU_ALPHA),
    Conv2D(
        128,
        kernel_size=3,
        strides=2,
        padding="same",
        kernel_initializer=INIT_WEIGHTS,
        use_bias=BIAS,
    ),
    BatchNormalization(momentum=BATCHNORM_MOMENTUM),
    Dropout(DROPOUT_RATE),
    LeakyReLU(LEAKY_RELU_ALPHA),
    Conv2D(
        256,
        kernel_size=3,
        strides=2,
        padding="same",
        kernel_initializer=INIT_WEIGHTS,
        use_bias=BIAS,
    ),
    BatchNormalization(momentum=BATCHNORM_MOMENTUM),
    Dropout(DROPOUT_RATE),
    LeakyReLU(LEAKY_RELU_ALPHA),
    Flatten(),
    Dense(1),
    Activation("sigmoid"),
]
if MINIBATCH_DISCRIMINATION:
    DISC_LAYER_STACK.insert(-3, MinibatchDiscrimination(100, 5))
"""
The layer stack configuration for the discriminator model.
"""

DISC_LOSS = "binary_crossentropy"
"""
The loss function for the discriminator model.
"""

DISC_OPTIMIZER = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)
"""
The optimizer for the discriminator model.
"""

# Generator
GEN_LAYER_STACK = [
    Dense(128 * 32 * 32, activation="relu", input_dim=100),
    Reshape((32, 32, 128)),
    UpSampling2D(),
    Conv2D(
        128,
        kernel_size=3,
        strides=1,
        padding="same",
        kernel_initializer=INIT_WEIGHTS,
        use_bias=BIAS,
    ),
    BatchNormalization(momentum=BATCHNORM_MOMENTUM),
    Dropout(DROPOUT_RATE),
    Activation("relu"),
    UpSampling2D(),
    Conv2D(
        64,
        kernel_size=3,
        strides=1,
        padding="same",
        kernel_initializer=INIT_WEIGHTS,
        use_bias=BIAS,
    ),
    BatchNormalization(momentum=BATCHNORM_MOMENTUM),
    Dropout(DROPOUT_RATE),
    Activation("relu"),
    Conv2D(
        1,
        kernel_size=3,
        strides=1,
        padding="same",
        kernel_initializer=INIT_WEIGHTS,
        use_bias=BIAS,
    ),
    Activation("tanh"),
]
"""
The layer stack configuration for the generator model.
"""

GEN_LOSS = "binary_crossentropy"
"""
The loss function for the generator model.
"""

GEN_OPTIMIZER = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)
"""
The optimizer for the generator model.
"""

NOISE_DIST = "gaussian"
"""
The distribution of the noise input to the generator model.
"""

NOISE_LENGTH = 100
"""
The length of the noise input to the generator model.
"""

# Critic
CRITIC_LAYER_STACK = [
    Conv2D(
        32,
        kernel_size=3,
        strides=2,
        padding="same",
        kernel_initializer=INIT_WEIGHTS,
        use_bias=BIAS,
    ),
    Dropout(DROPOUT_RATE),
    LeakyReLU(LEAKY_RELU_ALPHA),
    Conv2D(
        64,
        kernel_size=3,
        strides=2,
        padding="same",
        kernel_initializer=INIT_WEIGHTS,
        use_bias=BIAS,
    ),
    BatchNormalization(momentum=BATCHNORM_MOMENTUM),
    Dropout(DROPOUT_RATE),
    LeakyReLU(LEAKY_RELU_ALPHA),
    Conv2D(
        128,
        kernel_size=3,
        strides=2,
        padding="same",
        kernel_initializer=INIT_WEIGHTS,
        use_bias=BIAS,
    ),
    BatchNormalization(momentum=BATCHNORM_MOMENTUM),
    Dropout(DROPOUT_RATE),
    LeakyReLU(LEAKY_RELU_ALPHA),
    Conv2D(
        256,
        kernel_size=3,
        strides=2,
        padding="same",
        kernel_initializer=INIT_WEIGHTS,
        use_bias=BIAS,
    ),
    BatchNormalization(momentum=BATCHNORM_MOMENTUM),
    Dropout(DROPOUT_RATE),
    LeakyReLU(LEAKY_RELU_ALPHA),
    Flatten(),
    Dense(1),
    Activation("sigmoid"),
]
"""
The layer stack configuration for the critic model.
"""

CRITIC_TRAIN_EPOCHS = 200
"""
The number of epochs for training the critic model.
"""

CRITIC_LOSS = "binary_crossentropy"
"""
The loss function for the critic model.
"""

DISC_CRITIC_LOSS_MEAN = False
"""
Specifies whether to take the mean or otherwise the sum of the target and fake losses.
"""

CRITIC_OPTIMIZER = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)
"""
The optimizer for the critic model.
"""
