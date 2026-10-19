"""U-Net-style fully convolutional network (architecture of the thesis)."""

from keras.layers import (
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    Input,
    MaxPooling2D,
    concatenate,
)
from keras.models import Model

# Batch normalisation predicts (validation, test) with running averages of the
# batch statistics. The Keras default momentum 0.99 moves them only 1% towards
# each training batch: with 57 training images in batches of 16 (4 batches per
# epoch) they still largely reflect their initial values (mean 0, variance 1)
# after 100 epochs, and the predictions are poor although the training itself
# is unaffected. With 0.9 (the PyTorch default) they follow the data within
# about 20 epochs. Training does not use them: the weights are the same.
BN_MOMENTUM = 0.9


def fcn_8(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, dropout=0.5):
    """Build the (uncompiled) FCN; the loss is chosen at compile time."""
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    # Contracting path
    conv1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)  # noqa: E501
    conv1 = BatchNormalization(momentum=BN_MOMENTUM)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv1)  # noqa: E501
    conv1 = BatchNormalization(momentum=BN_MOMENTUM)(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool1)  # noqa: E501
    conv2 = BatchNormalization(momentum=BN_MOMENTUM)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv2)  # noqa: E501
    conv2 = BatchNormalization(momentum=BN_MOMENTUM)(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool2)  # noqa: E501
    conv3 = BatchNormalization(momentum=BN_MOMENTUM)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv3)  # noqa: E501
    conv3 = BatchNormalization(momentum=BN_MOMENTUM)(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool3)  # noqa: E501
    conv4 = BatchNormalization(momentum=BN_MOMENTUM)(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv4)  # noqa: E501
    conv4 = BatchNormalization(momentum=BN_MOMENTUM)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool4)  # noqa: E501
    conv5 = BatchNormalization(momentum=BN_MOMENTUM)(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv5)  # noqa: E501
    conv5 = BatchNormalization(momentum=BN_MOMENTUM)(conv5)

    # Expanding Path
    up6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5)
    up6 = concatenate([up6, conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up6)  # noqa: E501
    conv6 = BatchNormalization(momentum=BN_MOMENTUM)(conv6)

    up7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6)
    up7 = concatenate([up7, conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up7)  # noqa: E501
    conv7 = BatchNormalization(momentum=BN_MOMENTUM)(conv7)

    up8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7)
    up8 = concatenate([up8, conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up8)  # noqa: E501
    conv8 = BatchNormalization(momentum=BN_MOMENTUM)(conv8)

    up9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8)
    up9 = concatenate([up9, conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up9)  # noqa: E501
    conv9 = BatchNormalization(momentum=BN_MOMENTUM)(conv9)
    conv9 = Dropout(dropout)(conv9)  # Adding dropout for regularization

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    return Model(inputs=inputs, outputs=outputs)
