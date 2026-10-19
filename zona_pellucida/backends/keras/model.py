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

# Batch normalisation normalises with the statistics of each batch during
# training but predicts (validation, test) with running averages of them,
# which start at mean 0 and variance 1. The Keras default momentum 0.99
# moves them only 1% towards each batch: with 57 training images in batches
# of 16 (4 batches per epoch), 0.99^400 = 1.8% of the initial variance is
# still in them after 100 epochs. That is negligible in most layers, but in
# about half of the channels of the first layer the variance of the data is
# below 0.002, so there the running variance is still about ten times too
# large or more, and the predictions are poor. With 0.9 (PyTorch's default,
# written there as momentum 0.1) they follow the data within about 20
# epochs. The weights trained in each epoch do not depend on the momentum;
# the validation scores, and with them the kept epoch, do.
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
