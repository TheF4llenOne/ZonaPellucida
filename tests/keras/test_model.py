import pytest

pytest.importorskip("keras")

from keras.layers import (  # noqa: E402
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    Input,
    MaxPooling2D,
    concatenate,
)
from keras.models import Model  # noqa: E402

from zona_pellucida.backends.keras.model import fcn_8  # noqa: E402

pytestmark = pytest.mark.slow


def thesis_fcn_8(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    """fcn_8 as in the thesis notebook (compile/summary removed)."""
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    conv1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)  # noqa: E501
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv1)  # noqa: E501
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    conv2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool1)  # noqa: E501
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv2)  # noqa: E501
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    conv3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool2)  # noqa: E501
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv3)  # noqa: E501
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    conv4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool3)  # noqa: E501
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv4)  # noqa: E501
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    conv5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool4)  # noqa: E501
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv5)  # noqa: E501
    conv5 = BatchNormalization()(conv5)
    pool5 = MaxPooling2D((2, 2))(conv5)  # Not used  # noqa: F841
    up6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5)
    up6 = concatenate([up6, conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up6)  # noqa: E501
    conv6 = BatchNormalization()(conv6)
    up7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6)
    up7 = concatenate([up7, conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up7)  # noqa: E501
    conv7 = BatchNormalization()(conv7)
    up8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7)
    up8 = concatenate([up8, conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up8)  # noqa: E501
    conv8 = BatchNormalization()(conv8)
    up9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8)
    up9 = concatenate([up9, conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up9)  # noqa: E501
    conv9 = BatchNormalization()(conv9)
    conv9 = Dropout(0.5)(conv9)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    return Model(inputs=inputs, outputs=outputs)


def _signature(model):
    ignored = {"name"}
    return [
        (type(layer).__name__, layer.count_params(),
         {k: v for k, v in layer.get_config().items() if k not in ignored})
        for layer in model.layers
    ]


def test_architecture_is_unchanged():
    new, old = fcn_8(64, 64, 1), thesis_fcn_8(64, 64, 1)
    assert new.count_params() == old.count_params()
    assert _signature(new) == _signature(old)


def test_output_is_a_probability_map():
    model = fcn_8(32, 32, 1)
    assert model.output_shape == (None, 32, 32, 1)
    assert model.layers[-1].get_config()["activation"] == "sigmoid"
