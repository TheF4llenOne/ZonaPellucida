import numpy as np
import pytest

keras = pytest.importorskip("keras")

from zona_pellucida.backends.keras.losses import (  # noqa: E402
    WeightedBinaryCrossentropy,
    get_loss,
)


def _data():
    rng = np.random.default_rng(0)
    y_true = (rng.random((2, 8, 8, 1)) > 0.8).astype(np.float32)
    y_pred = rng.uniform(0.01, 0.99, (2, 8, 8, 1)).astype(np.float32)
    return y_true, y_pred


def test_unit_weights_equal_binary_crossentropy():
    y_true, y_pred = _data()
    weighted = WeightedBinaryCrossentropy(w0=1.0, w1=1.0)(y_true, y_pred)
    reference = keras.losses.BinaryCrossentropy()(y_true, y_pred)
    assert float(weighted) == pytest.approx(float(reference), rel=1e-5)


def test_weighted_loss_formula():
    y_true, y_pred = _data()
    w0, w1 = 0.6, 4.0
    expected = -np.mean(w1 * y_true * np.log(y_pred)
                        + w0 * (1 - y_true) * np.log(1 - y_pred))
    loss = WeightedBinaryCrossentropy(w0=w0, w1=w1)(y_true, y_pred)
    # Keras adds its epsilon (1e-7) inside the logarithm.
    assert float(loss) == pytest.approx(expected, rel=1e-4)


def test_probabilities_are_clipped():
    y_true = np.array([[[[1.0]], [[0.0]]]], dtype=np.float32)
    y_pred = np.array([[[[0.0]], [[1.0]]]], dtype=np.float32)
    loss = float(WeightedBinaryCrossentropy()(y_true, y_pred))
    reference = float(keras.losses.BinaryCrossentropy()(y_true, y_pred))
    assert np.isfinite(loss)
    assert loss == pytest.approx(reference, rel=1e-6)


def _one_fit_step(loss):
    """One SGD step of a 1x1 sigmoid conv whose logit is -20 on a zona
    pixel (confidently wrong). Returns (loss value, new kernel)."""
    keras.utils.set_random_seed(0)
    model = keras.Sequential([
        keras.Input((1, 1, 1)),
        keras.layers.Conv2D(
            1, (1, 1), activation="sigmoid", use_bias=False,
            kernel_initializer=keras.initializers.Constant(-20.0)),
    ])
    model.compile(optimizer=keras.optimizers.SGD(0.5), loss=loss)
    x = np.ones((1, 1, 1, 1), dtype=np.float32)
    history = model.fit(x, x, epochs=1, verbose=0)
    return history.history["loss"][0], float(model.get_weights()[0].ravel()[0])


def test_unit_weights_match_keras_bce_inside_fit_when_saturated():
    # Inside fit, Keras computes the BCE from the logits (no clipping); the
    # weighted loss must follow the same path, or saturated pixels would
    # stop learning in the weighted run only.
    bce_loss, bce_kernel = _one_fit_step(keras.losses.BinaryCrossentropy())
    w_loss, w_kernel = _one_fit_step(WeightedBinaryCrossentropy(1.0, 1.0))
    assert w_loss == pytest.approx(bce_loss, rel=1e-5)
    assert w_kernel == pytest.approx(bce_kernel, rel=1e-5)
    assert bce_kernel > -20.0  # the pixel is still learning


def test_config_round_trip():
    loss = WeightedBinaryCrossentropy(w0=0.5, w1=9.0)
    restored = WeightedBinaryCrossentropy.from_config(loss.get_config())
    assert (restored.w0, restored.w1) == (0.5, 9.0)


def test_get_loss():
    assert isinstance(get_loss("unweighted"), keras.losses.BinaryCrossentropy)
    loss = get_loss("weighted", {0: 0.5, 1: 10.0})
    assert (loss.w0, loss.w1) == (0.5, 10.0)
    with pytest.raises(ValueError):
        get_loss("weighted")
    with pytest.raises(ValueError):
        get_loss("focal")
