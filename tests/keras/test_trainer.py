import numpy as np
import pytest

keras = pytest.importorskip("keras")

from zona_pellucida.backends.keras import trainer  # noqa: E402
from zona_pellucida.backends.keras.metrics import ForegroundIoU  # noqa: E402
from zona_pellucida.config import load_config  # noqa: E402


def _cfg(epochs, **early_stopping):
    return load_config(overrides={"train": {
        "epochs": epochs, "batch_size": 4,
        "tensorboard": {"enabled": False},
        "early_stopping": early_stopping,
    }}).train


def test_callbacks_without_early_stopping_never_stop():
    callbacks, stop = trainer.make_callbacks(_cfg(100))
    assert stop in callbacks
    assert stop.patience == 100
    assert stop.start_from_epoch == 0
    assert stop.restore_best_weights
    assert stop.monitor == "val_iou"


def test_callbacks_with_early_stopping_pass_settings():
    _, stop = trainer.make_callbacks(
        _cfg(100, enabled=True, patience=7, start_from_epoch=30))
    assert (stop.patience, stop.start_from_epoch) == (7, 30)


def test_no_validation_no_early_stopping():
    callbacks, stop = trainer.make_callbacks(_cfg(100), has_validation=False)
    assert stop is None and callbacks == []


class _ScriptedValIoU(keras.callbacks.Callback):
    """Replaces val_iou by a scripted sequence and records the weights."""

    def __init__(self, values):
        super().__init__()
        self.values = values
        self.weights = []

    def on_epoch_end(self, epoch, logs=None):
        logs["val_iou"] = self.values[epoch]
        self.weights.append([w.copy() for w in self.model.get_weights()])


def _train_scripted(monkeypatch, values, train_cfg):
    scripted = _ScriptedValIoU(values)
    make_callbacks = trainer.make_callbacks

    def with_script(*args, **kwargs):
        callbacks, stop = make_callbacks(*args, **kwargs)
        return [scripted] + callbacks, stop

    monkeypatch.setattr(trainer, "make_callbacks", with_script)
    keras.utils.set_random_seed(0)
    model = keras.Sequential([keras.Input((4,)),
                              keras.layers.Dense(1, activation="sigmoid")])
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=[ForegroundIoU(name="iou")])
    rng = np.random.default_rng(0)
    x = rng.random((8, 4)).astype(np.float32)
    y = (rng.random((8, 1)) > 0.5).astype(np.float32)
    history, best_epoch = trainer.train_model(
        model, x, y, train_cfg, validation_data=(x, y))
    return model, history, best_epoch, scripted


@pytest.mark.slow
def test_all_epochs_trained_and_best_epoch_restored(monkeypatch):
    model, history, best_epoch, scripted = _train_scripted(
        monkeypatch, [0.1, 0.9, 0.2, 0.3], _cfg(4))
    assert len(history["loss"]) == 4  # no early stop
    assert best_epoch == 2  # 1-based
    for kept, best in zip(model.get_weights(), scripted.weights[1]):
        np.testing.assert_array_equal(kept, best)


@pytest.mark.slow
def test_early_stopping_ignores_warm_up_epochs(monkeypatch):
    _, history, best_epoch, _ = _train_scripted(
        monkeypatch, [0.9, 0.1, 0.5, 0.4, 0.3],
        _cfg(5, enabled=True, patience=1, start_from_epoch=1))
    # Epoch 1 (0.9) is in the warm-up and ignored; best is epoch 3 (0.5);
    # no improvement at epoch 4 -> stop.
    assert len(history["loss"]) == 4
    assert best_epoch == 3
