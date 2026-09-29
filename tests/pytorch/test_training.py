import numpy as np
import pytest

torch = pytest.importorskip("torch")
L = pytest.importorskip("lightning")

from zona_pellucida.backends.pytorch import trainer  # noqa: E402
from zona_pellucida.backends.pytorch.backend import backend  # noqa: E402
from zona_pellucida.backends.pytorch.data import (  # noqa: E402
    make_loader,
    to_numpy,
    to_tensor,
)
from zona_pellucida.config import load_config  # noqa: E402
from zona_pellucida.data.loading import load_dataset  # noqa: E402


def test_tensor_layout_round_trip():
    x = np.random.default_rng(0).random((3, 4, 5, 1)).astype(np.float32)
    assert tuple(to_tensor(x).shape) == (3, 1, 4, 5)
    np.testing.assert_array_equal(to_numpy(to_tensor(x)), x)


def _epoch_orders(seed, shuffle=True, epochs=3):
    x = np.arange(12, dtype=np.float32).reshape(12, 1, 1, 1)
    loader = make_loader(x, x, batch_size=12, shuffle=shuffle, seed=seed)
    return [next(iter(loader))[0].flatten().tolist() for _ in range(epochs)]


def test_shuffle_changes_every_epoch_and_is_reproducible():
    orders = _epoch_orders(seed=42)
    assert len({tuple(o) for o in orders}) == 3  # a new order each epoch
    assert _epoch_orders(seed=42) == orders  # same sequence for the seed
    assert _epoch_orders(seed=43) != orders


def test_no_shuffle_keeps_the_given_order():
    assert _epoch_orders(seed=None, shuffle=False)[0] == list(range(12))


def test_environment_reports_torch_and_lightning():
    assert {"torch", "lightning", "torchmetrics", "gpus", "cuda"} <= set(
        backend.environment())


def _cfg(tmp_path, epochs=2, shuffle=False, **early_stopping):
    return load_config(overrides={
        "output_dir": str(tmp_path),
        "data": {"size": 32},
        "train": {"epochs": epochs, "batch_size": 4, "shuffle": shuffle,
                  "tensorboard": {"enabled": False},
                  "early_stopping": early_stopping},
    })


def _train(dataset, cfg, name="run"):
    backend.start_run(cfg.seed, cfg.deterministic_ops)
    module = backend.build_model(cfg, "weighted", {0: 0.5, 1: 5.0})
    initial = {k: v.clone() for k, v in module.state_dict().items()}
    train = dataset.subset(np.arange(0, 16))
    val = dataset.subset(np.arange(16, 22))
    history, best_epoch, path = trainer.fit(
        module,
        make_loader(train.images, train.masks, 4, cfg.train.shuffle,
                    cfg.seed),
        make_loader(val.images, val.masks, 4),
        cfg.train, model_dir=cfg.output_dir, name=name,
        deterministic=cfg.deterministic_ops,
    )
    return initial, module, history, best_epoch, path


@pytest.mark.slow
@pytest.mark.parametrize("shuffle", [False, True])
def test_training_is_reproducible(synthetic_dirs, tmp_path, shuffle):
    # On a CUDA machine this runs on the GPU with deterministic algorithms.
    dataset = load_dataset(*synthetic_dirs, size=32)
    cfg = _cfg(tmp_path, shuffle=shuffle)
    init_a, model_a, hist_a, _, _ = _train(dataset, cfg, "a")
    init_b, model_b, hist_b, _, _ = _train(dataset, cfg, "b")
    for key in init_a:
        torch.testing.assert_close(init_a[key], init_b[key], rtol=0, atol=0)
    state_b = model_b.state_dict()
    for key, value in model_a.state_dict().items():
        torch.testing.assert_close(value, state_b[key], rtol=0, atol=0)
    assert hist_a == hist_b


@pytest.mark.slow
def test_other_seed_gives_other_initial_weights(synthetic_dirs, tmp_path):
    dataset = load_dataset(*synthetic_dirs, size=32)
    init_a, *_ = _train(dataset, _cfg(tmp_path, epochs=1), "a")
    cfg = _cfg(tmp_path, epochs=1)
    cfg.seed = 43
    init_b, *_ = _train(dataset, cfg, "b")
    key = "model.conv1.0.0.weight"
    assert not torch.equal(init_a[key], init_b[key])


@pytest.mark.slow
def test_history_and_best_checkpoint(synthetic_dirs, tmp_path):
    dataset = load_dataset(*synthetic_dirs, size=32)
    cfg = _cfg(tmp_path, epochs=3)
    _, _, history, best_epoch, path = _train(dataset, cfg)
    assert {"loss", "iou", "dice", "precision", "recall", "accuracy",
            "val_loss", "val_iou", "val_dice"} <= set(history)
    assert all(len(values) == 3 for values in history.values())
    assert best_epoch == int(np.argmax(history["val_iou"])) + 1
    assert path.name == "run.ckpt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] + 1 == best_epoch


class _ScriptedValIoU(L.Callback):
    """Overwrites val_iou with a scripted sequence before the other
    callbacks (checkpoint, early stopping) read it."""

    def __init__(self, values):
        super().__init__()
        self.values = values

    def _script(self, trainer_):
        trainer_.callback_metrics["val_iou"] = torch.tensor(
            self.values[trainer_.current_epoch])

    def on_validation_end(self, trainer_, pl_module):
        self._script(trainer_)

    def on_train_epoch_end(self, trainer_, pl_module):
        self._script(trainer_)


@pytest.mark.slow
def test_early_stopping_ignores_warm_up_epochs(synthetic_dirs, tmp_path,
                                               monkeypatch):
    make_callbacks = trainer.make_callbacks

    def with_script(*args, **kwargs):
        callbacks, history, checkpoint = make_callbacks(*args, **kwargs)
        script = _ScriptedValIoU([0.9, 0.1, 0.5, 0.4, 0.3])
        return [script] + callbacks, history, checkpoint

    monkeypatch.setattr(trainer, "make_callbacks", with_script)
    dataset = load_dataset(*synthetic_dirs, size=32)
    cfg = _cfg(tmp_path, epochs=5, enabled=True, patience=1,
               start_from_epoch=1)
    _, _, history, best_epoch, path = _train(dataset, cfg)
    # Epoch 1 (0.9) is in the warm-up and ignored; best is epoch 3 (0.5);
    # no improvement at epoch 4 -> stop. Like Keras, the checkpoint keeps
    # epoch 3, not the warm-up epoch.
    assert len(history["loss"]) == 4
    assert best_epoch == 3
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] + 1 == 3


@pytest.mark.slow
def test_predict_and_evaluate(synthetic_dirs, tmp_path):
    dataset = load_dataset(*synthetic_dirs, size=32)
    cfg = _cfg(tmp_path, epochs=1)
    _, _, _, _, path = _train(dataset, cfg)
    model = backend.load_model(path)
    assert model.hparams.loss == "weighted"
    assert (model.loss_fn.w0, model.loss_fn.w1) == (0.5, 5.0)
    probabilities = backend.predict(model, dataset.images[:3])
    assert probabilities.shape == (3, 32, 32, 1)
    assert 0 <= probabilities.min() and probabilities.max() <= 1
    scores = backend.evaluate(model, dataset.images[:5], dataset.masks[:5])
    assert {"loss", "accuracy", "iou", "dice", "precision",
            "recall"} <= set(scores)


def test_weighted_model_needs_class_weights(tmp_path):
    with pytest.raises(ValueError, match="class weights"):
        backend.build_model(_cfg(tmp_path), "weighted", None)
