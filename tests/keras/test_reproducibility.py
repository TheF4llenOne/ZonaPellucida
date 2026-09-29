import numpy as np
import pytest

pytest.importorskip("tensorflow")

from zona_pellucida.backends.keras.backend import backend  # noqa: E402
from zona_pellucida.backends.keras.losses import get_loss  # noqa: E402
from zona_pellucida.backends.keras.model import fcn_8  # noqa: E402
from zona_pellucida.backends.keras.trainer import (  # noqa: E402
    compile_model,
    shuffled_dataset,
    train_model,
)
from zona_pellucida.config import load_config  # noqa: E402
from zona_pellucida.data.loading import load_dataset  # noqa: E402


def _epoch_orders(seed, epochs=3):
    x = np.arange(12, dtype=np.float32)
    dataset = shuffled_dataset(x, x, batch_size=12, seed=seed)
    return [next(iter(dataset))[0].numpy().tolist() for _ in range(epochs)]


def test_shuffle_changes_every_epoch_and_is_reproducible():
    orders = _epoch_orders(seed=42)
    assert len({tuple(o) for o in orders}) == 3  # a new order each epoch
    assert _epoch_orders(seed=42) == orders  # same sequence for the seed
    assert _epoch_orders(seed=43) != orders


def test_environment_reports_tensorflow_and_keras():
    assert {"tensorflow", "keras", "gpus", "tf32"} <= set(
        backend.environment())


def _train_once(dataset, shuffle, seed=42):
    backend.start_run(seed, deterministic=True)
    cfg = load_config(overrides={
        "train": {"epochs": 2, "batch_size": 4, "shuffle": shuffle,
                  "tensorboard": {"enabled": False}},
    })
    model = compile_model(fcn_8(32, 32, 1), get_loss("unweighted"))
    initial = [w.copy() for w in model.get_weights()]
    history, _ = train_model(model, dataset.images, dataset.masks, cfg.train,
                             seed=seed)
    return initial, model.get_weights(), history


@pytest.mark.slow
@pytest.mark.parametrize("shuffle", [False, True])
def test_training_is_reproducible(synthetic_dirs, shuffle):
    dataset = load_dataset(*synthetic_dirs, size=32)
    init_a, final_a, hist_a = _train_once(dataset, shuffle)
    init_b, final_b, hist_b = _train_once(dataset, shuffle)
    for a, b in zip(init_a, init_b):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(final_a, final_b):
        np.testing.assert_array_equal(a, b)
    assert hist_a == hist_b


@pytest.mark.slow
def test_other_seed_gives_other_initial_weights(synthetic_dirs):
    dataset = load_dataset(*synthetic_dirs, size=32)
    init_a, _, _ = _train_once(dataset, False, seed=42)
    init_b, _, _ = _train_once(dataset, False, seed=43)
    assert not np.array_equal(init_a[0], init_b[0])
