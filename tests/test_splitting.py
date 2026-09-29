import numpy as np
import pytest

from zona_pellucida.data.loading import load_dataset
from zona_pellucida.data.splitting import (
    leave_one_embryo_out_splits,
    load_split,
    save_split,
    train_val_test_split,
)

# Same composition as the thesis dataset: 105 + 100 frames of two videos.
SAMPLES = np.array([1] * 105 + [2] * 100)


def test_sizes_match_thesis_dataset():
    split = train_val_test_split(SAMPLES, test_size=0.2, val_size=0.2)
    assert split.sizes() == {"train": 131, "val": 33, "test": 41}


def test_split_is_a_partition():
    split = train_val_test_split(SAMPLES)
    parts = np.concatenate([split.train, split.val, split.test])
    assert sorted(parts) == list(range(len(SAMPLES)))


def test_split_is_reproducible_and_seed_dependent():
    a = train_val_test_split(SAMPLES, seed=42)
    b = train_val_test_split(SAMPLES, seed=42)
    c = train_val_test_split(SAMPLES, seed=0)
    np.testing.assert_array_equal(a.test, b.test)
    np.testing.assert_array_equal(a.val, b.val)
    assert not np.array_equal(a.test, c.test)


def test_split_is_stratified_by_video():
    split = train_val_test_split(SAMPLES, stratify_by_sample=True)
    for part in (split.train, split.val, split.test):
        share = np.mean(SAMPLES[part] == 1)
        assert share == pytest.approx(105 / 205, abs=0.03)


def test_training_order_is_random_so_batches_mix_both_videos():
    # With shuffle=False the training order is the batch order of every
    # epoch; sorted indices would give batches of consecutive frames.
    split = train_val_test_split(SAMPLES, seed=42)
    assert np.any(np.diff(split.train) < 0)
    batches = [split.train[b:b + 16] for b in range(0, 128, 16)]
    assert all(len(np.unique(SAMPLES[batch])) == 2 for batch in batches)


def test_zero_validation_size():
    split = train_val_test_split(SAMPLES, val_size=0)
    assert split.sizes() == {"train": 164, "val": 0, "test": 41}


@pytest.mark.parametrize("val_size", [0.0, 0.2])
def test_leave_one_embryo_out_training_order_is_random(val_size):
    folds = leave_one_embryo_out_splits(SAMPLES, val_size=val_size, seed=42)
    again = leave_one_embryo_out_splits(SAMPLES, val_size=val_size, seed=42)
    for held_out, split in folds.items():
        assert np.any(np.diff(split.train) < 0)
        np.testing.assert_array_equal(split.train, again[held_out].train)


def test_leave_one_embryo_out():
    folds = leave_one_embryo_out_splits(SAMPLES, val_size=0.2)
    assert sorted(folds) == [1, 2]
    for held_out, split in folds.items():
        assert set(SAMPLES[split.test]) == {held_out}
        assert len(split.test) == np.sum(SAMPLES == held_out)
        assert held_out not in set(SAMPLES[split.train])
        assert held_out not in set(SAMPLES[split.val])


def test_save_and_load_split(synthetic_dirs, tmp_path):
    dataset = load_dataset(*synthetic_dirs, size=32)
    split = train_val_test_split(dataset.samples)
    save_split(dataset, split, tmp_path / "split.csv")
    loaded = load_split(dataset, tmp_path / "split.csv")
    for name in ("train", "val", "test"):
        np.testing.assert_array_equal(getattr(loaded, name),
                                      getattr(split, name))
