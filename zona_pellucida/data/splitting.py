"""Train / validation / test splits and their persistence."""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


@dataclass
class Split:
    """Row indices into a SegmentationDataset.

    `train` keeps the random order produced by train_test_split: with
    shuffle=False this is the (fixed) batch order of every epoch, as in the
    thesis. Sorting it would give batches of consecutive frames of one
    video. `val` and `test` are sorted (their order does not matter).
    """

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def sizes(self):
        return {
            "train": len(self.train),
            "val": len(self.val),
            "test": len(self.test),
        }


def _stratify(labels, indices, enabled):
    return labels[indices] if enabled else None


def train_val_test_split(samples, test_size=0.2, val_size=0.2, seed=42,
                         stratify_by_sample=True):
    """Random frame-level split into train / val / test.

    The test set is taken first (`test_size` of all pairs); validation is
    `val_size` of the remainder. `samples` (the video id of every pair) is
    used to stratify so each split holds the same share of each video.
    The training indices stay in random order (see `Split`).
    """
    samples = np.asarray(samples)
    indices = np.arange(len(samples))
    trainval, test = train_test_split(
        indices, test_size=test_size, random_state=seed,
        stratify=_stratify(samples, indices, stratify_by_sample),
    )
    if val_size > 0:
        train, val = train_test_split(
            trainval, test_size=val_size, random_state=seed,
            stratify=_stratify(samples, trainval, stratify_by_sample),
        )
    else:
        train, val = trainval, np.array([], dtype=int)
    split = Split(np.asarray(train), np.sort(val), np.sort(test))
    logger.info("Split sizes: %s", split.sizes())
    return split


def leave_one_embryo_out_splits(samples, val_size=0.2, seed=42):
    """One split per video: test on that video, train/val on the others.

    Frames of a time-lapse video are highly correlated, so this measures
    generalisation to an unseen embryo. Validation frames come from the
    training video(s).
    """
    samples = np.asarray(samples)
    folds = {}
    for held_out in np.unique(samples).tolist():
        test = np.flatnonzero(samples == held_out)
        trainval = np.flatnonzero(samples != held_out)
        if val_size > 0:
            train, val = train_test_split(
                trainval, test_size=val_size, random_state=seed
            )
        else:
            # Random training order, as train_test_split would give.
            train = np.random.default_rng(seed).permutation(trainval)
            val = np.array([], dtype=int)
        folds[held_out] = Split(np.asarray(train), np.sort(val), test)
        logger.info(
            "Leave-out sample %d: %s", held_out, folds[held_out].sizes()
        )
    return folds


def split_table(dataset, split):
    """One row per pair: filenames, sample, frame, its split and its
    position within the split (the training order)."""
    names = np.full(len(dataset), "", dtype=object)
    order = np.full(len(dataset), -1)
    for name in ("train", "val", "test"):
        indices = getattr(split, name)
        names[indices] = name
        order[indices] = np.arange(len(indices))
    return pd.DataFrame({
        "index": np.arange(len(dataset)),
        "image_file": dataset.image_files,
        "mask_file": dataset.mask_files,
        "sample": dataset.samples,
        "frame": dataset.frames,
        "split": names,
        "order": order,
    })


def save_split(dataset, split, path):
    """Write the split to CSV so it can be reported and reproduced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    split_table(dataset, split).to_csv(path, index=False)
    logger.info("Saved split to %s", path)


def load_split(dataset, path):
    """Rebuild a Split (including the training order) from a CSV written
    by `save_split`, matching rows by filename."""
    table = pd.read_csv(path)
    position = {name: i for i, name in enumerate(dataset.image_files)}
    missing = sorted(set(table["image_file"]) - set(position))
    if missing:
        raise ValueError(f"Split file lists unknown images: {missing[:5]}")
    parts = {}
    for name in ("train", "val", "test"):
        rows = table[table["split"] == name].sort_values("order")
        parts[name] = np.array(
            [position[f] for f in rows["image_file"]], dtype=int
        )
    return Split(**parts)
