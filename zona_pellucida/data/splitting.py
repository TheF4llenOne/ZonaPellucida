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


def _count(fraction, total):
    """`fraction` of `total`, rounded half up, at least 1."""
    return max(1, int(fraction * total + 0.5))


def _draw_samples(candidates, fraction, rng, purpose):
    """Draw `fraction` of the sample (embryo) ids `candidates` with `rng`,
    at least one, and leave at least one for training. Sorted ids."""
    count = _count(fraction, len(candidates))
    if count >= len(candidates):
        raise ValueError(
            f"{len(candidates)} embryos are too few to take {fraction:.0%} "
            f"of them for {purpose} and keep some for training")
    return np.sort(rng.choice(candidates, size=count, replace=False))


def _group_split(samples, test_size, val_size, seed):
    """Whole samples (embryos) go to one subset."""
    rng = np.random.default_rng(seed)
    embryos = np.unique(samples)
    test_ids = _draw_samples(embryos, test_size, rng, "testing")
    rest = np.setdiff1d(embryos, test_ids)
    val_ids = (_draw_samples(rest, val_size, rng, "validation")
               if val_size > 0 else rest[:0])
    test = np.flatnonzero(np.isin(samples, test_ids))
    val = np.flatnonzero(np.isin(samples, val_ids))
    train = np.flatnonzero(
        ~np.isin(samples, np.concatenate([test_ids, val_ids])))
    # Random training order: with shuffle=False it is the batch order.
    return Split(rng.permutation(train), val, test)


def train_val_test_split(samples, test_size=0.2, val_size=0.2, seed=42,
                         stratify_by_sample=True, group_by_sample=True):
    """Split the pairs into train / val / test.

    `samples` is the embryo (video) id of every pair. With
    `group_by_sample`, whole embryos go to one subset: `test_size` of the
    embryos for testing and `val_size` of the remaining embryos for
    validation (rounded half up, at least one each), so all time points of
    an embryo stay together. Without it, the split is frame-level (first
    thesis version): `test_size` of all pairs for testing and `val_size` of
    the remainder for validation, stratified by video if
    `stratify_by_sample`. The training indices stay in random order (see
    `Split`).
    """
    samples = np.asarray(samples)
    if group_by_sample:
        split = _group_split(samples, test_size, val_size, seed)
        logger.info("Split sizes: %s", split.sizes())
        return split
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


def cross_validation_splits(samples, folds=5, val_size=0.2, seed=42,
                            group_by_sample=True):
    """k-fold cross-validation by embryo: {fold number: Split}.

    The embryos are shuffled with `seed` and dealt into `folds` near-equal
    groups; each group is the test set of one fold, so every embryo is
    tested exactly once. With one fold per embryo the embryos are taken in
    order, which is leave-one-embryo-out (with group_by_sample False the
    splits are those of version 0.3.0). Validation: `val_size` of each
    fold's training embryos (group_by_sample) or of its training frames.
    """
    samples = np.asarray(samples)
    embryos = np.unique(samples)
    if not 2 <= folds <= len(embryos):
        raise ValueError(
            f"folds must be between 2 and the number of embryos "
            f"({len(embryos)}), not {folds}")
    if folds == len(embryos):
        groups = [embryos[i:i + 1] for i in range(len(embryos))]
    else:
        shuffled = np.random.default_rng(seed).permutation(embryos)
        groups = [np.sort(group) for group in np.array_split(shuffled, folds)]
    splits = {}
    for fold, test_ids in enumerate(groups, start=1):
        test = np.flatnonzero(np.isin(samples, test_ids))
        trainval = np.flatnonzero(~np.isin(samples, test_ids))
        if val_size <= 0:
            # Random training order, as train_test_split would give.
            train = np.random.default_rng(seed).permutation(trainval)
            val = np.array([], dtype=int)
        elif group_by_sample:
            rng = np.random.default_rng([seed, fold])
            val_ids = _draw_samples(np.unique(samples[trainval]), val_size,
                                    rng, "validation")
            val = np.flatnonzero(np.isin(samples, val_ids))
            train = rng.permutation(np.setdiff1d(trainval, val))
        else:
            train, val = train_test_split(trainval, test_size=val_size,
                                          random_state=seed)
        splits[fold] = Split(np.asarray(train), np.sort(val), test)
        logger.info("Cross-validation fold %d, test embryos %s: %s", fold,
                    test_ids.tolist(), splits[fold].sizes())
    return splits


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
