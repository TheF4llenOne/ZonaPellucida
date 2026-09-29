"""Zona pellucida segmentation with a U-Net-style FCN (TensorFlow/Keras).

Single-file export of fcn.ipynb (FRAMEWORK = 'keras') and the
zona_pellucida package, in the order of the notebook. Generated with
`python -m zona_pellucida.bin.export_pipeline`; do not edit by hand.
"""

import copy
import dataclasses
import datetime
import importlib.metadata
import importlib.util
import logging
import os
import pickle
import platform
import random
import re
import subprocess
import sys
import typing
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import yaml
from keras import ops
from keras.callbacks import EarlyStopping, TensorBoard
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
from PIL import Image
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

logger = logging.getLogger("zona_pellucida")

# ==== Zona pellucida segmentation with a U-Net-style FCN

# %% Notebook cell 3
# ---- Configuration: run this cell first ----
SEED = 42  # single seed for every source of randomness below
FRAMEWORK = 'keras'
DETERMINISTIC_OPS = True  # stop (naming the op) if an op is not deterministic

DRIVE_DIR = '/content/drive/MyDrive'
IMAGE_DIR = DRIVE_DIR + '/images'
MASK_DIR = DRIVE_DIR + '/masks'
REPO_DIR = DRIVE_DIR + '/ZonaPellucida'  # copy of this repository (code)
OUTPUT_DIR = DRIVE_DIR + '/ZonaPellucida_outputs/' + FRAMEWORK  # figures, tables, models, logs

SIZE = 512  # images and masks are resized to SIZE x SIZE
EPOCHS = 100  # all epochs are trained; the best validation-IoU epoch is kept
SHUFFLE = False  # see "Shuffling" in the training section
CLASS_WEIGHTING = 'inverse_frequency'  # or 'median_frequency'
FIGURE_RUN = 'weighted'  # model shown in the qualitative figures
USE_VAL_THRESHOLD = False  # figures at 0.5 (False) or at the val-selected threshold
RUN_LEAVE_ONE_EMBRYO_OUT = True
CONFIG_OVERRIDES = {}  # any other value of configs/config.yaml, e.g. {'train': {'batch_size': 8}}

if FRAMEWORK not in ('pytorch', 'keras'):
    raise ValueError(f"FRAMEWORK must be 'pytorch' or 'keras', not {FRAMEWORK!r}")

# ---- Reproducibility: seed everything before anything random happens ----
os.environ['PYTHONHASHSEED'] = str(SEED)  # only affects subprocesses

random.seed(SEED)
np.random.seed(SEED)

# Avoid OOM errors by setting GPU memory growth (before the GPU is used)
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
keras.utils.set_random_seed(SEED)  # same function as tf.keras.utils.set_random_seed
if DETERMINISTIC_OPS:
    tf.config.experimental.enable_op_determinism()

# ==============================================================================
# zona_pellucida/config.py
# Experiment configuration: a YAML file mapped onto typed dataclasses.
# ==============================================================================


# globals(): a copy of this code pasted into a notebook cell has no module
# file name and falls back to the defaults.
DEFAULT_CONFIG_PATH = (
    Path(globals().get("__file__", ".")).parent / "configs" / "config.yaml"
)


@dataclass
class DataConfig:
    image_dir: str = "/content/drive/MyDrive/images"
    mask_dir: str = "/content/drive/MyDrive/masks"
    image_pattern: str = r"^sample(?P<sample>\d+)_frame_(?P<frame>\d+)\.png$"
    mask_pattern: str = (
        r"^sample(?P<sample>\d+)_mask_frame_(?P<frame>\d+)\.png$"
    )
    size: int = 512
    mask_threshold: float = 0.5


@dataclass
class SplitConfig:
    test_size: float = 0.2
    val_size: float = 0.2
    stratify_by_sample: bool = True


@dataclass
class ModelConfig:
    dropout: float = 0.5


@dataclass
class EarlyStoppingConfig:
    enabled: bool = False
    patience: int = 20
    start_from_epoch: int = 40


@dataclass
class TensorBoardConfig:
    enabled: bool = True
    histogram_freq: int = 1


@dataclass
class TrainConfig:
    batch_size: int = 16
    epochs: int = 100
    learning_rate: float = 0.001
    shuffle: bool = False
    monitor: str = "val_iou"
    mode: str = "max"
    early_stopping: EarlyStoppingConfig = field(
        default_factory=EarlyStoppingConfig
    )
    tensorboard: TensorBoardConfig = field(default_factory=TensorBoardConfig)


@dataclass
class ClassWeightingConfig:
    method: str = "inverse_frequency"


@dataclass
class EvaluationConfig:
    threshold: float = 0.5
    tune_threshold: bool = True
    threshold_min: float = 0.05
    threshold_max: float = 0.95
    threshold_steps: int = 19
    tuning_metric: str = "dice"


@dataclass
class ExperimentsConfig:
    runs: list = field(default_factory=lambda: ["unweighted", "weighted"])
    figure_run: str = "weighted"


@dataclass
class LeaveOneEmbryoOutConfig:
    enabled: bool = True
    runs: list = field(default_factory=lambda: ["unweighted", "weighted"])


@dataclass
class Config:
    framework: str = "pytorch"
    seed: int = 42
    deterministic_ops: bool = True
    output_dir: str = "outputs"
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    class_weighting: ClassWeightingConfig = field(
        default_factory=ClassWeightingConfig
    )
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    experiments: ExperimentsConfig = field(default_factory=ExperimentsConfig)
    leave_one_embryo_out: LeaveOneEmbryoOutConfig = field(
        default_factory=LeaveOneEmbryoOutConfig
    )

    def to_dict(self):
        return dataclasses.asdict(self)


def merge_dicts(base, overrides):
    """Recursively merge `overrides` into a copy of `base`."""
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _build(cls, values, path="config"):
    """Build dataclass `cls` from a dict, rejecting unknown keys."""
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise KeyError(f"Unknown key(s) in {path}: {sorted(unknown)}")
    kwargs = {}
    for name, value in values.items():
        hint = hints[name]
        if dataclasses.is_dataclass(hint):
            kwargs[name] = _build(hint, value or {}, f"{path}.{name}")
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path=None, overrides=None):
    """Load the YAML config (default: the packaged one) and apply overrides.

    `overrides` is a nested dict, e.g.
    `{"seed": 42, "train": {"epochs": 50}}`. Without a YAML file (e.g. in
    the single-file export) the dataclass defaults, which mirror
    `configs/config.yaml`, are used.
    """
    values = {}
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if path.exists():
        with open(path, encoding="utf-8") as f:
            values = yaml.safe_load(f) or {}
    elif path != DEFAULT_CONFIG_PATH:
        raise FileNotFoundError(path)
    values = merge_dicts(values, overrides or {})
    return _build(Config, values)


def save_config(cfg, path):
    """Write the effective config next to the results (reproducibility)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
    logger.info("Saved config to %s", path)

# ==============================================================================
# zona_pellucida/reproducibility.py
# Seeding, op determinism and environment reporting (framework-neutral).
# ==============================================================================


# TensorFlow / PyTorch errors of ops without a deterministic implementation
# (not e.g. "random ops must have a seed when determinism is enabled").
DETERMINISM_ERROR = re.compile(
    r"deterministic|determinism is not yet supported", re.IGNORECASE)


class DeterminismError(RuntimeError):
    """An op has no deterministic implementation on the current device."""


def seed_python(seed):
    """Seed Python and NumPy. PYTHONHASHSEED only affects interpreters
    started after it is set (e.g. subprocesses)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def _nondeterministic_op(message):
    """Name of the op in a TensorFlow or PyTorch determinism error."""
    match = re.search(r"\{\{node ([^}]+)\}\}", message)  # TensorFlow
    if match is None:  # PyTorch: "<op> does not have a deterministic ..."
        match = re.search(r"(\S+) does not have a deterministic", message)
    return match.group(1) if match else "unknown (see message)"


def run_with_determinism_check(fn, *args, **kwargs):
    """Run `fn`; if an op has no deterministic kernel, name it and stop.

    Determinism is never switched off silently: set `deterministic_ops:
    false` in the config explicitly if you accept non-reproducible runs.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as err:  # TensorFlow and PyTorch raise other types
        message = getattr(err, "message", str(err))
        if isinstance(err, DeterminismError) or not DETERMINISM_ERROR.search(
                message):
            raise
        op = _nondeterministic_op(message)
        logger.error("Op without deterministic kernel: %s\n%s", op, message)
        raise DeterminismError(
            f"Op '{op}' has no deterministic implementation on this "
            f"device: {message}"
        ) from err


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _gpu_driver():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _git(repo, *args, timeout=10):
    try:
        result = subprocess.run(["git", "-C", str(repo), *args],
                                capture_output=True, text=True,
                                timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def code_version():
    """Git commit of this project's code ('-dirty' if tracked files other
    than the notebook changed), or the installed package version outside a
    git checkout of the project."""
    # git searches upward from the folder of this file (package module or
    # single-file export); a checkout of another project does not count.
    here = Path(globals().get("__file__", ".")).resolve().parent
    top = _git(here, "rev-parse", "--show-toplevel")
    if top is None or not (Path(top) / "zona_pellucida").is_dir():
        return package_version("zona-pellucida")
    head = _git(top, "rev-parse", "HEAD")
    if head is None:
        return package_version("zona-pellucida")
    # fcn.ipynb is left out: Colab saves outputs into it.
    status = _git(top, "status", "--porcelain", "--untracked-files=no",
                  "--", ".", ":(exclude)fcn.ipynb", timeout=60)
    if status is None:
        return head + "-unknown"
    return head + ("-dirty" if status else "")


def environment_versions(backend):
    """Versions and hardware that must be reported with the results.

    Bit-exact reruns need the same code, library versions and GPU type:
    Colab assigns A100, L4 or T4 per session, and A100/L4 use TF32 for
    convolutions by default.
    """
    return {
        "code_version": code_version(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "gpu_driver": _gpu_driver(),
        **backend.environment(),
    }


def save_environment(path, versions):
    """Write the environment report as YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(versions, f, sort_keys=False)


def log_versions(backend, path=None):
    """Log the environment and, if `path` is given, save it as YAML."""
    versions = environment_versions(backend)
    for name, version in versions.items():
        logger.info("%s: %s", name, version)
    if path is not None:
        save_environment(path, versions)
    return versions

# %% Notebook cell 6
# Show the log messages of the package in the cell outputs
logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True,
                    format='%(levelname)s %(name)s: %(message)s')

# Configuration = configs/config.yaml + the values of the first cell
cfg = load_config(overrides=merge_dicts({
    'framework': FRAMEWORK,
    'seed': SEED,
    'deterministic_ops': DETERMINISTIC_OPS,
    'output_dir': OUTPUT_DIR,
    'data': {'image_dir': IMAGE_DIR, 'mask_dir': MASK_DIR, 'size': SIZE},
    'train': {'epochs': EPOCHS, 'shuffle': SHUFFLE},
    'class_weighting': {'method': CLASS_WEIGHTING},
    'experiments': {'figure_run': FIGURE_RUN},
    'leave_one_embryo_out': {'enabled': RUN_LEAVE_ONE_EMBRYO_OUT},
}, CONFIG_OVERRIDES))
FIG_DIR = OUTPUT_DIR + '/figures'
os.makedirs(FIG_DIR, exist_ok=True)

# ==============================================================================
# zona_pellucida/data/loading.py
# Loading, pairing and preprocessing of the image / mask pairs.
# ==============================================================================


IMAGE_PATTERN = r"^sample(?P<sample>\d+)_frame_(?P<frame>\d+)\.png$"
MASK_PATTERN = r"^sample(?P<sample>\d+)_mask_frame_(?P<frame>\d+)\.png$"


@dataclass
class SegmentationDataset:
    """Preprocessed pairs; row i of every field describes the same pair."""

    images: np.ndarray  # (N, H, W, 1) float32 in [0, 1]
    masks: np.ndarray  # (N, H, W, 1) float32 in {0, 1}
    image_files: list
    mask_files: list
    samples: np.ndarray  # (N,) video / embryo id
    frames: np.ndarray  # (N,) frame number within the video

    def __len__(self):
        return len(self.images)

    def subset(self, indices):
        indices = np.asarray(indices, dtype=int)
        return SegmentationDataset(
            images=self.images[indices],
            masks=self.masks[indices],
            image_files=[self.image_files[i] for i in indices],
            mask_files=[self.mask_files[i] for i in indices],
            samples=self.samples[indices],
            frames=self.frames[indices],
        )


def parse_sample_frame(filename, pattern):
    """Return (sample, frame) from a filename, or None if it does not match."""
    match = re.match(pattern, filename)
    if match is None:
        return None
    return int(match.group("sample")), int(match.group("frame"))


def _index_by_key(filenames, pattern, kind):
    index = {}
    for name in filenames:
        key = parse_sample_frame(name, pattern)
        if key is None:
            logger.warning("Skipping %s file %s (name pattern)", kind, name)
            continue
        if key in index:
            raise ValueError(
                f"Two {kind} files for sample {key[0]}, frame {key[1]}: "
                f"{index[key]} and {name}"
            )
        index[key] = name
    return index


def pair_image_mask_files(image_filenames, mask_filenames,
                          image_pattern=IMAGE_PATTERN,
                          mask_pattern=MASK_PATTERN):
    """Pair images and masks on (sample, frame) number.

    Pairing on sorted-list position (zip) would silently shift every later
    pair if one file were missing, so any unmatched file is an error.
    Returns a list of (image_name, mask_name, sample, frame), sorted by
    sample and frame.
    """
    images = _index_by_key(image_filenames, image_pattern, "image")
    masks = _index_by_key(mask_filenames, mask_pattern, "mask")
    missing_masks = sorted(set(images) - set(masks))
    missing_images = sorted(set(masks) - set(images))
    if missing_masks or missing_images:
        raise ValueError(
            "Image/mask pairing failed. (sample, frame) without a mask: "
            f"{missing_masks}; without an image: {missing_images}"
        )
    pairs = []
    for key in sorted(images):
        image_key = parse_sample_frame(images[key], image_pattern)
        mask_key = parse_sample_frame(masks[key], mask_pattern)
        assert image_key == mask_key == key, (images[key], masks[key])
        pairs.append((images[key], masks[key], key[0], key[1]))
    return pairs


def _read_grayscale(path):
    array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if array is None:
        raise FileNotFoundError(f"Could not read image {path}")
    return array


def load_image(path, size):
    """Grayscale uint8 image resized to (size, size) with bicubic filtering."""
    image = Image.fromarray(_read_grayscale(path))
    image = image.resize((size, size), Image.BICUBIC)
    return np.array(image)


def binarize_mask(mask, threshold=0.5):
    """Map a 0-255 mask to {0, 1}: foreground where mask / 255 > threshold."""
    return (np.asarray(mask) / 255.0 > threshold).astype(np.uint8)


def load_mask(path, size, threshold=0.5):
    """Binary {0, 1} mask resized to (size, size).

    Nearest-neighbour resizing keeps the labels binary; the default
    (bicubic) filter creates intermediate values along the zona edges.
    """
    mask = Image.fromarray(_read_grayscale(path))
    mask = mask.resize((size, size), Image.NEAREST)
    return binarize_mask(np.array(mask), threshold)


def load_dataset(image_dir, mask_dir, size=512, image_pattern=IMAGE_PATTERN,
                 mask_pattern=MASK_PATTERN, mask_threshold=0.5):
    """Load, pair and preprocess every image / mask pair of two folders."""
    pairs = pair_image_mask_files(
        sorted(os.listdir(image_dir)), sorted(os.listdir(mask_dir)),
        image_pattern, mask_pattern,
    )
    if not pairs:
        raise ValueError(f"No image/mask pairs found in {image_dir}")
    images, masks = [], []
    for image_name, mask_name, _, _ in pairs:
        images.append(load_image(os.path.join(image_dir, image_name), size))
        masks.append(
            load_mask(os.path.join(mask_dir, mask_name), size, mask_threshold)
        )
    # Scale images to [0, 1] by the fixed 8-bit maximum (not per image).
    images = (np.stack(images).astype(np.float32) / 255.0)[..., np.newaxis]
    masks = np.stack(masks).astype(np.float32)[..., np.newaxis]
    dataset = SegmentationDataset(
        images=images,
        masks=masks,
        image_files=[p[0] for p in pairs],
        mask_files=[p[1] for p in pairs],
        samples=np.array([p[2] for p in pairs]),
        frames=np.array([p[3] for p in pairs]),
    )
    counts = dict(zip(*np.unique(dataset.samples, return_counts=True)))
    logger.info(
        "Loaded %d pairs, images %s, masks %s, pairs per sample %s",
        len(dataset), images.shape, masks.shape,
        {int(k): int(v) for k, v in counts.items()},
    )
    return dataset


def load_dataset_from_config(data_cfg):
    return load_dataset(
        data_cfg.image_dir, data_cfg.mask_dir, data_cfg.size,
        data_cfg.image_pattern, data_cfg.mask_pattern,
        data_cfg.mask_threshold,
    )

# %% Notebook cell 8
# Load, pair and preprocess all image / mask pairs
dataset = load_dataset_from_config(cfg.data)

image_dataset_normalized = dataset.images  # (N, SIZE, SIZE, 1), values in [0, 1]
mask_dataset = dataset.masks  # (N, SIZE, SIZE, 1), values in {0, 1}

print("Shape of image_dataset after normalization:", image_dataset_normalized.shape)
print("Shape of mask_dataset:", mask_dataset.shape)
print("Mask values:", np.unique(mask_dataset))

# %% Notebook cell 10
# Randomly select an image filename
image_name = random.Random(SEED).choice(dataset.image_files)
print('Image:', image_name)

# Load the original image
original_image = cv2.imread(os.path.join(IMAGE_DIR, image_name))

# Convert to grayscale
original_image_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Display the original grayscale image
plt.imshow(original_image_gray, cmap='gray')
plt.title('Original Grayscale Image (Before Preprocessing)')
plt.colorbar()
plt.savefig(FIG_DIR + '/original_image.png', dpi=200, bbox_inches='tight')
plt.show()

# ==============================================================================
# zona_pellucida/data/splitting.py
# Train / validation / test splits and their persistence.
# ==============================================================================


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

# %% Notebook cell 12
split = train_val_test_split(dataset.samples, test_size=cfg.split.test_size,
                             val_size=cfg.split.val_size, seed=cfg.seed,
                             stratify_by_sample=cfg.split.stratify_by_sample)
save_split(dataset, split, OUTPUT_DIR + '/split.csv')

X_train, y_train = dataset.images[split.train], dataset.masks[split.train]
X_val, y_val = dataset.images[split.val], dataset.masks[split.val]
X_test, y_test = dataset.images[split.test], dataset.masks[split.test]
print(split.sizes())

# ==== Class imbalance

# ==============================================================================
# zona_pellucida/data/class_balance.py
# Pixel-level class balance of the (binary) masks and class weights.
# ==============================================================================


def pixel_class_counts(masks, threshold=0.5):
    """Number of background (0) and foreground (1) pixels."""
    foreground = int(np.count_nonzero(np.asarray(masks) > threshold))
    total = int(np.asarray(masks).size)
    return {"background": total - foreground, "foreground": foreground,
            "total": total}


def foreground_fraction_per_image(masks, threshold=0.5):
    masks = np.asarray(masks) > threshold
    return masks.reshape(len(masks), -1).mean(axis=1)


def class_balance_summary(masks, threshold=0.5):
    """Overall class shares and per-image foreground-fraction statistics."""
    counts = pixel_class_counts(masks, threshold)
    fractions = foreground_fraction_per_image(masks, threshold)
    return {
        "n_images": len(fractions),
        "total_pixels": counts["total"],
        "background_pixels": counts["background"],
        "foreground_pixels": counts["foreground"],
        "background_percent": 100 * counts["background"] / counts["total"],
        "foreground_percent": 100 * counts["foreground"] / counts["total"],
        "background_to_foreground_ratio": (
            counts["background"] / max(counts["foreground"], 1)
        ),
        "fg_fraction_min": float(fractions.min()),
        "fg_fraction_mean": float(fractions.mean()),
        "fg_fraction_median": float(np.median(fractions)),
        "fg_fraction_max": float(fractions.max()),
        "fg_fraction_std": float(fractions.std()),
    }


def log_class_balance(summary):
    logger.info(
        "Background: %d px (%.2f%%), zona pellucida: %d px (%.2f%%), "
        "ratio %.1f:1",
        summary["background_pixels"], summary["background_percent"],
        summary["foreground_pixels"], summary["foreground_percent"],
        summary["background_to_foreground_ratio"],
    )
    logger.info(
        "Per-image foreground fraction over %d images: min %.4f, "
        "mean %.4f, max %.4f",
        summary["n_images"], summary["fg_fraction_min"],
        summary["fg_fraction_mean"], summary["fg_fraction_max"],
    )


def plot_class_balance(summary, fractions, save_path=None):
    """Bar chart of the class pixel shares + histogram of the per-image
    foreground fraction (training split)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    shares = [summary["background_percent"], summary["foreground_percent"]]
    bars = ax1.bar(["Background", "Zona pellucida"], shares,
                   color=["#9e9e9e", "#1f77b4"])
    ax1.bar_label(bars, fmt="%.2f%%")
    ax1.set_ylabel("Share of training pixels (%)")
    ax1.set_ylim(0, 105)
    ax1.set_title("Class pixel shares")
    ax2.hist(np.asarray(fractions) * 100, bins=20, color="#1f77b4",
             edgecolor="white")
    ax2.set_xlabel("Zona pellucida pixels per image (%)")
    ax2.set_ylabel("Number of images")
    ax2.set_title(f"Per-image foreground fraction (n={len(fractions)})")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        logger.info("Saved figure %s", save_path)
    plt.show()
    plt.close(fig)


def save_class_balance(masks, image_files, output_dir, threshold=0.5):
    """Write class_balance.csv, class_balance_per_image.csv and
    class_balance.png to `output_dir`; return the summary dict."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = class_balance_summary(masks, threshold)
    fractions = foreground_fraction_per_image(masks, threshold)
    log_class_balance(summary)
    pd.DataFrame(
        {"statistic": list(summary), "value": list(summary.values())}
    ).to_csv(output_dir / "class_balance.csv", index=False)
    pd.DataFrame(
        {"image_file": image_files, "foreground_fraction": fractions}
    ).to_csv(output_dir / "class_balance_per_image.csv", index=False)
    plot_class_balance(summary, fractions,
                       save_path=output_dir / "class_balance.png")
    return summary


CLASS_WEIGHT_METHODS = ("inverse_frequency", "median_frequency")


def compute_class_weights(masks, method="inverse_frequency", threshold=0.5):
    """Class weights {0: w_background, 1: w_zona} from pixel frequencies.

    inverse_frequency (default): w_c = N_total / (C * n_c), C = 2, so the
        weighted pixel count sum_c w_c * n_c equals N_total.
    median_frequency (SegNet, Eigen & Fergus): w_c = median_freq / freq_c,
        where freq_c = n_c / (pixels of the images that contain class c).
        With two classes present in every image this equals
        inverse_frequency; they differ only if some masks are empty.
    Must be computed on the training split only.
    """
    masks = np.asarray(masks) > threshold
    n_classes = 2
    per_image = masks.reshape(len(masks), -1)
    pixels_per_image = per_image.shape[1]
    n_fg = per_image.sum(axis=1)
    n_c = np.array([per_image.size - n_fg.sum(), n_fg.sum()], dtype=float)
    if np.any(n_c == 0):
        raise ValueError(f"A class has no pixels in the training masks: {n_c}")
    if method == "inverse_frequency":
        weights = n_c.sum() / (n_classes * n_c)
    elif method == "median_frequency":
        images_with_c = np.array(
            [np.sum(n_fg < pixels_per_image), np.sum(n_fg > 0)], dtype=float
        )
        freq = n_c / (images_with_c * pixels_per_image)
        weights = np.median(freq) / freq
    else:
        raise ValueError(
            f"Unknown class weighting '{method}', use {CLASS_WEIGHT_METHODS}"
        )
    class_weights = {0: float(weights[0]), 1: float(weights[1])}
    logger.info(
        "Class weights (%s): background %.4f, zona pellucida %.4f",
        method, class_weights[0], class_weights[1],
    )
    return class_weights

# %% Notebook cell 14
# Class balance of the training split: printed, plotted and saved to OUTPUT_DIR
class_balance = save_class_balance(y_train, [dataset.image_files[i] for i in split.train], OUTPUT_DIR)
print(pd.Series(class_balance).to_string())

# ==== Frequency-normalised class weighting

# %% Notebook cell 16
class_weights = compute_class_weights(y_train, method=cfg.class_weighting.method)
print(f"w_background = {class_weights[0]:.4f}, w_zona = {class_weights[1]:.4f}")

# Check: the weighted pixel count equals the total pixel count (inverse frequency)
n_zona = y_train.sum()
n_background = y_train.size - n_zona
print("weighted / total pixels:",
      (class_weights[0] * n_background + class_weights[1] * n_zona) / y_train.size)

# %% Notebook cell 18
#Check, view few images
image_number = random.Random(SEED).randint(0, len(X_train) - 1)
print('Image:', dataset.image_files[split.train[image_number]])
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(X_train[image_number], (SIZE, SIZE)), cmap='gray')
plt.subplot(122)
plt.imshow(np.reshape(y_train[image_number], (SIZE, SIZE)), cmap='gray')
plt.savefig(FIG_DIR + '/fig5_1_train_sample.png', dpi=200, bbox_inches='tight')
plt.show()

# ==============================================================================
# zona_pellucida/backends/keras/losses.py
# Weighted binary cross-entropy (Keras).
# ==============================================================================


@keras.saving.register_keras_serializable(package="zona_pellucida")
class WeightedBinaryCrossentropy(keras.losses.Loss):
    """Per-pixel weighted binary cross-entropy on sigmoid probabilities.

    loss = -[w1 * y * log(p) + w0 * (1 - y) * log(1 - p)], averaged over
    pixels and batch. The per-pixel cross-entropy is the one of the
    unweighted Keras baseline: inside `fit` and `evaluate` it is computed
    from the logits of the sigmoid layer (on predicted probabilities it
    clips them to [1e-7, 1 - 1e-7]), so the two runs differ only in the
    weights. Keras
    `class_weight=` does not support per-pixel (4-D) targets, hence a loss.
    """

    def __init__(self, w0=1.0, w1=1.0, name="weighted_binary_crossentropy",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.w0 = float(w0)
        self.w1 = float(w1)

    def call(self, y_true, y_pred):
        y_true = ops.cast(y_true, y_pred.dtype)
        # Element-wise -[y log(p) + (1 - y) log(1 - p)]
        bce = ops.binary_crossentropy(y_true, y_pred)
        weights = self.w1 * y_true + self.w0 * (1.0 - y_true)
        loss = weights * bce
        # Mean over the pixels of each image -> one value per image.
        return ops.mean(loss, axis=tuple(range(1, len(loss.shape))))

    def get_config(self):
        config = super().get_config()
        config.update(w0=self.w0, w1=self.w1)
        return config


def get_loss(run, class_weights=None):
    """Loss of an experiment run: 'unweighted' or 'weighted'."""
    if run == "unweighted":
        return keras.losses.BinaryCrossentropy()
    if run == "weighted":
        if class_weights is None:
            raise ValueError("The weighted run needs class weights")
        return WeightedBinaryCrossentropy(
            w0=class_weights[0], w1=class_weights[1]
        )
    raise ValueError(f"Unknown run '{run}', use 'unweighted' or 'weighted'")

# ==============================================================================
# zona_pellucida/backends/keras/metrics.py
# Imbalance-aware Keras metrics monitored during training.
# ==============================================================================


class _PixelConfusion(keras.metrics.Metric):
    """Accumulates foreground TP, FP and FN over all pixels seen.

    Predictions are binarised with `threshold`; the ground truth with 0.5.
    Only sums are used, so the metric is deterministic on the GPU.
    """

    def __init__(self, threshold=0.5, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = float(threshold)
        self.tp = self.add_variable(shape=(), initializer="zeros", name="tp")
        self.fp = self.add_variable(shape=(), initializer="zeros", name="fp")
        self.fn = self.add_variable(shape=(), initializer="zeros", name="fn")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = ops.cast(ops.greater(y_true, 0.5), "float32")
        y_pred = ops.cast(ops.greater(y_pred, self.threshold), "float32")
        self.tp.assign_add(ops.sum(y_true * y_pred))
        self.fp.assign_add(ops.sum((1.0 - y_true) * y_pred))
        self.fn.assign_add(ops.sum(y_true * (1.0 - y_pred)))

    def _ratio(self, numerator, denominator):
        # 0 when there is no foreground in either mask, as in torchmetrics
        # and zona_pellucida.evaluation.metrics.
        return ops.where(denominator > 0,
                         numerator / ops.maximum(denominator, 1.0), 0.0)

    def reset_state(self):
        for variable in (self.tp, self.fp, self.fn):
            variable.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update(threshold=self.threshold)
        return config


@keras.saving.register_keras_serializable(package="zona_pellucida")
class DiceCoefficient(_PixelConfusion):
    """Foreground Dice = 2TP / (2TP + FP + FN)."""

    def __init__(self, threshold=0.5, name="dice", **kwargs):
        super().__init__(threshold=threshold, name=name, **kwargs)

    def result(self):
        return self._ratio(2.0 * self.tp, 2.0 * self.tp + self.fp + self.fn)


@keras.saving.register_keras_serializable(package="zona_pellucida")
class ForegroundIoU(_PixelConfusion):
    """Foreground IoU = TP / (TP + FP + FN).

    Same value as keras.metrics.BinaryIoU(target_class_ids=[1]), whose
    confusion matrix uses scatter_nd; with op determinism enabled TensorFlow
    runs scatter_nd on the CPU, which slows down GPU training.
    """

    def __init__(self, threshold=0.5, name="iou", **kwargs):
        super().__init__(threshold=threshold, name=name, **kwargs)

    def result(self):
        return self._ratio(self.tp, self.tp + self.fp + self.fn)


def training_metrics(threshold=0.5):
    """Accuracy plus foreground IoU, Dice, precision and recall."""
    return [
        keras.metrics.BinaryAccuracy(name="accuracy", threshold=threshold),
        ForegroundIoU(threshold=threshold, name="iou"),
        DiceCoefficient(threshold=threshold, name="dice"),
        keras.metrics.Precision(thresholds=threshold, name="precision"),
        keras.metrics.Recall(thresholds=threshold, name="recall"),
    ]

# ==============================================================================
# zona_pellucida/backends/keras/model.py
# U-Net-style fully convolutional network (architecture of the thesis).
# ==============================================================================


def fcn_8(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, dropout=0.5):
    """Build the (uncompiled) FCN; the loss is chosen at compile time."""
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    # Contracting path
    conv1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)

    # Expanding Path
    up6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5)
    up6 = concatenate([up6, conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up6)
    conv6 = BatchNormalization()(conv6)

    up7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6)
    up7 = concatenate([up7, conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up7)
    conv7 = BatchNormalization()(conv7)

    up8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7)
    up8 = concatenate([up8, conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8)
    up9 = concatenate([up9, conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Dropout(dropout)(conv9)  # Adding dropout for regularization

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    return Model(inputs=inputs, outputs=outputs)

# ==============================================================================
# zona_pellucida/backends/keras/trainer.py
# Compiling, training, saving and reloading the Keras FCN.
# ==============================================================================


# Needed by keras.models.load_model for models trained with these objects.
CUSTOM_OBJECTS = {
    "WeightedBinaryCrossentropy": WeightedBinaryCrossentropy,
    "DiceCoefficient": DiceCoefficient,
    "ForegroundIoU": ForegroundIoU,
}


def compile_model(model, loss, learning_rate=0.001, threshold=0.5):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=training_metrics(threshold),
    )
    return model


def make_callbacks(train_cfg, log_dir=None, has_validation=True):
    callbacks = []
    early_stopping = None
    if has_validation:
        # Restores the weights of the best validation epoch at the end of
        # training. Without early stopping, patience = epochs means it never
        # stops, so all epochs are trained.
        stop = train_cfg.early_stopping
        early_stopping = EarlyStopping(
            monitor=train_cfg.monitor,
            mode=train_cfg.mode,
            patience=stop.patience if stop.enabled else train_cfg.epochs,
            start_from_epoch=stop.start_from_epoch if stop.enabled else 0,
            restore_best_weights=True,
            verbose=1,
        )
        callbacks.append(early_stopping)
    tensorboard = train_cfg.tensorboard.enabled and log_dir is not None
    if tensorboard and importlib.util.find_spec("tensorboard") is None:
        logger.warning("TensorBoard is not installed; logging disabled")
        tensorboard = False
    if tensorboard:
        callbacks.append(TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=train_cfg.tensorboard.histogram_freq,
            write_graph=True, update_freq="epoch",
        ))
    return callbacks, early_stopping


def shuffled_dataset(X, y, batch_size, seed):
    """Batches in a new (seeded, reproducible) order every epoch.

    Keras 3 `fit(..., shuffle=True)` on NumPy arrays repeats the same
    permutation in every epoch once a global seed is set, so the
    reshuffling is done with tf.data instead.
    """
    return (tf.data.Dataset.from_tensor_slices((X, y))
            .shuffle(len(X), seed=seed, reshuffle_each_iteration=True)
            .batch(batch_size))


def train_model(model, X_train, y_train, train_cfg, validation_data=None,
                log_dir=None, seed=None):
    """Fit the compiled model. Returns (history dict, best epoch, 1-based).

    Validation data (never the test set) selects the best epoch, whose
    weights are kept (and drives the optional early stopping). With
    `train_cfg.shuffle` the training set is reshuffled every epoch
    (reproducibly, from `seed`); otherwise the batch order is fixed.
    """
    callbacks, early_stopping = make_callbacks(
        train_cfg, log_dir, has_validation=validation_data is not None
    )
    if train_cfg.shuffle:
        # The dataset reshuffles itself; shuffle=False only silences the
        # Keras warning about shuffling a tf.data dataset.
        data = {"x": shuffled_dataset(X_train, y_train,
                                      train_cfg.batch_size, seed),
                "shuffle": False}
    else:
        data = {"x": X_train, "y": y_train,
                "batch_size": train_cfg.batch_size, "shuffle": False}
    history = model.fit(
        **data,
        epochs=train_cfg.epochs,
        verbose=1,
        validation_data=validation_data,
        callbacks=callbacks,
    )
    epochs_run = len(history.history["loss"])
    if early_stopping is not None and early_stopping.best_weights is not None:
        best_epoch = early_stopping.best_epoch + 1
    else:
        best_epoch = epochs_run
    logger.info("Trained %d epochs; weights of epoch %d kept",
                epochs_run, best_epoch)
    return history.history, best_epoch


def save_model(model, path):
    """Keras 3 needs the `.keras` extension for full-model saving."""
    path = Path(path)
    if path.suffix != ".keras":
        raise ValueError(f"Model path must end in .keras, got {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    logger.info("Saved model to %s", path)


def load_trained_model(path, compile=True):
    """Reload a saved model including the custom loss and metrics."""
    return keras.models.load_model(
        path, custom_objects=CUSTOM_OBJECTS, compile=compile
    )

# ==============================================================================
# zona_pellucida/backends/keras/backend.py
# TensorFlow / Keras implementation of the backend interface.
# ==============================================================================


class KerasBackend:
    """What the experiments need from a deep-learning framework."""

    name = "keras"
    model_suffix = ".keras"

    def prepare_device(self):
        """Avoid OOM errors: allocate GPU memory on demand. Must run
        before the GPU is initialised."""
        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as err:
                logger.warning("No memory growth on %s: %s", gpu, err)

    def start_run(self, seed, deterministic=True):
        """Fresh Keras session and seeds, so every run starts identically."""
        keras.backend.clear_session()
        seed_python(seed)
        # Seeds python `random`, NumPy and TensorFlow.
        keras.utils.set_random_seed(seed)
        if deterministic:
            tf.config.experimental.enable_op_determinism()
        logger.info("Seed %d (op determinism: %s)", seed, deterministic)

    def environment(self):
        build = tf.sysconfig.get_build_info()
        return {
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
            "gpus": [
                tf.config.experimental.get_device_details(d).get(
                    "device_name", d.name)
                for d in tf.config.list_physical_devices("GPU")
            ],
            # CUDA/cuDNN TensorFlow was built for; the libraries actually
            # loaded are the nvidia-* pip packages.
            "cuda_build": build.get("cuda_version"),
            "cudnn_build": build.get("cudnn_version"),
            "cuda_runtime": package_version("nvidia-cuda-runtime-cu12"),
            "cudnn_runtime": package_version("nvidia-cudnn-cu12"),
            "tf32": tf.config.experimental.tensor_float_32_execution_enabled(),
        }

    def build_model(self, cfg):
        size = cfg.data.size
        return fcn_8(size, size, 1, dropout=cfg.model.dropout)

    def summary(self, model):
        model.summary()

    def train(self, loss, class_weights, train, val, cfg, name, log_dir=None):
        """Train a new model; returns (model, history, best epoch, path)."""
        model = self.build_model(cfg)
        compile_model(model, get_loss(loss, class_weights),
                      learning_rate=cfg.train.learning_rate,
                      threshold=cfg.evaluation.threshold)
        history, best_epoch = train_model(
            model, train.images, train.masks, cfg.train,
            validation_data=(val.images, val.masks) if len(val) else None,
            log_dir=log_dir, seed=cfg.seed,
        )
        model_path = Path(cfg.output_dir) / "models" / f"{name}.keras"
        save_model(model, model_path)
        return model, history, best_epoch, model_path

    def predict(self, model, images, batch_size=16):
        """Sigmoid probabilities, shape (N, H, W, 1)."""
        return model.predict(images, batch_size=batch_size, verbose=0)

    def evaluate(self, model, images, masks, batch_size=16):
        """Loss and Keras metrics (threshold 0.5), evaluated once."""
        return model.evaluate(images, masks, batch_size=batch_size,
                              return_dict=True, verbose=0)

    def load_model(self, path):
        return load_trained_model(path)


backend = KerasBackend()

# %% Notebook cell 20
model = backend.build_model(cfg)
backend.summary(model)

# %% Notebook cell 21
# Python, framework, NumPy, CUDA/cuDNN versions and the GPU of this session
# (saved together with the results after training)
versions = log_versions(backend)

# %% Notebook cell 22
# Check data types
print(X_train.dtype, y_train.dtype)

# Check pixel values range
print(np.min(X_train), np.max(X_train))

# ==== Training: unweighted baseline and weighted model

# ==============================================================================
# zona_pellucida/evaluation/metrics.py
# Pixel-level segmentation metrics on binarised masks (NumPy).
# ==============================================================================


def binarize(array, threshold=0.5):
    return np.asarray(array) > threshold


def confusion_counts(y_true, y_pred):
    """TN, FP, FN, TP of two boolean arrays, pooled over all pixels."""
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    tp = int(np.count_nonzero(y_true & y_pred))
    fp = int(np.count_nonzero(~y_true & y_pred))
    fn = int(np.count_nonzero(y_true & ~y_pred))
    tn = int(y_true.size - tp - fp - fn)
    return tn, fp, fn, tp


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else 0.0


def dice_coefficient(y_true, y_pred, smooth=1):
    """Dice of one image, smoothed as in the thesis."""
    y_true_f = np.asarray(y_true, dtype=float).flatten()
    y_pred_f = np.asarray(y_pred, dtype=float).flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        np.sum(y_true_f) + np.sum(y_pred_f) + smooth
    )


def iou_coefficient(y_true, y_pred, smooth=1):
    """IoU of one image, smoothed like the Dice above."""
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    intersection = np.count_nonzero(y_true & y_pred)
    union = np.count_nonzero(y_true | y_pred)
    return (intersection + smooth) / (union + smooth)


def per_image_scores(y_true, y_pred, score=dice_coefficient):
    """Score of each image; both inputs are binarised arrays (N, H, W, 1)."""
    return np.array([score(t, p) for t, p in zip(y_true, y_pred)])


def segmentation_metrics(y_true, y_prob, threshold=0.5):
    """Test-set metrics of probabilities `y_prob` at `threshold`.

    The ground truth is binarised with > 0.5 first. Accuracy, precision,
    recall, F1 and IoU are pooled over every pixel of every image (F1 is
    then identical to the pooled Dice). `dice` and `iou_per_image` are the
    means (and `*_std` the standard deviations) of the per-image scores;
    the mean per-image Dice is the "Average Dice" of the thesis.
    """
    y_true = binarize(y_true, 0.5)
    y_pred = binarize(y_prob, threshold)
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    dice = per_image_scores(y_true, y_pred)
    iou = per_image_scores(y_true, y_pred, iou_coefficient)
    return {
        "threshold": float(threshold),
        "accuracy": _ratio(tp + tn, tp + tn + fp + fn),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "iou": _ratio(tp, tp + fp + fn),
        "dice": float(np.mean(dice)),
        "dice_std": float(np.std(dice)),
        "iou_per_image": float(np.mean(iou)),
        "iou_per_image_std": float(np.std(iou)),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }

# ==============================================================================
# zona_pellucida/evaluation/thresholds.py
# Decision-threshold selection on the validation set only.
# ==============================================================================


TUNING_METRICS = ("dice", "iou")


def threshold_grid(eval_cfg):
    # Rounded so that e.g. 0.5 is exactly 0.5 (linspace round-off).
    return np.round(np.linspace(eval_cfg.threshold_min,
                                eval_cfg.threshold_max,
                                eval_cfg.threshold_steps), 6)


def select_threshold(y_true, y_prob, thresholds, metric="dice"):
    """Threshold maximising the pooled Dice (= F1) or IoU.

    Ties go to the threshold closest to 0.5. Returns (threshold, table of
    the metric at every candidate threshold).
    """
    if metric not in TUNING_METRICS:
        raise ValueError(f"metric must be one of {TUNING_METRICS}")
    y_true = binarize(y_true, 0.5)
    rows = []
    for threshold in thresholds:
        _, fp, fn, tp = confusion_counts(y_true, binarize(y_prob, threshold))
        denominator = (2 * tp + fp + fn) if metric == "dice" else (
            tp + fp + fn)
        numerator = 2 * tp if metric == "dice" else tp
        score = numerator / denominator if denominator > 0 else 0.0
        rows.append({"threshold": float(threshold), metric: float(score)})
    table = pd.DataFrame(rows)
    best = table[table[metric] == table[metric].max()]
    best_threshold = float(
        best.loc[(best["threshold"] - 0.5).abs().idxmin(), "threshold"]
    )
    logger.info("Validation-selected threshold %.3f (%s %.4f)",
                best_threshold, metric, table[metric].max())
    return best_threshold, table

# ==============================================================================
# zona_pellucida/experiments.py
# Unweighted vs weighted runs and the leave-one-embryo-out check.
# ==============================================================================


RESULT_COLUMNS = [
    "framework", "run", "loss", "held_out_sample", "threshold_source", "threshold",
    "accuracy", "precision", "recall", "f1", "iou", "dice", "dice_std",
    "iou_per_image", "iou_per_image_std", "tn", "fp", "fn", "tp",
    "test_loss", "best_epoch",
    "epochs_trained", "w_background", "w_zona",
]


@dataclass
class RunResult:
    framework: str
    name: str
    loss: str
    model: object
    model_path: Path
    history: dict
    best_epoch: int
    class_weights: dict
    test_scores: dict
    test_metrics: list
    val_threshold: float = None
    threshold_table: pd.DataFrame = None
    held_out_sample: int = None


def tensorboard_log_dir(output_dir, run_name):
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(output_dir) / "logs" / "fit" / f"{run_name}-{stamp}"


def save_history(history, path):
    """Pickle the history dict (as in the thesis) and also write a CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(history, f)
    table = pd.DataFrame(history)
    table.index = table.index + 1
    table.to_csv(path.with_suffix(".csv"), index_label="epoch")
    logger.info("Saved training history to %s", path)


def load_history(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def run_experiment(loss, dataset, split, cfg, backend, name=None,
                   held_out_sample=None, keep_model=True):
    """Train one model from scratch and evaluate it on the test split.

    Every run starts from a fresh session seeded with `cfg.seed`, so the
    unweighted and weighted runs start from identical initial weights and
    see the same batches, whatever order they are run in.
    """
    name = name or loss
    output_dir = Path(cfg.output_dir)
    backend.start_run(cfg.seed, cfg.deterministic_ops)

    train = dataset.subset(split.train)
    val = dataset.subset(split.val)
    test = dataset.subset(split.test)
    logger.info("Run %s (%s): train %d, val %d, test %d images", name,
                backend.name, len(train), len(val), len(test))

    # Class weights always come from the training split only.
    class_weights = compute_class_weights(
        train.masks, cfg.class_weighting.method
    )
    model, history, best_epoch, model_path = run_with_determinism_check(
        backend.train, loss, class_weights, train, val, cfg, name,
        log_dir=tensorboard_log_dir(output_dir, name),
    )
    # Config and environment that produced this model, saved next to it.
    save_config(cfg, model_path.with_suffix(".config.yaml"))
    save_environment(model_path.with_suffix(".environment.yaml"),
                     environment_versions(backend))
    save_history(history, output_dir / "histories" / f"{name}.pkl")

    batch_size = cfg.train.batch_size
    thresholds = [("fixed", cfg.evaluation.threshold)]
    val_threshold, threshold_table = None, None
    if cfg.evaluation.tune_threshold and len(val):
        val_prob = run_with_determinism_check(
            backend.predict, model, val.images, batch_size)
        val_threshold, threshold_table = select_threshold(
            val.masks, val_prob,
            threshold_grid(cfg.evaluation), cfg.evaluation.tuning_metric,
        )
        thresholds.append(("val_tuned", val_threshold))
        threshold_path = output_dir / "thresholds" / f"{name}.csv"
        threshold_path.parent.mkdir(parents=True, exist_ok=True)
        threshold_table.to_csv(threshold_path, index=False)

    # Evaluate once; the returned dict names every value.
    test_scores = run_with_determinism_check(
        backend.evaluate, model, test.images, test.masks, batch_size)
    test_prob = run_with_determinism_check(
        backend.predict, model, test.images, batch_size)
    test_metrics = [
        segmentation_metrics(test.masks, test_prob, threshold)
        | {"threshold_source": source}
        for source, threshold in thresholds
    ]
    for row in test_metrics:
        logger.info(
            "Run %s test @%.2f (%s): acc %.4f, precision %.4f, recall %.4f,"
            " F1 %.4f, IoU %.4f, Dice %.4f", name, row["threshold"],
            row["threshold_source"], row["accuracy"], row["precision"],
            row["recall"], row["f1"], row["iou"], row["dice"],
        )
    return RunResult(
        framework=backend.name, name=name, loss=loss, model=model if keep_model else None,
        model_path=model_path,
        history=history, best_epoch=best_epoch, class_weights=class_weights,
        test_scores=test_scores, test_metrics=test_metrics,
        val_threshold=val_threshold, threshold_table=threshold_table,
        held_out_sample=held_out_sample,
    )


def results_table(results):
    """One row per run and threshold source."""
    rows = []
    for result in results:
        # Loss weights actually used (1 / 1 for the unweighted baseline).
        weights = (result.class_weights if result.loss == "weighted"
                   else {0: 1.0, 1: 1.0})
        for metrics in result.test_metrics:
            rows.append(metrics | {
                "framework": result.framework,
                "run": result.name,
                "loss": result.loss,
                "held_out_sample": result.held_out_sample,
                "test_loss": result.test_scores["loss"],
                "best_epoch": result.best_epoch,
                "epochs_trained": len(result.history["loss"]),
                "w_background": weights[0],
                "w_zona": weights[1],
            })
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def save_results(results, path):
    table = results_table(results)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    logger.info("Saved results to %s", path)
    return table


def run_comparison(dataset, split, cfg, backend):
    """Unweighted baseline vs weighted model, same seed and split."""
    results = {
        loss: run_experiment(loss, dataset, split, cfg, backend)
        for loss in cfg.experiments.runs
    }
    table = save_results(
        list(results.values()),
        Path(cfg.output_dir) / "results_comparison.csv",
    )
    return results, table


def run_leave_one_embryo_out(dataset, cfg, backend):
    """Train on one video, test on the other (robustness check only)."""
    folds = leave_one_embryo_out_splits(
        dataset.samples, cfg.split.val_size, cfg.seed
    )
    output_dir = Path(cfg.output_dir)
    results = []
    for held_out, split in folds.items():
        save_split(dataset, split,
                   output_dir / "splits" / f"loeo_test_sample{held_out}.csv")
        for loss in cfg.leave_one_embryo_out.runs:
            results.append(run_experiment(
                loss, dataset, split, cfg, backend,
                name=f"loeo_test_sample{held_out}_{loss}",
                held_out_sample=held_out, keep_model=False,
            ))
    table = save_results(results, output_dir / "leave_one_embryo_out.csv")
    return results, table


def summarize_leave_one_embryo_out(table):
    """Mean and std over the held-out videos, per loss and threshold."""
    metrics = ["accuracy", "precision", "recall", "f1", "iou", "dice"]
    return table.groupby(["loss", "threshold_source"])[metrics].agg(
        ["mean", "std"]
    )

# %% Notebook cell 24
# Unweighted baseline: binary cross-entropy (as in the thesis)
results = {}
results['unweighted'] = run_experiment('unweighted', dataset, split, cfg, backend)

# %% Notebook cell 25
# Weighted model: frequency-weighted binary cross-entropy, same seed and split
results['weighted'] = run_experiment('weighted', dataset, split, cfg, backend)

# %% Notebook cell 26
# Test-set comparison of both runs for Chapter 5 (saved to OUTPUT_DIR/results_comparison.csv)
comparison = save_results(list(results.values()), OUTPUT_DIR + '/results_comparison.csv')
# The configuration and environment that produced these results
save_config(cfg, OUTPUT_DIR + '/config_used.yaml')
log_versions(backend, OUTPUT_DIR + '/environment.yaml')
print(comparison[['run', 'threshold_source', 'threshold', 'accuracy', 'precision',
                  'recall', 'f1', 'iou', 'dice', 'best_epoch', 'epochs_trained']].to_string(index=False))

# %% Notebook cell 28
# Load the trained model and training history of FIGURE_RUN
model = backend.load_model(OUTPUT_DIR + f'/models/{FIGURE_RUN}{backend.model_suffix}')
history = load_history(OUTPUT_DIR + f'/histories/{FIGURE_RUN}.pkl')

comparison = pd.read_csv(OUTPUT_DIR + '/results_comparison.csv')
val_rows = comparison[(comparison['run'] == FIGURE_RUN) & (comparison['threshold_source'] == 'val_tuned')]
THRESHOLD = float(val_rows['threshold'].iloc[0]) if USE_VAL_THRESHOLD and len(val_rows) else cfg.evaluation.threshold
print("Decision threshold used below:", THRESHOLD)

# %% Notebook cell 30
# Evaluate the model once; each metric is returned by name
test_scores = backend.evaluate(model, X_test, y_test, batch_size=cfg.train.batch_size)
test_loss, test_acc = test_scores['loss'], test_scores['accuracy']
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)
print("Test IoU:", test_scores['iou'], " Dice:", test_scores['dice'],
      " Precision:", test_scores['precision'], " Recall:", test_scores['recall'])

# ==============================================================================
# zona_pellucida/plots.py
# Figures of the notebook; each can also be saved for the thesis.
# ==============================================================================


def _finish(fig, save_path=None):
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        logger.info("Saved figure %s", save_path)
    plt.show()
    plt.close(fig)


METRIC_LABELS = {"loss": "Loss", "accuracy": "Accuracy", "iou": "IoU",
                 "dice": "Dice", "precision": "Precision",
                 "recall": "Recall"}


def plot_training_curves(history, metric="loss", title=None, save_path=None,
                         best_epoch=None):
    """Training and validation curve of one metric per epoch."""
    label = METRIC_LABELS.get(metric, metric)
    values = history[metric]
    epochs = range(1, len(values) + 1)
    fig = plt.figure(figsize=(12, 6))
    plt.plot(epochs, values, 'y', label=f'Training {label.lower()}')
    if f'val_{metric}' in history:
        plt.plot(epochs, history[f'val_{metric}'], 'r',
                 label=f'Validation {label.lower()}')
    if best_epoch is not None:
        plt.axvline(best_epoch, color='grey', linestyle='--',
                    label=f'Best validation epoch ({best_epoch})')
    plt.title(title or f'Training and Validation {label}')
    plt.xlabel('Epochs')
    plt.ylabel(label)
    plt.legend()
    _finish(fig, save_path)


def plot_threshold_curve(table, metric, selected, save_path=None):
    """Validation metric as a function of the decision threshold."""
    fig = plt.figure(figsize=(8, 5))
    plt.plot(table["threshold"], table[metric], marker="o")
    plt.axvline(selected, color="r", linestyle="--",
                label=f"selected = {selected:.2f}")
    plt.axvline(0.5, color="grey", linestyle=":", label="0.5")
    plt.xlabel("Decision threshold")
    plt.ylabel(f"Validation {metric}")
    plt.title("Threshold selection on the validation set")
    plt.legend()
    _finish(fig, save_path)


def plot_sample_prediction(image, ground_truth, prediction, title=None,
                           panel_titles=('Testing Image', 'Ground Truth',
                                         'Prediction'),
                           save_path=None):
    """Image, ground-truth mask and predicted mask side by side."""
    fig = plt.figure(figsize=(16, 8) if title is None else (12, 6))
    if title is not None:
        plt.suptitle(title, fontsize=16)
    panels = zip((image, ground_truth, prediction), panel_titles)
    for i, (array, name) in enumerate(panels):
        plt.subplot(1, 3, i + 1)
        plt.title(name)
        plt.imshow(np.squeeze(array).astype(float), cmap='gray')
    _finish(fig, save_path)


def plot_prediction_overlay(image, prediction, alpha=0.1, save_path=None):
    """Original image with the predicted mask overlaid in white."""
    image = np.squeeze(image)
    # Set the overlay region to white where the prediction is positive
    overlay = np.zeros_like(image, dtype=float)
    overlay[np.squeeze(prediction) == 1] = 255
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray')
    # Overlay the prediction with transparency
    plt.imshow(overlay, cmap='gray', alpha=alpha)
    plt.title('Original Image with Prediction Overlay')
    plt.axis('off')
    _finish(fig, save_path)


def plot_confusion_matrix(conf_matrix, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(conf_matrix).plot(ax=ax)
    plt.title('Confusion Matrix')
    _finish(fig, save_path)


def plot_confusion_matrix_heatmap(conf_matrix, save_path=None):
    """Confusion matrix with seaborn for better visualization."""
    conf_matrix_df = pd.DataFrame(
        conf_matrix, index=['Actual Negative', 'Actual Positive'],
        columns=['Predicted Negative', 'Predicted Positive'])
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    _finish(fig, save_path)


def plot_distance_transform_overlay(image, ground_truth, prediction,
                                    alpha=0.3, save_path=None):
    """Prediction coloured by its distance transform over the image."""
    image = np.squeeze(image)
    prediction = np.squeeze(prediction).astype(np.uint8)
    # Compute the distance transform on the predicted mask
    dist_transform = distance_transform_edt(prediction)
    # Normalize the distance transform for better visualization
    dist_transform_normalized = cv2.normalize(
        dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Create a color map for overlay
    overlay = cv2.applyColorMap(dist_transform_normalized, cv2.COLORMAP_JET)
    # Convert the test image to uint8 and then to 3 channels
    test_img_uint8 = (image * 255).astype(np.uint8)
    test_img_color = cv2.cvtColor(test_img_uint8, cv2.COLOR_GRAY2BGR)
    # Overlay the distance transform on the original image
    mask = dist_transform_normalized > 0
    overlayed_img = test_img_color.copy()
    overlayed_img[mask] = cv2.addWeighted(
        test_img_color, 1 - alpha, overlay, alpha, 0)[mask]

    fig = plt.figure(figsize=(16, 8))
    panels = [(image, 'Testing Image'),
              (np.squeeze(ground_truth), 'Ground Truth'),
              (prediction, 'Prediction')]
    for i, (array, name) in enumerate(panels):
        plt.subplot(2, 4, i + 1)
        plt.title(name)
        plt.imshow(array.astype(float), cmap='gray')
    plt.subplot(244)
    plt.title('Distance Transform Overlay')
    # OpenCV images are BGR; matplotlib expects RGB.
    plt.imshow(cv2.cvtColor(overlayed_img, cv2.COLOR_BGR2RGB))
    _finish(fig, save_path)

# %% Notebook cell 31
#plot the training and validation accuracy, loss, IoU and Dice at each epoch (both runs)
# Figures 5.3 (accuracy) and 5.4 (loss); the dashed line marks the epoch whose weights were kept
figure_names = {'accuracy': 'fig5_3_accuracy', 'loss': 'fig5_4_loss', 'iou': 'iou', 'dice': 'dice'}
for run in ['unweighted', 'weighted']:
    run_history = load_history(OUTPUT_DIR + f'/histories/{run}.pkl')
    best_epoch = int(comparison.loc[comparison['run'] == run, 'best_epoch'].iloc[0])
    for metric, figure_name in figure_names.items():
        plot_training_curves(run_history, metric, best_epoch=best_epoch,
                             title=f'Training and Validation {METRIC_LABELS[metric]} ({run})',
                             save_path=FIG_DIR + f'/{figure_name}_{run}.png')

# ==== Decision threshold chosen on the validation set

# %% Notebook cell 33
if cfg.evaluation.tune_threshold and len(split.val):
    for run in ['unweighted', 'weighted']:
        table = pd.read_csv(OUTPUT_DIR + f'/thresholds/{run}.csv')
        selected = comparison.loc[(comparison['run'] == run) & (comparison['threshold_source'] == 'val_tuned'), 'threshold'].iloc[0]
        plot_threshold_curve(table, cfg.evaluation.tuning_metric, selected,
                             save_path=FIG_DIR + f'/{run}_threshold_selection.png')

# %% Notebook cell 35
#IOU
y_prob = backend.predict(model, X_test, cfg.train.batch_size)  # probabilities (N, SIZE, SIZE, 1)
y_pred_thresholded = binarize(y_prob, THRESHOLD)
y_test_binary = binarize(y_test, 0.5)  # ground truth binarised before the metrics

intersection = np.logical_and(y_test_binary, y_pred_thresholded)
union = np.logical_or(y_test_binary, y_pred_thresholded)
iou_score = np.sum(intersection) / np.sum(union)
print("IoU score is: ", iou_score)

# %% Notebook cell 36
# Calculate Dice coefficient for each test sample (dice_coefficient in zona_pellucida/evaluation/metrics.py)
dice_scores = per_image_scores(y_test_binary, y_pred_thresholded, dice_coefficient)
average_dice = np.mean(dice_scores)
print("Dice Coefficient for each test sample: ", np.round(dice_scores, 4).tolist())
print("Average Dice Coefficient: ", average_dice)

# %% Notebook cell 37
# Sort indices based on Dice coefficients
sorted_indices = np.argsort(dice_scores)
panel_titles = ('Original Image', 'Original Mask', 'Predicted Mask')
n_show = min(5, len(sorted_indices))

# Display a few samples with the lowest Dice coefficients
print("Samples with lowest Dice coefficients:")
for i in range(n_show):
    index = sorted_indices[i]
    plot_sample_prediction(X_test[index], y_test_binary[index], y_pred_thresholded[index],
                           title=f'Dice Coefficient: {dice_scores[index]:.4f}',
                           panel_titles=panel_titles)

# Display a few samples with the highest Dice coefficients
print("Samples with highest Dice coefficients:")
for i in range(n_show):
    index = sorted_indices[-(i + 1)]
    plot_sample_prediction(X_test[index], y_test_binary[index], y_pred_thresholded[index],
                           title=f'Dice Coefficient: {dice_scores[index]:.4f}',
                           panel_titles=panel_titles)

# %% Notebook cell 38
# Figures 5.6-5.8: the test images with the lowest, median and highest Dice coefficient
examples = {'fig5_6_lowest_dice': sorted_indices[0],
            'fig5_7_median_dice': sorted_indices[len(sorted_indices) // 2],
            'fig5_8_highest_dice': sorted_indices[-1]}
for figure_name, index in examples.items():
    print(f"{figure_name}: {dataset.image_files[split.test[index]]}, Dice {dice_scores[index]:.4f}")
    plot_sample_prediction(X_test[index], y_test_binary[index], y_pred_thresholded[index],
                           title=f'Dice Coefficient: {dice_scores[index]:.4f}',
                           panel_titles=panel_titles,
                           save_path=FIG_DIR + f'/{figure_name}.png')

# %% Notebook cell 40
# Randomly select four images from the test set (Figure 5.9)
for k, test_img_number in enumerate(random.Random(SEED).sample(range(len(X_test)), min(4, len(X_test))), start=1):
    print(f'fig5_9_random_test_{k}:', dataset.image_files[split.test[test_img_number]])
    test_img = X_test[test_img_number]
    ground_truth = y_test[test_img_number]

    # Add the batch dimension: (1, SIZE, SIZE, 1)
    test_img_input = test_img[None, ...]

    # Perform prediction
    prediction = (backend.predict(model, test_img_input)[0, :, :, 0] > THRESHOLD).astype(np.uint8)

    # Visualize the image, ground truth mask, and prediction
    plot_sample_prediction(test_img, ground_truth, prediction,
                           save_path=FIG_DIR + f'/fig5_9_random_test_{k}.png')

# %% Notebook cell 41
# Randomly select an image from the test set
random_index = random.Random(SEED + 1).randint(0, len(X_test) - 1)
print('fig5_10_prediction_overlay:', dataset.image_files[split.test[random_index]])
test_img = X_test[random_index]

# Perform prediction
prediction = (backend.predict(model, test_img[None, ...])[0, :, :, 0] > THRESHOLD).astype(np.uint8)

# Overlay the prediction on the original image (white, transparency alpha)
plot_prediction_overlay(test_img, prediction, alpha=0.1, save_path=FIG_DIR + '/fig5_10_prediction_overlay.png')

# %% Notebook cell 42
# Confusion matrix over all test pixels (ground truth binarised)
tn, fp, fn, tp = confusion_counts(y_test_binary, y_pred_thresholded)
conf_matrix = np.array([[tn, fp], [fn, tp]])
plot_confusion_matrix(conf_matrix, save_path=FIG_DIR + '/confusion_matrix_display.png')

# Print the confusion matrix values
print(f'True Negatives (TN): {tn}')
print(f'False Positives (FP): {fp}')
print(f'False Negatives (FN): {fn}')
print(f'True Positives (TP): {tp}')

# Metrics from the confusion matrix (zona pellucida = positive class)
test_metrics = segmentation_metrics(y_test, y_prob, THRESHOLD)
print(f"Accuracy: {test_metrics['accuracy']:.4f}")
print(f"Precision: {test_metrics['precision']:.4f}")
print(f"Recall: {test_metrics['recall']:.4f}")
print(f"F1 Score: {test_metrics['f1']:.4f}")
print(f"IoU: {test_metrics['iou']:.4f}")
print(f"Per-image Dice (mean ± SD): {test_metrics['dice']:.4f} ± {test_metrics['dice_std']:.4f}")
print(f"Per-image IoU (mean ± SD): {test_metrics['iou_per_image']:.4f} ± {test_metrics['iou_per_image_std']:.4f}")

# %% Notebook cell 43
# Plotting the confusion matrix using seaborn for better visualization (Figure 5.5)
plot_confusion_matrix_heatmap(conf_matrix, save_path=FIG_DIR + '/fig5_5_confusion_matrix.png')

# %% Notebook cell 44
# Inspect the saved training history of FIGURE_RUN (.pkl; a .csv copy is saved next to it)
file_path = OUTPUT_DIR + f'/histories/{FIGURE_RUN}.pkl'
with open(file_path, 'rb') as f:
    data = pickle.load(f)

print("Keys:", data.keys())
for key, value in data.items():
    print(f"{key}: {len(value)} entries")
print("Loss (first 5 entries):", data['loss'][:5])

# %% Notebook cell 45
# Randomly select an image from the test set
test_img_number = random.Random(SEED + 2).randint(0, len(X_test) - 1)
print('distance_transform_overlay:', dataset.image_files[split.test[test_img_number]])
test_img = X_test[test_img_number]
ground_truth = y_test[test_img_number]

# Perform prediction
prediction = (backend.predict(model, test_img[None, ...])[0, :, :, 0] > THRESHOLD).astype(np.uint8)

# Image, ground truth, prediction and the distance transform of the prediction
plot_distance_transform_overlay(test_img, ground_truth, prediction, alpha=0.3,
                                save_path=FIG_DIR + '/distance_transform_overlay.png')

# ==== Leave-one-embryo-out (robustness check)

# %% Notebook cell 47
if RUN_LEAVE_ONE_EMBRYO_OUT:
    loeo_results, loeo_table = run_leave_one_embryo_out(dataset, cfg, backend)
    print(loeo_table[['run', 'held_out_sample', 'threshold_source', 'threshold', 'accuracy',
                      'precision', 'recall', 'f1', 'iou', 'dice']].to_string())
    loeo_summary = summarize_leave_one_embryo_out(loeo_table)
    print(loeo_summary.to_string())
