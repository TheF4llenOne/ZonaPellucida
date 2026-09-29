"""Experiment configuration: a YAML file mapped onto typed dataclasses."""

import copy
import dataclasses
import logging
import typing
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

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
    shuffle: bool = True
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


def overlapping_keys(base, overrides, prefix=""):
    """Dotted keys that are set in both nested dicts."""
    keys = []
    for key, value in overrides.items():
        if key not in base:
            continue
        if isinstance(value, dict) and isinstance(base[key], dict):
            keys += overlapping_keys(base[key], value, f"{prefix}{key}.")
        else:
            keys.append(f"{prefix}{key}")
    return keys


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
