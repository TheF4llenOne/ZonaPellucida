import pytest

from zona_pellucida.config import Config, load_config, merge_dicts
from zona_pellucida.data.loading import IMAGE_PATTERN, MASK_PATTERN


def test_yaml_matches_dataclass_defaults():
    # The single-file export has no YAML and relies on the defaults.
    assert load_config() == Config()


def test_default_patterns_match_loader():
    assert Config().data.image_pattern == IMAGE_PATTERN
    assert Config().data.mask_pattern == MASK_PATTERN


def test_overrides_are_merged_recursively():
    cfg = load_config(overrides={"seed": 7, "train": {"epochs": 3}})
    assert cfg.seed == 7
    assert cfg.train.epochs == 3
    assert cfg.train.batch_size == 16
    assert cfg.train.monitor == "val_iou"
    assert cfg.train.early_stopping.enabled is False


def test_unknown_key_raises():
    with pytest.raises(KeyError, match="train"):
        load_config(overrides={"train": {"epoch": 3}})


def test_missing_explicit_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_merge_dicts_does_not_modify_inputs():
    base = {"a": {"b": 1}}
    merged = merge_dicts(base, {"a": {"c": 2}})
    assert merged == {"a": {"b": 1, "c": 2}}
    assert base == {"a": {"b": 1}}


def test_load_backend_rejects_unknown_framework():
    from zona_pellucida.backends import load_backend

    with pytest.raises(ValueError, match="tensorflow"):
        load_backend("tensorflow")


def test_overlapping_keys():
    from zona_pellucida.config import overlapping_keys

    base = {"seed": 1, "train": {"epochs": 100, "shuffle": True}}
    assert overlapping_keys(base, {"train": {"batch_size": 8}}) == []
    assert overlapping_keys(base, {"seed": 2, "train": {"epochs": 5}}) == [
        "seed", "train.epochs"]
