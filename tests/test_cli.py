import pandas as pd
import pytest
import yaml

from zona_pellucida.bin.run_experiments import main
from zona_pellucida.config import DEFAULT_CONFIG_PATH

pytestmark = pytest.mark.slow


def test_command_line_run(framework, synthetic_dirs, tmp_path):
    image_dir, mask_dir = synthetic_dirs
    cfg = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["framework"] = framework
    cfg["output_dir"] = str(tmp_path / "outputs")
    cfg["data"].update(image_dir=str(image_dir), mask_dir=str(mask_dir),
                       size=32)
    cfg["train"].update(epochs=2, batch_size=4)
    cfg["train"]["tensorboard"]["enabled"] = False
    cfg["cross_validation"]["enabled"] = False
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    main(config_path)

    # Everything goes to <output_dir>/<framework>.
    outputs = tmp_path / "outputs" / framework
    table = pd.read_csv(outputs / "results_comparison.csv")
    assert set(table["framework"]) == {framework}
    for name in ("environment.yaml", "config_used.yaml", "split.csv",
                 "class_balance.png"):
        assert (outputs / name).exists(), name


def test_command_line_runs_single_folds(framework, synthetic_dirs, tmp_path):
    """--cv-fold: only those cross-validation folds, nothing else."""
    image_dir, mask_dir = synthetic_dirs
    cfg = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["framework"] = framework
    cfg["output_dir"] = str(tmp_path / "outputs")
    cfg["data"].update(image_dir=str(image_dir), mask_dir=str(mask_dir),
                       size=32)
    cfg["train"].update(epochs=2, batch_size=4)
    cfg["train"]["tensorboard"]["enabled"] = False
    cfg["cross_validation"].update(enabled=False, folds=2,
                                   runs=["unweighted"])
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    outputs = tmp_path / "outputs" / framework
    main(config_path, cv_folds=[2])
    table = pd.read_csv(outputs / "cross_validation.csv")
    assert table["fold"].unique().tolist() == [2]
    assert (outputs / "cross_validation" / "fold2.csv").exists()
    # The outputs of the main run are left alone
    for name in ("results_comparison.csv", "config_used.yaml",
                 "environment.yaml", "split.csv"):
        assert not (outputs / name).exists(), name

    main(config_path, cv_folds=[1])
    table = pd.read_csv(outputs / "cross_validation.csv")
    assert table["fold"].unique().tolist() == [1, 2]
