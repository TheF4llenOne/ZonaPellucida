from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from zona_pellucida.data.loading import (
    SegmentationDataset,
    load_dataset_from_config,
)
from zona_pellucida.data.splitting import train_val_test_split
from zona_pellucida.evaluation.metrics import segmentation_metrics
from zona_pellucida.experiments import (
    RESULT_COLUMNS,
    load_history,
    per_image_table,
    run_comparison,
    run_cross_validation,
    summarize_by_time,
)


@pytest.mark.slow
def test_comparison_writes_results_and_reloadable_models(small_cfg, backend):
    dataset = load_dataset_from_config(small_cfg.data)
    split = train_val_test_split(dataset.samples, seed=small_cfg.seed)
    results, table = run_comparison(dataset, split, small_cfg, backend)

    assert list(table.columns) == RESULT_COLUMNS
    assert set(table["run"]) == {"unweighted", "weighted"}
    assert set(table["threshold_source"]) == {"fixed", "val_tuned"}
    saved = pd.read_csv(f"{small_cfg.output_dir}/results_comparison.csv")
    assert len(saved) == 4

    weighted = results["weighted"]
    assert weighted.class_weights[1] > 1 > weighted.class_weights[0]
    assert weighted.model_path.suffix == backend.model_suffix
    per_image = pd.read_csv(f"{small_cfg.output_dir}/per_image/weighted.csv")
    assert len(per_image) == 2 * len(split.test)
    fixed = per_image[per_image["threshold_source"] == "fixed"]
    row = table[(table["run"] == "weighted")
                & (table["threshold_source"] == "fixed")].iloc[0]
    assert fixed["dice"].mean() == pytest.approx(row["dice"])
    by_time = pd.read_csv(f"{small_cfg.output_dir}/results_by_time.csv")
    assert set(by_time["frame"]) == set(dataset.frames[split.test].tolist())
    reloaded = backend.load_model(weighted.model_path)
    x = dataset.images[split.test[:2]]
    np.testing.assert_allclose(backend.predict(reloaded, x),
                               backend.predict(weighted.model, x),
                               atol=1e-6)
    history = load_history(
        f"{small_cfg.output_dir}/histories/weighted.pkl")
    assert {"loss", "val_loss", "iou", "val_iou", "dice"} <= set(history)
    assert len(history["val_iou"]) == small_cfg.train.epochs
    # Config and environment are saved next to every model.
    for suffix in (".config.yaml", ".environment.yaml"):
        assert weighted.model_path.with_suffix(suffix).exists()


@pytest.mark.slow
def test_evaluate_returns_named_scores(small_cfg, backend):
    dataset = load_dataset_from_config(small_cfg.data)
    split = train_val_test_split(dataset.samples, seed=small_cfg.seed)
    small_cfg.experiments.runs = ["weighted"]
    results, _ = run_comparison(dataset, split, small_cfg, backend)
    scores = results["weighted"].test_scores
    assert {"loss", "accuracy", "iou", "dice", "precision",
            "recall"} <= set(scores)


@pytest.mark.slow
def test_runs_are_reproducible(small_cfg, backend):
    dataset = load_dataset_from_config(small_cfg.data)
    split = train_val_test_split(dataset.samples, seed=small_cfg.seed)
    small_cfg.experiments.runs = ["weighted"]
    _, first = run_comparison(dataset, split, small_cfg, backend)
    _, second = run_comparison(dataset, split, small_cfg, backend)
    pd.testing.assert_frame_equal(first, second)


@pytest.mark.slow
def test_cross_validation(small_cfg, backend):
    dataset = load_dataset_from_config(small_cfg.data)
    small_cfg.cross_validation.runs = ["unweighted"]
    results, table = run_cross_validation(dataset, small_cfg, backend)
    assert sorted(table["fold"].unique()) == [1, 2, 3, 4, 5]
    # Every embryo is tested exactly once
    tested = [int(s) for r in results for s in r.test_samples.split()]
    assert sorted(tested) == sorted(np.unique(dataset.samples).tolist())
    size = small_cfg.data.size
    total = 0
    for result in results:
        m = result.test_metrics[0]
        ids = [int(s) for s in result.test_samples.split()]
        n_test = int(np.isin(dataset.samples, ids).sum())
        assert m["tn"] + m["fp"] + m["fn"] + m["tp"] == n_test * size ** 2
        total += n_test
    assert total == len(dataset)
    assert all(r.model is None for r in results)
    outputs = Path(small_cfg.output_dir)
    for name in ("cross_validation.csv", "cross_validation_summary.csv",
                 "cross_validation_by_time.csv", "splits/cv_fold1.csv"):
        assert (outputs / name).exists(), name
    summary = pd.read_csv(outputs / "cross_validation_summary.csv")
    assert {"dice_mean", "dice_std"} <= set(summary.columns)


def test_results_table_reports_the_loss_weights_used():
    from zona_pellucida.experiments import RunResult, results_table

    rows = [{"threshold": 0.5, "threshold_source": "fixed"}]
    common = dict(framework="pytorch", model=None, model_path=None,
                  history={"loss": [1.0]}, best_epoch=1,
                  class_weights={0: 0.5, 1: 10.0}, test_scores={"loss": 1.0},
                  test_metrics=rows, fold=None, test_samples="101 102")
    table = results_table([RunResult(name="u", loss="unweighted", **common),
                           RunResult(name="w", loss="weighted", **common)])
    assert table["w_zona"].tolist() == [1.0, 10.0]
    assert table["test_samples"].tolist() == ["101 102"] * 2


def _tiny_dataset():
    masks = np.zeros((2, 4, 4, 1), np.float32)
    masks[0, :2] = 1
    masks[1, :, :2] = 1
    return SegmentationDataset(
        images=np.zeros_like(masks), masks=masks,
        image_files=["embryo101_25h.png", "embryo101_69h.png"],
        mask_files=["embryo101_25h_mask.png", "embryo101_69h_mask.png"],
        samples=np.array([101, 101]), frames=np.array([25, 69]))


def test_per_image_table_matches_the_results_metrics():
    dataset = _tiny_dataset()
    probabilities = np.where(dataset.masks > 0, 0.9, 0.1).astype(np.float32)
    probabilities[1, 0, 2] = 0.9  # a false positive in the second image
    table = per_image_table(dataset, probabilities,
                            [("fixed", 0.5), ("val_tuned", 0.95)])
    assert len(table) == 4
    fixed = table[table["threshold_source"] == "fixed"]
    assert fixed["frame"].tolist() == [25, 69]
    assert fixed["sample"].tolist() == [101, 101]
    metrics = segmentation_metrics(dataset.masks, probabilities, 0.5)
    assert fixed["dice"].mean() == pytest.approx(metrics["dice"])
    assert fixed["iou"].mean() == pytest.approx(metrics["iou_per_image"])
    assert fixed["dice"].iloc[0] == pytest.approx(1.0)


def test_summarize_by_time():
    per_image = pd.DataFrame({
        "loss": ["weighted"] * 4, "threshold_source": ["fixed"] * 4,
        "frame": [25, 25, 69, 117], "dice": [0.8, 0.6, 0.9, 0.5],
        "iou": [0.7, 0.5, 0.8, 0.4]})
    table = summarize_by_time(per_image)
    assert table["frame"].tolist() == [25, 69, 117]
    assert table["n"].tolist() == [2, 1, 1]
    assert table["dice"].tolist() == pytest.approx([0.7, 0.9, 0.5])
    # Population SD, like dice_std of the result tables
    assert table["dice_std"].tolist() == pytest.approx([0.1, 0.0, 0.0])
