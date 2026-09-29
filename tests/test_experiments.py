import numpy as np
import pandas as pd
import pytest

from zona_pellucida.data.loading import load_dataset_from_config
from zona_pellucida.data.splitting import train_val_test_split
from zona_pellucida.experiments import (
    RESULT_COLUMNS,
    load_history,
    run_comparison,
    run_leave_one_embryo_out,
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
def test_leave_one_embryo_out(small_cfg, backend):
    dataset = load_dataset_from_config(small_cfg.data)
    small_cfg.leave_one_embryo_out.runs = ["unweighted"]
    results, table = run_leave_one_embryo_out(dataset, small_cfg, backend)
    assert sorted(table["held_out_sample"].unique()) == [1, 2]
    # Each fold is evaluated on exactly the frames of its held-out video.
    size = small_cfg.data.size
    for result in results:
        m = result.test_metrics[0]
        n_test = int(np.sum(dataset.samples == result.held_out_sample))
        assert m["tn"] + m["fp"] + m["fn"] + m["tp"] == n_test * size ** 2
    assert all(r.model is None for r in results)


def test_results_table_reports_the_loss_weights_used():
    from zona_pellucida.experiments import RunResult, results_table

    rows = [{"threshold": 0.5, "threshold_source": "fixed"}]
    common = dict(framework="pytorch", model=None, model_path=None,
                  history={"loss": [1.0]}, best_epoch=1,
                  class_weights={0: 0.5, 1: 10.0}, test_scores={"loss": 1.0},
                  test_metrics=rows)
    table = results_table([RunResult(name="u", loss="unweighted", **common),
                           RunResult(name="w", loss="weighted", **common)])
    assert table["w_zona"].tolist() == [1.0, 10.0]
