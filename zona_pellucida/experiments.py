"""Unweighted vs weighted runs and the cross-validation by embryo.

Framework-neutral: the model is built, trained, saved and applied by a
`backend` (PyTorch Lightning or Keras, see zona_pellucida/backends).
"""

import datetime
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from zona_pellucida.config import save_config
from zona_pellucida.data.class_balance import compute_class_weights
from zona_pellucida.data.splitting import (
    cross_validation_splits,
    save_split,
)
from zona_pellucida.evaluation.metrics import (
    binarize,
    dice_coefficient,
    iou_coefficient,
    per_image_scores,
    segmentation_metrics,
)
from zona_pellucida.evaluation.thresholds import (
    select_threshold,
    threshold_grid,
)
from zona_pellucida.reproducibility import (
    environment_versions,
    run_with_determinism_check,
    save_environment,
)

logger = logging.getLogger(__name__)

RESULT_COLUMNS = [
    "framework", "run", "loss", "fold", "test_samples", "threshold_source",
    "threshold", "accuracy", "precision", "recall", "f1", "iou", "dice",
    "dice_std", "iou_per_image", "iou_per_image_std", "tn", "fp", "fn", "tp",
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
    fold: int = None
    test_samples: str = ""
    per_image: pd.DataFrame = None


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


def per_image_table(dataset, probabilities, thresholds):
    """Dice and IoU of every image at every (source, threshold) pair.

    Their means are the `dice` and `iou_per_image` columns of the results.
    """
    y_true = binarize(dataset.masks, 0.5)
    tables = []
    for source, threshold in thresholds:
        y_pred = binarize(probabilities, threshold)
        tables.append(pd.DataFrame({
            "image_file": dataset.image_files,
            "sample": dataset.samples,
            "frame": dataset.frames,
            "threshold_source": source,
            "threshold": float(threshold),
            "dice": per_image_scores(y_true, y_pred, dice_coefficient),
            "iou": per_image_scores(y_true, y_pred, iou_coefficient),
        }))
    return pd.concat(tables, ignore_index=True)


def summarize_by_time(per_image):
    """Per loss, threshold source and time point (`frame`, in hours): the
    number of images and the mean and (population) standard deviation of
    the per-image Dice and IoU."""
    def spread(values):
        return values.std(ddof=0)

    return (per_image.groupby(["loss", "threshold_source", "frame"])
            .agg(n=("dice", "size"), dice=("dice", "mean"),
                 dice_std=("dice", spread), iou=("iou", "mean"),
                 iou_std=("iou", spread))
            .reset_index())


def save_time_summary(results, path):
    """Write summarize_by_time of the per-image scores of `results`."""
    table = summarize_by_time(
        pd.concat([r.per_image for r in results], ignore_index=True))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    logger.info("Saved the scores per time point to %s", path)
    return table


def run_experiment(loss, dataset, split, cfg, backend, name=None,
                   fold=None, keep_model=True):
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
            " F1 %.4f, IoU %.4f, Dice (per-image mean) %.4f", name,
            row["threshold"],
            row["threshold_source"], row["accuracy"], row["precision"],
            row["recall"], row["f1"], row["iou"], row["dice"],
        )
    per_image = per_image_table(test, test_prob, thresholds)
    per_image.insert(0, "run", name)
    per_image.insert(1, "loss", loss)
    per_image.insert(2, "fold", fold)
    per_image_path = output_dir / "per_image" / f"{name}.csv"
    per_image_path.parent.mkdir(parents=True, exist_ok=True)
    per_image.to_csv(per_image_path, index=False)
    test_samples = " ".join(str(s) for s in np.unique(test.samples))
    return RunResult(
        framework=backend.name, name=name, loss=loss, model=model if keep_model else None,
        model_path=model_path,
        history=history, best_epoch=best_epoch, class_weights=class_weights,
        test_scores=test_scores, test_metrics=test_metrics,
        val_threshold=val_threshold, threshold_table=threshold_table,
        fold=fold, test_samples=test_samples, per_image=per_image,
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
                "fold": result.fold,
                "test_samples": result.test_samples,
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
    save_time_summary(list(results.values()),
                      Path(cfg.output_dir) / "results_by_time.csv")
    return results, table


def run_cross_validation(dataset, cfg, backend):
    """k-fold cross-validation by embryo: every embryo is tested once.

    A robustness check next to the main split, whose test metrics depend
    on which embryos were drawn for testing.
    """
    folds = cross_validation_splits(
        dataset.samples, cfg.cross_validation.folds, cfg.split.val_size,
        cfg.seed, cfg.split.group_by_sample,
    )
    output_dir = Path(cfg.output_dir)
    results = []
    for fold, split in folds.items():
        save_split(dataset, split, output_dir / "splits" / f"cv_fold{fold}.csv")
        for loss in cfg.cross_validation.runs:
            results.append(run_experiment(
                loss, dataset, split, cfg, backend,
                name=f"cv_fold{fold}_{loss}", fold=fold, keep_model=False,
            ))
    table = save_results(results, output_dir / "cross_validation.csv")
    summarize_cross_validation(table).to_csv(
        output_dir / "cross_validation_summary.csv", index=False)
    save_time_summary(results, output_dir / "cross_validation_by_time.csv")
    return results, table


def summarize_cross_validation(table):
    """Mean and standard deviation over the folds, per loss and threshold."""
    metrics = ["accuracy", "precision", "recall", "f1", "iou", "dice"]
    summary = table.groupby(["loss", "threshold_source"])[metrics].agg(
        ["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()
