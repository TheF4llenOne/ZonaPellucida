"""Decision-threshold selection on the validation set only."""

import logging

import numpy as np
import pandas as pd

from zona_pellucida.evaluation.metrics import binarize, confusion_counts

logger = logging.getLogger(__name__)

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
    logger.info("Validation-selected threshold %.3f (pooled %s %.4f)",
                best_threshold, metric, table[metric].max())
    return best_threshold, table
