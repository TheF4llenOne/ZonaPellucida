import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from zona_pellucida.evaluation.metrics import (
    confusion_counts,
    dice_coefficient,
    segmentation_metrics,
)


def _data(seed=0):
    rng = np.random.default_rng(seed)
    y_true = (rng.random((4, 16, 16, 1)) > 0.85).astype(np.float32)
    y_prob = np.clip(y_true * 0.6 + rng.random(y_true.shape) * 0.5, 0, 1)
    return y_true, y_prob.astype(np.float32)


def test_confusion_counts_match_sklearn():
    y_true, y_prob = _data()
    y_pred = y_prob > 0.5
    expected = confusion_matrix(y_true.ravel(), y_pred.ravel()).ravel()
    assert confusion_counts(y_true > 0.5, y_pred) == tuple(expected)


def test_segmentation_metrics_match_sklearn():
    y_true, y_prob = _data()
    t, p = y_true.ravel(), (y_prob > 0.5).ravel()
    metrics = segmentation_metrics(y_true, y_prob, 0.5)
    assert metrics["accuracy"] == pytest.approx(accuracy_score(t, p))
    assert metrics["precision"] == pytest.approx(precision_score(t, p))
    assert metrics["recall"] == pytest.approx(recall_score(t, p))
    assert metrics["f1"] == pytest.approx(f1_score(t, p))
    tp, fp, fn = metrics["tp"], metrics["fp"], metrics["fn"]
    assert metrics["iou"] == pytest.approx(tp / (tp + fp + fn))


def test_ground_truth_is_binarised():
    # Grey edge values (e.g. 0.3) must count as background, not zona.
    y_true = np.array([1.0, 0.3, 0.0, 0.7]).reshape(1, 2, 2, 1)
    y_prob = np.array([0.9, 0.1, 0.1, 0.9]).reshape(1, 2, 2, 1)
    metrics = segmentation_metrics(y_true, y_prob, 0.5)
    assert metrics["iou"] == 1.0
    assert (metrics["fp"], metrics["fn"]) == (0, 0)


def test_mean_dice_uses_thesis_formula():
    y_true, y_prob = _data()
    y_pred = y_prob > 0.5
    expected = np.mean([dice_coefficient(t, p)
                        for t, p in zip(y_true, y_pred)])
    assert segmentation_metrics(y_true, y_prob)["dice"] == pytest.approx(
        expected)


def test_per_image_standard_deviation():
    y_true, y_prob = _data()
    y_pred = y_prob > 0.5
    scores = [dice_coefficient(t, p) for t, p in zip(y_true, y_pred)]
    metrics = segmentation_metrics(y_true, y_prob)
    assert metrics["dice_std"] == pytest.approx(np.std(scores))
