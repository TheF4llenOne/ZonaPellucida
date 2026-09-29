import numpy as np
import pytest

keras = pytest.importorskip("keras")

from zona_pellucida.backends.keras.metrics import (  # noqa: E402
    DiceCoefficient,
    ForegroundIoU,
    training_metrics,
)
from zona_pellucida.evaluation.metrics import (  # noqa: E402
    confusion_counts,
    segmentation_metrics,
)


def _data(seed=0):
    rng = np.random.default_rng(seed)
    y_true = (rng.random((4, 16, 16, 1)) > 0.85).astype(np.float32)
    y_prob = np.clip(y_true * 0.6 + rng.random(y_true.shape) * 0.5, 0, 1)
    return y_true, y_prob.astype(np.float32)


def test_dice_metric_matches_numpy_and_accumulates():
    y_true, y_prob = _data()
    metric = DiceCoefficient(threshold=0.5)
    metric.update_state(y_true[:2], y_prob[:2])
    metric.update_state(y_true[2:], y_prob[2:])
    tn, fp, fn, tp = confusion_counts(y_true > 0.5, y_prob > 0.5)
    assert float(metric.result()) == pytest.approx(
        2 * tp / (2 * tp + fp + fn), rel=1e-6)
    metric.reset_state()
    metric.update_state(y_true[:1], y_true[:1])
    assert float(metric.result()) == pytest.approx(1.0)


def test_iou_metric_equals_keras_binary_iou():
    y_true, y_prob = _data()
    iou = next(m for m in training_metrics() if m.name == "iou")
    assert isinstance(iou, ForegroundIoU)
    iou.update_state(y_true[:2], y_prob[:2])
    iou.update_state(y_true[2:], y_prob[2:])
    reference = keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5)
    reference.update_state(y_true, y_prob)
    expected = segmentation_metrics(y_true, y_prob, 0.5)["iou"]
    assert float(iou.result()) == pytest.approx(expected, rel=1e-6)
    assert float(iou.result()) == pytest.approx(float(reference.result()),
                                                rel=1e-5)


def test_no_foreground_anywhere_gives_zero():
    empty = np.zeros((1, 4, 4, 1), dtype=np.float32)
    for metric in (DiceCoefficient(), ForegroundIoU()):
        metric.update_state(empty, empty)
        assert float(metric.result()) == 0.0
