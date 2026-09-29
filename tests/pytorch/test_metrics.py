import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchmetrics")

from zona_pellucida.backends.pytorch.module import (  # noqa: E402
    segmentation_metric_collection,
)
from zona_pellucida.evaluation.metrics import segmentation_metrics  # noqa: E402


def _data(seed=0):
    rng = np.random.default_rng(seed)
    y_true = (rng.random((4, 16, 16, 1)) > 0.85).astype(np.float32)
    y_prob = np.clip(y_true * 0.6 + rng.random(y_true.shape) * 0.5, 0, 1)
    return y_true, y_prob.astype(np.float32)


def test_training_metrics_match_numpy_test_metrics():
    y_true, y_prob = _data()
    metrics = segmentation_metric_collection(0.5)
    for batch in (slice(0, 2), slice(2, 4)):  # pooled over batches
        metrics.update(torch.from_numpy(y_prob[batch]),
                       torch.from_numpy(y_true[batch]).int())
    result = {name: float(value) for name, value in metrics.compute().items()}
    expected = segmentation_metrics(y_true, y_prob, 0.5)
    for name in ("accuracy", "precision", "recall", "iou"):
        assert result[name] == pytest.approx(expected[name], rel=1e-6)
    # "dice" of the training metrics is the pooled Dice = F1.
    assert result["dice"] == pytest.approx(expected["f1"], rel=1e-6)


def test_no_foreground_anywhere_gives_zero():
    empty = torch.zeros(1, 1, 4, 4)
    metrics = segmentation_metric_collection(0.5)
    metrics.update(empty, empty.int())
    result = metrics.compute()
    assert float(result["iou"]) == 0.0
    assert float(result["dice"]) == 0.0
