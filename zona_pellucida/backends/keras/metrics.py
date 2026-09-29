"""Imbalance-aware Keras metrics monitored during training."""

import keras
from keras import ops


class _PixelConfusion(keras.metrics.Metric):
    """Accumulates foreground TP, FP and FN over all pixels seen.

    Predictions are binarised with `threshold`; the ground truth with 0.5.
    Only sums are used, so the metric is deterministic on the GPU.
    """

    def __init__(self, threshold=0.5, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = float(threshold)
        self.tp = self.add_variable(shape=(), initializer="zeros", name="tp")
        self.fp = self.add_variable(shape=(), initializer="zeros", name="fp")
        self.fn = self.add_variable(shape=(), initializer="zeros", name="fn")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = ops.cast(ops.greater(y_true, 0.5), "float32")
        y_pred = ops.cast(ops.greater(y_pred, self.threshold), "float32")
        self.tp.assign_add(ops.sum(y_true * y_pred))
        self.fp.assign_add(ops.sum((1.0 - y_true) * y_pred))
        self.fn.assign_add(ops.sum(y_true * (1.0 - y_pred)))

    def _ratio(self, numerator, denominator):
        # 0 when there is no foreground in either mask, as in torchmetrics
        # and zona_pellucida.evaluation.metrics.
        return ops.where(denominator > 0,
                         numerator / ops.maximum(denominator, 1.0), 0.0)

    def reset_state(self):
        for variable in (self.tp, self.fp, self.fn):
            variable.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update(threshold=self.threshold)
        return config


@keras.saving.register_keras_serializable(package="zona_pellucida")
class DiceCoefficient(_PixelConfusion):
    """Foreground Dice = 2TP / (2TP + FP + FN)."""

    def __init__(self, threshold=0.5, name="dice", **kwargs):
        super().__init__(threshold=threshold, name=name, **kwargs)

    def result(self):
        return self._ratio(2.0 * self.tp, 2.0 * self.tp + self.fp + self.fn)


@keras.saving.register_keras_serializable(package="zona_pellucida")
class ForegroundIoU(_PixelConfusion):
    """Foreground IoU = TP / (TP + FP + FN).

    Same value as keras.metrics.BinaryIoU(target_class_ids=[1]), whose
    confusion matrix uses scatter_nd; with op determinism enabled TensorFlow
    runs scatter_nd on the CPU, which slows down GPU training.
    """

    def __init__(self, threshold=0.5, name="iou", **kwargs):
        super().__init__(threshold=threshold, name=name, **kwargs)

    def result(self):
        return self._ratio(self.tp, self.tp + self.fp + self.fn)


def training_metrics(threshold=0.5):
    """Accuracy plus foreground IoU, Dice, precision and recall."""
    return [
        keras.metrics.BinaryAccuracy(name="accuracy", threshold=threshold),
        ForegroundIoU(threshold=threshold, name="iou"),
        DiceCoefficient(threshold=threshold, name="dice"),
        keras.metrics.Precision(thresholds=threshold, name="precision"),
        keras.metrics.Recall(thresholds=threshold, name="recall"),
    ]
