"""Weighted binary cross-entropy (Keras)."""

import keras
from keras import ops


@keras.saving.register_keras_serializable(package="zona_pellucida")
class WeightedBinaryCrossentropy(keras.losses.Loss):
    """Per-pixel weighted binary cross-entropy on sigmoid probabilities.

    loss = -[w1 * y * log(p) + w0 * (1 - y) * log(1 - p)], averaged over
    pixels and batch. The per-pixel cross-entropy is the one of the
    unweighted Keras baseline: inside `fit` and `evaluate` it is computed
    from the logits of the sigmoid layer (on predicted probabilities it
    clips them to [1e-7, 1 - 1e-7]), so the two runs differ only in the
    weights. Keras
    `class_weight=` does not support per-pixel (4-D) targets, hence a loss.
    """

    def __init__(self, w0=1.0, w1=1.0, name="weighted_binary_crossentropy",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.w0 = float(w0)
        self.w1 = float(w1)

    def call(self, y_true, y_pred):
        y_true = ops.cast(y_true, y_pred.dtype)
        # Element-wise -[y log(p) + (1 - y) log(1 - p)]
        bce = ops.binary_crossentropy(y_true, y_pred)
        weights = self.w1 * y_true + self.w0 * (1.0 - y_true)
        loss = weights * bce
        # Mean over the pixels of each image -> one value per image.
        return ops.mean(loss, axis=tuple(range(1, len(loss.shape))))

    def get_config(self):
        config = super().get_config()
        config.update(w0=self.w0, w1=self.w1)
        return config


def get_loss(run, class_weights=None):
    """Loss of an experiment run: 'unweighted' or 'weighted'."""
    if run == "unweighted":
        return keras.losses.BinaryCrossentropy()
    if run == "weighted":
        if class_weights is None:
            raise ValueError("The weighted run needs class weights")
        return WeightedBinaryCrossentropy(
            w0=class_weights[0], w1=class_weights[1]
        )
    raise ValueError(f"Unknown run '{run}', use 'unweighted' or 'weighted'")
