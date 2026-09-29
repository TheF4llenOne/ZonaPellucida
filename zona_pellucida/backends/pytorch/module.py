"""LightningModule: the FCN with its loss, optimiser and metrics."""

import lightning as L
import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryJaccardIndex,
    BinaryPrecision,
    BinaryRecall,
)

from zona_pellucida.backends.pytorch.losses import get_loss
from zona_pellucida.backends.pytorch.model import UNet


def segmentation_metric_collection(threshold=0.5):
    """Accuracy plus zona pellucida IoU, Dice (= F1), precision and recall,
    pooled over all pixels of an epoch."""
    return MetricCollection({
        "accuracy": BinaryAccuracy(threshold=threshold),
        "iou": BinaryJaccardIndex(threshold=threshold),
        "dice": BinaryF1Score(threshold=threshold),
        "precision": BinaryPrecision(threshold=threshold),
        "recall": BinaryRecall(threshold=threshold),
    })


class SegmentationModule(L.LightningModule):
    """UNet trained with the (weighted) binary cross-entropy and Adam."""

    def __init__(self, loss="unweighted", w0=1.0, w1=1.0,
                 learning_rate=1e-3, dropout=0.5, threshold=0.5):
        super().__init__()
        # Stored in every checkpoint, so load_from_checkpoint rebuilds it.
        self.save_hyperparameters()
        self.model = UNet(dropout=dropout)
        self.loss_fn = get_loss(loss, {0: w0, 1: w1})
        self.train_metrics = segmentation_metric_collection(threshold)
        self.val_metrics = self.train_metrics.clone(prefix="val_")

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, metrics, loss_name):
        images, masks = batch
        logits = self(images)
        loss = self.loss_fn(logits, masks)
        metrics.update(torch.sigmoid(logits), masks.int())
        # Epoch values, averaged over batches like Keras' history.
        self.log(loss_name, loss, on_step=False, on_epoch=True,
                 prog_bar=True, batch_size=len(images))
        self.log_dict(metrics, on_step=False, on_epoch=True,
                      batch_size=len(images))
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, self.train_metrics, "loss")

    def validation_step(self, batch, batch_idx):
        self._step(batch, self.val_metrics, "val_loss")

    def configure_optimizers(self):
        # Epsilon of Keras' Adam (1e-7 instead of PyTorch's 1e-8). Keras adds
        # it before the bias correction, PyTorch after, so the first updates
        # differ very slightly between the frameworks.
        return torch.optim.Adam(self.parameters(),
                                lr=self.hparams.learning_rate, eps=1e-7)
