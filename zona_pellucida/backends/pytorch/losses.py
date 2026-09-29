"""Weighted binary cross-entropy (PyTorch)."""

import torch.nn.functional as F
from torch import nn


class WeightedBCEWithLogitsLoss(nn.Module):
    """Per-pixel weighted binary cross-entropy on logits.

    loss = -[w1 * y * log(p) + w0 * (1 - y) * log(1 - p)], p = sigmoid(z),
    averaged over all pixels. Computed from the logits z, which is
    numerically stable (no clipping needed). With w0 = w1 = 1 it is the
    unweighted binary cross-entropy.
    """

    def __init__(self, w0=1.0, w1=1.0):
        super().__init__()
        self.w0 = float(w0)
        self.w1 = float(w1)

    def forward(self, logits, targets):
        targets = targets.to(logits.dtype)
        weights = self.w1 * targets + self.w0 * (1.0 - targets)
        return F.binary_cross_entropy_with_logits(logits, targets,
                                                  weight=weights)


def get_loss(run, class_weights=None):
    """Loss of an experiment run: 'unweighted' or 'weighted'."""
    if run == "unweighted":
        return nn.BCEWithLogitsLoss()
    if run == "weighted":
        if class_weights is None:
            raise ValueError("The weighted run needs class weights")
        return WeightedBCEWithLogitsLoss(w0=class_weights[0],
                                         w1=class_weights[1])
    raise ValueError(f"Unknown run '{run}', use 'unweighted' or 'weighted'")
