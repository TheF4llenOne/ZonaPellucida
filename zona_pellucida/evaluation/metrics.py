"""Pixel-level segmentation metrics on binarised masks (NumPy)."""

import numpy as np


def binarize(array, threshold=0.5):
    return np.asarray(array) > threshold


def confusion_counts(y_true, y_pred):
    """TN, FP, FN, TP of two boolean arrays, pooled over all pixels."""
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    tp = int(np.count_nonzero(y_true & y_pred))
    fp = int(np.count_nonzero(~y_true & y_pred))
    fn = int(np.count_nonzero(y_true & ~y_pred))
    tn = int(y_true.size - tp - fp - fn)
    return tn, fp, fn, tp


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else 0.0


def dice_coefficient(y_true, y_pred, smooth=1):
    """Dice of one image, smoothed as in the thesis."""
    y_true_f = np.asarray(y_true, dtype=float).flatten()
    y_pred_f = np.asarray(y_pred, dtype=float).flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        np.sum(y_true_f) + np.sum(y_pred_f) + smooth
    )


def iou_coefficient(y_true, y_pred, smooth=1):
    """IoU of one image, smoothed like the Dice above."""
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    intersection = np.count_nonzero(y_true & y_pred)
    union = np.count_nonzero(y_true | y_pred)
    return (intersection + smooth) / (union + smooth)


def per_image_scores(y_true, y_pred, score=dice_coefficient):
    """Score of each image; both inputs are binarised arrays (N, H, W, 1)."""
    return np.array([score(t, p) for t, p in zip(y_true, y_pred)])


def segmentation_metrics(y_true, y_prob, threshold=0.5):
    """Test-set metrics of probabilities `y_prob` at `threshold`.

    The ground truth is binarised with > 0.5 first. Accuracy, precision,
    recall, F1 and IoU are pooled over every pixel of every image (F1 is
    then identical to the pooled Dice). `dice` and `iou_per_image` are the
    means (and `*_std` the standard deviations) of the per-image scores;
    the mean per-image Dice is the "Average Dice" of the thesis.
    """
    y_true = binarize(y_true, 0.5)
    y_pred = binarize(y_prob, threshold)
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    dice = per_image_scores(y_true, y_pred)
    iou = per_image_scores(y_true, y_pred, iou_coefficient)
    return {
        "threshold": float(threshold),
        "accuracy": _ratio(tp + tn, tp + tn + fp + fn),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "iou": _ratio(tp, tp + fp + fn),
        "dice": float(np.mean(dice)),
        "dice_std": float(np.std(dice)),
        "iou_per_image": float(np.mean(iou)),
        "iou_per_image_std": float(np.std(iou)),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }
