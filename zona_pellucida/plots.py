"""Figures of the notebook; each can also be saved for the thesis."""

import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import ConfusionMatrixDisplay

logger = logging.getLogger(__name__)


def _finish(fig, save_path=None):
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        logger.info("Saved figure %s", save_path)
    plt.show()
    plt.close(fig)


METRIC_LABELS = {"loss": "Loss", "accuracy": "Accuracy", "iou": "IoU",
                 "dice": "Dice", "precision": "Precision",
                 "recall": "Recall"}


def plot_training_curves(history, metric="loss", title=None, save_path=None,
                         best_epoch=None):
    """Training and validation curve of one metric per epoch."""
    label = METRIC_LABELS.get(metric, metric)
    values = history[metric]
    epochs = range(1, len(values) + 1)
    fig = plt.figure(figsize=(12, 6))
    plt.plot(epochs, values, 'y', label=f'Training {label.lower()}')
    if f'val_{metric}' in history:
        plt.plot(epochs, history[f'val_{metric}'], 'r',
                 label=f'Validation {label.lower()}')
    if best_epoch is not None:
        plt.axvline(best_epoch, color='grey', linestyle='--',
                    label=f'Best validation epoch ({best_epoch})')
    plt.title(title or f'Training and Validation {label}')
    plt.xlabel('Epochs')
    plt.ylabel(label)
    plt.legend()
    _finish(fig, save_path)


def plot_threshold_curve(table, metric, selected, save_path=None):
    """Validation metric as a function of the decision threshold."""
    fig = plt.figure(figsize=(8, 5))
    plt.plot(table["threshold"], table[metric], marker="o")
    plt.axvline(selected, color="r", linestyle="--",
                label=f"selected = {selected:.2f}")
    plt.axvline(0.5, color="grey", linestyle=":", label="0.5")
    plt.xlabel("Decision threshold")
    plt.ylabel(f"Validation {metric}")
    plt.title("Threshold selection on the validation set")
    plt.legend()
    _finish(fig, save_path)


def plot_sample_prediction(image, ground_truth, prediction, title=None,
                           panel_titles=('Testing Image', 'Ground Truth',
                                         'Prediction'),
                           save_path=None):
    """Image, ground-truth mask and predicted mask side by side."""
    fig = plt.figure(figsize=(16, 8) if title is None else (12, 6))
    if title is not None:
        plt.suptitle(title, fontsize=16)
    panels = zip((image, ground_truth, prediction), panel_titles)
    for i, (array, name) in enumerate(panels):
        plt.subplot(1, 3, i + 1)
        plt.title(name)
        plt.imshow(np.squeeze(array).astype(float), cmap='gray')
    _finish(fig, save_path)


def plot_prediction_overlay(image, prediction, alpha=0.1, save_path=None):
    """Original image with the predicted mask overlaid in white."""
    image = np.squeeze(image)
    # Set the overlay region to white where the prediction is positive
    overlay = np.zeros_like(image, dtype=float)
    overlay[np.squeeze(prediction) == 1] = 255
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray')
    # Overlay the prediction with transparency
    plt.imshow(overlay, cmap='gray', alpha=alpha)
    plt.title('Original Image with Prediction Overlay')
    plt.axis('off')
    _finish(fig, save_path)


def plot_confusion_matrix(conf_matrix, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(conf_matrix).plot(ax=ax)
    plt.title('Confusion Matrix')
    _finish(fig, save_path)


def plot_confusion_matrix_heatmap(conf_matrix, save_path=None):
    """Confusion matrix with seaborn for better visualization."""
    conf_matrix_df = pd.DataFrame(
        conf_matrix, index=['Actual Negative', 'Actual Positive'],
        columns=['Predicted Negative', 'Predicted Positive'])
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    _finish(fig, save_path)


def plot_distance_transform_overlay(image, ground_truth, prediction,
                                    alpha=0.3, save_path=None):
    """Prediction coloured by its distance transform over the image."""
    image = np.squeeze(image)
    prediction = np.squeeze(prediction).astype(np.uint8)
    # Compute the distance transform on the predicted mask
    dist_transform = distance_transform_edt(prediction)
    # Normalize the distance transform for better visualization
    dist_transform_normalized = cv2.normalize(
        dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Create a color map for overlay
    overlay = cv2.applyColorMap(dist_transform_normalized, cv2.COLORMAP_JET)
    # Convert the test image to uint8 and then to 3 channels
    test_img_uint8 = (image * 255).astype(np.uint8)
    test_img_color = cv2.cvtColor(test_img_uint8, cv2.COLOR_GRAY2BGR)
    # Overlay the distance transform on the original image
    mask = dist_transform_normalized > 0
    overlayed_img = test_img_color.copy()
    overlayed_img[mask] = cv2.addWeighted(
        test_img_color, 1 - alpha, overlay, alpha, 0)[mask]

    fig = plt.figure(figsize=(16, 8))
    panels = [(image, 'Testing Image'),
              (np.squeeze(ground_truth), 'Ground Truth'),
              (prediction, 'Prediction')]
    for i, (array, name) in enumerate(panels):
        plt.subplot(2, 4, i + 1)
        plt.title(name)
        plt.imshow(array.astype(float), cmap='gray')
    plt.subplot(244)
    plt.title('Distance Transform Overlay')
    # OpenCV images are BGR; matplotlib expects RGB.
    plt.imshow(cv2.cvtColor(overlayed_img, cv2.COLOR_BGR2RGB))
    _finish(fig, save_path)
