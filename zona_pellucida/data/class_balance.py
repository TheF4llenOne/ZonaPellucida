"""Pixel-level class balance of the (binary) masks and class weights."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def pixel_class_counts(masks, threshold=0.5):
    """Number of background (0) and foreground (1) pixels."""
    foreground = int(np.count_nonzero(np.asarray(masks) > threshold))
    total = int(np.asarray(masks).size)
    return {"background": total - foreground, "foreground": foreground,
            "total": total}


def foreground_fraction_per_image(masks, threshold=0.5):
    masks = np.asarray(masks) > threshold
    return masks.reshape(len(masks), -1).mean(axis=1)


def class_balance_summary(masks, threshold=0.5):
    """Overall class shares and per-image foreground-fraction statistics."""
    counts = pixel_class_counts(masks, threshold)
    fractions = foreground_fraction_per_image(masks, threshold)
    return {
        "n_images": len(fractions),
        "total_pixels": counts["total"],
        "background_pixels": counts["background"],
        "foreground_pixels": counts["foreground"],
        "background_percent": 100 * counts["background"] / counts["total"],
        "foreground_percent": 100 * counts["foreground"] / counts["total"],
        "background_to_foreground_ratio": (
            counts["background"] / max(counts["foreground"], 1)
        ),
        "fg_fraction_min": float(fractions.min()),
        "fg_fraction_mean": float(fractions.mean()),
        "fg_fraction_median": float(np.median(fractions)),
        "fg_fraction_max": float(fractions.max()),
        "fg_fraction_std": float(fractions.std()),
    }


def log_class_balance(summary):
    logger.info(
        "Background: %d px (%.2f%%), zona pellucida: %d px (%.2f%%), "
        "ratio %.1f:1",
        summary["background_pixels"], summary["background_percent"],
        summary["foreground_pixels"], summary["foreground_percent"],
        summary["background_to_foreground_ratio"],
    )
    logger.info(
        "Per-image foreground fraction over %d images: min %.4f, "
        "mean %.4f, max %.4f",
        summary["n_images"], summary["fg_fraction_min"],
        summary["fg_fraction_mean"], summary["fg_fraction_max"],
    )


def plot_class_balance(summary, fractions, save_path=None):
    """Bar chart of the class pixel shares + histogram of the per-image
    foreground fraction (training split)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    shares = [summary["background_percent"], summary["foreground_percent"]]
    bars = ax1.bar(["Background", "Zona pellucida"], shares,
                   color=["#9e9e9e", "#1f77b4"])
    ax1.bar_label(bars, fmt="%.2f%%")
    ax1.set_ylabel("Share of training pixels (%)")
    ax1.set_ylim(0, 105)
    ax1.set_title("Class pixel shares")
    ax2.hist(np.asarray(fractions) * 100, bins=20, color="#1f77b4",
             edgecolor="white")
    ax2.set_xlabel("Zona pellucida pixels per image (%)")
    ax2.set_ylabel("Number of images")
    ax2.set_title(f"Per-image foreground fraction (n={len(fractions)})")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        logger.info("Saved figure %s", save_path)
    plt.show()
    plt.close(fig)


def save_class_balance(masks, image_files, output_dir, threshold=0.5):
    """Write class_balance.csv, class_balance_per_image.csv and
    class_balance.png to `output_dir`; return the summary dict."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = class_balance_summary(masks, threshold)
    fractions = foreground_fraction_per_image(masks, threshold)
    log_class_balance(summary)
    pd.DataFrame(
        {"statistic": list(summary), "value": list(summary.values())}
    ).to_csv(output_dir / "class_balance.csv", index=False)
    pd.DataFrame(
        {"image_file": image_files, "foreground_fraction": fractions}
    ).to_csv(output_dir / "class_balance_per_image.csv", index=False)
    plot_class_balance(summary, fractions,
                       save_path=output_dir / "class_balance.png")
    return summary


CLASS_WEIGHT_METHODS = ("inverse_frequency", "median_frequency")


def compute_class_weights(masks, method="inverse_frequency", threshold=0.5):
    """Class weights {0: w_background, 1: w_zona} from pixel frequencies.

    inverse_frequency (default): w_c = N_total / (C * n_c), C = 2, so the
        weighted pixel count sum_c w_c * n_c equals N_total.
    median_frequency (SegNet, Eigen & Fergus): w_c = median_freq / freq_c,
        where freq_c = n_c / (pixels of the images that contain class c).
        With two classes present in every image this equals
        inverse_frequency; they differ only if some masks are empty.
    Must be computed on the training split only.
    """
    masks = np.asarray(masks) > threshold
    n_classes = 2
    per_image = masks.reshape(len(masks), -1)
    pixels_per_image = per_image.shape[1]
    n_fg = per_image.sum(axis=1)
    n_c = np.array([per_image.size - n_fg.sum(), n_fg.sum()], dtype=float)
    if np.any(n_c == 0):
        raise ValueError(f"A class has no pixels in the training masks: {n_c}")
    if method == "inverse_frequency":
        weights = n_c.sum() / (n_classes * n_c)
    elif method == "median_frequency":
        images_with_c = np.array(
            [np.sum(n_fg < pixels_per_image), np.sum(n_fg > 0)], dtype=float
        )
        freq = n_c / (images_with_c * pixels_per_image)
        weights = np.median(freq) / freq
    else:
        raise ValueError(
            f"Unknown class weighting '{method}', use {CLASS_WEIGHT_METHODS}"
        )
    class_weights = {0: float(weights[0]), 1: float(weights[1])}
    logger.info(
        "Class weights (%s): background %.4f, zona pellucida %.4f",
        method, class_weights[0], class_weights[1],
    )
    return class_weights
