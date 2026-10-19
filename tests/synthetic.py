"""Synthetic image/mask pairs named like the real dataset (for tests)."""

import re
from pathlib import Path

import numpy as np
from PIL import Image


def ring_mask(height, width, center, r_inner, r_outer):
    yy, xx = np.mgrid[:height, :width]
    dist = np.hypot(yy - center[0], xx - center[1])
    return (dist >= r_inner) & (dist <= r_outer)


# File names of the G: embryo images (default) and of the video frames of
# the first thesis version; {sample} is the embryo or video, {frame} the
# time point in hours or the frame number.
EMBRYO_NAMES = ("embryo{sample}_{frame}h.png",
                "embryo{sample}_{frame}h_mask.png")
VIDEO_NAMES = ("sample{sample}_frame_{frame}.png",
               "sample{sample}_mask_frame_{frame}.png")
# 10 embryos (dish 1, P01-P10) at the three time points of the G: data
DEFAULT_FRAMES = {sample: (25, 69, 117) for sample in range(101, 111)}


def make_synthetic_dataset(root, frames=None, names=EMBRYO_NAMES, width=160,
                           height=90, seed=0):
    """Write RGB images with a bright ring (the "zona") and 0/255 masks.

    `frames` maps each sample id to its frame numbers (default
    DEFAULT_FRAMES). Returns (image_dir, mask_dir).
    """
    frames = DEFAULT_FRAMES if frames is None else frames
    rng = np.random.default_rng(seed)
    image_dir = Path(root) / "images"
    mask_dir = Path(root) / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    image_name, mask_name = names
    for sample, sample_frames in frames.items():
        for frame in sample_frames:
            center = (height / 2 + rng.uniform(-5, 5),
                      width / 2 + rng.uniform(-5, 5))
            r_outer = rng.uniform(25, 35)
            ring = ring_mask(height, width, center, r_outer - 6, r_outer)
            gray = rng.normal(90, 20, (height, width))
            gray[ring] += 90
            gray = np.clip(gray, 0, 255).astype(np.uint8)
            rgb = np.stack([gray] * 3, axis=-1)
            Image.fromarray(rgb).save(
                image_dir / image_name.format(sample=sample, frame=frame))
            Image.fromarray((ring * 255).astype(np.uint8)).save(
                mask_dir / mask_name.format(sample=sample, frame=frame))
    return image_dir, mask_dir


def set_parameters(source, **values):
    """Replace `NAME = ...` assignments (first cell / top of the export)."""
    for name, value in values.items():
        pattern = re.compile(rf"^{name} = .*$", re.MULTILINE)
        if not pattern.search(source):
            raise KeyError(f"No assignment of {name}")
        source = pattern.sub(lambda _: f"{name} = {value!r}", source,
                             count=1)
    return source


def synthetic_parameters(image_dir, mask_dir, output_dir):
    """Parameter values for a fast CPU run of the notebook / export."""
    return {
        "IMAGE_DIR": str(image_dir),
        "MASK_DIR": str(mask_dir),
        "OUTPUT_DIR": str(output_dir),
        "SIZE": 32,
        "EPOCHS": 2,
        "CONFIG_OVERRIDES": {
            "train": {"batch_size": 4},
        },
    }
