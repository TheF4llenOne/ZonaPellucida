"""Loading, pairing and preprocessing of the image / mask pairs."""

import logging
import os
import re
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

IMAGE_PATTERN = r"^sample(?P<sample>\d+)_frame_(?P<frame>\d+)\.png$"
MASK_PATTERN = r"^sample(?P<sample>\d+)_mask_frame_(?P<frame>\d+)\.png$"


@dataclass
class SegmentationDataset:
    """Preprocessed pairs; row i of every field describes the same pair."""

    images: np.ndarray  # (N, H, W, 1) float32 in [0, 1]
    masks: np.ndarray  # (N, H, W, 1) float32 in {0, 1}
    image_files: list
    mask_files: list
    samples: np.ndarray  # (N,) video / embryo id
    frames: np.ndarray  # (N,) frame number within the video

    def __len__(self):
        return len(self.images)

    def subset(self, indices):
        indices = np.asarray(indices, dtype=int)
        return SegmentationDataset(
            images=self.images[indices],
            masks=self.masks[indices],
            image_files=[self.image_files[i] for i in indices],
            mask_files=[self.mask_files[i] for i in indices],
            samples=self.samples[indices],
            frames=self.frames[indices],
        )


def parse_sample_frame(filename, pattern):
    """Return (sample, frame) from a filename, or None if it does not match."""
    match = re.match(pattern, filename)
    if match is None:
        return None
    return int(match.group("sample")), int(match.group("frame"))


def _index_by_key(filenames, pattern, kind):
    index = {}
    for name in filenames:
        key = parse_sample_frame(name, pattern)
        if key is None:
            logger.warning("Skipping %s file %s (name pattern)", kind, name)
            continue
        if key in index:
            raise ValueError(
                f"Two {kind} files for sample {key[0]}, frame {key[1]}: "
                f"{index[key]} and {name}"
            )
        index[key] = name
    return index


def pair_image_mask_files(image_filenames, mask_filenames,
                          image_pattern=IMAGE_PATTERN,
                          mask_pattern=MASK_PATTERN):
    """Pair images and masks on (sample, frame) number.

    Pairing on sorted-list position (zip) would silently shift every later
    pair if one file were missing, so any unmatched file is an error.
    Returns a list of (image_name, mask_name, sample, frame), sorted by
    sample and frame.
    """
    images = _index_by_key(image_filenames, image_pattern, "image")
    masks = _index_by_key(mask_filenames, mask_pattern, "mask")
    missing_masks = sorted(set(images) - set(masks))
    missing_images = sorted(set(masks) - set(images))
    if missing_masks or missing_images:
        raise ValueError(
            "Image/mask pairing failed. (sample, frame) without a mask: "
            f"{missing_masks}; without an image: {missing_images}"
        )
    pairs = []
    for key in sorted(images):
        image_key = parse_sample_frame(images[key], image_pattern)
        mask_key = parse_sample_frame(masks[key], mask_pattern)
        assert image_key == mask_key == key, (images[key], masks[key])
        pairs.append((images[key], masks[key], key[0], key[1]))
    return pairs


def _read_grayscale(path):
    array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if array is None:
        raise FileNotFoundError(f"Could not read image {path}")
    return array


def load_image(path, size):
    """Grayscale uint8 image resized to (size, size) with bicubic filtering."""
    image = Image.fromarray(_read_grayscale(path))
    image = image.resize((size, size), Image.BICUBIC)
    return np.array(image)


def binarize_mask(mask, threshold=0.5):
    """Map a 0-255 mask to {0, 1}: foreground where mask / 255 > threshold."""
    return (np.asarray(mask) / 255.0 > threshold).astype(np.uint8)


def load_mask(path, size, threshold=0.5):
    """Binary {0, 1} mask resized to (size, size).

    Nearest-neighbour resizing keeps the labels binary; the default
    (bicubic) filter creates intermediate values along the zona edges.
    """
    mask = Image.fromarray(_read_grayscale(path))
    mask = mask.resize((size, size), Image.NEAREST)
    return binarize_mask(np.array(mask), threshold)


def load_dataset(image_dir, mask_dir, size=512, image_pattern=IMAGE_PATTERN,
                 mask_pattern=MASK_PATTERN, mask_threshold=0.5):
    """Load, pair and preprocess every image / mask pair of two folders."""
    pairs = pair_image_mask_files(
        sorted(os.listdir(image_dir)), sorted(os.listdir(mask_dir)),
        image_pattern, mask_pattern,
    )
    if not pairs:
        raise ValueError(f"No image/mask pairs found in {image_dir}")
    images, masks = [], []
    for image_name, mask_name, _, _ in pairs:
        images.append(load_image(os.path.join(image_dir, image_name), size))
        masks.append(
            load_mask(os.path.join(mask_dir, mask_name), size, mask_threshold)
        )
    # Scale images to [0, 1] by the fixed 8-bit maximum (not per image).
    images = (np.stack(images).astype(np.float32) / 255.0)[..., np.newaxis]
    masks = np.stack(masks).astype(np.float32)[..., np.newaxis]
    dataset = SegmentationDataset(
        images=images,
        masks=masks,
        image_files=[p[0] for p in pairs],
        mask_files=[p[1] for p in pairs],
        samples=np.array([p[2] for p in pairs]),
        frames=np.array([p[3] for p in pairs]),
    )
    counts = dict(zip(*np.unique(dataset.samples, return_counts=True)))
    logger.info(
        "Loaded %d pairs, images %s, masks %s, pairs per sample %s",
        len(dataset), images.shape, masks.shape,
        {int(k): int(v) for k, v in counts.items()},
    )
    return dataset


def load_dataset_from_config(data_cfg):
    return load_dataset(
        data_cfg.image_dir, data_cfg.mask_dir, data_cfg.size,
        data_cfg.image_pattern, data_cfg.mask_pattern,
        data_cfg.mask_threshold,
    )
