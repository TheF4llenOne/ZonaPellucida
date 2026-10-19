import numpy as np
import pytest
from PIL import Image

from tests.synthetic import VIDEO_NAMES, make_synthetic_dataset
from zona_pellucida.data.loading import (
    VIDEO_IMAGE_PATTERN,
    VIDEO_MASK_PATTERN,
    binarize_mask,
    load_dataset,
    load_mask,
    pair_image_mask_files,
)


def test_pairs_on_embryo_and_time_not_position():
    images = ["embryo102_25h.png", "embryo101_117h.png", "embryo101_69h.png"]
    masks = ["embryo101_69h_mask.png", "embryo102_25h_mask.png",
             "embryo101_117h_mask.png"]
    pairs = pair_image_mask_files(images, masks)
    assert [(p[2], p[3]) for p in pairs] == [(101, 69), (101, 117),
                                             (102, 25)]
    for image_name, mask_name, sample, frame in pairs:
        assert image_name == f"embryo{sample}_{frame}h.png"
        assert mask_name == f"embryo{sample}_{frame}h_mask.png"


def test_missing_mask_raises_instead_of_shifting_pairs():
    images = ["embryo101_25h.png", "embryo101_69h.png", "embryo101_117h.png"]
    masks = ["embryo101_25h_mask.png", "embryo101_117h_mask.png"]
    with pytest.raises(ValueError, match=r"\(101, 69\)"):
        pair_image_mask_files(images, masks)


def test_duplicate_frame_raises():
    images = ["embryo101_25h.png", "embryo101_025h.png"]
    with pytest.raises(ValueError, match="Two image files"):
        pair_image_mask_files(images, ["embryo101_25h_mask.png"])


def test_non_matching_files_are_skipped():
    # A mask among the images must not be read as an image
    pairs = pair_image_mask_files(
        ["embryo101_25h.png", "desktop.ini", "embryo101_25h_mask.png"],
        ["embryo101_25h_mask.png", "Thumbs.db"],
    )
    assert len(pairs) == 1


def test_video_frames_of_the_first_version_still_load(tmp_path):
    image_dir, mask_dir = make_synthetic_dataset(
        tmp_path, frames={1: (1, 2, 3), 2: (1, 2)}, names=VIDEO_NAMES)
    dataset = load_dataset(image_dir, mask_dir, size=32,
                           image_pattern=VIDEO_IMAGE_PATTERN,
                           mask_pattern=VIDEO_MASK_PATTERN)
    assert dataset.samples.tolist() == [1, 1, 1, 2, 2]
    assert dataset.frames.tolist() == [1, 2, 3, 1, 2]


def test_binarize_mask_threshold():
    mask = np.array([0, 127, 128, 255], dtype=np.uint8)
    np.testing.assert_array_equal(binarize_mask(mask), [0, 0, 1, 1])


def test_mask_resize_stays_binary(tmp_path):
    # A thin ring: bicubic resizing would create many grey edge values.
    yy, xx = np.mgrid[:90, :160]
    ring = np.abs(np.hypot(yy - 45, xx - 80) - 30) < 3
    path = tmp_path / "mask.png"
    Image.fromarray((ring * 255).astype(np.uint8)).save(path)
    bicubic = np.array(Image.open(path).resize((64, 64)))
    assert len(np.unique(bicubic)) > 2  # the old behaviour
    mask = load_mask(str(path), 64)
    assert set(np.unique(mask)) <= {0, 1}
    assert mask.dtype == np.uint8


def test_load_dataset(synthetic_dirs):
    image_dir, mask_dir = synthetic_dirs
    dataset = load_dataset(image_dir, mask_dir, size=32)
    assert len(dataset) == 30
    assert dataset.images.shape == (30, 32, 32, 1)
    assert dataset.masks.shape == (30, 32, 32, 1)
    assert dataset.images.dtype == np.float32
    assert 0 <= dataset.images.min() and dataset.images.max() <= 1
    assert set(np.unique(dataset.masks)) == {0.0, 1.0}
    samples, counts = np.unique(dataset.samples, return_counts=True)
    assert samples.tolist() == list(range(101, 111))
    assert set(counts.tolist()) == {3}
    assert dataset.frames[:3].tolist() == [25, 69, 117]
    subset = dataset.subset([0, 29])
    assert subset.image_files == ["embryo101_25h.png", "embryo110_117h.png"]
    assert subset.frames.tolist() == [25, 117]
