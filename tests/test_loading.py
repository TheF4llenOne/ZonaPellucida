import numpy as np
import pytest
from PIL import Image

from zona_pellucida.data.loading import (
    binarize_mask,
    load_dataset,
    load_mask,
    pair_image_mask_files,
)


def test_pairs_on_sample_and_frame_not_position():
    images = ["sample2_frame_1.png", "sample1_frame_10.png",
              "sample1_frame_2.png"]
    masks = ["sample1_mask_frame_2.png", "sample2_mask_frame_1.png",
             "sample1_mask_frame_10.png"]
    pairs = pair_image_mask_files(images, masks)
    assert [(p[2], p[3]) for p in pairs] == [(1, 2), (1, 10), (2, 1)]
    for image_name, mask_name, sample, frame in pairs:
        assert image_name == f"sample{sample}_frame_{frame}.png"
        assert mask_name == f"sample{sample}_mask_frame_{frame}.png"


def test_missing_mask_raises_instead_of_shifting_pairs():
    images = ["sample1_frame_1.png", "sample1_frame_2.png",
              "sample1_frame_3.png"]
    masks = ["sample1_mask_frame_1.png", "sample1_mask_frame_3.png"]
    with pytest.raises(ValueError, match=r"\(1, 2\)"):
        pair_image_mask_files(images, masks)


def test_duplicate_frame_raises():
    images = ["sample1_frame_1.png", "sample1_frame_01.png"]
    with pytest.raises(ValueError, match="Two image files"):
        pair_image_mask_files(images, ["sample1_mask_frame_1.png"])


def test_non_matching_files_are_skipped():
    pairs = pair_image_mask_files(
        ["sample1_frame_1.png", "desktop.ini"],
        ["sample1_mask_frame_1.png", "Thumbs.db"],
    )
    assert len(pairs) == 1


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
    assert len(dataset) == 22
    assert dataset.images.shape == (22, 32, 32, 1)
    assert dataset.masks.shape == (22, 32, 32, 1)
    assert dataset.images.dtype == np.float32
    assert 0 <= dataset.images.min() and dataset.images.max() <= 1
    assert set(np.unique(dataset.masks)) == {0.0, 1.0}
    assert np.bincount(dataset.samples).tolist() == [0, 12, 10]
    subset = dataset.subset([0, 21])
    assert subset.image_files == [dataset.image_files[0],
                                  dataset.image_files[21]]
    assert subset.frames.tolist() == [1, 10]
