import numpy as np
import pytest

from zona_pellucida.data.class_balance import compute_class_weights


def _masks(n_fg_per_image):
    masks = np.zeros((len(n_fg_per_image), 10, 10, 1), dtype=np.float32)
    for i, n_fg in enumerate(n_fg_per_image):
        masks[i].flat[:n_fg] = 1
    return masks


def test_inverse_frequency_weights():
    masks = _masks([10, 30])  # 40 foreground, 160 background pixels
    weights = compute_class_weights(masks, "inverse_frequency")
    assert weights[0] == pytest.approx(200 / (2 * 160))
    assert weights[1] == pytest.approx(200 / (2 * 40))
    # Weighted pixel count equals the total pixel count.
    assert weights[0] * 160 + weights[1] * 40 == pytest.approx(200)


def test_median_frequency_equals_inverse_when_classes_in_every_image():
    masks = _masks([10, 30, 5])
    inverse = compute_class_weights(masks, "inverse_frequency")
    median = compute_class_weights(masks, "median_frequency")
    assert median == pytest.approx(inverse)


def test_median_frequency_with_an_empty_mask():
    masks = _masks([10, 30, 0])
    weights = compute_class_weights(masks, "median_frequency")
    freq_bg = 260 / 300  # background appears in all 3 images
    freq_fg = 40 / 200  # foreground appears in 2 images only
    median = np.median([freq_bg, freq_fg])
    assert weights[0] == pytest.approx(median / freq_bg)
    assert weights[1] == pytest.approx(median / freq_fg)


def test_class_without_pixels_raises():
    with pytest.raises(ValueError):
        compute_class_weights(_masks([0, 0]))


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        compute_class_weights(_masks([1]), "sqrt")
