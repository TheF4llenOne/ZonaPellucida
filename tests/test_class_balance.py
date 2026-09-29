import numpy as np
import pandas as pd
import pytest

from zona_pellucida.data.class_balance import (
    class_balance_summary,
    foreground_fraction_per_image,
    pixel_class_counts,
    save_class_balance,
)


def _masks():
    masks = np.zeros((3, 4, 4, 1), dtype=np.float32)
    masks[0, :1] = 1  # 4 of 16 pixels
    masks[1, :2] = 1  # 8 of 16
    return masks  # image 2 is empty


def test_pixel_class_counts():
    assert pixel_class_counts(_masks()) == {
        "background": 36, "foreground": 12, "total": 48}


def test_foreground_fraction_per_image():
    np.testing.assert_allclose(foreground_fraction_per_image(_masks()),
                               [0.25, 0.5, 0.0])


def test_summary():
    summary = class_balance_summary(_masks())
    assert summary["foreground_percent"] == pytest.approx(25.0)
    assert summary["background_percent"] == pytest.approx(75.0)
    assert summary["background_to_foreground_ratio"] == pytest.approx(3.0)
    assert summary["fg_fraction_min"] == 0.0
    assert summary["fg_fraction_max"] == 0.5
    assert summary["fg_fraction_mean"] == pytest.approx(0.25)


def test_save_class_balance(tmp_path):
    save_class_balance(_masks(), ["a", "b", "c"], tmp_path)
    assert (tmp_path / "class_balance.png").stat().st_size > 0
    table = pd.read_csv(tmp_path / "class_balance.csv")
    value = table.set_index("statistic")["value"]
    assert value["foreground_pixels"] == 12
    per_image = pd.read_csv(tmp_path / "class_balance_per_image.csv")
    assert per_image["image_file"].tolist() == ["a", "b", "c"]
