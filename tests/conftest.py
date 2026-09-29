import os

import matplotlib
import pytest

matplotlib.use("Agg")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from tests.synthetic import make_synthetic_dataset  # noqa: E402
from zona_pellucida.config import load_config  # noqa: E402

FRAMEWORK_MODULES = {"pytorch": "lightning", "keras": "keras"}


@pytest.fixture(params=list(FRAMEWORK_MODULES))
def framework(request):
    """Each framework whose library is installed (the others are skipped)."""
    pytest.importorskip(FRAMEWORK_MODULES[request.param])
    return request.param


@pytest.fixture
def backend(framework):
    from zona_pellucida.backends import load_backend

    return load_backend(framework)


@pytest.fixture(scope="session")
def synthetic_dirs(tmp_path_factory):
    """(image_dir, mask_dir) with 12 + 10 synthetic pairs."""
    return make_synthetic_dataset(
        tmp_path_factory.mktemp("data"), frames_per_sample=(12, 10)
    )


@pytest.fixture
def small_cfg(synthetic_dirs, tmp_path):
    """Config for fast CPU runs on the synthetic data."""
    image_dir, mask_dir = synthetic_dirs
    return load_config(overrides={
        "output_dir": str(tmp_path / "outputs"),
        "data": {"image_dir": str(image_dir), "mask_dir": str(mask_dir),
                 "size": 32},
        "train": {"epochs": 2, "batch_size": 4,
                  "tensorboard": {"enabled": False}},
    })
