import pytest

from zona_pellucida.reproducibility import (
    DeterminismError,
    environment_versions,
    run_with_determinism_check,
)


class _OpError(Exception):
    """Like tf.errors.UnimplementedError: not a RuntimeError."""


def test_determinism_error_names_tensorflow_op():
    def fails():
        raise _OpError(
            "Deterministic GPU implementation of unsorted segment reduction "
            "op not available. [[{{node UnsortedSegmentSum}}]]")

    with pytest.raises(DeterminismError, match="UnsortedSegmentSum"):
        run_with_determinism_check(fails)


def test_determinism_error_names_pytorch_op():
    def fails():
        raise RuntimeError(
            "adaptive_max_pool2d_backward_cuda does not have a deterministic "
            "implementation, but you set "
            "'torch.use_deterministic_algorithms(True)'.")

    with pytest.raises(DeterminismError,
                       match="adaptive_max_pool2d_backward_cuda"):
        run_with_determinism_check(fails)


def test_tensorflow_not_yet_supported_is_a_determinism_error():
    def fails():
        raise _OpError("Determinism is not yet supported in GPU "
                       "implementation of Bincount. [[{{node Bincount}}]]")

    with pytest.raises(DeterminismError, match="Bincount"):
        run_with_determinism_check(fails)


def test_seed_errors_are_not_determinism_errors():
    def fails():
        raise _OpError("When determinism is enabled, random ops must have a "
                       "seed specified.")

    with pytest.raises(_OpError):
        run_with_determinism_check(fails)


def test_other_errors_are_not_masked():
    def fails():
        raise _OpError("something else")

    with pytest.raises(_OpError):
        run_with_determinism_check(fails)


class _FakeBackend:
    def environment(self):
        return {"framework_version": "1.0"}


def test_environment_versions_merge_backend_versions():
    versions = environment_versions(_FakeBackend())
    assert {"code_version", "python", "numpy", "gpu_driver",
            "framework_version"} <= set(versions)
