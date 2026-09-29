"""Seeding, op determinism and environment reporting (framework-neutral).

The framework-specific parts (TensorFlow / PyTorch seeds, determinism flags
and library versions) live in the backends.
"""

import importlib.metadata
import logging
import os
import platform
import random
import re
import subprocess
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# TensorFlow / PyTorch errors of ops without a deterministic implementation
# (not e.g. "random ops must have a seed when determinism is enabled").
DETERMINISM_ERROR = re.compile(
    r"deterministic|determinism is not yet supported", re.IGNORECASE)


class DeterminismError(RuntimeError):
    """An op has no deterministic implementation on the current device."""


def seed_python(seed):
    """Seed Python and NumPy. PYTHONHASHSEED only affects interpreters
    started after it is set (e.g. subprocesses)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def _nondeterministic_op(message):
    """Name of the op in a TensorFlow or PyTorch determinism error."""
    match = re.search(r"\{\{node ([^}]+)\}\}", message)  # TensorFlow
    if match is None:  # PyTorch: "<op> does not have a deterministic ..."
        match = re.search(r"(\S+) does not have a deterministic", message)
    return match.group(1) if match else "unknown (see message)"


def run_with_determinism_check(fn, *args, **kwargs):
    """Run `fn`; if an op has no deterministic kernel, name it and stop.

    Determinism is never switched off silently: set `deterministic_ops:
    false` in the config explicitly if you accept non-reproducible runs.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as err:  # TensorFlow and PyTorch raise other types
        message = getattr(err, "message", str(err))
        if isinstance(err, DeterminismError) or not DETERMINISM_ERROR.search(
                message):
            raise
        op = _nondeterministic_op(message)
        logger.error("Op without deterministic kernel: %s\n%s", op, message)
        raise DeterminismError(
            f"Op '{op}' has no deterministic implementation on this "
            f"device: {message}"
        ) from err


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _gpu_driver():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _git(repo, *args, timeout=10):
    try:
        result = subprocess.run(["git", "-C", str(repo), *args],
                                capture_output=True, text=True,
                                timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def code_version():
    """Git commit of this project's code ('-dirty' if tracked files other
    than the notebook changed), or the installed package version outside a
    git checkout of the project."""
    # git searches upward from the folder of this file (package module or
    # single-file export); a checkout of another project does not count.
    here = Path(globals().get("__file__", ".")).resolve().parent
    top = _git(here, "rev-parse", "--show-toplevel")
    if top is None or not (Path(top) / "zona_pellucida").is_dir():
        return package_version("zona-pellucida")
    head = _git(top, "rev-parse", "HEAD")
    if head is None:
        return package_version("zona-pellucida")
    # fcn.ipynb is left out: Colab saves outputs into it.
    status = _git(top, "status", "--porcelain", "--untracked-files=no",
                  "--", ".", ":(exclude)fcn.ipynb", timeout=60)
    if status is None:
        return head + "-unknown"
    return head + ("-dirty" if status else "")


def environment_versions(backend):
    """Versions and hardware that must be reported with the results.

    Bit-exact reruns need the same code, library versions and GPU type:
    Colab assigns A100, L4 or T4 per session, and A100/L4 use TF32 for
    convolutions by default.
    """
    return {
        "code_version": code_version(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "gpu_driver": _gpu_driver(),
        **backend.environment(),
    }


def save_environment(path, versions):
    """Write the environment report as YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(versions, f, sort_keys=False)


def log_versions(backend, path=None):
    """Log the environment and, if `path` is given, save it as YAML."""
    versions = environment_versions(backend)
    for name, version in versions.items():
        logger.info("%s: %s", name, version)
    if path is not None:
        save_environment(path, versions)
    return versions
