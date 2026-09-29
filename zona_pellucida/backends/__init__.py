"""Framework backends: PyTorch Lightning (default) and TensorFlow/Keras.

Both implement the same interface (start_run, environment, build_model,
summary, train, predict, evaluate, load_model), used by experiments.py.
"""

FRAMEWORKS = ("pytorch", "keras")


def load_backend(framework):
    """Import only the requested framework."""
    if framework == "pytorch":
        from zona_pellucida.backends.pytorch.backend import backend
    elif framework == "keras":
        from zona_pellucida.backends.keras.backend import backend
    else:
        raise ValueError(f"Unknown framework '{framework}', use {FRAMEWORKS}")
    return backend
