"""TensorFlow / Keras implementation of the backend interface."""

import logging
from pathlib import Path

import keras
import tensorflow as tf

from zona_pellucida.backends.keras.losses import get_loss
from zona_pellucida.backends.keras.model import fcn_8
from zona_pellucida.backends.keras.trainer import (
    compile_model,
    load_trained_model,
    save_model,
    train_model,
)
from zona_pellucida.reproducibility import package_version, seed_python

logger = logging.getLogger(__name__)


class KerasBackend:
    """What the experiments need from a deep-learning framework."""

    name = "keras"
    model_suffix = ".keras"

    def prepare_device(self):
        """Avoid OOM errors: allocate GPU memory on demand. Must run
        before the GPU is initialised."""
        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as err:
                logger.warning("No memory growth on %s: %s", gpu, err)

    def start_run(self, seed, deterministic=True):
        """Fresh Keras session and seeds, so every run starts identically."""
        keras.backend.clear_session()
        seed_python(seed)
        # Seeds python `random`, NumPy and TensorFlow.
        keras.utils.set_random_seed(seed)
        if deterministic:
            tf.config.experimental.enable_op_determinism()
        logger.info("Seed %d (op determinism: %s)", seed, deterministic)

    def environment(self):
        build = tf.sysconfig.get_build_info()
        return {
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
            "gpus": [
                tf.config.experimental.get_device_details(d).get(
                    "device_name", d.name)
                for d in tf.config.list_physical_devices("GPU")
            ],
            # CUDA/cuDNN TensorFlow was built for; the libraries actually
            # loaded are the nvidia-* pip packages.
            "cuda_build": build.get("cuda_version"),
            "cudnn_build": build.get("cudnn_version"),
            "cuda_runtime": package_version("nvidia-cuda-runtime-cu12"),
            "cudnn_runtime": package_version("nvidia-cudnn-cu12"),
            "tf32": tf.config.experimental.tensor_float_32_execution_enabled(),
        }

    def build_model(self, cfg):
        size = cfg.data.size
        return fcn_8(size, size, 1, dropout=cfg.model.dropout)

    def summary(self, model):
        model.summary()

    def train(self, loss, class_weights, train, val, cfg, name, log_dir=None):
        """Train a new model; returns (model, history, best epoch, path)."""
        model = self.build_model(cfg)
        compile_model(model, get_loss(loss, class_weights),
                      learning_rate=cfg.train.learning_rate,
                      threshold=cfg.evaluation.threshold)
        history, best_epoch = train_model(
            model, train.images, train.masks, cfg.train,
            validation_data=(val.images, val.masks) if len(val) else None,
            log_dir=log_dir, seed=cfg.seed,
        )
        model_path = Path(cfg.output_dir) / "models" / f"{name}.keras"
        save_model(model, model_path)
        return model, history, best_epoch, model_path

    def predict(self, model, images, batch_size=16):
        """Sigmoid probabilities, shape (N, H, W, 1)."""
        return model.predict(images, batch_size=batch_size, verbose=0)

    def evaluate(self, model, images, masks, batch_size=16):
        """Loss and Keras metrics (threshold 0.5), evaluated once."""
        return model.evaluate(images, masks, batch_size=batch_size,
                              return_dict=True, verbose=0)

    def load_model(self, path):
        return load_trained_model(path)


backend = KerasBackend()
