"""PyTorch Lightning implementation of the backend interface."""

import logging
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torchmetrics
from lightning.pytorch.utilities.model_summary import ModelSummary

from zona_pellucida.backends.pytorch.data import make_loader, to_numpy
from zona_pellucida.backends.pytorch.module import (
    SegmentationModule,
    segmentation_metric_collection,
)
from zona_pellucida.backends.pytorch.trainer import fit
from zona_pellucida.reproducibility import seed_python

logger = logging.getLogger(__name__)


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TorchBackend:
    """What the experiments need from a deep-learning framework."""

    name = "pytorch"
    model_suffix = ".ckpt"

    def prepare_device(self):
        pass  # PyTorch allocates GPU memory on demand

    def start_run(self, seed, deterministic=True):
        """Seeds and determinism flags, so every run starts identically."""
        seed_python(seed)
        # Seeds python `random`, NumPy and PyTorch (CPU and CUDA).
        L.seed_everything(seed, workers=True, verbose=False)
        torch.use_deterministic_algorithms(deterministic)
        torch.backends.cudnn.benchmark = False
        logger.info("Seed %d (deterministic algorithms: %s)", seed,
                    deterministic)

    def environment(self):
        gpus = [torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())]
        cudnn = torch.backends.cudnn.version() if gpus else None
        if cudnn is not None:  # e.g. 91900 -> 9.19.0
            cudnn = f"{cudnn // 10000}.{cudnn // 100 % 100}.{cudnn % 100}"
        return {
            "torch": str(torch.__version__),
            "lightning": str(L.__version__),
            "torchmetrics": str(torchmetrics.__version__),
            "gpus": gpus,
            "cuda": torch.version.cuda,
            "cudnn": cudnn,
            # TF32 convolutions on Ampere or newer GPUs (A100, L4).
            "tf32": torch.backends.cudnn.allow_tf32 if gpus else None,
        }

    def build_model(self, cfg, loss="unweighted", class_weights=None):
        if loss == "unweighted":
            class_weights = {0: 1.0, 1: 1.0}  # stored in the checkpoint
        elif class_weights is None:
            raise ValueError("The weighted run needs class weights")
        return SegmentationModule(
            loss=loss, w0=class_weights[0], w1=class_weights[1],
            learning_rate=cfg.train.learning_rate, dropout=cfg.model.dropout,
            threshold=cfg.evaluation.threshold,
        )

    def summary(self, model):
        logger.info("Model summary:\n%s", ModelSummary(model, max_depth=3))

    def train(self, loss, class_weights, train, val, cfg, name, log_dir=None):
        """Train a new model; returns (model with the best weights,
        history, best epoch, checkpoint path)."""
        module = self.build_model(cfg, loss, class_weights)
        batch_size = cfg.train.batch_size
        train_loader = make_loader(train.images, train.masks, batch_size,
                                   shuffle=cfg.train.shuffle, seed=cfg.seed)
        val_loader = None
        if len(val):
            val_loader = make_loader(val.images, val.masks, batch_size)
        history, best_epoch, model_path = fit(
            module, train_loader, val_loader, cfg.train,
            model_dir=Path(cfg.output_dir) / "models", name=name,
            log_dir=log_dir, deterministic=cfg.deterministic_ops,
        )
        return self.load_model(model_path), history, best_epoch, model_path

    def predict(self, model, images, batch_size=16):
        """Sigmoid probabilities, shape (N, H, W, 1) like Keras."""
        model = model.to(_device()).eval()
        loader = make_loader(images, np.zeros_like(images), batch_size)
        with torch.no_grad():
            probabilities = [torch.sigmoid(model(x.to(model.device)))
                             for x, _ in loader]
        return to_numpy(torch.cat(probabilities))

    def evaluate(self, model, images, masks, batch_size=16):
        """Loss and metrics (threshold 0.5) on a data set, evaluated once."""
        model = model.to(_device()).eval()
        metrics = segmentation_metric_collection(
            model.hparams.threshold).to(model.device)
        total, count = 0.0, 0
        with torch.no_grad():
            for x, y in make_loader(images, masks, batch_size):
                x, y = x.to(model.device), y.to(model.device)
                logits = model(x)
                total += float(model.loss_fn(logits, y)) * len(x)
                count += len(x)
                metrics.update(torch.sigmoid(logits), y.int())
        scores = {name: float(value)
                  for name, value in metrics.compute().items()}
        return {"loss": total / count, **scores}

    def load_model(self, path):
        """Rebuild the module (loss, weights) from a checkpoint."""
        return SegmentationModule.load_from_checkpoint(
            path, map_location="cpu")


backend = TorchBackend()
