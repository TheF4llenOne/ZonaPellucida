"""Training the LightningModule: callbacks, trainer, best checkpoint."""

import importlib.util
import logging
import warnings
from collections import defaultdict
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

logger = logging.getLogger(__name__)

# The data are small in-memory tensors: worker processes are not needed.
warnings.filterwarnings("ignore", ".*does not have many workers.*")


class History(L.Callback):
    """Per-epoch metrics with the names of Keras' history (loss, iou, ...,
    val_loss, val_iou, ...), so plots and files are the same for both
    frameworks. Validation runs before `on_train_epoch_end`."""

    def __init__(self):
        super().__init__()
        self.history = defaultdict(list)

    def on_train_epoch_end(self, trainer, pl_module):
        for name, value in sorted(trainer.callback_metrics.items()):
            self.history[name].append(float(value))


class WeightHistograms(L.Callback):
    """Histograms of all weights in TensorBoard every `every_n_epochs`
    epochs (Keras: TensorBoard(histogram_freq=...))."""

    def __init__(self, every_n_epochs=1):
        super().__init__()
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if not isinstance(trainer.logger, TensorBoardLogger) or (
                epoch % self.every_n_epochs):
            return
        for name, parameter in pl_module.named_parameters():
            trainer.logger.experiment.add_histogram(name, parameter, epoch)


class _AfterWarmup:
    """Skips a callback's checks during the first `start_from_epoch`
    epochs (like Keras' `start_from_epoch`). Lightning checks in one of
    these two hooks, depending on how often validation runs."""

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch >= self.start_from_epoch:
            super().on_train_epoch_end(trainer, pl_module)

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch >= self.start_from_epoch:
            super().on_validation_end(trainer, pl_module)


class WarmupEarlyStopping(_AfterWarmup, EarlyStopping):
    def __init__(self, start_from_epoch=0, **kwargs):
        super().__init__(**kwargs)
        self.start_from_epoch = start_from_epoch


class WarmupModelCheckpoint(_AfterWarmup, ModelCheckpoint):
    """Best-epoch checkpoint that, like Keras, never keeps a warm-up
    epoch."""

    def __init__(self, start_from_epoch=0, **kwargs):
        super().__init__(**kwargs)
        self.start_from_epoch = start_from_epoch


def make_callbacks(train_cfg, model_dir, name, has_validation=True):
    """History, best-epoch checkpoint, optional early stopping and weight
    histograms."""
    history = History()
    stop = train_cfg.early_stopping
    early_stopping = stop.enabled and has_validation
    # Keeps the weights of the epoch with the best validation value of
    # `monitor` (without validation: the last epoch) as
    # <model_dir>/<name>.ckpt.
    checkpoint = WarmupModelCheckpoint(
        start_from_epoch=stop.start_from_epoch if early_stopping else 0,
        dirpath=model_dir, filename=name,
        monitor=train_cfg.monitor if has_validation else None,
        mode=train_cfg.mode, save_top_k=1, save_weights_only=True,
        auto_insert_metric_name=False, enable_version_counter=False,
    )
    callbacks = [history, checkpoint]
    if early_stopping:
        callbacks.append(WarmupEarlyStopping(
            monitor=train_cfg.monitor, mode=train_cfg.mode,
            patience=stop.patience, start_from_epoch=stop.start_from_epoch,
            verbose=True,
        ))
    if train_cfg.tensorboard.enabled and train_cfg.tensorboard.histogram_freq:
        callbacks.append(
            WeightHistograms(train_cfg.tensorboard.histogram_freq))
    return callbacks, history, checkpoint


def make_logger(train_cfg, log_dir):
    if not train_cfg.tensorboard.enabled or log_dir is None:
        return False
    if importlib.util.find_spec("tensorboard") is None:
        logger.warning("TensorBoard is not installed; logging disabled")
        return False
    log_dir = Path(log_dir)
    return TensorBoardLogger(save_dir=log_dir.parent.parent,
                             name=log_dir.parent.name, version=log_dir.name)


def fit(module, train_loader, val_loader, train_cfg, model_dir, name,
        log_dir=None, deterministic=True):
    """Train; returns (history dict, best epoch (1-based), checkpoint path).

    Validation data (never the test set) selects the best epoch, whose
    weights are saved (and drives the optional early stopping).
    """
    callbacks, history, checkpoint = make_callbacks(
        train_cfg, model_dir, name, has_validation=val_loader is not None)
    trainer = L.Trainer(
        max_epochs=train_cfg.epochs,
        accelerator="auto",
        devices=1,
        # torch.use_deterministic_algorithms(True): an op without a
        # deterministic implementation raises instead of running.
        deterministic=deterministic,
        callbacks=callbacks,
        logger=make_logger(train_cfg, log_dir),
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        enable_model_summary=False,
    )
    trainer.fit(module, train_loader, val_loader)
    history = dict(history.history)
    epochs_run = len(history["loss"])
    path = Path(checkpoint.best_model_path) if checkpoint.best_model_path \
        else None
    if path is None:
        # Nothing selected (training ended in the warm-up): keep the last
        # epoch, as Keras does.
        path = Path(model_dir) / f"{name}.ckpt"
        trainer.save_checkpoint(path, weights_only=True)
        best_epoch = epochs_run
    else:
        # Our own checkpoint file, so loading it fully is safe.
        best_epoch = torch.load(path, map_location="cpu",
                                weights_only=False)["epoch"] + 1
    logger.info("Trained %d epochs; weights of epoch %d kept",
                epochs_run, best_epoch)
    return history, best_epoch, path
