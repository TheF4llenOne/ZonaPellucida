"""Compiling, training, saving and reloading the Keras FCN."""

import importlib.util
import logging
from pathlib import Path

import keras
import tensorflow as tf
from keras.callbacks import EarlyStopping, TensorBoard

from zona_pellucida.backends.keras.losses import WeightedBinaryCrossentropy
from zona_pellucida.backends.keras.metrics import (
    DiceCoefficient,
    ForegroundIoU,
    training_metrics,
)

logger = logging.getLogger(__name__)

# Needed by keras.models.load_model for models trained with these objects.
CUSTOM_OBJECTS = {
    "WeightedBinaryCrossentropy": WeightedBinaryCrossentropy,
    "DiceCoefficient": DiceCoefficient,
    "ForegroundIoU": ForegroundIoU,
}


def compile_model(model, loss, learning_rate=0.001, threshold=0.5):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=training_metrics(threshold),
    )
    return model


def make_callbacks(train_cfg, log_dir=None, has_validation=True):
    callbacks = []
    early_stopping = None
    if has_validation:
        # Restores the weights of the best validation epoch at the end of
        # training. Without early stopping, patience = epochs means it never
        # stops, so all epochs are trained.
        stop = train_cfg.early_stopping
        early_stopping = EarlyStopping(
            monitor=train_cfg.monitor,
            mode=train_cfg.mode,
            patience=stop.patience if stop.enabled else train_cfg.epochs,
            start_from_epoch=stop.start_from_epoch if stop.enabled else 0,
            restore_best_weights=True,
            verbose=1,
        )
        callbacks.append(early_stopping)
    tensorboard = train_cfg.tensorboard.enabled and log_dir is not None
    if tensorboard and importlib.util.find_spec("tensorboard") is None:
        logger.warning("TensorBoard is not installed; logging disabled")
        tensorboard = False
    if tensorboard:
        callbacks.append(TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=train_cfg.tensorboard.histogram_freq,
            write_graph=True, update_freq="epoch",
        ))
    return callbacks, early_stopping


def shuffled_dataset(X, y, batch_size, seed):
    """Batches in a new (seeded, reproducible) order every epoch.

    Keras 3 `fit(..., shuffle=True)` on NumPy arrays repeats the same
    permutation in every epoch once a global seed is set, so the
    reshuffling is done with tf.data instead.
    """
    return (tf.data.Dataset.from_tensor_slices((X, y))
            .shuffle(len(X), seed=seed, reshuffle_each_iteration=True)
            .batch(batch_size))


def train_model(model, X_train, y_train, train_cfg, validation_data=None,
                log_dir=None, seed=None):
    """Fit the compiled model. Returns (history dict, best epoch, 1-based).

    Validation data (never the test set) selects the best epoch, whose
    weights are kept (and drives the optional early stopping). With
    `train_cfg.shuffle` the training set is reshuffled every epoch
    (reproducibly, from `seed`); otherwise the batch order is fixed.
    """
    callbacks, early_stopping = make_callbacks(
        train_cfg, log_dir, has_validation=validation_data is not None
    )
    if train_cfg.shuffle:
        # The dataset reshuffles itself; shuffle=False only silences the
        # Keras warning about shuffling a tf.data dataset.
        data = {"x": shuffled_dataset(X_train, y_train,
                                      train_cfg.batch_size, seed),
                "shuffle": False}
    else:
        data = {"x": X_train, "y": y_train,
                "batch_size": train_cfg.batch_size, "shuffle": False}
    history = model.fit(
        **data,
        epochs=train_cfg.epochs,
        verbose=1,
        validation_data=validation_data,
        callbacks=callbacks,
    )
    epochs_run = len(history.history["loss"])
    if early_stopping is not None and early_stopping.best_weights is not None:
        best_epoch = early_stopping.best_epoch + 1
    else:
        best_epoch = epochs_run
    logger.info("Trained %d epochs; weights of epoch %d kept",
                epochs_run, best_epoch)
    return history.history, best_epoch


def save_model(model, path):
    """Keras 3 needs the `.keras` extension for full-model saving."""
    path = Path(path)
    if path.suffix != ".keras":
        raise ValueError(f"Model path must end in .keras, got {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    logger.info("Saved model to %s", path)


def load_trained_model(path, compile=True):
    """Reload a saved model including the custom loss and metrics."""
    return keras.models.load_model(
        path, custom_objects=CUSTOM_OBJECTS, compile=compile
    )
