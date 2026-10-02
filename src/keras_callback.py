"""
Custom Keras callbacks shared across MLP, CNN, and DeepConvLSTM pipelines.

Provides:
  - CSVEpochLogger  — append per-epoch metrics to a CSV file
  - TimingCallback  — measure wall-clock time per epoch
  - get_default_callbacks() — convenience factory used by all run_*.py scripts
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

from tensorflow import keras


class CSVEpochLogger(keras.callbacks.Callback):
    """Append per-epoch metrics to a CSV file.

    The header row is written on the first epoch; subsequent epochs are
    appended without re-opening the file.

    Parameters
    ----------
    filepath:
        Destination CSV file.  Parent directories are created automatically.
    separator:
        Column delimiter (default ``,``).
    """

    def __init__(self, filepath: str | Path, separator: str = ",") -> None:
        super().__init__()
        self.filepath  = Path(filepath)
        self.separator = separator
        self._file:   object | None = None
        self._writer: object | None = None

    def on_train_begin(self, logs=None) -> None:
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._file   = open(self.filepath, "w", newline="", encoding="utf-8")
        self._writer = None  # header deferred until we know the metric names

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        logs = logs or {}
        if self._writer is None:
            fieldnames = ["epoch"] + sorted(logs.keys())
            self._writer = csv.DictWriter(
                self._file, fieldnames=fieldnames, delimiter=self.separator
            )
            self._writer.writeheader()

        row = {
            "epoch": epoch + 1,
            **{k: logs.get(k, "") for k in self._writer.fieldnames[1:]},
        }
        self._writer.writerow(row)
        self._file.flush()

    def on_train_end(self, logs=None) -> None:
        if self._file is not None:
            self._file.close()
            self._file   = None
            self._writer = None


class TimingCallback(keras.callbacks.Callback):
    """Record wall-clock time for each training epoch.

    After training, ``callback.epoch_times`` contains a list of elapsed
    seconds (one entry per epoch).  The elapsed time is also injected into
    the ``logs`` dict as ``epoch_time_s`` so CSV loggers pick it up.
    """

    def __init__(self) -> None:
        super().__init__()
        self.epoch_times: list[float] = []
        self._t0: float = 0.0

    def on_epoch_begin(self, epoch: int, logs=None) -> None:
        self._t0 = time.perf_counter()

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        elapsed = time.perf_counter() - self._t0
        self.epoch_times.append(elapsed)
        if logs is not None:
            logs["epoch_time_s"] = round(elapsed, 3)


def get_default_callbacks(
    log_csv_path: str | Path | None = None,
    early_stopping_patience: int = 15,
    reduce_lr_patience: int = 7,
    model_checkpoint_path: str | Path | None = None,
) -> list[keras.callbacks.Callback]:
    """Return a standard callback stack for HAPT model training.

    Parameters
    ----------
    log_csv_path:
        If provided, per-epoch metrics are streamed to this CSV file.
    early_stopping_patience:
        Epochs without improvement in ``val_loss`` before stopping.
    reduce_lr_patience:
        Epochs without improvement before halving the learning rate.
    model_checkpoint_path:
        If provided, best weights (by ``val_accuracy``) are saved here.

    Returns
    -------
    list[keras.callbacks.Callback]
    """
    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-6,
            verbose=1,
        ),
        TimingCallback(),
    ]

    if log_csv_path is not None:
        callbacks.append(CSVEpochLogger(log_csv_path))

    if model_checkpoint_path is not None:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=str(model_checkpoint_path),
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            )
        )

    return callbacks
