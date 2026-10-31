"""Load HAPT raw accelerometer/gyroscope files into labeled sensor windows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

TRAIN_SUBJECTS = (1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30)
TEST_SUBJECTS = (2, 4, 9, 10, 12, 13, 18, 20, 24)

FILE_PATTERN = re.compile(r"^(acc|gyro)_exp(\d+)_user(\d+)\.txt$", re.IGNORECASE)


@dataclass(frozen=True)
class WindowedHapt:
    windows: np.ndarray
    labels: np.ndarray
    subjects: np.ndarray


def _read_sensor_file(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError(f"Expected at least three sensor columns in {path}")
    values = values[:, :3]
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite sensor values in {path}")
    # Match the common HAPT conditioning pipeline: centered five-sample median
    # smoothing followed by a third-order, 20 Hz low-pass Butterworth filter.
    filtered = pd.DataFrame(values).rolling(window=5, center=True, min_periods=1).median().to_numpy()
    if len(filtered) > 24:
        numerator, denominator = butter(3, 20, btype="low", fs=50)
        filtered = filtfilt(numerator, denominator, filtered, axis=0)
    return filtered.astype(np.float32)


def load_hapt_windows(
    data_dir: str | Path,
    window_size: int = 128,
    stride: int = 64,
) -> WindowedHapt:
    """Build majority-labeled windows from UCI HAPT raw files.

    ``data_dir`` may point to the extracted archive root or its ``RawData``
    directory. Label intervals in ``labels.txt`` use one-based inclusive sample
    indices and activity IDs 1..12; returned targets are zero-based.
    Windows do not cross recording boundaries, and only fully labeled windows
    are retained.
    """
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"HAPT data directory does not exist: {root}")
    if window_size < 1 or stride < 1:
        raise ValueError("window_size and stride must be positive")

    label_files = list(root.rglob("labels.txt"))
    if len(label_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one HAPT labels.txt below {root}, found {len(label_files)}"
        )
    raw_labels = np.loadtxt(label_files[0], dtype=np.int64)
    if raw_labels.ndim == 1:
        raw_labels = raw_labels.reshape(1, -1)
    if raw_labels.ndim != 2 or raw_labels.shape[1] < 5:
        raise ValueError("labels.txt must contain experiment, user, activity, start, and end columns")

    annotations: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for experiment, user, activity, start, end, *_ in raw_labels.tolist():
        if not 1 <= activity <= 12 or start < 1 or end < start:
            raise ValueError(f"Invalid HAPT label row: {(experiment, user, activity, start, end)}")
        annotations.setdefault((experiment, user), []).append((activity, start, end))

    recordings: dict[tuple[str, int, int], Path] = {}
    for path in root.rglob("*.txt"):
        match = FILE_PATTERN.match(path.name)
        if match:
            sensor, experiment, user = match.groups()
            recordings[(sensor.lower(), int(experiment), int(user))] = path

    experiment_users = sorted({(exp, user) for _, exp, user in recordings})
    windows: list[np.ndarray] = []
    labels: list[int] = []
    subjects: list[int] = []
    for experiment, user in experiment_users:
        acc_path = recordings.get(("acc", experiment, user))
        gyro_path = recordings.get(("gyro", experiment, user))
        if acc_path is None or gyro_path is None:
            raise FileNotFoundError(
                f"Missing acc/gyro pair for experiment {experiment}, user {user}"
            )
        acc, gyro = _read_sensor_file(acc_path), _read_sensor_file(gyro_path)
        if len(acc) != len(gyro):
            raise ValueError(f"Sample count mismatch for experiment {experiment}, user {user}")
        signal = np.concatenate((acc, gyro), axis=1)
        sample_labels = np.zeros(len(signal), dtype=np.int16)
        for activity, start, end in annotations.get((experiment, user), []):
            if end > len(signal):
                raise ValueError(f"Label interval ends past recording length for user {user}")
            sample_labels[start - 1 : end] = activity

        for begin in range(0, len(signal) - window_size + 1, stride):
            end = begin + window_size
            labels_in_window = sample_labels[begin:end]
            if np.any(labels_in_window == 0):
                continue
            counts = np.bincount(labels_in_window, minlength=13)[1:]
            windows.append(signal[begin:end])
            labels.append(int(np.argmax(counts)))
            subjects.append(user)

    if not windows:
        raise ValueError(f"No fully labeled {window_size}-sample windows found below {root}")
    return WindowedHapt(
        windows=np.asarray(windows, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        subjects=np.asarray(subjects, dtype=np.int64),
    )


def normalize_windows(
    train_windows: np.ndarray,
    *other_windows: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Normalize each sensor channel using training data only."""
    mean, std = channel_statistics(train_windows)
    return tuple(apply_channel_statistics(windows, mean, std) for windows in (train_windows, *other_windows))


def channel_statistics(train_windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return training-only per-channel mean and standard deviation."""
    mean = train_windows.mean(axis=(0, 1), keepdims=True)
    std = train_windows.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_channel_statistics(
    windows: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Apply previously fitted per-channel z-score statistics."""
    return ((windows - mean) / std).astype(np.float32)
