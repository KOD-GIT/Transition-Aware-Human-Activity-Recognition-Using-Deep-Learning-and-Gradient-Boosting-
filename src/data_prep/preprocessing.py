"""
Windowing and normalisation utilities for HAPT inertial signals.

Used by:
  - run_mlp.py  (raw-window mode)
  - run_cnn.py  (teammates)
  - run_deep_conv_lstm.py (teammates)
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


def sliding_window(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int = 128,
    step_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Segment a continuous sensor stream into fixed-length windows.

    Parameters
    ----------
    X:
        Raw sensor data, shape ``(N_samples, n_channels)``.
    y:
        Per-sample label array, shape ``(N_samples,)``.
    window_size:
        Number of time-steps per window.  128 ≈ 2.56 s at 50 Hz.
    step_size:
        Stride between consecutive window starts.  64 → 50 % overlap.

    Returns
    -------
    X_windows : ndarray, shape ``(n_windows, window_size, n_channels)``
    y_windows : ndarray, shape ``(n_windows,)``
        Majority-vote label assigned to each window.
    """
    if X.ndim != 2:
        raise ValueError(f"sliding_window expects 2-D input (N, C), got {X.ndim}-D")

    n_samples, n_channels = X.shape
    windows: list[np.ndarray] = []
    labels:  list[int]       = []

    start = 0
    while start + window_size <= n_samples:
        end = start + window_size
        windows.append(X[start:end])
        # majority vote — handles multi-class gracefully
        counts = np.bincount(y[start:end])
        labels.append(int(counts.argmax()))
        start += step_size

    if not windows:
        raise ValueError(
            f"No windows extracted. n_samples={n_samples} < window_size={window_size}."
        )

    return (
        np.array(windows, dtype=np.float32),
        np.array(labels,  dtype=np.int32),
    )


def normalize_features(
    X_train: np.ndarray,
    X_test:  np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Standardise feature matrices (zero mean, unit variance).

    The scaler is *fit on training data only* to prevent data leakage.

    Parameters
    ----------
    X_train, X_test:
        2-D feature matrices of shape ``(N, F)``.

    Returns
    -------
    X_train_scaled, X_test_scaled : float32 ndarrays
    scaler : fitted StandardScaler (kept for inference / persistence)
    """
    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError(
            "normalize_features expects 2-D matrices. "
            "Call flatten_windows() first if input is 3-D."
        )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s  = scaler.transform(X_test).astype(np.float32)
    return X_train_s, X_test_s, scaler


def flatten_windows(X_windows: np.ndarray) -> np.ndarray:
    """Flatten ``(n_windows, T, C)`` → ``(n_windows, T*C)`` for MLP input."""
    if X_windows.ndim != 3:
        raise ValueError(f"Expected 3-D array, got shape {X_windows.shape}")
    n = X_windows.shape[0]
    return X_windows.reshape(n, -1).astype(np.float32)
