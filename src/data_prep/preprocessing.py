"""
Windowing and normalisation utilities for HAPT inertial signals.

Used by:
  - run_mlp.py  (raw-window mode)
  - run_cnn.py  (teammates)
  - run_deep_conv_lstm.py (teammates)
  - src/data_prep/create_features.py (LightGBM feature engineering)
"""
from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from logging import getLogger
import math
import traceback
from typing import List, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import norm
from scipy import stats
from scipy.fftpack import fft
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler

logger = getLogger(__name__)


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


class Preprocess:
    """Handcrafted-feature helpers used by the LightGBM feature pipeline."""

    def __init__(self, fs: int = 50) -> None:
        """
        Args:
            fs (int, default=50): Sampling frequency of sensor signals
        """
        self.fs = fs

    def apply_filter(
        self, signal: pd.DataFrame, filter: str = "median", window: int = 5
    ) -> pd.DataFrame:
        """A denosing filter is applied to remove noise in signals.
        Args:
            signal (pd.DataFrame): Raw signal
            filter (str, default='median'): Filter name is chosen from 'mean', 'median', or 'butterworth'
            window (int, default=5): Length of filter
        Returns:
            signal (pd.DataFrame): Filtered signal
        See Also:
            'butterworth' applies a 3rd order low-pass Butterworth filter with a corner frequency of 20 Hz.
        """
        if filter == "mean":
            signal = signal.rolling(window=window, center=True, min_periods=1).mean()
        elif filter == "median":
            signal = signal.rolling(window=window, center=True, min_periods=1).median()
        elif filter == "butterworth":
            fc = 20  # cutoff frequency
            w = fc / (self.fs / 2)  # Normalize the frequency
            b, a = butter(3, w, "low")  # 3rd order low-pass Butterworth filter
            signal = pd.DataFrame(filtfilt(b, a, signal, axis=0), columns=signal.columns)
        else:
            try:
                raise ValueError("Not defined filter. See Args.")
            except ValueError:
                logger.error(traceback.format_exc())

        return signal

    def normalize(self, signal: pd.DataFrame) -> pd.DataFrame:
        """Apply normalization
        Args:
            signal (pd.DataFrame): Raw signal
        Returns:
            signal (pd.DataFrame): Normalized signal
        """
        df_mean = signal.mean()
        df_std = signal.std()
        signal = (signal - df_mean) / df_std
        return signal

    def segment_signal(
        self,
        signal: pd.DataFrame,
        window_size: int = 128,
        overlap_rate: int = 0.5,
        res_type: str = "dataframe",
    ) -> List[pd.DataFrame]:
        """Sample sensor signals in fixed-width sliding windows of 2.56 sec and 50% overlap (128 readings/window).
        Args:
            signal (pandas.DataFrame): Raw signal
            window_size (int, default=128): Window size of sliding window to segment raw signals.
            overlap_rate (float, default=0.5): Overlap rate of sliding window to segment raw signals.
            res_type (str, default='dataframe'): Type of return value; 'array' or 'dataframe'
        Returns:
            signal_seg (list of pandas.DataFrame): List of segmented sigmal.
        """
        signal_seg = []

        for start_idx in range(0, len(signal) - window_size, int(window_size * overlap_rate)):
            seg = signal.iloc[start_idx : start_idx + window_size].reset_index(drop=True)
            if res_type == "array":
                seg = seg.values
            signal_seg.append(seg)

        if res_type == "array":
            signal_seg = np.array(signal_seg)

        return signal_seg

    def separate_gravity(self, acc: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Separate acceleration signal into body and gravity acceleration signal.
        Another low pass Butterworth filter with a corner frequency of 0.3 Hz is applied.
        Args:
            acc (pd.DataFrame): Segmented acceleration signal
        Returns:
            acc_body (pd.DataFrame): Body acceleration signal
            acc_grav (pd.DataFrame): Gravity acceleration signal
        """
        fc = 0.3  # cutoff frequency
        w = fc / (self.fs / 2)  # Normalize the frequency
        b, a = butter(3, w, "low")  # 3rd order low pass Butterworth filter
        acc_grav = pd.DataFrame(
            filtfilt(b, a, acc, axis=0), columns=acc.columns
        )  # Apply Butterworth filter

        # Substract gravity acceleration from acceleration sigal.
        acc_body = acc - acc_grav
        return acc_body, acc_grav

    def obtain_jerk_signal(self, signal: pd.DataFrame) -> pd.DataFrame:
        """Derive signal to obtain Jerk signals
        Args:
            signal (pd.DataFrame)
        Returns:
            jerk_signal (pd.DataFrame):
        """
        jerk_signal = signal.diff(periods=1)  # Calculate difference
        jerk_signal.iloc[0] = jerk_signal.iloc[1]  # Fillna
        jerk_signal = jerk_signal / (1 / self.fs)  # Derive in time (1 / sampling frequency)
        return jerk_signal

    def obtain_magnitude(self, signal):
        """Calculate the magnitude of these three-dimensional signals using the Euclidean norm
        Args:
            signal (pandas.DataFrame): Three-dimensional signals
        Returns:
            res (pandas.DataFrame): Magnitude of three-dimensional signals
        """
        return pd.DataFrame(norm(signal, ord=2, axis=1))

    def obtain_spectrum(self, signal):
        """Obtain spectrum using Fast Fourier Transform (FFT).
        Args:
            signal (pandas.DataFrame): Time domain signals
        Returns:
            amp (pandas.DataFrame): Amplitude spectrum
            phase (pandas.DataFrame): Phase spectrum
        """
        N = len(signal)
        columns = signal.columns

        for col in columns:
            signal[col] = signal[col] * np.hamming(N)  # hamming window

        F = fft(signal, axis=0)  # Apply FFT
        F = F[: N // 2, :]  # Remove the overlapping part

        amp = np.abs(F)  # Obtain the amplitude spectrum
        amp = amp / N * 2
        amp[0] = amp[0] / 2
        amp = pd.DataFrame(amp, columns=columns)  # Convert array to DataFrame
        phase = np.angle(F)
        phase = pd.DataFrame(phase, columns=columns)  # Convert array to DataFrame

        return amp, phase

    def obtain_ecdf_percentile(self, signal, n_bins=10):
        """Obtain ECDF (empirical cumulative distribution function) percentile values.
        Args:
            signal (DataFrame): Time domain signals
            n_bins (int, default: 10): How many percentiles to use as a feature
        Returns:
            features (array): ECDF percentile values.
        """
        from statsmodels.distributions.empirical_distribution import ECDF

        idx = np.linspace(0, signal.shape[0] - 1, n_bins)  # Take n_bins linspace percentile.
        idx = [int(Decimal(str(ix)).quantize(Decimal("0"), rounding=ROUND_HALF_UP)) for ix in idx]
        features = np.array([])
        for col in signal.columns:
            ecdf = ECDF(signal[col].values)  # fit
            x = ecdf.x[1:]  # Remove -inf
            feat = x[idx]
            features = np.hstack([features, feat])

        return features

    def obtain_mean(self, signal) -> np.ndarray:
        return signal.mean().values

    def obtain_std(self, signal) -> np.ndarray:
        return signal.std().values

    def obtain_mad(self, signal) -> np.ndarray:
        if hasattr(stats, "median_abs_deviation"):
            return stats.median_abs_deviation(signal, axis=0)
        return stats.median_absolute_deviation(signal, axis=0)

    def obtain_max(self, signal) -> np.ndarray:
        return signal.max().values

    def obtain_min(self, signal) -> np.ndarray:
        return signal.min().values

    def obtain_sma(self, signal, window_size=128) -> np.ndarray:
        window_second = window_size / self.fs
        return sum(signal.sum().values - self.obtain_min(signal) * len(signal)) / window_second

    def obtain_energy(self, signal) -> np.ndarray:
        return norm(signal, ord=2, axis=0) ** 2 / len(signal)

    def obtain_iqr(self, signal) -> np.ndarray:
        return signal.quantile(0.75).values - signal.quantile(0.25).values

    def obtain_entropy(self, signal) -> np.ndarray:
        signal = signal - signal.min()
        return stats.entropy(signal)

    def obtain_arCoeff(self, signal) -> np.ndarray:
        from statsmodels.regression.linear_model import burg

        arCoeff = np.array([])
        for col in signal.columns:
            val, _ = burg(signal[col], order=4)
            arCoeff = np.hstack((arCoeff, val))
        return arCoeff

    def obtain_correlation(self, signal) -> np.ndarray:
        if signal.shape[1] == 1:  # Signal dimension is 1
            correlation = np.array([])
        else:  # Signal dimension is 3
            xy = np.corrcoef(signal["x"], signal["y"])[0][1]
            yz = np.corrcoef(signal["y"], signal["z"])[0][1]
            zx = np.corrcoef(signal["z"], signal["x"])[0][1]
            correlation = np.hstack((xy, yz, zx))
        return correlation

    def obtain_maxInds(self, signal) -> np.ndarray:
        return signal.idxmax().values

    def obtain_meanFreq(self, signal) -> np.ndarray:
        meanFreq = np.array([])
        for col in signal.columns:
            val = np.mean(signal[col] * np.arange(len(signal)))
            meanFreq = np.hstack((meanFreq, val))
        return meanFreq

    def obtain_skewness(self, signal) -> np.ndarray:
        return signal.skew().values

    def obtain_kurtosis(self, signal) -> np.ndarray:
        return signal.kurt().values

    def obtain_bandsEnergy(self, signal) -> np.ndarray:
        bandsEnergy = np.array([])
        bins = [0, 4, 8, 12, 16, 20, 24, 29, 34, 39, 44, 49, 54, 59, 64]
        for i in range(len(bins) - 1):
            df = signal.iloc[bins[i] : bins[i + 1]]
            arr = self.obtain_energy(df)
            bandsEnergy = np.hstack((bandsEnergy, arr))
        return bandsEnergy

    def obtain_angle(self, v1, v2) -> np.ndarray:
        length = lambda v: math.sqrt(np.dot(v, v))
        return math.acos(np.dot(v1, v2) / (length(v1) * length(v2)))
