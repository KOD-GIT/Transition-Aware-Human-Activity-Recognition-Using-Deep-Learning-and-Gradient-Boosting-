"""
Load the HAPT dataset — all 12 activity + postural-transition classes.

Expected dataset layout::

    data/hapt_data_set/
        Train/
            X_train.txt       # pre-computed 561-feature matrix
            y_train.txt       # integer labels 1–12
            Inertial Signals/ # optional raw signal files
        Test/
            X_test.txt
            y_test.txt
            Inertial Signals/

Labels are stored 1-indexed in the files; this loader converts them to
0-indexed (0–11) for Keras / scikit-learn compatibility.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# ── Label catalogue (1-indexed, as stored in y_*.txt) ─────────────────────── #
ACTIVITY_LABELS: dict[int, str] = {
    1:  "WALKING",
    2:  "WALKING_UPSTAIRS",
    3:  "WALKING_DOWNSTAIRS",
    4:  "SITTING",
    5:  "STANDING",
    6:  "LAYING",
    # Postural transitions ↓
    7:  "STAND_TO_SIT",
    8:  "SIT_TO_STAND",
    9:  "SIT_TO_LIE",
    10: "LIE_TO_SIT",
    11: "STAND_TO_LIE",
    12: "LIE_TO_STAND",
}

TRANSITION_CLASSES: tuple[int, ...] = (7, 8, 9, 10, 11, 12)
BASIC_ACTIVITY_CLASSES: tuple[int, ...] = (1, 2, 3, 4, 5, 6)

# Raw inertial signal names (9 channels)
_SIGNAL_NAMES = [
    "body_acc_x",  "body_acc_y",  "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]


# ── Private helpers ────────────────────────────────────────────────────────── #

def _find_split_dir(root: Path, split: str) -> Path:
    """Locate the train/test split directory, tolerating case variants."""
    for candidate in (split, split.lower(), split.upper(), split.capitalize()):
        d = root / candidate
        if d.exists():
            return d
    raise FileNotFoundError(
        f"Split directory '{split}' not found under {root}.\n"
        "Expected one of: Train/, train/, TEST/, test/"
    )


def _load_raw_signals(split_dir: Path) -> np.ndarray | None:
    """Load individual inertial-signal files → ndarray (N, T, 9)."""
    # Try both 'Inertial Signals/' and flat layout
    for sig_dir in (split_dir / "Inertial Signals", split_dir):
        split_tag = split_dir.name.lower()  # 'train' or 'test'
        arrays = []
        for sig in _SIGNAL_NAMES:
            fname = sig_dir / f"{sig}_{split_tag}.txt"
            if not fname.exists():
                break
            arrays.append(np.loadtxt(fname))
        if len(arrays) == len(_SIGNAL_NAMES):
            return np.stack(arrays, axis=-1).astype(np.float32)
    return None


def _load_feature_matrix(split_dir: Path) -> np.ndarray | None:
    """Load pre-computed feature matrix (X_train.txt / X_test.txt)."""
    cap = split_dir.name.capitalize()  # e.g. 'Train'
    for fname in (
        split_dir / f"X_{cap}.txt",
        split_dir / f"X_{cap.lower()}.txt",
        split_dir / "X.txt",
    ):
        if fname.exists():
            return np.loadtxt(fname, dtype=np.float32)
    return None


def _load_labels(split_dir: Path) -> np.ndarray:
    """Load labels from y_*.txt and convert to 0-indexed."""
    cap = split_dir.name.capitalize()
    for fname in (
        split_dir / f"y_{cap}.txt",
        split_dir / f"y_{cap.lower()}.txt",
        split_dir / "y.txt",
    ):
        if fname.exists():
            labels = np.loadtxt(fname, dtype=np.int32)
            return labels - 1  # 1-indexed → 0-indexed
    raise FileNotFoundError(f"Label file not found in {split_dir}")


# ── Public API ─────────────────────────────────────────────────────────────── #

def load_hapt(
    data_root: str | Path = "data/hapt_data_set",
    prefer_raw_signals: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load HAPT for all 12 classes (6 activities + 6 postural transitions).

    Parameters
    ----------
    data_root:
        Root directory containing ``Train/`` and ``Test/`` sub-folders.
    prefer_raw_signals:
        When *True*, attempt to load the per-channel inertial signal files
        (shape ``N × T × 9``) instead of the pre-computed 561-feature
        matrix.  Falls back to the feature matrix if signal files are absent.

    Returns
    -------
    X_train, y_train, X_test, y_test
        Arrays with labels **0-indexed** (0–11).
    """
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {root}\n"
            "Place the HAPT dataset under data/hapt_data_set/ — see README."
        )

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for split in ("Train", "Test"):
        split_dir = _find_split_dir(root, split)

        X: np.ndarray | None = None
        if prefer_raw_signals:
            X = _load_raw_signals(split_dir)
            if X is None:
                print(
                    f"[load_hapt] Raw signals not found for '{split}'; "
                    "falling back to pre-computed features."
                )

        if X is None:
            X = _load_feature_matrix(split_dir)
        if X is None:
            raise FileNotFoundError(
                f"No feature matrix (X_*.txt) found in {split_dir}"
            )

        y = _load_labels(split_dir)
        result[split] = (X, y)

    X_train, y_train = result["Train"]
    X_test,  y_test  = result["Test"]

    # ── Sanity-check: all 12 classes ──────────────────────────────────────── #
    all_classes = set(np.unique(y_train)) | set(np.unique(y_test))
    if len(all_classes) < 12:
        missing_1idx = sorted({i + 1 for i in range(12)} - {c + 1 for c in all_classes})
        missing_names = [ACTIVITY_LABELS[c] for c in missing_1idx]
        print(
            f"[load_hapt] WARNING: {len(all_classes)}/12 classes found. "
            f"Missing 1-indexed classes {missing_1idx}: {missing_names}"
        )
    else:
        print(f"[load_hapt] All 12 classes present — "
              f"train: {len(y_train)}, test: {len(y_test)}")

    return X_train, y_train, X_test, y_test


def load_features() -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Dict[int, str],
    Dict[str, int],
]:
    """Load handcrafted features generated by ``run_generate_features.py``.

    This loader is used by the LightGBM pipeline. It reads pickle/npy files
    under ``data/my_dataset/`` and currently keeps the six basic activity
    classes that pipeline was trained on. The 12-class ``load_hapt()`` API
    used by MLP / DeepConvLSTM is unchanged.

    Returns:
        X_train, X_test, y_train, y_test, label2act, act2label
    """
    data_dir = Path(__file__).resolve().parents[2] / "data"
    X_train = pd.read_pickle(data_dir / "my_dataset" / "X_train.pickle")
    y_train = pd.DataFrame(np.load(data_dir / "my_dataset" / "y_train.npy"))
    X_test = pd.read_pickle(data_dir / "my_dataset" / "X_test.pickle")
    y_test = pd.DataFrame(np.load(data_dir / "my_dataset" / "y_test.npy"))

    activity_labels = pd.read_table(
        data_dir / "hapt_data_set" / "activity_labels.txt", header=None
    ).values.flatten()
    activity_labels = np.array([label.rstrip().split() for label in activity_labels])
    label2act, act2label = {}, {}
    for label, activity in activity_labels:
        label2act[int(label)] = activity
        act2label[activity] = int(label)

    class_names_inc = [
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING",
    ]
    class_ids_inc = [act2label[c] for c in class_names_inc]

    idx_train = y_train[y_train[0].isin(class_ids_inc)].index
    X_train = X_train.iloc[idx_train].reset_index(drop=True)
    y_train = y_train.iloc[idx_train].reset_index(drop=True)

    idx_test = y_test[y_test[0].isin(class_ids_inc)].index
    X_test = X_test.iloc[idx_test].reset_index(drop=True)
    y_test = y_test.iloc[idx_test].reset_index(drop=True)

    # LightGBM expects contiguous ids 0-5; remap LAYING from 6 to 0.
    rep_activity = label2act[6]
    label2act[0] = rep_activity
    label2act.pop(6)
    act2label[rep_activity] = 0

    y_train = y_train.replace(6, 0)
    y_test = y_test.replace(6, 0)

    return X_train, X_test, y_train, y_test, label2act, act2label
