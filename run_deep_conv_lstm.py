"""Train and evaluate DeepConvLSTM using the shared subject-independent HAPT split."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold

from models.deep_conv_lstm import build_deep_conv_lstm
from src.data_prep.hapt_windows import (
    TEST_SUBJECTS,
    TRAIN_SUBJECTS,
    apply_channel_statistics,
    channel_statistics,
    load_hapt_windows,
)


def metric_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(12), zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "transition_recall_mean": float(recall[6:].mean()),
        "transition_f1_mean": float(f1[6:].mean()),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=np.arange(12)).tolist(),
    }


def class_weights(labels: np.ndarray) -> dict[int, float]:
    """Compute normalized inverse-frequency weights for labels 0 through 11."""
    counts = np.bincount(labels, minlength=12).astype(np.float64)
    present = counts > 0
    weights = np.zeros(12, dtype=np.float64)
    weights[present] = len(labels) / (present.sum() * counts[present])
    return {class_id: float(weight) for class_id, weight in enumerate(weights)}


def fit_fold(
    windows: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    args: argparse.Namespace,
    fold_number: int,
) -> tuple[tf.keras.Model, np.ndarray, np.ndarray, int, float, float, np.ndarray]:
    raw_train = windows[train_indices]
    mean, std = channel_statistics(raw_train)
    train_x = apply_channel_statistics(raw_train, mean, std)
    validation_x = apply_channel_statistics(windows[validation_indices], mean, std)
    train_y = labels[train_indices]
    validation_y = labels[validation_indices]

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(args.seed + fold_number)
    model = build_deep_conv_lstm(
        window_size=train_x.shape[1],
        input_channels=train_x.shape[2],
        num_classes=12,
        learning_rate=args.learning_rate,
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        restore_best_weights=True,
        mode="min",
        verbose=1,
    )
    started = time.perf_counter()
    history = model.fit(
        train_x,
        train_y,
        validation_data=(validation_x, validation_y),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weights(train_y),
        callbacks=[early_stopping],
        shuffle=True,
        verbose=args.verbose,
    )
    duration = time.perf_counter() - started
    validation_losses = np.asarray(history.history["val_loss"], dtype=np.float64)
    best_epoch = int(validation_losses.argmin()) + 1
    best_loss = float(validation_losses.min())
    validation_probabilities = model.predict(validation_x, batch_size=args.batch_size, verbose=0)
    return model, mean, std, best_epoch, best_loss, duration, validation_probabilities


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/hapt_data_set"))
    parser.add_argument("--output", type=Path, default=Path("logs/deep_conv_lstm_results.json"))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=1)
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.folds < 2:
        parser.error("epochs, patience, batch size, and folds must be positive (folds >= 2)")

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    dataset = load_hapt_windows(args.data_dir)
    print(
        f"Loaded {len(dataset.labels):,} windows from {len(np.unique(dataset.subjects))} subjects; "
        f"window shape={dataset.windows.shape[1:]}"
    )

    # Match the fixed 21/9 user partition already used in this repository.
    train_mask = np.isin(dataset.subjects, TRAIN_SUBJECTS)
    test_mask = np.isin(dataset.subjects, TEST_SUBJECTS)
    if not np.all(train_mask | test_mask) or np.any(train_mask & test_mask):
        raise RuntimeError("Dataset contains a subject outside the shared train/test split")
    train_indices = np.flatnonzero(train_mask)
    test_indices = np.flatnonzero(test_mask)
    train_subjects = np.unique(dataset.subjects[train_indices])
    test_subjects = np.unique(dataset.subjects[test_indices])
    if np.intersect1d(train_subjects, test_subjects).size:
        raise RuntimeError("Subject leakage between training and test partitions")
    print(f"Train subjects: {train_subjects.tolist()}; held-out test subjects: {test_subjects.tolist()}")

    train_x = dataset.windows[train_indices]
    train_y = dataset.labels[train_indices]
    train_groups = dataset.subjects[train_indices]
    cross_validator = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof_predictions = np.full(len(train_indices), -1, dtype=np.int64)
    test_probabilities: list[np.ndarray] = []
    fold_records: list[dict[str, object]] = []
    fold_subjects: list[dict[str, list[int]]] = []

    for fold, (fold_train_local, fold_val_local) in enumerate(
        cross_validator.split(train_x, train_y, groups=train_groups), start=1
    ):
        model, mean, std, best_epoch, best_val_loss, duration, val_probabilities = fit_fold(
            dataset.windows,
            dataset.labels,
            train_indices[fold_train_local],
            train_indices[fold_val_local],
            args,
            fold,
        )
        val_actual = dataset.labels[train_indices[fold_val_local]]
        oof_predictions[fold_val_local] = val_probabilities.argmax(axis=1)
        test_x = apply_channel_statistics(dataset.windows[test_indices], mean, std)
        test_probabilities.append(
            model.predict(test_x, batch_size=args.batch_size, verbose=0)
        )
        fold_records.append(
            {
                "fold": fold,
                "best_epoch": best_epoch,
                "best_validation_loss": best_val_loss,
                "training_seconds": duration,
                "validation": metric_summary(val_actual, val_probabilities.argmax(axis=1)),
            }
        )
        fold_subjects.append(
            {
                "train": np.unique(train_groups[fold_train_local]).tolist(),
                "validation": np.unique(train_groups[fold_val_local]).tolist(),
            }
        )
        del model
        print(f"Completed fold {fold}/{args.folds}: best epoch {best_epoch}")

    if np.any(oof_predictions < 0):
        raise RuntimeError("Cross-validation did not produce one out-of-fold prediction per training window")
    ensemble_probabilities = np.mean(test_probabilities, axis=0)
    test_predictions = ensemble_probabilities.argmax(axis=1)
    class_names = [
        "walking", "walking_upstairs", "walking_downstairs", "sitting", "standing", "lying",
        "stand_to_sit", "sit_to_stand", "sit_to_lie", "lie_to_sit", "stand_to_lie", "lie_to_stand",
    ]
    results: dict[str, object] = {
        "model": "DeepConvLSTM",
        "seed": args.seed,
        "window_size": 128,
        "stride": 64,
        "input_channels": ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
        "class_ids": {str(index): name for index, name in enumerate(class_names)},
        "split": {
            "train_subjects": train_subjects.tolist(),
            "test_subjects": test_subjects.tolist(),
            "cross_validation_subjects": fold_subjects,
        },
        "hyperparameters": {
            "epochs_max": args.epochs,
            "early_stopping_patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "gradient_clip_norm": 1.0,
            "folds": args.folds,
            "class_weighting": "inverse frequency, computed from each training fold",
            "normalization": "per-channel z-score fitted on each training fold",
        },
        "out_of_fold": metric_summary(train_y, oof_predictions),
        "held_out_test_ensemble": metric_summary(dataset.labels[test_indices], test_predictions),
        "folds": fold_records,
        "inference_ensemble_size": len(test_probabilities),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "out_of_fold": results["out_of_fold"],
        "held_out_test_ensemble": results["held_out_test_ensemble"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
