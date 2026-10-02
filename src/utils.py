"""
Shared evaluation and plotting utilities for the 12-class HAPT project.

Provides:
  - Classification report (all 12 classes)
  - Transition-class precision / recall / F1
  - Confusion-matrix plot
  - Keras training-history plot
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.data_prep.load import ACTIVITY_LABELS, TRANSITION_CLASSES


# ── Helpers ────────────────────────────────────────────────────────────────── #

def get_class_names(num_classes: int = 12) -> list[str]:
    """Return ordered list of class names (0-indexed internally)."""
    return [ACTIVITY_LABELS[i + 1] for i in range(num_classes)]


# ── Classification reports ─────────────────────────────────────────────────── #

def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 12,
) -> None:
    """Print full sklearn classification report for all 12 HAPT classes."""
    target_names = get_class_names(num_classes)
    print("\n" + "=" * 72)
    print("CLASSIFICATION REPORT — ALL 12 CLASSES")
    print("=" * 72)
    print(
        classification_report(
            y_true, y_pred,
            target_names=target_names,
            digits=4,
            zero_division=0,
        )
    )


def print_transition_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """Print per-class precision / recall / F1 for the 6 postural transitions.

    These classes are typically underrepresented and merit individual analysis.
    """
    # TRANSITION_CLASSES is 1-indexed → convert to 0-indexed
    transition_indices = [c - 1 for c in TRANSITION_CLASSES]
    names = [ACTIVITY_LABELS[c] for c in TRANSITION_CLASSES]

    print("\n" + "=" * 72)
    print("TRANSITION-CLASS METRICS (postural transitions only)")
    print("=" * 72)
    print(f"{'Class':<22} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 64)

    for idx, name in zip(transition_indices, names):
        mask_true = y_true == idx
        mask_pred = y_pred == idx
        tp = int((mask_true & mask_pred).sum())
        fp = int((~mask_true & mask_pred).sum())
        fn = int((mask_true & ~mask_pred).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        support = int(mask_true.sum())
        print(f"{name:<22} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {support:>10}")

    # Macro averages over transition samples only
    transition_mask = np.isin(y_true, transition_indices)
    if transition_mask.sum() > 0:
        yt = y_true[transition_mask]
        yp = y_pred[transition_mask]
        macro_p  = precision_score(yt, yp, labels=transition_indices,
                                   average="macro", zero_division=0)
        macro_r  = recall_score(yt, yp, labels=transition_indices,
                                average="macro", zero_division=0)
        macro_f1 = f1_score(yt, yp, labels=transition_indices,
                            average="macro", zero_division=0)
        print("-" * 64)
        print(
            f"{'Macro (transitions)':<22} "
            f"{macro_p:>10.4f} {macro_r:>10.4f} {macro_f1:>10.4f} "
            f"{int(transition_mask.sum()):>10}"
        )


# ── Plots ──────────────────────────────────────────────────────────────────── #

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 12,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot and optionally save the 12-class confusion matrix."""
    labels = get_class_names(num_classes)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(14, 12))
    disp.plot(ax=ax, colorbar=True, xticks_rotation=45)
    ax.set_title("Confusion Matrix — 12-Class HAPT", fontsize=14, pad=12)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[utils] Confusion matrix saved → {save_path}")

    if show:
        plt.show()
    plt.close(fig)


def plot_training_history(
    history,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot Keras training history (accuracy and loss curves)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    for ax, metric, title in zip(
        axes,
        ["accuracy", "loss"],
        ["Accuracy", "Loss"],
    ):
        ax.plot(history.history[metric], label=f"train_{metric}")
        val_key = f"val_{metric}"
        if val_key in history.history:
            ax.plot(history.history[val_key], label=f"val_{metric}", linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle("MLP Training History — 12-Class HAPT", fontsize=13)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[utils] Training history saved → {save_path}")

    if show:
        plt.show()
    plt.close(fig)
