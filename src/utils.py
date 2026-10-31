"""
Shared evaluation and plotting utilities for the 12-class HAPT project.

Provides:
  - Classification report (all 12 classes)
  - Transition-class precision / recall / F1
  - Confusion-matrix plot
  - Keras training-history plot
  - LightGBM class-balance, SHAP, and CV confusion-matrix helpers
"""
from __future__ import annotations

from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.data_prep.load import ACTIVITY_LABELS, TRANSITION_CLASSES

try:
    shap.initjs()
except Exception:
    pass

logger = getLogger(__name__)


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


def color_generator(i: int) -> str:
    palette = ["#FFAF6D", "#DC4195", "#F1E898", "#6DCBB9", "#3E89C4", "#6F68CF"]
    return palette[i]


def round_float(f: float, r: float = 0.000001) -> float:
    return float(Decimal(str(f)).quantize(Decimal(str(r)), rounding=ROUND_HALF_UP))


def round_list(l: List[float], r: float = 0.000001) -> List[float]:
    return [round_float(f, r) for f in l]


def round_dict(d: Dict[Any, Any], r: float = 0.000001) -> Dict[Any, Any]:
    return {key: round(d[key], r) for key in d.keys()}


def round(arg: Any, r: float = 0.000001) -> Any:
    if type(arg) == float or type(arg) == np.float64 or type(arg) == np.float32:
        return round_float(arg, r)
    elif type(arg) == list or type(arg) == np.ndarray:
        return round_list(arg, r)
    elif type(arg) == dict:
        return round_dict(arg, r)
    else:
        logger.error(f"Arg type {type(arg)} is not supported")
        return arg


def check_class_balance(
    y_train: np.ndarray, y_test: np.ndarray, label2act: Dict[int, str], n_class: int = 6
) -> None:
    c_train = Counter(y_train)
    c_test = Counter(y_test)

    for c, mode in zip([c_train, c_test], ["train", "test"]):
        logger.debug(f"{mode} labels")
        len_y = sum(c.values())
        for label_id in range(n_class):
            logger.debug(
                f"{label2act[label_id]} ({label_id}): {c[label_id]} samples ({c[label_id] / len_y * 100:.04} %)"
            )


def plot_feature_importance(
    models: List[Any],
    num_features: int,
    cols: List[str],
    importance_type: str = "gain",
    path: str = "importance.png",
    figsize: Tuple[int, int] = (16, 10),
    max_display: int = -1,
) -> None:
    """
    Args:
        importance_type: chosen from "gain" or "split"
    """
    importances = np.zeros((len(models), num_features))
    for i, model in enumerate(models):
        importances[i] = model.feature_importance(importance_type=importance_type)

    importance = np.mean(importances, axis=0)
    importance_df = pd.DataFrame({"Feature": cols, "Value": importance})
    importance_df = importance_df.sort_values(by="Value", ascending=False)[:max_display]

    plt.figure(figsize=figsize)
    sns.barplot(x="Value", y="Feature", data=importance_df)
    plt.title("Feature Importance (avg over folds)")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_shap_summary(
    models: List[Any],
    X_train: pd.DataFrame,
    class_names: List[str],
    path: str = "shap_summary_plot.png",
    max_display: Optional[int] = None,
) -> None:
    shap_values_list = []
    for model in models:
        try:
            explainer = shap.TreeExplainer(
                model,
                feature_perturbation="tree_path_dependent",
            )
        except Exception:
            explainer = shap.TreeExplainer(model)
        shap_value_oof = explainer.shap_values(X_train)
        if isinstance(shap_value_oof, np.ndarray) and shap_value_oof.ndim == 3:
            shap_value_oof = [shap_value_oof[:, :, i] for i in range(shap_value_oof.shape[2])]
        shap_values_list.append(shap_value_oof)

    shap_values = [np.zeros(shap_values_list[0][0].shape) for _ in range(len(class_names))]
    for shap_value_oof in shap_values_list:
        for i in range(len(class_names)):
            shap_values[i] += shap_value_oof[i]

    for i in range(len(class_names)):
        shap_values[i] /= len(models)

    shap.summary_plot(
        shap_values,
        X_train,
        max_display=max_display,
        class_names=class_names,
        color=color_generator,
        show=False,
    )
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_cv_confusion_matrices(
    cms: Dict[str, np.ndarray],
    labels: Optional[List[str]] = None,
    path: str = "confusion_matrix.png",
) -> None:
    """Plot LightGBM train/valid/test confusion matrices averaged over CV folds."""
    cms = [np.mean(cms[mode], axis=0) for mode in ["train", "valid", "test"]]

    fig, ax = plt.subplots(ncols=3, figsize=(20, 7))
    for i, (cm, mode) in enumerate(zip(cms, ["train", "valid", "test"])):
        sns.heatmap(
            cm,
            annot=True,
            cmap="Blues",
            square=True,
            vmin=0,
            vmax=1.0,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax[i],
        )
        ax[i].set_xlabel("Predicted label")
        ax[i].set_ylabel("True label")
        ax[i].set_title(f"Normalized confusion matrix - {mode}")

    plt.tight_layout()
    fig.savefig(path)
    plt.close()

