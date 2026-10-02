"""
Train and evaluate the MLP on the HAPT dataset (12-class).

Two input strategies are available via --mode:

  features (default)
      Uses the pre-computed 561-dimensional feature vector from X_train.txt.
      Fast to load, no windowing needed.

  raw
      Loads the 9-channel raw inertial signals, applies a sliding window
      (default 128 steps, 50 % overlap), then flattens each window for the MLP.

Usage examples
--------------
    # Default: engineered features
    poetry run python run_mlp.py

    # Raw-window mode, custom epochs
    poetry run python run_mlp.py --mode raw --epochs 100 --batch-size 128
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from models.mlp import build_mlp
from src.data_prep.load import load_hapt
from src.data_prep.preprocessing import (
    flatten_windows,
    normalize_features,
    sliding_window,
)
from src.keras_callback import get_default_callbacks
from src.utils import (
    plot_confusion_matrix,
    plot_training_history,
    print_classification_report,
    print_transition_metrics,
)

CONFIG_PATH = Path("configs/default.json")
LOG_DIR = Path("logs")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train MLP for 12-class HAPT (activities + transitions)"
    )
    p.add_argument(
        "--mode",
        choices=["features", "raw"],
        default="features",
        help="Input strategy: 'features' (561-dim engineered) or 'raw' (windowed signals)",
    )
    p.add_argument("--data-root",  default="data/hapt_data_set",
                   help="Path to the HAPT dataset root directory")
    p.add_argument("--epochs",     type=int, default=150)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--window-size", type=int, default=128,
                   help="[raw mode] Sliding-window length in timesteps")
    p.add_argument("--step-size",  type=int, default=64,
                   help="[raw mode] Stride between consecutive windows")
    p.add_argument("--no-plot",    action="store_true",
                   help="Skip generating and saving plots")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    LOG_DIR.mkdir(exist_ok=True)

    # ── Config ──────────────────────────────────────────────────────────────── #
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    num_classes: int  = config.get("num_classes", 12)
    mlp_params:  dict = config.get("mlp_params", {})

    print(f"\n{'='*60}")
    print(f"  MLP — 12-class HAPT  |  mode={args.mode}  |  classes={num_classes}")
    print(f"{'='*60}\n")

    # ── 1. Load ─────────────────────────────────────────────────────────────── #
    X_train, y_train, X_test, y_test = load_hapt(
        data_root=args.data_root,
        prefer_raw_signals=(args.mode == "raw"),
    )

    # ── 2. Preprocess ───────────────────────────────────────────────────────── #
    if args.mode == "raw":
        if X_train.ndim == 3:
            # Raw signal files already give shape (N, T, C) — just flatten
            X_tr_flat = flatten_windows(X_train)
            X_te_flat = flatten_windows(X_test)
        else:
            # Continuous 2-D signal → window → flatten
            print(f"[run_mlp] Applying sliding window: "
                  f"size={args.window_size}, step={args.step_size}")
            X_tr_w, y_train = sliding_window(
                X_train, y_train, args.window_size, args.step_size
            )
            X_te_w, y_test = sliding_window(
                X_test, y_test, args.window_size, args.step_size
            )
            X_tr_flat = flatten_windows(X_tr_w)
            X_te_flat = flatten_windows(X_te_w)

        X_train_s, X_test_s, _ = normalize_features(X_tr_flat, X_te_flat)

    else:  # 'features' mode
        if X_train.ndim != 2:
            raise RuntimeError(
                f"Expected 2-D feature matrix in 'features' mode, "
                f"got shape {X_train.shape}"
            )
        X_train_s, X_test_s, _ = normalize_features(X_train, X_test)

    input_dim = X_train_s.shape[1]
    print(f"[run_mlp] input_dim={input_dim} | "
          f"train={X_train_s.shape} | test={X_test_s.shape}")

    # ── 3. Build ─────────────────────────────────────────────────────────────── #
    model = build_mlp(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_units=mlp_params.get("hidden_units"),
        dropout_rate=mlp_params.get("dropout_rate", 0.3),
        use_batch_norm=mlp_params.get("use_batch_norm", True),
        learning_rate=mlp_params.get("learning_rate", 1e-3),
    )
    model.summary()

    # ── 4. Train ─────────────────────────────────────────────────────────────── #
    tag = f"mlp_{args.mode}"
    callbacks = get_default_callbacks(
        log_csv_path=LOG_DIR / f"{tag}_training.csv",
        early_stopping_patience=mlp_params.get("early_stopping_patience", 15),
        reduce_lr_patience=mlp_params.get("reduce_lr_patience", 7),
        model_checkpoint_path=LOG_DIR / f"{tag}_best.keras",
    )

    history = model.fit(
        X_train_s, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )

    # ── 5. Evaluate ──────────────────────────────────────────────────────────── #
    loss, acc = model.evaluate(X_test_s, y_test, verbose=0)
    print(f"\n[run_mlp] Test  loss={loss:.4f}  acc={acc:.4f}")

    y_pred = np.argmax(model.predict(X_test_s, verbose=0), axis=1)

    print_classification_report(y_test, y_pred, num_classes=num_classes)
    print_transition_metrics(y_test, y_pred)

    # ── 6. Plots ─────────────────────────────────────────────────────────────── #
    if not args.no_plot:
        plot_training_history(
            history,
            save_path=LOG_DIR / f"{tag}_history.png",
            show=False,
        )
        plot_confusion_matrix(
            y_test, y_pred,
            num_classes=num_classes,
            save_path=LOG_DIR / f"{tag}_confusion_matrix.png",
            show=False,
        )

    print(f"\n[run_mlp] Done. Logs and plots saved to {LOG_DIR}/")


if __name__ == "__main__":
    main()
