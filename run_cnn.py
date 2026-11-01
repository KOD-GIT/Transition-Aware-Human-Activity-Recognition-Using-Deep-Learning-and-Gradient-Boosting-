from datetime import datetime
import json
from logging import basicConfig, getLogger, StreamHandler, DEBUG, WARNING
import os
import sys
import glob
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from src.data_prep.load import load_hapt, ACTIVITY_LABELS
from src.utils import plot_confusion_matrix
from models.cnn import train_and_predict

CUR_DIR = os.path.dirname(os.path.abspath(__file__))  # Path to current directory

# Logging settings
EXEC_TIME = "cnn-" + datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_DIR = os.path.join(CUR_DIR, f"logs/{EXEC_TIME}")
os.makedirs(LOG_DIR, exist_ok=True)  # Create log directory

formatter = "%(levelname)s: %(asctime)s: %(filename)s: %(funcName)s: %(message)s"
basicConfig(filename=f"{LOG_DIR}/{EXEC_TIME}.log", level=DEBUG, format=formatter)
mpl_logger = getLogger("matplotlib")  # Suppress matplotlib logging
mpl_logger.setLevel(WARNING)
# Handle logging to both logging and stdout.
getLogger().addHandler(StreamHandler(sys.stdout))

logger = getLogger(__name__)
logger.setLevel(DEBUG)
logger.debug(f"{LOG_DIR}/{EXEC_TIME}.log")

# --- NEW CUSTOM RAW DATA LOADER ---
def load_hapt_raw_windows(data_dir):
    raw_dir = os.path.join(data_dir, "RawData")
    labels = pd.read_csv(os.path.join(raw_dir, "labels.txt"), sep='\s+', header=None, names=["exp", "user", "act", "start", "end"])
    
    # Subject-independent split matching the project report
    train_users = [1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30]

    X_tr, y_tr, X_te, y_te = [], [], [], []

    for exp in range(1, 62):
        acc_matches = glob.glob(os.path.join(raw_dir, f"acc_exp{exp:02d}_*.txt"))
        gyro_matches = glob.glob(os.path.join(raw_dir, f"gyro_exp{exp:02d}_*.txt"))
        if not acc_matches or not gyro_matches: continue

        user_id = int(acc_matches[0].split("user")[1][:2])
        acc_data = pd.read_csv(acc_matches[0], sep='\s+', header=None).values
        gyro_data = pd.read_csv(gyro_matches[0], sep='\s+', header=None).values
        combined = np.hstack((acc_data, gyro_data)) # Keep only the 6 required channels

        exp_labels = labels[labels['exp'] == exp]
        for _, row in exp_labels.iterrows():
            segment = combined[row['start'] : row['end'] + 1]
            act_idx = row['act'] - 1 # Shift 1-12 to 0-11 for Keras

            # 128 window size, 64 step size (50% overlap)
            for i in range(0, len(segment) - 128 + 1, 64):
                if user_id in train_users:
                    X_tr.append(segment[i : i + 128])
                    y_tr.append(act_idx)
                else:
                    X_te.append(segment[i : i + 128])
                    y_te.append(act_idx)

    # Channel-wise normalization based on training set statistics
    X_train_arr = np.array(X_tr)
    X_test_arr = np.array(X_te)
    means = X_train_arr.mean(axis=(0, 1))
    stds = X_train_arr.std(axis=(0, 1))

    return (X_train_arr - means) / stds, np.array(y_tr), (X_test_arr - means) / stds, np.array(y_te)

# 1. Load, window, and normalize the 3D raw signals directly from RawData
data_path = os.path.join(CUR_DIR, "data", "hapt_data_set")
X_train, y_train, X_test, y_test = load_hapt_raw_windows(data_path)

# 2. Reconstruct the 0-indexed label mappings that the rest of the script relies on
label2act = {k-1: v for k, v in ACTIVITY_LABELS.items()}
act2label = {v: k-1 for k, v in ACTIVITY_LABELS.items()}

# ADDED: Calculate the specific class weights to combat the severe class imbalance
classes = np.unique(y_train.flatten())
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train.flatten())
class_weights_dict = dict(zip(classes, weights))

logger.debug(f"{X_train.shape=} {X_test.shape=}")
logger.debug(f"{y_train.shape=} {y_test.shape=}")

# Split data by preserving the percentage of samples for each class.
n_splits = 5
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=71)
valid_preds = np.zeros((X_train.shape[0], 12)) # CHANGED to 12
test_preds = np.zeros((n_splits, X_test.shape[0], 12)) # CHANGED to 12
models = []
scores: Dict[str, Dict[str, List[Any]]] = {
    "logloss": {"train": [], "valid": [], "test": []},
    "accuracy": {"train": [], "valid": [], "test": []},
    "precision": {"train": [], "valid": [], "test": []},
    "recall": {"train": [], "valid": [], "test": []},
    "f1": {"train": [], "valid": [], "test": []},
    "cm": {"train": [], "valid": [], "test": []},
    "per_class_f1": {"train": [], "valid": [], "test": []},
}
# Load hyper-parameters
with open(os.path.join(CUR_DIR, "configs/default.json"), "r") as f:
    cnn_params = json.load(f)["cnn_params"]
    logger.debug(f"{cnn_params=}")

y_test = keras.utils.to_categorical(y_test, 12)

for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train, y_train)):
    X_tr = X_train[train_index, :]
    X_val = X_train[valid_index, :]
    y_tr = y_train[train_index]
    y_val = y_train[valid_index]
    
    y_tr = keras.utils.to_categorical(y_tr, 12)
    y_val = keras.utils.to_categorical(y_val, 12)

    logger.debug(f"{X_tr.shape=} {X_val.shape=} {X_test.shape=}")
    logger.debug(f"{y_tr.shape=} {y_val.shape=} {y_test.shape=}")

    pred_tr, pred_val, pred_test, model = train_and_predict(
        LOG_DIR, fold_id, X_tr, X_val, X_test, y_tr, y_val, cnn_params,
        class_weights=class_weights_dict
    )
    models.append(model)

    valid_preds[valid_index] = pred_val
    test_preds[fold_id] = pred_test

    for pred, X, y, mode in zip(
        [pred_tr, pred_val, pred_test], [X_tr, X_val, X_test], [y_tr, y_val, y_test], ["train", "valid", "test"]
    ):
        loss, acc = model.evaluate(X, y, verbose=0)
        pred = pred.argmax(axis=1)
        y = y.argmax(axis=1)
        scores["logloss"][mode].append(loss)
        scores["accuracy"][mode].append(acc)
        scores["precision"][mode].append(precision_score(y, pred, average="macro"))
        scores["recall"][mode].append(recall_score(y, pred, average="macro"))
        scores["f1"][mode].append(f1_score(y, pred, average="macro"))
        scores["cm"][mode].append(confusion_matrix(y, pred, normalize="true"))
        scores["per_class_f1"][mode].append(f1_score(y, pred, average=None))

# Output Cross Validation Scores
logger.debug("---Cross Validation Scores---")
for mode in ["train", "valid", "test"]:
    logger.debug(f"---{mode}---")
    for metric in ["logloss", "accuracy", "precision", "recall", "f1"]:
        logger.debug(f"{metric}={round(np.mean(scores[metric][mode]))}")

    class_f1_mat = scores["per_class_f1"][mode]
    class_f1_result = {}
    for class_id in range(12):
        mean_class_f1 = np.mean([class_f1_mat[i][class_id] for i in range(n_splits)])
        class_f1_result[label2act[class_id]] = mean_class_f1
    logger.debug(f"per-class f1={class_f1_result}")

# Output Final Scores Averaged over Folds
logger.debug("---Final Test Scores Averaged over Folds---")
test_pred = np.mean(test_preds, axis=0).argmax(axis=1)  # average over folds
y_test = y_test.argmax(axis=1)
logger.debug(f"accuracy={accuracy_score(y_test, test_pred)}")
logger.debug(f"precision={precision_score(y_test, test_pred, average='macro')}")
logger.debug(f"recall={recall_score(y_test, test_pred, average='macro')}")
logger.debug(f"f1={f1_score(y_test, test_pred, average='macro')}")
logger.debug(f"per-class f1={f1_score(y_test, test_pred, average=None)}")

# Plot comfusion matrix
# Plot confusion matrix
plot_confusion_matrix(
    y_true=y_test,
    y_pred=test_pred,
    num_classes=12,
    save_path=f"{LOG_DIR}/confusion_matrix.png",
    show=False
)

np.save(f"{LOG_DIR}/valid_oof.npy", valid_preds)
np.save(f"{LOG_DIR}/test_oof.npy", np.mean(test_preds, axis=0))  # Averaging

