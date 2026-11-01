"""Functions for training Convolutional Neural Network (CNN)"""
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Activation, Dropout, Conv1D, 
    BatchNormalization, MaxPooling1D, GlobalAveragePooling1D
)
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K

from src.keras_callback import get_default_callbacks

tf.random.set_seed(0)

def train_and_predict(
    LOG_DIR: str,
    fold_id: int,
    X_train: np.ndarray,
    X_valid: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    cnn_params: Dict[str, Any],
    class_weights: Dict[int, float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Model]:
    
    # CHANGED: Call the new 1D-CNN builder
    model = build_1d_cnn(
        input_shape=X_train.shape[1:], output_dim=y_train.shape[1], lr=cnn_params["lr"]
    )

    callbacks = get_default_callbacks(
        early_stopping_patience=20,
        model_checkpoint_path=f"{LOG_DIR}/trained_model_fold{fold_id}.h5"
    )

    fit = model.fit(
        X_train,
        y_train,
        batch_size=cnn_params["batch_size"],
        epochs=cnn_params["epochs"],
        verbose=cnn_params["verbose"],
        validation_data=(X_valid, y_valid),
        callbacks=callbacks,
        class_weight=class_weights 
    )

    model = keras.models.load_model(f"{LOG_DIR}/trained_model_fold{fold_id}.h5")

    pred_train = model.predict(X_train)
    pred_valid = model.predict(X_valid)
    pred_test = model.predict(X_test)

    K.clear_session()
    return pred_train, pred_valid, pred_test, model


def build_1d_cnn(
    input_shape: Tuple[int, int] = (128, 6), # CHANGED: 1D-CNN uses (timesteps, channels)
    output_dim: int = 12, 
    lr: float = 0.001
) -> Model:
    model = Sequential()
    
    # Block 1
    model.add(Conv1D(64, kernel_size=5, padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling1D(pool_size=2))
    
    # Block 2
    model.add(Conv1D(128, kernel_size=5, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling1D(pool_size=2))
    
    # Block 3
    model.add(Conv1D(256, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(GlobalAveragePooling1D())
    
    # Classification Head
    model.add(Dropout(0.4, seed=0))
    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(Dense(output_dim))
    model.add(Activation("softmax"))
    
    model.compile(
        loss="categorical_crossentropy", 
        optimizer=optimizers.Adam(learning_rate=lr), 
        metrics=["accuracy"]
    )
    return model