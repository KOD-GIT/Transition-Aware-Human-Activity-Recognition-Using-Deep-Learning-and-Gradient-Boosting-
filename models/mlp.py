"""
MLP model for 12-class HAPT (Human Activities and Postural Transitions).

Supports both input strategies:
  - Engineered features  (561-dim from X_train.txt)
  - Flattened raw window (window_size × n_channels)

Branch  : feature/mlp
Author  : Vinayak Shankar
"""
from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers


def build_mlp(
    input_dim: int,
    num_classes: int = 12,
    hidden_units: list[int] | None = None,
    dropout_rate: float = 0.3,
    use_batch_norm: bool = True,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Build and compile a Multi-Layer Perceptron for 12-class HAR.

    Parameters
    ----------
    input_dim:
        Number of input features (engineered) or flattened window length.
    num_classes:
        Output classes — 12 for the full HAPT label set (6 activities +
        6 postural transitions).
    hidden_units:
        Ordered list of hidden-layer widths.  Defaults to [512, 256, 128].
    dropout_rate:
        Dropout probability applied after every Dense+BN block.
    use_batch_norm:
        Insert a BatchNormalisation layer between Dense and Dropout when True.
    learning_rate:
        Initial learning rate for the Adam optimiser.

    Returns
    -------
    keras.Model
        Compiled model ready for ``model.fit()``.
    """
    if hidden_units is None:
        hidden_units = [512, 256, 128]

    inputs = keras.Input(shape=(input_dim,), name="sensor_features")
    x = inputs

    for i, units in enumerate(hidden_units):
        x = layers.Dense(units, name=f"dense_{i}")(x)
        if use_batch_norm:
            x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.Activation("relu", name=f"relu_{i}")(x)
        x = layers.Dropout(dropout_rate, name=f"dropout_{i}")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="mlp_har")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
