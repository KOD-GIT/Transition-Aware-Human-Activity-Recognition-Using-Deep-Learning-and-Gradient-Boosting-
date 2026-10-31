"""DeepConvLSTM model for six-channel, 12-class HAPT sensor windows."""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers


def build_deep_conv_lstm(
    window_size: int = 128,
    input_channels: int = 6,
    num_classes: int = 12,
    conv_filters: int = 64,
    lstm_units: int = 128,
    dropout_rate: float = 0.2,
    head_dropout: float = 0.5,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Build and compile the report's four-convolution, two-LSTM network.

    The input shape is ``(timesteps, channels)``. The convolution blocks use
    stride one and same padding, so they preserve the temporal order and length
    for the recurrent layers. The final layer returns 12-class probabilities.
    """
    if min(window_size, input_channels, num_classes, conv_filters, lstm_units) < 1:
        raise ValueError("Model dimensions must be positive")
    if not 0 <= dropout_rate < 1 or not 0 <= head_dropout < 1:
        raise ValueError("Dropout probabilities must be in [0, 1)")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    inputs = keras.Input(shape=(window_size, input_channels), name="sensor_window")
    x = inputs
    for index in range(4):
        x = layers.Conv1D(
            filters=conv_filters,
            kernel_size=5,
            strides=1,
            padding="same",
            activation="relu",
            name=f"conv1d_{index + 1}",
        )(x)

    x = layers.LSTM(
        lstm_units,
        return_sequences=True,
        dropout=dropout_rate,
        name="lstm_1",
    )(x)
    x = layers.LSTM(
        lstm_units,
        return_sequences=False,
        dropout=dropout_rate,
        name="lstm_2",
    )(x)
    x = layers.Dropout(head_dropout, name="head_dropout")(x)
    x = layers.Dense(128, activation="relu", name="dense_128")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="activity_probabilities")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="deep_convlstm_hapt")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
