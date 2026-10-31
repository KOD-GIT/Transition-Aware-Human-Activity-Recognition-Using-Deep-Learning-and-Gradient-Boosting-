import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

from models.deep_conv_lstm import build_deep_conv_lstm
from src.data_prep.hapt_windows import (
    apply_channel_statistics,
    channel_statistics,
    load_hapt_windows,
)


class DeepConvLSTMTests(unittest.TestCase):
    def test_logits_and_gradients(self) -> None:
        model = build_deep_conv_lstm()
        inputs = tf.random.normal((2, 128, 6))
        with tf.GradientTape() as tape:
            probabilities = model(inputs, training=True)
            loss = tf.reduce_sum(probabilities)
        gradients = tape.gradient(loss, model.trainable_variables)
        self.assertEqual(tuple(probabilities.shape), (2, 12))
        self.assertTrue(all(gradient is not None for gradient in gradients))

    def test_hapt_annotations_create_transition_aware_windows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            raw = Path(temporary_directory) / "RawData"
            raw.mkdir()
            time = np.arange(320, dtype=np.float32)
            acc = np.column_stack((np.sin(time / 8), np.cos(time / 11), time / 320))
            gyro = np.column_stack((np.cos(time / 13), np.sin(time / 9), -time / 320))
            np.savetxt(raw / "acc_exp01_user01.txt", acc)
            np.savetxt(raw / "gyro_exp01_user01.txt", gyro)
            (raw / "labels.txt").write_text("1 1 1 1 160\n1 1 7 161 320\n", encoding="utf-8")

            dataset = load_hapt_windows(temporary_directory)
            self.assertEqual(dataset.windows.shape, (4, 128, 6))
            self.assertEqual(dataset.labels.tolist(), [0, 0, 6, 6])
            self.assertEqual(dataset.subjects.tolist(), [1, 1, 1, 1])

    def test_normalization_uses_training_statistics(self) -> None:
        training = np.ones((2, 4, 6), dtype=np.float32)
        training[:, :, 0] = 3.0
        mean, std = channel_statistics(training)
        validation = np.ones((1, 4, 6), dtype=np.float32) * 5.0
        normalized = apply_channel_statistics(validation, mean, std)
        self.assertEqual(normalized.shape, validation.shape)
        self.assertTrue(np.isfinite(normalized).all())
        self.assertTrue((normalized[:, :, 0] == 2.0).all())


if __name__ == "__main__":
    unittest.main()
