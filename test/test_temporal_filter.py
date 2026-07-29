import numpy as np

from digit_sdk.temporal import NeuralFeelsFIR, PersistenceCutoff


def _weights(count: int) -> np.ndarray:
    values = np.exp(np.arange(1, count + 1, dtype=np.float64) / count)
    return values / values.sum()


def test_neuralfeels_fir_matches_balanced_study_weights():
    temporal = NeuralFeelsFIR()
    frames = [
        np.full((2, 3), value, dtype=np.float32)
        for value in range(1, 7)
    ]

    outputs = [temporal(frame) for frame in frames]

    for count in range(1, 5):
        expected = np.dot(_weights(count), np.arange(1, count + 1))
        np.testing.assert_allclose(outputs[count - 1], expected)
    expected = np.dot(_weights(5), np.arange(2, 7))
    np.testing.assert_allclose(outputs[-1], expected)


def test_neuralfeels_fir_reset_discards_history():
    temporal = NeuralFeelsFIR()
    temporal(np.zeros((2, 2), dtype=np.float32))
    temporal.reset()

    output = temporal(np.ones((2, 2), dtype=np.float32))

    np.testing.assert_array_equal(output, np.ones((2, 2), dtype=np.float32))


def test_neuralfeels_fir_resets_on_shape_change():
    temporal = NeuralFeelsFIR()
    temporal(np.zeros((2, 2), dtype=np.float32))

    output = temporal(np.ones((3, 2), dtype=np.float32))

    np.testing.assert_array_equal(output, np.ones((3, 2), dtype=np.float32))


def test_persistence_requires_two_hits_in_three_frames():
    persistence = PersistenceCutoff(0.1)
    active = np.array([[0.2, 0.0]], dtype=np.float32)
    inactive = np.zeros((1, 2), dtype=np.float32)

    np.testing.assert_array_equal(persistence(active), inactive)
    np.testing.assert_array_equal(persistence(active), active)
    np.testing.assert_array_equal(persistence(inactive), inactive)
    np.testing.assert_array_equal(persistence(inactive), inactive)


def test_persistence_reset_discards_hits():
    persistence = PersistenceCutoff(0.1)
    active = np.array([[0.2]], dtype=np.float32)
    persistence(active)
    persistence.reset()

    np.testing.assert_array_equal(
        persistence(active), np.zeros_like(active)
    )
