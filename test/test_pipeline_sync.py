import unittest.mock

import numpy as np
import pytest

from digit_sdk.processing_engine import ProcessingEngine
from digit_sdk.shm_protocol import CameraFrameSnapshot
from digit_sdk.temporal import NeuralFeelsFIR


def _snapshot(sequence: int, timestamp_ns: int):
    return CameraFrameSnapshot(
        sequence=sequence,
        timestamp_ns=timestamp_ns,
        image=np.zeros((8, 8), dtype=np.uint8),
    )


def _engine(estimator) -> ProcessingEngine:
    engine = ProcessingEngine.__new__(ProcessingEngine)
    engine._serials = ("left", "right")
    engine._estimator = estimator
    engine._depth_cutoff_mm = 0.0
    engine._temporal_filters = {
        "left": NeuralFeelsFIR(),
        "right": NeuralFeelsFIR(),
    }
    engine._persistence_filters = {}
    engine._temporal_timestamps_ns = {}
    engine._processed_sequences = {"left": -1, "right": -1}
    engine._latest_results = {}
    return engine


def test_process_snapshots_preserves_source_identity():
    estimator = unittest.mock.Mock()
    estimator.prepare.side_effect = lambda image: image
    estimator.estimate_prepared_batch.side_effect = (
        lambda prepared: {
            serial: np.array([[5.0]], dtype=np.float32)
            for serial in prepared
        }
    )
    engine = _engine(estimator)

    results = engine.process_snapshots(
        {"left": _snapshot(sequence=7, timestamp_ns=123)}
    )

    assert results["left"].source_sequence == 7
    assert results["left"].timestamp_ns == 123
    assert engine._processed_sequences["left"] == 7
    assert engine._latest_results["left"] is results["left"]
    assert engine._processed_sequences["right"] == -1


def test_process_snapshots_rejects_unknown_serials():
    engine = _engine(unittest.mock.Mock())

    with pytest.raises(KeyError):
        engine.process_snapshots(
            {"ghost": _snapshot(sequence=1, timestamp_ns=1)}
        )


def test_read_frame_dedups_same_sequence():
    engine = ProcessingEngine.__new__(ProcessingEngine)
    engine._shms = {"left": unittest.mock.Mock()}
    engine._last_read_sequences = {"left": -1}
    snapshot = _snapshot(sequence=4, timestamp_ns=9)

    with unittest.mock.patch(
        "digit_sdk.processing_engine.read_camera_frame",
        return_value=snapshot,
    ):
        assert engine.read_frame("left") is snapshot
        assert engine.read_frame("left") is None
