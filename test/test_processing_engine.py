import numpy as np

from digit_sdk.processing_engine import ProcessingEngine, _PendingDepthInput
from digit_sdk.temporal import NeuralFeelsFIR, PersistenceCutoff


def _input(serial: str, sequence: int) -> _PendingDepthInput:
    return _PendingDepthInput(
        source_sequence=sequence,
        timestamp_ns=sequence,
        prepared=f"{serial}-{sequence}",
    )


def _result_engine(cutoff: float = 0.0) -> ProcessingEngine:
    engine = ProcessingEngine.__new__(ProcessingEngine)
    engine._depth_cutoff_mm = cutoff
    engine._temporal_filters = {"sensor": NeuralFeelsFIR()}
    engine._persistence_filters = (
        {"sensor": PersistenceCutoff(cutoff)} if cutoff > 0 else {}
    )
    engine._temporal_timestamps_ns = {}
    return engine


def test_inference_batch_is_fixed_size_after_initialization():
    engine = ProcessingEngine.__new__(ProcessingEngine)
    engine._serials = ("left", "right")
    engine._latest_inputs = {}

    assert engine._build_inference_batch({"left": _input("left", 1)}) is None
    assert engine._build_inference_batch({"right": _input("right", 1)}) == {
        "left": "left-1",
        "right": "right-1",
    }

    assert engine._build_inference_batch({"left": _input("left", 2)}) == {
        "left": "left-2",
        "right": "right-1",
    }


def test_zero_cutoff_preserves_filtered_values():
    engine = _result_engine()
    depth = np.array([[0.05, 0.2]], dtype=np.float32)

    result = engine._make_result(
        "sensor", depth, source_sequence=2, timestamp_ns=3
    )

    np.testing.assert_array_equal(
        result.depth_mm, np.array([[0.05, 0.2]], dtype=np.float32)
    )


def test_positive_cutoff_and_persistence_are_applied():
    engine = _result_engine(0.1)
    depth = np.array([[0.05, 0.2]], dtype=np.float32)

    first = engine._make_result(
        "sensor", depth, source_sequence=2, timestamp_ns=3
    )
    result = engine._make_result(
        "sensor", depth, source_sequence=3, timestamp_ns=4
    )

    np.testing.assert_array_equal(first.depth_mm, np.zeros_like(depth))
    np.testing.assert_allclose(
        result.depth_mm, np.array([[0.0, 0.2]], dtype=np.float32)
    )


def test_temporal_filter_runs_before_depth_cutoff():
    engine = _result_engine(0.1)
    engine._make_result(
        "sensor",
        np.array([[0.0]], dtype=np.float32),
        source_sequence=1,
        timestamp_ns=1,
    )

    result = engine._make_result(
        "sensor",
        np.array([[0.15]], dtype=np.float32),
        source_sequence=2,
        timestamp_ns=2,
    )

    assert result.depth_mm.item() == 0.0


def test_temporal_filter_resets_after_timestamp_gap():
    engine = _result_engine()
    engine._make_result(
        "sensor",
        np.array([[0.0]], dtype=np.float32),
        source_sequence=1,
        timestamp_ns=1,
    )

    result = engine._make_result(
        "sensor",
        np.array([[1.0]], dtype=np.float32),
        source_sequence=2,
        timestamp_ns=100_000_002,
    )

    assert result.depth_mm.item() == 1.0


def test_temporal_filters_are_independent_per_sensor():
    engine = ProcessingEngine.__new__(ProcessingEngine)
    engine._depth_cutoff_mm = 0.0
    engine._temporal_filters = {
        "left": NeuralFeelsFIR(),
        "right": NeuralFeelsFIR(),
    }
    engine._persistence_filters = {}
    engine._temporal_timestamps_ns = {}
    engine._make_result(
        "left",
        np.array([[0.0]], dtype=np.float32),
        source_sequence=1,
        timestamp_ns=1,
    )

    result = engine._make_result(
        "right",
        np.array([[1.0]], dtype=np.float32),
        source_sequence=1,
        timestamp_ns=1,
    )

    assert result.depth_mm.item() == 1.0
