import json

import cv2
import numpy as np
import pytest

try:
    from fix.record_temporal import (
        DEFAULT_PHASES,
        FFV1Writer,
        capture_interruption,
        phase_at,
        verify_recording,
    )
except ModuleNotFoundError:
    _FIX_AVAILABLE = False
else:
    _FIX_AVAILABLE = True


pytestmark = pytest.mark.skipif(
    not _FIX_AVAILABLE,
    reason="fix.record_temporal is not tracked in digit_sdk",
)


def test_phase_protocol_has_unambiguous_evaluation_intervals():
    names = [phase.name for phase in DEFAULT_PHASES]

    assert names == [
        "background",
        "prepare_shallow",
        "shallow_hold",
        "prepare_deep",
        "deep_hold",
        "moving",
        "prepare_release",
        "released",
    ]
    assert phase_at(0.0, DEFAULT_PHASES).name == "background"
    assert phase_at(8.0, DEFAULT_PHASES).name == "prepare_shallow"
    assert phase_at(50.9, DEFAULT_PHASES).name == "released"
    assert phase_at(51.0, DEFAULT_PHASES) is None
    assert not any(
        phase.evaluate for phase in DEFAULT_PHASES
        if phase.name.startswith("prepare_")
    )


def test_capture_interruption_detects_large_timestamp_gap():
    assert capture_interruption(
        index=10,
        previous_sequence=20,
        previous_timestamp_ns=1_000_000_000,
        sequence=21,
        timestamp_ns=1_016_000_000,
        maximum_gap_ms=100.0,
    ) is None

    assert capture_interruption(
        index=11,
        previous_sequence=21,
        previous_timestamp_ns=1_016_000_000,
        sequence=22,
        timestamp_ns=11_516_000_000,
        maximum_gap_ms=100.0,
    ) == {
        "index": 11,
        "previous_sequence": 21,
        "sequence": 22,
        "gap_ms": 10_500.0,
    }


def test_ffv1_recording_round_trip_is_byte_exact(tmp_path):
    rng = np.random.default_rng(7)
    frames = rng.integers(0, 256, (8, 24, 32, 3), dtype=np.uint8)
    video = tmp_path / "raw.mkv"
    metadata = tmp_path / "frames.jsonl"

    with FFV1Writer(video, width=32, height=24, fps=60.0) as writer:
        with metadata.open("w") as stream:
            for index, frame in enumerate(frames):
                digest = writer.write(frame)
                stream.write(json.dumps({
                    "index": index,
                    "sequence": index,
                    "timestamp_ns": index,
                    "phase": "background",
                    "sha256": digest,
                }) + "\n")

    result = verify_recording(video, metadata)

    assert result == {"frames": len(frames), "byte_exact": True}
    capture = cv2.VideoCapture(str(video))
    decoded = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        decoded.append(frame)
    capture.release()
    np.testing.assert_array_equal(np.asarray(decoded), frames)
