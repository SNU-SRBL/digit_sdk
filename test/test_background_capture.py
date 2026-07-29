import json

import cv2
import numpy as np
import pytest

from calibration.collect_background import (
    average_background,
    parse_args,
    save_background_reference,
)


def test_background_collection_defaults_to_sixty_frames_one_second_apart():
    args = parse_args(["--serial", "DTEST"])

    assert args.frame_count == 60
    assert args.interval_seconds == 1.0


def test_background_reference_is_float_mean_with_preserved_sources(tmp_path):
    frames = [
        np.full((4, 5, 3), value, dtype=np.uint8)
        for value in (10, 20, 10, 20, 10, 20, 10, 20, 10, 20)
    ]

    record = save_background_reference(
        tmp_path,
        frames,
        serial="DTEST",
        capture_session="session-1",
        interval_seconds=1.0,
    )

    reference = cv2.imread(str(tmp_path / "reference.png"))
    assert np.array_equal(reference, np.full((4, 5, 3), 15, dtype=np.uint8))
    assert len(list((tmp_path / "frames/session-1").glob("*.png"))) == 10
    assert record["sample_count"] == 10
    assert record["reduction"] == "float32_mean_round_uint8"
    assert json.loads((tmp_path / "metadata.json").read_text()) == record


def test_background_reference_refuses_to_overwrite(tmp_path):
    frames = [np.zeros((4, 5, 3), dtype=np.uint8)] * 2
    save_background_reference(
        tmp_path,
        frames,
        serial="DTEST",
        capture_session="session-1",
        interval_seconds=1.0,
    )

    with pytest.raises(ValueError, match="already exists"):
        save_background_reference(
            tmp_path,
            frames,
            serial="DTEST",
            capture_session="session-2",
            interval_seconds=1.0,
        )


def test_average_background_rejects_inconsistent_frames():
    with pytest.raises(ValueError, match="same shape"):
        average_background([
            np.zeros((4, 5, 3), dtype=np.uint8),
            np.zeros((5, 4, 3), dtype=np.uint8),
        ])
