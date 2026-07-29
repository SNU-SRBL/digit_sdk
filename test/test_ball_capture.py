import json
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from calibration.collect_background import save_background_reference
from calibration.collect_ball import (
    background_difference,
    load_background_reference,
    overlay_guide,
    parse_args,
    target_for,
    validate_replacement_source,
)
from digit_sdk.camera import Camera


def test_guided_targets_cover_five_depths_then_advance_cell():
    assert target_for(0) == (0, 0, 1)
    assert target_for(4) == (0, 0, 5)
    assert target_for(5) == (0, 1, 1)
    assert target_for(24) == (0, 4, 5)
    assert target_for(25) == (1, 4, 1)
    assert target_for(49) == (1, 0, 5)
    assert target_for(50) == (2, 0, 1)
    assert target_for(124) == (4, 4, 5)


def test_ball_collection_does_not_require_a_partition():
    args = parse_args([
        "--serial", "DTEST", "--ball-diameter-mm", "3",
    ])
    assert not hasattr(args, "partition")


def test_targeted_replacement_arguments_select_one_guide_target():
    args = parse_args([
        "--serial", "DTEST", "--ball-diameter-mm", "3",
        "--target-row", "0", "--target-column", "4",
        "--target-depth", "5", "--replaces", "bad-sample",
    ])

    assert (args.target_row, args.target_column, args.target_depth) == (0, 4, 5)
    assert args.replaces == "bad-sample"


def test_replacement_source_must_be_rejected_and_not_already_replaced(tmp_path):
    source = {
        "sample_id": "bad-sample", "kind": "ball", "serial": "DTEST",
        "ball_diameter_mm": 3.0, "guide_grid_row": 0,
        "guide_grid_column": 4, "guide_depth_level": 5, "rejected": True,
    }
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text(json.dumps(source) + "\n", encoding="utf-8")

    result = validate_replacement_source(
        tmp_path, "bad-sample", "DTEST", 3.0, (0, 4, 5)
    )

    assert result == source
    replacement = {**source, "sample_id": "replacement", "rejected": False,
                   "replaces_sample_id": "bad-sample"}
    metadata.write_text(
        json.dumps(source) + "\n" + json.dumps(replacement) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="already has a replacement"):
        validate_replacement_source(
            tmp_path, "bad-sample", "DTEST", 3.0, (0, 4, 5)
        )


def test_background_difference_option_is_available():
    args = parse_args([
        "--serial", "DTEST", "--ball-diameter-mm", "3",
        "--display-difference",
    ])
    assert args.display_difference is True


def test_ball_collection_loads_shared_background_reference(tmp_path):
    background_root = (
        tmp_path / "DTEST/calibration/inbox/background"
    )
    expected = np.full((4, 5, 3), 17, dtype=np.uint8)
    save_background_reference(
        background_root,
        [expected] * 2,
        serial="DTEST",
        capture_session="session-1",
        interval_seconds=1.0,
    )

    reference, background_id = load_background_reference(tmp_path, "DTEST")

    assert np.array_equal(reference, expected)
    assert background_id == "DTEST_background_reference"


def test_guide_does_not_modify_captured_frame():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    output = overlay_guide(frame, target_index=0, saved=0)

    assert not np.array_equal(output, frame)
    assert np.count_nonzero(frame) == 0


def test_background_difference_is_amplified_without_modifying_inputs():
    background = np.full((4, 5, 3), 10, dtype=np.uint8)
    frame = background.copy()
    frame[1, 2] = 20

    output = background_difference(frame, background)

    assert np.array_equal(output[1, 2], [30, 30, 30])
    assert np.array_equal(background, np.full((4, 5, 3), 10, dtype=np.uint8))
    assert np.array_equal(frame[1, 2], [20, 20, 20])


def test_camera_framerate_argument_overrides_sensor_config(monkeypatch):
    config = {
        "device_type": "DIGIT",
        "imgh": 320,
        "imgw": 240,
        "raw_imgh": 320,
        "raw_imgw": 240,
        "framerate": 60,
    }
    monkeypatch.setattr("digit_sdk.camera.load_config", Mock(return_value=config))
    monkeypatch.setattr(
        "digit_sdk.camera.DigitHandler.find_digit",
        Mock(return_value={"dev_name": "/dev/video1"}),
    )

    camera = Camera(serial="DTEST", sensors_root="sensors", framerate=30)

    assert camera.framerate == 30
    assert camera._wd_dt == 0.05
