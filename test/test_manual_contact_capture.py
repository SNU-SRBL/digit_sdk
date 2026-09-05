import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from calibration.capture_session import CollectionCamera, StagedCaptureWriter
from calibration.collect_manual_contacts import (
    default_output_root,
    parse_args,
    staged_counts,
)


def test_save_frame_stages_unpartitioned_lossless_capture(tmp_path):
    frame = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
    captured_at = datetime(2026, 7, 16, 1, 2, 3, 456789, tzinfo=timezone.utc)

    writer = StagedCaptureWriter(
        tmp_path, serial="DTEST", capture_session="session-1"
    )
    record = writer.save(frame, "touch", captured_at=captured_at)

    assert "partition" not in record
    assert record["sample_id"].startswith("DTEST_touch_")
    assert record["split_group"] == record["sample_id"]
    assert record["path"].startswith("touches/session-1/")
    assert record["capture_fps"] == 30
    assert record["captured_at_utc"] == "2026-07-16T01:02:03.456789+00:00"
    saved_path = tmp_path / record["path"]
    assert np.array_equal(cv2.imread(str(saved_path)), frame)

    rows = [json.loads(line) for line in (tmp_path / "metadata.jsonl").read_text().splitlines()]
    assert rows == [record]
    assert not any("led" in key.lower() or "intensity" in key.lower() for key in record)


def test_manual_capture_writer_rejects_backgrounds(tmp_path):
    writer = StagedCaptureWriter(
        tmp_path,
        serial="DTEST",
        capture_session="session-1",
    )
    with pytest.raises(ValueError, match="unsupported capture kind"):
        writer.save(
            np.zeros((4, 5, 3), dtype=np.uint8),
            "background",
        )


def test_collection_has_no_capture_partition_argument():
    args = parse_args(["--serial", "DTEST"])
    assert args.serial == "DTEST"
    assert default_output_root("sensors", "DTEST") == Path(
        "sensors/DTEST/calibration/inbox/manual_mask"
    )

    with pytest.raises(SystemExit):
        parse_args(["--serial", "DTEST", "--partition", "train"])


def test_staged_counts_include_existing_records_for_requested_sensor(tmp_path):
    rows = [
        {"serial": "DTEST", "kind": "background"},
        {"serial": "DTEST", "kind": "touch"},
        {"serial": "DTEST", "kind": "touch"},
        {"serial": "OTHER", "kind": "touch"},
        {"serial": "DTEST", "kind": "ball"},
    ]
    (tmp_path / "metadata.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    counts = staged_counts(tmp_path, "DTEST")

    assert counts == {"touch": 2}


def test_save_refuses_overwrite_and_invalid_frames(tmp_path):
    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    writer = StagedCaptureWriter(
        tmp_path, serial="DTEST", capture_session="session-1"
    )
    timestamp = datetime(2026, 7, 16, tzinfo=timezone.utc)
    writer.save(frame, "touch", captured_at=timestamp)
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        writer.save(frame, "touch", captured_at=timestamp)
    with pytest.raises(ValueError, match="uint8 HxWx3"):
        writer.save(
            np.zeros((4, 5), dtype=np.uint8),
            "touch",
        )


def test_collection_camera_reconnects_by_serial_after_failed_read(
    monkeypatch, tmp_path
):
    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    failed = Mock()
    failed.cap.isOpened.return_value = True
    failed.cap.get.return_value = 30
    failed.get_image.return_value = None
    recovered = Mock()
    recovered.cap.isOpened.return_value = True
    recovered.cap.get.return_value = 30
    recovered.get_image.return_value = frame
    camera_factory = Mock(side_effect=[failed, recovered])
    monkeypatch.setattr("calibration.capture_session.Camera", camera_factory)
    monkeypatch.setattr("calibration.capture_session.time.sleep", Mock())

    camera = CollectionCamera("DTEST", tmp_path, warmup_frames=0)
    camera.connect()

    assert camera.read() is None
    assert camera.read() is frame
    failed.release.assert_called_once()
    recovered.connect.assert_called_once_with(verbose=False)
    assert camera_factory.call_count == 2


def test_collection_camera_retries_until_serial_returns(monkeypatch, tmp_path):
    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    failed = Mock()
    failed.cap.isOpened.return_value = True
    failed.cap.get.return_value = 30
    failed.get_image.return_value = None
    recovered = Mock()
    recovered.cap.isOpened.return_value = True
    recovered.cap.get.return_value = 30
    recovered.get_image.return_value = frame
    camera_factory = Mock(
        side_effect=[failed, RuntimeError("DIGIT absent"), recovered]
    )
    monkeypatch.setattr("calibration.capture_session.Camera", camera_factory)
    monkeypatch.setattr("calibration.capture_session.time.sleep", Mock())

    camera = CollectionCamera("DTEST", tmp_path, warmup_frames=0)
    camera.connect()

    assert camera.read() is None
    assert camera.read() is None
    assert camera.read() is frame
    assert camera_factory.call_count == 3
