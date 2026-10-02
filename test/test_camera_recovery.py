import pytest
from unittest.mock import Mock, patch

import numpy as np

from digit_sdk.camera import Camera, DigitHandler


def _disconnected_digit() -> Camera:
    camera = Camera.__new__(Camera)
    camera.device_type = "DIGIT"
    camera.serial = "DTEST"
    camera.device = "/dev/video99"
    camera.dev_id = 99
    camera.raw_imgw = 320
    camera.raw_imgh = 240
    camera.framerate = 60
    return camera


def test_digit_discovery_retries_during_initial_udev_enumeration():
    found = {
        "dev_name": "/dev/video8",
        "manufacturer": "Facebook",
        "model": "DIGIT",
        "revision": "0200",
        "serial": "D21275",
    }

    with patch.object(
        DigitHandler,
        "list_digits",
        side_effect=[[], [], [found]],
    ), patch("digit_sdk.camera.time.sleep") as sleep:
        result = DigitHandler.find_digit(
            "D21275", attempts=3, retry_interval=0.1
        )

    assert result == found
    assert sleep.call_count == 2


def test_connect_uses_stable_serial_path_and_discards_startup_frames():
    camera = _disconnected_digit()
    capture = Mock()
    capture.isOpened.return_value = True
    capture.read.return_value = (True, np.zeros((240, 320, 3), np.uint8))

    with patch("digit_sdk.camera.Path.exists", return_value=True), patch(
        "digit_sdk.camera.os.path.realpath", return_value="/dev/video8"
    ), patch("digit_sdk.camera.V4L2Capture", return_value=capture) as open_:
        camera.connect(verbose=False)

    args, kwargs = open_.call_args
    assert args == ("/dev/v4l/by-id/usb-Facebook_DIGIT_DTEST-video-index0", 320, 240, 60)
    assert kwargs["phase_timings"] == camera._last_connect_timings
    assert capture.read.call_count == 10


def test_connect_rejects_failed_startup_read():
    camera = _disconnected_digit()
    capture = Mock()
    capture.isOpened.return_value = True
    capture.read.return_value = (False, None)

    with patch("digit_sdk.camera.Path.exists", return_value=True), patch(
        "digit_sdk.camera.os.path.realpath", return_value="/dev/video8"
    ), patch("digit_sdk.camera.V4L2Capture", return_value=capture):
        with pytest.raises(RuntimeError, match="warm-up failed"):
            camera.connect(verbose=False)

    capture.release.assert_called_once_with()


def test_recover_reopens_without_extra_reads():
    camera = Camera.__new__(Camera)
    camera._recovery_sleep = 0.0
    camera._recovery_retry_sleep = 0.0
    camera._recovery_started_at = None
    camera._recovery_reason = None
    camera._recovery_failures = 0
    camera.serial = "DTEST"
    camera.device = "/dev/video0"
    camera.cap = Mock()
    camera.release = Mock()
    camera.connect = Mock()

    assert camera._recover("read_failed") is True

    camera.release.assert_called_once_with()
    camera.connect.assert_called_once_with(verbose=False)
    camera.cap.read.assert_not_called()


def test_failed_read_triggers_recovery_and_resets_watchdog():
    camera = Camera.__new__(Camera)
    camera.cap = Mock()
    camera.cap.read.return_value = (False, None)
    camera._recover = Mock()
    camera._wd_slow_count = 7
    camera._wd_last_time = 123.0

    assert camera.get_image() is None

    camera._recover.assert_called_once_with("read_failed")
    assert camera._wd_slow_count == 0
    assert camera._wd_last_time == 0.0


def test_isolated_slow_frame_does_not_trigger_recovery():
    camera = Camera.__new__(Camera)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    camera.cap = Mock()
    camera.cap.read.return_value = (True, frame)
    camera._is_corrupt = Mock(return_value=False)
    camera._recover = Mock()
    camera._wd_dt = 0.025
    camera._wd_max_slow = 10
    camera._wd_slow_count = 3
    camera._wd_last_time = 10.0

    with patch("digit_sdk.camera.time.monotonic", return_value=10.2):
        result = camera.get_image()

    np.testing.assert_array_equal(result, frame)
    camera._recover.assert_not_called()
    assert camera._wd_slow_count == 4
    assert camera._wd_last_time == 10.2


def test_failed_reconnect_is_retryable_without_raising():
    camera = Camera.__new__(Camera)
    camera._recovery_sleep = 0.0
    camera._recovery_retry_sleep = 0.0
    camera._recovery_started_at = None
    camera._recovery_reason = None
    camera._recovery_failures = 0
    camera.cap = Mock()
    camera.release = Mock()
    camera.connect = Mock(side_effect=RuntimeError("camera absent"))

    assert camera._recover("read_failed") is False
    assert camera._recovery_reason == "read_failed"
    assert camera._recovery_failures == 1


def _sensor_config(**overrides):
    config = {
        "device_type": "DIGIT",
        "imgh": 240,
        "imgw": 320,
        "raw_imgh": 240,
        "raw_imgw": 320,
        "framerate": 60,
    }
    config.update(overrides)
    return config


def test_capture_fps_defaults_to_sensor_yaml():
    with patch("digit_sdk.camera.load_config", return_value=_sensor_config()), patch.object(
        DigitHandler, "find_digit", return_value={"dev_name": "/dev/video8"}
    ):
        camera = Camera(serial="DTEST")

    assert camera.framerate == 60


def test_capture_fps_override_wins_over_sensor_yaml():
    with patch("digit_sdk.camera.load_config", return_value=_sensor_config()), patch.object(
        DigitHandler, "find_digit", return_value={"dev_name": "/dev/video8"}
    ):
        camera = Camera(serial="DTEST", framerate=30)

    assert camera.framerate == 30


def test_corruption_detector_flags_row_discontinuity():
    frame = np.full((240, 320, 3), 20, dtype=np.uint8)
    frame[120:] = 220

    assert Camera._is_corrupt(frame) is True
