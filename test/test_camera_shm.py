import importlib.util
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


_SOURCE = Path(__file__).parents[1] / "ros2" / "camera_shm.py"
_SPEC = importlib.util.spec_from_file_location("camera_shm", _SOURCE)
camera_shm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(camera_shm)

_LAUNCH_SOURCE = (
    Path(__file__).parents[1]
    / "ros2"
    / "launch"
    / "multi_sensor_tactile_streamer.launch.py"
)


def test_connect_with_retry_retries_then_connects():
    camera = Mock()
    camera.serial = "DTEST"
    camera.connect.side_effect = [RuntimeError("not ready"), None]

    with patch.object(camera_shm.time, "sleep") as sleep:
        camera_shm.connect_with_retry(camera, verbose=False, timeout_s=1.0)

    assert camera.connect.call_count == 2
    camera.release.assert_called_once_with()
    sleep.assert_called_once_with(camera_shm.STARTUP_CONNECT_RETRY_S)


def test_connect_with_retry_reports_timeout():
    camera = Mock()
    camera.serial = "DTEST"
    camera.connect.side_effect = RuntimeError("not ready")

    with patch.object(camera_shm.time, "monotonic", side_effect=[0.0, 1.0]):
        try:
            camera_shm.connect_with_retry(camera, verbose=False, timeout_s=1.0)
        except RuntimeError as error:
            assert "startup failed after 1 attempts" in str(error)
        else:
            raise AssertionError("expected startup timeout")


def _run_once(camera, **kwargs):
    """Run camera_shm.run with the capture loop short-circuited after setup."""
    with patch.object(camera_shm, "Camera", return_value=camera) as camera_cls, patch.object(
        camera_shm, "connect_with_retry"
    ), patch.object(camera_shm.shared_memory, "SharedMemory") as shm, patch.object(
        camera_shm.signal, "signal", side_effect=lambda _sig, handler: handler()
    ):
        shm.return_value.buf = b""
        camera_shm.run("DTEST", "/root/sensors", **kwargs)
    return camera_cls


def _fake_camera():
    camera = Mock()
    camera.serial = "DTEST"
    camera.device = "/dev/video99"
    camera.dev_id = 99
    camera.raw_imgh = 240
    camera.raw_imgw = 320
    camera.framerate = 60.0
    return camera


def test_run_defaults_keep_sensor_yaml_framerate():
    camera = _fake_camera()
    camera_cls = _run_once(camera)

    camera_cls.assert_called_once_with(
        serial="DTEST", sensors_root="/root/sensors", framerate=None
    )


def test_run_passes_capture_fps_when_configured():
    camera = _fake_camera()
    camera_cls = _run_once(camera, capture_fps=30.0)

    camera_cls.assert_called_once_with(
        serial="DTEST", sensors_root="/root/sensors", framerate=30.0
    )


def _launch_module():
    pytest.importorskip("launch")
    spec = importlib.util.spec_from_file_location(
        "multi_sensor_tactile_streamer", _LAUNCH_SOURCE
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _camera_commands(module, serials=("DTEST",), **overrides):
    launch = pytest.importorskip("launch")
    from launch.utilities import perform_substitutions

    values = {
        "sensors_root": "/root/sensors",
        "serials": ",".join(serials),
        "model_device": "cpu",
        "rate": "60.0",
        "depth_cutoff": "0.2",
        "publish_raw": "false",
        "publish_depth": "false",
        "publish_pointcloud": "false",
        "publish_force": "false",
        "color_pointcloud_with_force": "false",
        "models_root": "/root/models",
        "force_rate": "30.0",
        "point_sample_mm": "0.2",
        "shm_connect_timeout": "30.0",
        "capture_fps": "",
    }
    values.update(overrides)
    context = launch.LaunchContext()
    # LaunchContext.launch_configurations is a read-only property; the
    # supported way to seed it is to mutate the returned dict.
    context.launch_configurations.update(values)

    with patch.object(module, "_owned_serials", return_value=list(serials)), patch.object(
        module, "_serial_scoped_cleanup"
    ), patch.object(module, "get_package_prefix", return_value="/pkg/digit_sdk"), patch.object(
        module, "_get_tactile_affinity", return_value={"camera_shm": [0]}
    ):
        nodes = module.launch_setup(context)

    return [
        [perform_substitutions(context, part) for part in node.cmd]
        for node in nodes[: len(serials)]
    ]


def test_launch_omits_capture_flags_when_unset():
    args = _camera_commands(_launch_module())[0]

    assert "--capture-fps" not in args


def test_launch_passes_capture_flags_only_when_configured():
    args = _camera_commands(
        _launch_module(),
        capture_fps="30",
    )[0]

    assert args[args.index("--capture-fps") + 1] == "30"
