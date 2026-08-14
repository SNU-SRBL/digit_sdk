"""Unit tests for digit_sdk SHM publisher helpers and publishers."""

import multiprocessing
from multiprocessing import resource_tracker
from pathlib import Path
import types
from unittest.mock import patch

import pytest

from digit_sdk.publisher_shm import (
    ShmConnectError,
    connect_shm_with_retry,
    open_shared_memory_safely,
    reopen_shm_if_stale,
)


def _unlink_shm(shm):
    try:
        shm.unlink()
    except FileNotFoundError:
        pass
    try:
        resource_tracker.unregister(shm._name, "shared_memory")
    except KeyError:
        pass
    shm.close()


def _shm_for_producer(name, size):
    """Segment owned by a fake producer; not attached through the helper."""
    return multiprocessing.shared_memory.SharedMemory(
        name=name, create=True, size=size
    )


@pytest.mark.timeout(30)
def test_connect_shm_with_retry_raises_after_bounded_timeout():
    name = "tactile_retry_missing"
    _unlink_shm(multiprocessing.shared_memory.SharedMemory(
        name=name, create=True, size=64
    ))
    with pytest.raises(ShmConnectError, match=name):
        connect_shm_with_retry(name, timeout_s=0.2, poll_s=0.05)


@pytest.mark.timeout(30)
def test_connect_shm_with_retry_attaches_once_available():
    name = "tactile_retry_late"
    existing = multiprocessing.shared_memory.SharedMemory(
        name=name, create=True, size=64
    )
    shm = connect_shm_with_retry(name, timeout_s=2.0, poll_s=0.05)
    try:
        assert shm.name == name
    finally:
        _unlink_shm(existing)
        _unlink_shm(shm)


@pytest.mark.timeout(30)
def test_open_shared_memory_safely_returns_none_when_missing():
    assert open_shared_memory_safely("tactile_definitely_missing") is None


@pytest.mark.timeout(30)
def test_reopen_shm_after_producer_restart():
    name = "tactile_reopen_restart"
    first = _shm_for_producer(name, 64)
    try:
        reader = open_shared_memory_safely(name)
        fresh = reopen_shm_if_stale(name, reader, liveness_timeout_s=2.0)
        assert fresh is not None
        assert fresh.name == name
        assert fresh is not reader
        fresh.close()

        _unlink_shm(first)
        second = _shm_for_producer(name, 64)
        try:
            reader = open_shared_memory_safely(name)
            fresh = reopen_shm_if_stale(
                name, reader, liveness_timeout_s=2.0
            )
            assert fresh is not None
            assert fresh.name == name
            assert fresh is not reader
            fresh.close()
        finally:
            _unlink_shm(second)
    finally:
        _unlink_shm(first)


@pytest.mark.timeout(30)
def test_reopen_shm_returns_none_when_name_absent():
    name = "tactile_reopen_absent"
    shm = multiprocessing.shared_memory.SharedMemory(
        name=name, create=True, size=64
    )
    _unlink_shm(shm)
    assert reopen_shm_if_stale(name, shm, liveness_timeout_s=0.2) is None


@pytest.mark.timeout(30)
def test_raw_publisher_exits_1_on_connect_failure():
    from ros2.raw_publisher import main

    with patch("ros2.raw_publisher.RawPublisher") as node:
        node.side_effect = ShmConnectError("camera SHM unavailable")
        with patch("ros2.raw_publisher.rclpy") as rclpy_mock:
            rclpy_mock.init.return_value = None
            rclpy_mock.ok.return_value = False
            with pytest.raises(SystemExit) as exc:
                main()
    assert exc.value.code == 1


@pytest.mark.timeout(30)
def test_force_publisher_exits_1_on_connect_failure():
    from ros2.force_publisher import main

    with patch("ros2.force_publisher.ForcePublisher") as node:
        node.side_effect = ShmConnectError("force SHM unavailable")
        with patch("ros2.force_publisher.rclpy") as rclpy_mock:
            rclpy_mock.init.return_value = None
            rclpy_mock.ok.return_value = False
            with pytest.raises(SystemExit) as exc:
                main()
    assert exc.value.code == 1


@pytest.mark.timeout(30)
def test_surface_publisher_qos_depth_is_10():
    from ros2.surface_publisher import _BE_QOS

    assert _BE_QOS.depth == 10


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    ("rate", "oversample", "pointcloud"),
    [(60.0, 2.0, False), (120.0, 4.0, True)],
)
def test_surface_publisher_timer_period_uses_poll_oversample(
    rate, oversample, pointcloud
):
    import rclpy
    from ros2.surface_publisher import SurfacePublisher

    args = [
        "--ros-args",
        "-p", "serial:=unit",
        "-p", "rate:={}".format(rate),
        "-p", "poll_oversample:={}".format(oversample),
    ]
    if pointcloud:
        args += ["-p", "publish_pointcloud:=true", "-p", "ppmm:=10.0"]
    rclpy.init(args=args)
    node = None
    fake_shm = types.SimpleNamespace(buf=b"", close=lambda: None)
    try:
        with patch(
            "ros2.surface_publisher.connect_shm_with_retry",
            return_value=fake_shm,
        ):
            node = SurfacePublisher()
        periods = {timer.timer_period_ns for timer in node._timers}
        assert periods == {int(1e9 / (rate * oversample))}
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


@pytest.mark.timeout(30)
def test_launch_cleanup_is_serial_scoped():
    launch_path = Path(__file__).parents[1] / "ros2" / "launch" / (
        "multi_sensor_tactile_streamer.launch.py"
    )
    assert launch_path.is_file(), launch_path
    source = launch_path.read_text()

    assert (
        "camera_shm|pipeline_node|raw_publisher|surface_publisher"
        not in source
    )
    assert "rm -f /dev/shm/tactile_*" not in source
    assert "tactile_*" not in source
    assert 'pkill' in source
    assert "--serial {serial}" in source
    assert "/dev/shm/tactile_{serial}" in source
    assert '"_surface"' in source
    assert '"_force"' in source
