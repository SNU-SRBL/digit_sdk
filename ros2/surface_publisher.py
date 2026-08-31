#!/usr/bin/env python3
"""Per-sensor depth and point-cloud publisher from latest surface SHM."""

import array
import os
import sys
import time

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy._rclpy_pybind11 import RCLError
from sensor_msgs.msg import Image, PointCloud2, PointField

from digit_sdk.depth_geometry import depth_to_pointcloud
from digit_sdk.publisher_shm import (
    ShmConnectError,
    connect_shm_with_retry,
    reopen_shm_if_stale,
)
from digit_sdk.shm_protocol import read_surface_frame


_BE_QOS = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)


def _set_stamp(stamp, timestamp_ns: int) -> None:
    seconds, nanoseconds = divmod(int(timestamp_ns), 1_000_000_000)
    stamp.sec = seconds
    stamp.nanosec = nanoseconds


class SurfacePublisher(Node):
    """Publish independently selectable depth and point-cloud paths."""

    def __init__(self):
        super().__init__("surface_publisher")
        self.declare_parameter("serial", value="")
        self.declare_parameter("rate", 60.0)
        self.declare_parameter("poll_oversample", 2.0)
        self.declare_parameter("liveness_timeout", 2.0)
        self.declare_parameter("cpu_affinity", "")
        self.declare_parameter("publish_depth", True)
        self.declare_parameter("publish_pointcloud", False)
        self.declare_parameter("shm_connect_timeout", 10.0)
        self.declare_parameter("ppmm", 0.0)
        self.declare_parameter("point_sample_mm", 0.2)

        serial = self.get_parameter("serial").value
        rate = float(self.get_parameter("rate").value)
        poll_oversample = float(self.get_parameter("poll_oversample").value)
        self._liveness_timeout = float(
            self.get_parameter("liveness_timeout").value
        )
        self._publish_depth = bool(self.get_parameter("publish_depth").value)
        self._publish_pointcloud = bool(
            self.get_parameter("publish_pointcloud").value
        )
        self._ppmm = float(self.get_parameter("ppmm").value)
        self._point_sample_mm = float(
            self.get_parameter("point_sample_mm").value
        )
        self._shm_connect_timeout = float(
            self.get_parameter("shm_connect_timeout").value
        )
        self._apply_affinity(self.get_parameter("cpu_affinity").value)

        if not serial:
            raise ValueError("serial parameter is required")
        if not self._publish_depth and not self._publish_pointcloud:
            raise ValueError("at least one surface output must be enabled")
        if self._publish_pointcloud and self._ppmm <= 0:
            raise ValueError("positive ppmm is required for point-cloud output")
        if poll_oversample <= 0:
            raise ValueError("poll_oversample must be positive")

        self._serial = serial
        self._last_depth_seq = -1
        self._last_pointcloud_seq = -1
        self._last_fresh = time.monotonic()
        self._shm = connect_shm_with_retry(
            f"tactile_{serial}_surface",
            timeout_s=self._shm_connect_timeout,
        )

        self._pub_depth = (
            self.create_publisher(Image, f"tactile/{serial}/depth", _BE_QOS)
            if self._publish_depth else None
        )
        self._pub_pc = (
            self.create_publisher(
                PointCloud2, f"tactile/{serial}/pointcloud", _BE_QOS
            )
            if self._publish_pointcloud else None
        )
        self._timers = []
        poll_period = 1.0 / (rate * poll_oversample)
        if self._publish_depth:
            self._depth_group = MutuallyExclusiveCallbackGroup()
            self._timers.append(self.create_timer(
                poll_period,
                self._publish_latest_depth,
                callback_group=self._depth_group,
            ))
        if self._publish_pointcloud:
            self._pointcloud_group = MutuallyExclusiveCallbackGroup()
            self._timers.append(self.create_timer(
                poll_period,
                self._publish_latest_pointcloud,
                callback_group=self._pointcloud_group,
            ))
        enabled = ", ".join(
            name for name, value in (
                ("depth", self._publish_depth),
                ("pointcloud", self._publish_pointcloud),
            ) if value
        )
        self.get_logger().info(
            f"Surface publisher ready for {serial}: {enabled} @ {rate:.0f}Hz"
        )

    @staticmethod
    def _apply_affinity(value: str) -> None:
        if not value:
            return
        cores = set()
        for part in value.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-", 1)
                cores.update(range(int(lo), int(hi) + 1))
            else:
                cores.add(int(part))
        if cores:
            os.sched_setaffinity(0, cores)

    def _publish_latest_depth(self):
        if not rclpy.ok():
            return
        snapshot = read_surface_frame(self._shm.buf)
        if snapshot is None or snapshot.sequence == self._last_depth_seq:
            if time.monotonic() - self._last_fresh > self._liveness_timeout:
                self._recover_shm()
            return
        self._last_fresh = time.monotonic()
        self._last_depth_seq = snapshot.sequence
        frame_id = f"tactile_{self._serial}_depth_frame"
        height, width = snapshot.depth.shape
        depth_m = np.ascontiguousarray(snapshot.depth / 1000.0, dtype=np.float32)
        message = Image()
        message.height = height
        message.width = width
        message.encoding = "32FC1"
        message.is_bigendian = False
        message.step = width * 4
        message.data = array.array("B", depth_m.tobytes())
        message.header.frame_id = frame_id
        _set_stamp(message.header.stamp, snapshot.timestamp_ns)
        try:
            self._pub_depth.publish(message)
        except RCLError:
            if rclpy.ok():
                raise

    def _publish_latest_pointcloud(self):
        if not rclpy.ok():
            return
        snapshot = read_surface_frame(self._shm.buf)
        if snapshot is None or snapshot.sequence == self._last_pointcloud_seq:
            if time.monotonic() - self._last_fresh > self._liveness_timeout:
                self._recover_shm()
            return
        self._last_fresh = time.monotonic()
        self._last_pointcloud_seq = snapshot.sequence
        points = depth_to_pointcloud(
            snapshot.depth, self._ppmm, self._point_sample_mm
        )
        message = PointCloud2()
        message.header.frame_id = f"tactile_{self._serial}_depth_frame"
        _set_stamp(message.header.stamp, snapshot.timestamp_ns)
        message.height = 1
        message.width = len(points)
        message.fields = [
            PointField(
                name="x", offset=0, datatype=PointField.FLOAT32, count=1
            ),
            PointField(
                name="y", offset=4, datatype=PointField.FLOAT32, count=1
            ),
            PointField(
                name="z", offset=8, datatype=PointField.FLOAT32, count=1
            ),
        ]
        message.is_bigendian = False
        message.point_step = 12
        message.row_step = 12 * len(points)
        message.is_dense = True
        message.data = array.array("B", points.tobytes())
        try:
            self._pub_pc.publish(message)
        except RCLError:
            if rclpy.ok():
                raise

    def _recover_shm(self):
        name = f"tactile_{self._serial}_surface"
        fresh = reopen_shm_if_stale(
            name, self._shm, liveness_timeout_s=self._liveness_timeout
        )
        if fresh is None:
            try:
                fresh = connect_shm_with_retry(
                    name, timeout_s=self._liveness_timeout
                )
            except ShmConnectError as exc:
                self.get_logger().fatal(str(exc))
                sys.exit(1)
        self._shm = fresh
        self._last_depth_seq = -1
        self._last_pointcloud_seq = -1
        self._last_fresh = time.monotonic()
        self.get_logger().warn(f"Reattached surface SHM {name}")

    def destroy_node(self):
        for timer in self._timers:
            timer.cancel()
        if self._shm is not None:
            self._shm.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    try:
        node = SurfacePublisher()
    except ShmConnectError as exc:
        print(f"FATAL: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
    if node._publish_depth and node._publish_pointcloud:
        executor = MultiThreadedExecutor(num_threads=2)
    else:
        executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
