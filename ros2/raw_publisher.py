#!/usr/bin/env python3
"""Raw publisher: reads SHM, publishes /tactile/{serial}/raw over DDS.

One process per sensor — independent Python GIL for each camera.
Launched by multi_sensor_tactile_streamer.launch.py with serial:=... param.
"""
import array
import os
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image

from digit_sdk.publisher_shm import (
    ShmConnectError,
    connect_shm_with_retry,
    reopen_shm_if_stale,
)
from digit_sdk.shm_protocol import read_camera_frame

_BE_QOS = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)


class RawPublisher(Node):
    """Reads a single sensor's SHM block, publishes BGR raw frames over DDS.

    One instance per camera — no GIL contention with other sensors.
    """

    def __init__(self):
        super().__init__('raw_publisher')

        self.declare_parameter('serial', value='')
        self.declare_parameter('rate', 60.0)
        self.declare_parameter('liveness_timeout', 2.0)
        self.declare_parameter('cpu_affinity', '')

        serial = self.get_parameter('serial').value
        rate = self.get_parameter('rate').value
        self._liveness_timeout = float(
            self.get_parameter('liveness_timeout').value
        )
        cpu_affinity = self.get_parameter('cpu_affinity').value

        if cpu_affinity:
            cores = set()
            for part in cpu_affinity.split(','):
                part = part.strip()
                if '-' in part:
                    lo, hi = part.split('-', 1)
                    cores.update(range(int(lo), int(hi) + 1))
                else:
                    cores.add(int(part))
            if cores:
                os.sched_setaffinity(0, cores)

        if not serial:
            self.get_logger().error('No serial parameter — nothing to do')
            return

        self._serial = serial
        self._last_seq = -1
        self._last_fresh = time.monotonic()
        self._shm = connect_shm_with_retry(f'tactile_{serial}')

        # Publisher
        self._pub = self.create_publisher(
            Image, f'tactile/{serial}/raw', _BE_QOS)

        # Single timer — one sensor, no GIL contention
        self._timer = self.create_timer(1.0 / rate, self._handle_sensor)

        self.get_logger().info(f'Raw publisher ready for {serial} @ {rate:.0f}Hz')

    def _handle_sensor(self):
        """Read latest frame from SHM, publish as BGR Image."""
        if self._shm is None:
            return
        snapshot = read_camera_frame(self._shm.buf)
        if snapshot is None or snapshot.sequence == self._last_seq:
            if time.monotonic() - self._last_fresh > self._liveness_timeout:
                self._recover_shm()
            return
        self._last_fresh = time.monotonic()
        self._last_seq = snapshot.sequence
        bgr = snapshot.image
        h, w = bgr.shape[:2]

        msg = Image()
        msg.height = h
        msg.width = w
        msg.encoding = 'bgr8'
        msg.is_bigendian = False
        msg.step = w * 3
        msg.data = array.array('B', bgr.tobytes())
        msg.header.frame_id = f'tactile_{self._serial}_depth_frame'
        seconds, nanoseconds = divmod(snapshot.timestamp_ns, 1_000_000_000)
        msg.header.stamp.sec = seconds
        msg.header.stamp.nanosec = nanoseconds
        self._pub.publish(msg)

    def _recover_shm(self):
        name = f'tactile_{self._serial}'
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
        self._last_seq = -1
        self._last_fresh = time.monotonic()
        self.get_logger().warn(f'Reattached camera SHM {name}')

    def destroy_node(self):
        if self._shm is not None:
            self._shm.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    try:
        node = RawPublisher()
    except ShmConnectError as exc:
        print(f'FATAL: {exc}', file=sys.stderr, flush=True)
        sys.exit(1)
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


if __name__ == '__main__':
    main()
