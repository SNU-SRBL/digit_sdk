#!/usr/bin/env python3
"""Raw publisher: reads SHM, publishes /tactile/{serial}/raw over DDS.

One process per sensor — independent Python GIL for each camera.
Launched by multi_sensor_tactile_streamer.launch.py with serial:=... param.
"""
import array
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image

from digit_sdk.shm_protocol import open_shared_memory, read_camera_frame

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
        self.declare_parameter('cpu_affinity', '')

        serial = self.get_parameter('serial').value
        rate = self.get_parameter('rate').value
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

        # Connect to SHM (retry up to 10s)
        self._shm = None
        for _ in range(100):
            try:
                self._shm = open_shared_memory(f'tactile_{serial}')
                break
            except FileNotFoundError:
                time.sleep(0.1)
        if self._shm is None:
            self.get_logger().error(f'SHM not found for {serial} — exiting')
            return

        self._serial = serial
        self._last_seq = -1

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
            return
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
        msg.header.frame_id = f'tactile_{self._serial}_optical_frame'
        seconds, nanoseconds = divmod(snapshot.timestamp_ns, 1_000_000_000)
        msg.header.stamp.sec = seconds
        msg.header.stamp.nanosec = nanoseconds
        self._pub.publish(msg)

    def destroy_node(self):
        if self._shm is not None:
            self._shm.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RawPublisher()
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
