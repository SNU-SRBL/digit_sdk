#!/usr/bin/env python3
"""Force publisher: reads SHM tactile_{serial}_force, publishes force DDS.

One process per sensor — independent GIL.  Reads force_field (normal+shear)
and force_vector from shared memory written by pipeline_node.

Topics:
  /tactile/{serial}/force_field      — sensor_msgs/Image (32FC3)
  /tactile/{serial}/force_field_viz  — sensor_msgs/Image (rgb8)
  /tactile/{serial}/force_vector     — geometry_msgs/WrenchStamped
"""

import os
import struct
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import WrenchStamped
import numpy as np

from digit_sdk.viz_utils import force_field_to_rgb
from digit_sdk.publisher_shm import (
    ShmConnectError,
    connect_shm_with_retry,
    reopen_shm_if_stale,
)

_BE_QOS = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)

SHM_HEADER = 40


class ForcePublisher(Node):
    """Reads force SHM, publishes force_field + force_vector over DDS."""

    def __init__(self):
        super().__init__('force_publisher')

        self.declare_parameter('serial', value='')
        self.declare_parameter('rate', 30.0)
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
        self._last_fresh = time.monotonic()
        self._shm = connect_shm_with_retry(f'tactile_{serial}_force')

        # Publishers
        self._pub_field = self.create_publisher(
            Image, f'tactile/{serial}/force_field', _BE_QOS)
        self._pub_field_viz = self.create_publisher(
            Image, f'tactile/{serial}/force_field_viz', _BE_QOS)
        self._pub_vector = self.create_publisher(
            WrenchStamped, f'tactile/{serial}/force_vector', _BE_QOS)

        self._timer = self.create_timer(1.0 / rate, self._handle_sensor)

        self.get_logger().info(
            f'Force publisher ready for {serial} @ {rate:.0f}Hz')

    def _handle_sensor(self):
        """Read latest force data from SHM, publish."""
        if self._shm is None:
            return
        buf = self._shm.buf

        if not buf[36]:
            if time.monotonic() - self._last_fresh > self._liveness_timeout:
                self._recover_shm()
            return
        self._last_fresh = time.monotonic()

        h, w = struct.unpack_from('<II', buf, 16)
        fx, fy, fz = struct.unpack_from('<fff', buf, 24)
        if h == 0 or w == 0:
            return

        offset = SHM_HEADER  # 40
        normal = np.frombuffer(buf[offset:offset + h * w * 4],
                               dtype=np.float32).reshape(h, w)
        offset += h * w * 4
        shear_x = np.frombuffer(buf[offset:offset + h * w * 4],
                                dtype=np.float32).reshape(h, w)
        offset += h * w * 4
        shear_y = np.frombuffer(buf[offset:offset + h * w * 4],
                                dtype=np.float32).reshape(h, w)

        header = self.get_clock().now().to_msg()
        frame = f'tactile_{self._serial}_depth_frame'

        # Force field: pack normal+shear into 32FC3
        force_rgb = np.zeros((h, w, 3), dtype=np.float32)
        force_rgb[:, :, 0] = shear_x
        force_rgb[:, :, 1] = shear_y
        force_rgb[:, :, 2] = normal
        field_msg = Image()
        field_msg.height = h
        field_msg.width = w
        field_msg.encoding = '32FC3'
        field_msg.is_bigendian = False
        field_msg.step = w * 4 * 3
        field_msg.data = np.ascontiguousarray(force_rgb).tobytes()
        field_msg.header.frame_id = frame
        field_msg.header.stamp = header
        self._pub_field.publish(field_msg)

        # Force field viz: force_field_to_rgb
        viz_msg = Image()
        viz_msg.height = h
        viz_msg.width = w
        viz_msg.encoding = 'rgb8'
        viz_msg.is_bigendian = False
        viz_msg.step = w * 3
        viz_msg.data = np.ascontiguousarray(
            force_field_to_rgb(normal, np.stack([shear_x, shear_y], axis=-1))).tobytes()
        viz_msg.header.frame_id = frame
        viz_msg.header.stamp = header
        self._pub_field_viz.publish(viz_msg)

        # Force vector
        vec_msg = WrenchStamped()
        vec_msg.header.frame_id = frame
        vec_msg.header.stamp = header
        vec_msg.wrench.force.x = float(fx)
        vec_msg.wrench.force.y = float(fy)
        vec_msg.wrench.force.z = float(fz)
        self._pub_vector.publish(vec_msg)

    def _recover_shm(self):
        name = f'tactile_{self._serial}_force'
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
        self._last_fresh = time.monotonic()
        self.get_logger().warn(f'Reattached force SHM {name}')

    def destroy_node(self):
        if self._shm is not None:
            self._shm.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    try:
        node = ForcePublisher()
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
