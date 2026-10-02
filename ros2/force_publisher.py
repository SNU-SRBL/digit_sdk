#!/usr/bin/env python3
"""Estimate force from camera SHM and publish it over DDS."""

import os
import sys
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import WrenchStamped
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from digit_sdk.force_estimator import ForceEstimator
from digit_sdk.publisher_shm import (
    ShmConnectError,
    connect_shm_with_retry,
    reopen_shm_if_stale,
)
from digit_sdk.shm_protocol import read_camera_frame
from digit_sdk.utils import load_config
from digit_sdk.viz_utils import force_field_to_rgb


_BE_QOS = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)
_FORCE_MODEL_FILES = {
    "sparsh-dino-base": "sparsh_dino_base_encoder.ckpt",
    "sparsh-digit-forcefield": "sparsh_digit_forcefield_decoder.pth",
}


def _parse_affinity(value):
    cores = set()
    for part in value.split(","):
        if not part:
            continue
        start, *end = part.split("-", 1)
        cores.update(range(int(start), int(end[0]) + 1) if end else [int(start)])
    return cores


def _force_model_paths(config, models_root):
    force = (config or {}).get("force") or {}
    try:
        encoder = _FORCE_MODEL_FILES[force.get("encoder", "sparsh-dino-base")]
        decoder = _FORCE_MODEL_FILES[
            force.get("decoder", "sparsh-digit-forcefield")
        ]
    except KeyError as error:
        raise ValueError(f"unsupported force model {error.args[0]!r}") from error
    return os.path.join(models_root, encoder), os.path.join(models_root, decoder)


class ForcePublisher(Node):
    """One force estimator per enabled sensor."""

    def __init__(self):
        super().__init__("force_publisher")
        self.declare_parameter("serial", "")
        self.declare_parameter("sensors_root", "")
        self.declare_parameter("models_root", "")
        self.declare_parameter("model_device", "cuda")
        self.declare_parameter("rate", 30.0)
        self.declare_parameter("liveness_timeout", 2.0)
        self.declare_parameter("shm_connect_timeout", 30.0)
        self.declare_parameter("cpu_affinity", "")

        self._serial = self.get_parameter("serial").value
        sensors_root = self.get_parameter("sensors_root").value
        models_root = self.get_parameter("models_root").value
        device = self.get_parameter("model_device").value
        rate = float(self.get_parameter("rate").value)
        self._liveness_timeout = float(
            self.get_parameter("liveness_timeout").value
        )
        connect_timeout = float(
            self.get_parameter("shm_connect_timeout").value
        )
        affinity = self.get_parameter("cpu_affinity").value

        if not self._serial or not sensors_root or not models_root:
            raise ValueError("serial, sensors_root, and models_root are required")
        if affinity:
            os.sched_setaffinity(0, _parse_affinity(affinity))

        config = load_config(serial=self._serial, sensors_root=sensors_root)
        force = config.get("force") or {}
        if not force.get("enable_force", False):
            raise ValueError(f"force is disabled for {self._serial}")

        encoder_path, decoder_path = _force_model_paths(config, models_root)
        missing = [
            path for path in (encoder_path, decoder_path) if not os.path.isfile(path)
        ]
        if missing:
            raise FileNotFoundError(
                "force model files not found; set models_root to the downloaded "
                f"checkpoint directory: {', '.join(missing)}"
            )

        # Attach first: a disabled/missing camera cannot allocate the force model.
        self._shm_name = f"tactile_{self._serial}"
        self._shm = connect_shm_with_retry(
            self._shm_name, timeout_s=connect_timeout
        )
        self._last_sequence = None
        self._last_fresh = time.monotonic()

        background_path = os.path.join(
            sensors_root,
            self._serial,
            "calibration",
            "background",
            "reference.png",
        )
        background = cv2.imread(background_path, cv2.IMREAD_COLOR)
        if background is None:
            raise FileNotFoundError(
                f"force background not found or unreadable: {background_path}"
            )
        self._estimator = ForceEstimator(
            encoder_path,
            decoder_path,
            temporal_stride=int(force.get("temporal_stride", 5)),
            bg_offset=float(force.get("bg_offset", 0.5)),
            device=device,
            force_vector_scale=force.get("force_vector_scale"),
        )
        self._estimator.load_background(background)

        self._pub_field = self.create_publisher(
            Image, f"tactile/{self._serial}/force_field", _BE_QOS
        )
        self._pub_field_viz = self.create_publisher(
            Image, f"tactile/{self._serial}/force_field_rgb", _BE_QOS
        )
        self._pub_vector = self.create_publisher(
            WrenchStamped, f"tactile/{self._serial}/force_vector", _BE_QOS
        )
        self._timer = self.create_timer(1.0 / rate, self._handle_sensor)
        self.get_logger().info(
            f"Force publisher ready for {self._serial} @ {rate:.0f}Hz"
        )

    def _handle_sensor(self):
        frame = read_camera_frame(self._shm.buf)
        if frame is None or frame.sequence == self._last_sequence:
            if time.monotonic() - self._last_fresh > self._liveness_timeout:
                self._recover_shm()
            return

        self._last_sequence = frame.sequence
        self._last_fresh = time.monotonic()
        result = self._estimator.estimate(frame.image, frame.timestamp_ns / 1e9)
        if result is None:
            return

        field = result["force_field"]
        normal, shear = field["normal"], field["shear"]
        height, width = normal.shape
        header = self.get_clock().now().to_msg()
        frame_id = f"tactile_{self._serial}_depth_frame"

        field_data = np.dstack((shear[:, :, 0], shear[:, :, 1], normal))
        message = Image()
        message.height, message.width = height, width
        message.encoding = "32FC3"
        message.step = width * 12
        message.data = np.ascontiguousarray(field_data, dtype=np.float32).tobytes()
        message.header.frame_id, message.header.stamp = frame_id, header
        self._pub_field.publish(message)

        viz = Image()
        viz.height, viz.width = height, width
        viz.encoding = "rgb8"
        viz.step = width * 3
        viz.data = np.ascontiguousarray(
            force_field_to_rgb(normal, shear)
        ).tobytes()
        viz.header.frame_id, viz.header.stamp = frame_id, header
        self._pub_field_viz.publish(viz)

        vector = result["force_vector_physical"]
        wrench = WrenchStamped()
        wrench.header.frame_id, wrench.header.stamp = frame_id, header
        wrench.wrench.force.x = vector["fx"]
        wrench.wrench.force.y = vector["fy"]
        wrench.wrench.force.z = vector["fz"]
        self._pub_vector.publish(wrench)

    def _recover_shm(self):
        fresh = reopen_shm_if_stale(
            self._shm_name, self._shm, self._liveness_timeout
        )
        if fresh is None:
            try:
                fresh = connect_shm_with_retry(
                    self._shm_name, timeout_s=self._liveness_timeout
                )
            except ShmConnectError as error:
                self.get_logger().error(str(error))
                return
        self._shm = fresh
        self._last_sequence = None
        self._last_fresh = time.monotonic()
        self.get_logger().warning(f"Reattached camera SHM {self._shm_name}")

    def destroy_node(self):
        self._shm.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    try:
        node = ForcePublisher()
    except (FileNotFoundError, RuntimeError, ShmConnectError, ValueError) as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        rclpy.shutdown()
        raise SystemExit(1) from error
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
