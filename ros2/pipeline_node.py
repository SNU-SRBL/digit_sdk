#!/usr/bin/env python3
"""Batch depth coordinator: camera SHM -> GPU inference -> surface SHM."""

import os
from multiprocessing import shared_memory
import threading
import time
from typing import Dict

import rclpy
import torch
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from digit_sdk.processing_engine import ProcessingEngine
from digit_sdk.shm_protocol import SURFACE_HEADER_SIZE, write_surface_frame


_MAX_HEIGHT = 320
_MAX_WIDTH = 240


def _parse_affinity(value: str):
    cores = set()
    for part in value.split(","):
        part = part.strip()
        if "-" in part:
            low, high = map(int, part.split("-", 1))
            cores.update(range(low, high + 1))
        elif part:
            cores.add(int(part))
    return cores


class PipelineNode(Node):
    """Schedule one shared-encoder batch and publish latest depth through SHM."""

    def __init__(self):
        super().__init__("pipeline_node")
        self.declare_parameter(
            "serials",
            value=[""],
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING_ARRAY
            ),
        )
        self.declare_parameter("sensors_root", "../sensors")
        self.declare_parameter("model_device", "cuda")
        self.declare_parameter("depth_cutoff", 0.1)
        self.declare_parameter("rate", 60.0)
        self.declare_parameter("cpu_affinity", "")

        affinity = self.get_parameter("cpu_affinity").value
        if affinity:
            cores = _parse_affinity(affinity)
            os.sched_setaffinity(0, cores)
            # torch was imported before Node construction and may have sized
            # its pools for every system CPU. Match the production affinity to
            # prevent oversubscribed workers from starving camera processes.
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)

        serials_raw = self.get_parameter("serials").value
        serials = [] if not serials_raw or serials_raw == [""] else list(serials_raw)
        self._engine = ProcessingEngine(
            serials=serials,
            sensors_root=self.get_parameter("sensors_root").value,
            model_device=self.get_parameter("model_device").value,
            depth_cutoff_mm=self.get_parameter("depth_cutoff").value,
        )
        if not self._engine.serials:
            self.get_logger().fatal("No sensors initialized")
            raise RuntimeError("No sensors initialized")

        self._surface_shms: Dict[str, shared_memory.SharedMemory] = {}
        self._surface_generations: Dict[str, int] = {}
        self._last_written_sequences: Dict[str, int] = {}
        for serial in self._engine.serials:
            name = f"tactile_{serial}_surface"
            try:
                stale = shared_memory.SharedMemory(name=name, create=False)
                stale.close()
                stale.unlink()
            except FileNotFoundError:
                pass
            self._surface_shms[serial] = shared_memory.SharedMemory(
                name=name,
                create=True,
                size=SURFACE_HEADER_SIZE + _MAX_HEIGHT * _MAX_WIDTH * 4,
            )
            self._surface_generations[serial] = 0
            self._last_written_sequences[serial] = -1

        rate = float(self.get_parameter("rate").value)
        self._stop = threading.Event()
        self._coordinator_thread = threading.Thread(
            target=self._coordinator,
            daemon=True,
            name="pipeline-coordinator",
        )
        self._coordinator_thread.start()
        self.get_logger().info(
            f"Depth pipeline ready for {len(self._engine.serials)} sensors "
            f"({', '.join(self._engine.serials)}) @ {rate:.0f}Hz "
            "(NeuralFeels FIR window=5)"
        )

    def _coordinator(self):
        rate = float(self.get_parameter("rate").value)
        period = 1.0 / rate
        while True:
            started = time.monotonic()
            try:
                snapshots = self._engine.read_available_frames()
                if snapshots:
                    results = self._engine.process_snapshots(snapshots)
                    for serial, result in results.items():
                        generation = write_surface_frame(
                            self._surface_shms[serial].buf,
                            generation=self._surface_generations[serial],
                            sequence=result.source_sequence,
                            timestamp_ns=result.timestamp_ns,
                            depth=result.depth_mm,
                            pointcloud=None,
                        )
                        self._surface_generations[serial] = generation
                        self._last_written_sequences[serial] = (
                            result.source_sequence
                        )
            except Exception:
                self.get_logger().exception("pipeline cycle failed")
            delay = period - (time.monotonic() - started)
            if delay > 0:
                self._stop.wait(delay)
            if self._stop.is_set():
                break

    def destroy_node(self):
        self._stop.set()
        self._coordinator_thread.join(timeout=2.0)
        self._engine.shutdown()
        for shm in self._surface_shms.values():
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PipelineNode()
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
