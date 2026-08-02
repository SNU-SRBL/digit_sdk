"""Multi-sensor production depth scheduling.

Camera capture remains isolated in one process per sensor.  This engine reads
stable camera snapshots, keeps only each sensor's latest frame, executes one
shared-encoder batch, and exposes the latest metric result per sensor.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from multiprocessing import shared_memory
from pathlib import Path
import threading
import time
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import torch

from .depth import DepthEstimator, DepthInput
from .shm_protocol import (
    CameraFrameSnapshot,
    open_shared_memory,
    read_camera_frame,
)
from .temporal import NeuralFeelsFIR, PersistenceCutoff


logger = logging.getLogger(__name__)
_TEMPORAL_RESET_GAP_NS = 100_000_000


@dataclass(frozen=True)
class DepthFrame:
    """One completed result tied to its source camera frame."""

    source_sequence: int
    timestamp_ns: int
    depth_mm: np.ndarray


@dataclass(frozen=True)
class _PendingDepthInput:
    source_sequence: int
    timestamp_ns: int
    prepared: DepthInput


class ProcessingEngine:
    """Own the production estimator and latest-frame batch scheduler."""

    def __init__(
        self,
        serials: Sequence[str],
        sensors_root: str | Path,
        model_device: str = "cuda",
        depth_cutoff_mm: float = 0.1,
        shm_connect_timeout: float = 10.0,
    ):
        if not serials:
            raise ValueError("at least one sensor serial is required")
        if len(set(serials)) != len(serials):
            raise ValueError("sensor serials must be unique")
        depth_cutoff_mm = float(depth_cutoff_mm)
        if not np.isfinite(depth_cutoff_mm) or depth_cutoff_mm < 0:
            raise ValueError("depth_cutoff_mm must be finite and non-negative")

        self._configured_serials = tuple(serials)
        self._depth_cutoff_mm = depth_cutoff_mm
        self._shms: Dict[str, shared_memory.SharedMemory] = {}
        self._last_read_sequences: Dict[str, int] = {}
        self._connect_shms(shm_connect_timeout)
        self._serials = tuple(
            serial for serial in self._configured_serials if serial in self._shms
        )
        self._estimator = (
            DepthEstimator(self._serials, sensors_root, model_device)
            if self._serials else None
        )

        self._condition = threading.Condition()
        self._running = False
        self._worker: Optional[threading.Thread] = None
        self._pending: Dict[str, _PendingDepthInput] = {}
        self._processed_sequences: Dict[str, int] = {
            serial: -1 for serial in self._serials
        }
        self._latest_results: Dict[str, DepthFrame] = {}
        self._latest_inputs: Dict[str, _PendingDepthInput] = {}
        self._temporal_filters = {
            serial: NeuralFeelsFIR() for serial in self._serials
        }
        self._persistence_filters = {
            serial: PersistenceCutoff(self._depth_cutoff_mm)
            for serial in self._serials
            if self._depth_cutoff_mm > 0
        }
        self._temporal_timestamps_ns: Dict[str, int] = {}

    @property
    def serials(self):
        return list(self._serials)

    def read_frame(self, serial: str) -> Optional[CameraFrameSnapshot]:
        """Return the latest unseen stable camera snapshot."""
        shm = self._shms.get(serial)
        if shm is None:
            return None
        snapshot = read_camera_frame(shm.buf)
        if snapshot is None:
            return None
        if snapshot.sequence == self._last_read_sequences.get(serial, -1):
            return None
        self._last_read_sequences[serial] = snapshot.sequence
        return snapshot

    def read_available_frames(self) -> Dict[str, CameraFrameSnapshot]:
        """Read at most one latest unseen frame from every active sensor."""
        snapshots = {}
        for serial in self._serials:
            snapshot = self.read_frame(serial)
            if snapshot is not None:
                snapshots[serial] = snapshot
        return snapshots

    def process_snapshots(
        self, snapshots: Mapping[str, CameraFrameSnapshot]
    ) -> Dict[str, DepthFrame]:
        unknown = set(snapshots) - set(self._serials)
        if unknown:
            raise KeyError(f"inactive sensor serials: {sorted(unknown)}")
        if not snapshots:
            return {}
        prepared = {
            serial: _PendingDepthInput(
                source_sequence=snapshot.sequence,
                timestamp_ns=snapshot.timestamp_ns,
                prepared=self._estimator.prepare(snapshot.image),
            )
            for serial, snapshot in snapshots.items()
        }
        prepared_batch = {
            serial: item.prepared for serial, item in prepared.items()
        }
        if getattr(self._estimator, "device", None) is not None and (
            self._estimator.device.type == "cuda"
        ):
            raw_depths = self._estimator.estimate_prepared_batch_tensors(
                prepared_batch
            )
        else:
            raw_depths = self._estimator.estimate_prepared_batch(
                prepared_batch
            )
        results: Dict[str, DepthFrame] = {}
        for serial, item in prepared.items():
            if serial not in raw_depths:
                continue
            result = self._make_result(
                serial,
                raw_depths[serial],
                source_sequence=item.source_sequence,
                timestamp_ns=item.timestamp_ns,
            )
            self._processed_sequences[serial] = result.source_sequence
            self._latest_results[serial] = result
            results[serial] = result
        return results

    def submit_frames(
        self, snapshots: Mapping[str, CameraFrameSnapshot]
    ) -> None:
        """Atomically replace pending frames; never queue stale history."""
        unknown = set(snapshots) - set(self._serials)
        if unknown:
            raise KeyError(f"inactive sensor serials: {sorted(unknown)}")
        prepared = {
            serial: _PendingDepthInput(
                source_sequence=snapshot.sequence,
                timestamp_ns=snapshot.timestamp_ns,
                prepared=self._estimator.prepare(snapshot.image),
            )
            for serial, snapshot in snapshots.items()
        }
        with self._condition:
            for serial, item in prepared.items():
                if item.source_sequence > self._processed_sequences[serial]:
                    self._pending[serial] = item
            self._condition.notify()

    def submit_frame(
        self,
        serial: str,
        frame: np.ndarray,
        *,
        source_sequence: Optional[int] = None,
        timestamp_ns: Optional[int] = None,
    ) -> None:
        """Submit one frame while retaining source identity when provided."""
        if source_sequence is None:
            source_sequence = self._processed_sequences.get(serial, -1) + 1
        if timestamp_ns is None:
            timestamp_ns = time.time_ns()
        self.submit_frames({
            serial: CameraFrameSnapshot(source_sequence, timestamp_ns, frame)
        })

    def start_workers(self) -> None:
        """Start the single batch worker."""
        with self._condition:
            if self._running:
                return
            if self._estimator is None:
                raise RuntimeError("no active sensors")
            self._running = True
            self._worker = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name="depth-batch-worker",
            )
            self._worker.start()

    def get_latest_result(self, serial: str) -> Optional[DepthFrame]:
        """Return the newest completed immutable result, if available."""
        with self._condition:
            return self._latest_results.get(serial)

    def process_frames_sync(
        self, frames: Mapping[str, np.ndarray]
    ) -> Dict[str, DepthFrame]:
        """Synchronous inference helper for tests and diagnostics."""
        if self._estimator is None:
            return {}
        timestamp_ns = time.time_ns()
        raw_depths = self._estimator.estimate_batch(frames)
        return {
            serial: self._make_result(
                serial, depth, source_sequence=0, timestamp_ns=timestamp_ns
            )
            for serial, depth in raw_depths.items()
        }

    def shutdown(self) -> None:
        with self._condition:
            self._running = False
            self._condition.notify_all()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None
        for shm in self._shms.values():
            shm.close()
        self._shms.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.shutdown()

    def _connect_shms(self, timeout: float) -> None:
        deadline = time.monotonic() + max(0.0, timeout)
        remaining = set(self._configured_serials)
        while remaining:
            for serial in tuple(remaining):
                try:
                    self._shms[serial] = open_shared_memory(
                        f"tactile_{serial}"
                    )
                    self._last_read_sequences[serial] = -1
                    remaining.remove(serial)
                except FileNotFoundError:
                    pass
            if not remaining or time.monotonic() >= deadline:
                break
            time.sleep(0.05)
        for serial in sorted(remaining):
            logger.warning("camera SHM unavailable; skipping sensor %s", serial)

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: not self._running or self._pending)
                if not self._running:
                    return
                snapshots = self._pending
                self._pending = {}

            inference_batch = self._build_inference_batch(snapshots)
            if inference_batch is None:
                continue

            try:
                raw_depths = self._estimator.estimate_prepared_batch(inference_batch)
            except Exception:
                logger.exception("depth batch inference failed")
                continue

            completed = {
                serial: self._make_result(
                    serial,
                    raw_depths[serial],
                    source_sequence=item.source_sequence,
                    timestamp_ns=item.timestamp_ns,
                )
                for serial, item in snapshots.items()
                if serial in raw_depths
            }
            with self._condition:
                for serial, result in completed.items():
                    if result.source_sequence > self._processed_sequences[serial]:
                        self._processed_sequences[serial] = result.source_sequence
                        self._latest_results[serial] = result

    def _build_inference_batch(
        self,
        snapshots: Dict[str, _PendingDepthInput],
    ) -> Optional[Dict[str, DepthInput]]:
        """Return one fixed-size batch, reusing only internal stale inputs."""
        self._latest_inputs.update(snapshots)
        if any(serial not in self._latest_inputs for serial in self._serials):
            return None
        return {
            serial: self._latest_inputs[serial].prepared
            for serial in self._serials
        }

    def _make_result(
        self,
        serial: str,
        depth_raw_mm: np.ndarray,
        *,
        source_sequence: int,
        timestamp_ns: int,
    ) -> DepthFrame:
        previous_timestamp_ns = self._temporal_timestamps_ns.get(serial)
        if (
            previous_timestamp_ns is not None
            and (
                timestamp_ns <= previous_timestamp_ns
                or timestamp_ns - previous_timestamp_ns
                > _TEMPORAL_RESET_GAP_NS
            )
        ):
            self._temporal_filters[serial].reset()
            persistence = self._persistence_filters.get(serial)
            if persistence is not None:
                persistence.reset()
        self._temporal_timestamps_ns[serial] = timestamp_ns
        depth_filtered_mm = self._temporal_filters[serial](depth_raw_mm)
        persistence = self._persistence_filters.get(serial)
        if persistence is not None:
            depth_filtered_mm = persistence(depth_filtered_mm)
        if isinstance(depth_filtered_mm, torch.Tensor):
            depth_filtered_mm = depth_filtered_mm.detach().cpu().numpy()
        return DepthFrame(
            source_sequence=source_sequence,
            timestamp_ns=timestamp_ns,
            depth_mm=depth_filtered_mm,
        )

    def __repr__(self) -> str:
        return (
            f"ProcessingEngine(serials={list(self._serials)!r}, "
            f"depth_cutoff_mm={self._depth_cutoff_mm}, "
            "temporal=neuralfeels_fir_5+persistence_2_of_3)"
        )
