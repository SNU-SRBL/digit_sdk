"""Single-writer SHM layouts with bounded seqlock readers.

Generation zero means no committed payload. Odd generations mean a writer is
active; even generations identify stable commits. Readers always return owned
NumPy arrays, never live shared-memory views.
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import resource_tracker, shared_memory
import struct
import time
from typing import Callable, Optional, TypeVar

import numpy as np


CAMERA_HEADER_SIZE = 32
SURFACE_HEADER_SIZE = 40
DEFAULT_READ_RETRIES = 2
DEFAULT_RETRY_DELAY_S = 0.0001

_U64 = struct.Struct("<Q")
_CAMERA_META = struct.Struct("<QQII")
_SURFACE_META = struct.Struct("<QQIII")
_T = TypeVar("_T")


def open_shared_memory(name: str) -> shared_memory.SharedMemory:
    """Attach without claiming cleanup ownership on Python 3.10.

    Writer processes own unlinking.  Without unregistering, an attached reader
    can unlink the writer's live segment when the reader exits.
    """
    shm = shared_memory.SharedMemory(name=name, create=False)
    resource_tracker.unregister(shm._name, "shared_memory")
    return shm


@dataclass(frozen=True)
class CameraFrameSnapshot:
    sequence: int
    timestamp_ns: int
    image: np.ndarray


@dataclass(frozen=True)
class SurfaceFrameSnapshot:
    """Stable surface payload; depth is float32 millimetres."""

    sequence: int
    timestamp_ns: int
    depth: np.ndarray
    pointcloud: np.ndarray


def _load_generation(buf) -> int:
    return _U64.unpack_from(buf, 0)[0]


def _store_generation(buf, generation: int) -> None:
    _U64.pack_into(buf, 0, generation)


def _read_consistent(
    buf,
    snapshot: Callable[[], _T],
    *,
    retries: int = DEFAULT_READ_RETRIES,
    retry_delay_s: float = DEFAULT_RETRY_DELAY_S,
) -> Optional[_T]:
    """Return a stable snapshot or None after bounded contention retries."""
    for attempt in range(retries + 1):
        generation_before = _load_generation(buf)
        if generation_before != 0 and not generation_before & 1:
            try:
                value = snapshot()
            except (BufferError, ValueError, struct.error):
                value = None
            generation_after = _load_generation(buf)
            if (
                value is not None
                and generation_before == generation_after
                and not generation_after & 1
            ):
                return value
        if attempt < retries and retry_delay_s > 0:
            time.sleep(retry_delay_s)
    return None


def _begin_write(buf, generation: int) -> int:
    if generation < 0 or generation & 1:
        raise ValueError("committed generation must be a non-negative even integer")
    odd_generation = generation + 1
    _store_generation(buf, odd_generation)
    return odd_generation


def write_camera_frame(
    buf,
    *,
    generation: int,
    sequence: int,
    timestamp_ns: int,
    image: np.ndarray,
) -> int:
    image = np.ascontiguousarray(image, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("camera image must have shape (height, width, 3)")
    height, width = image.shape[:2]
    end = CAMERA_HEADER_SIZE + image.nbytes
    if end > len(buf):
        raise ValueError("camera frame exceeds shared-memory capacity")

    odd_generation = _begin_write(buf, generation)
    _CAMERA_META.pack_into(buf, 8, sequence, timestamp_ns, height, width)
    buf[CAMERA_HEADER_SIZE:end] = image.tobytes()
    committed_generation = odd_generation + 1
    _store_generation(buf, committed_generation)
    return committed_generation


def read_camera_frame(
    buf,
    *,
    retries: int = DEFAULT_READ_RETRIES,
    retry_delay_s: float = DEFAULT_RETRY_DELAY_S,
) -> Optional[CameraFrameSnapshot]:
    def snapshot() -> Optional[CameraFrameSnapshot]:
        sequence, timestamp_ns, height, width = _CAMERA_META.unpack_from(buf, 8)
        if height == 0 or width == 0:
            return None
        size = height * width * 3
        end = CAMERA_HEADER_SIZE + size
        if end > len(buf):
            return None
        image = np.frombuffer(
            buf[CAMERA_HEADER_SIZE:end], dtype=np.uint8
        ).copy().reshape(height, width, 3)
        return CameraFrameSnapshot(sequence, timestamp_ns, image)

    return _read_consistent(
        buf, snapshot, retries=retries, retry_delay_s=retry_delay_s
    )


def write_surface_frame(
    buf,
    *,
    generation: int,
    sequence: int,
    timestamp_ns: int,
    depth: np.ndarray,
    pointcloud: Optional[np.ndarray],
) -> int:
    """Commit float32 millimetre depth and an optional metric point cloud."""
    depth = np.ascontiguousarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("surface depth must have shape (height, width)")
    if pointcloud is None:
        pointcloud = np.empty((0, 3), dtype=np.float32)
    else:
        pointcloud = np.ascontiguousarray(pointcloud, dtype=np.float32)
        if pointcloud.ndim != 2 or pointcloud.shape[1] != 3:
            raise ValueError("pointcloud must have shape (count, 3)")

    height, width = depth.shape
    point_count = pointcloud.shape[0]
    depth_end = SURFACE_HEADER_SIZE + depth.nbytes
    pointcloud_end = depth_end + pointcloud.nbytes
    if pointcloud_end > len(buf):
        raise ValueError("surface frame exceeds shared-memory capacity")

    odd_generation = _begin_write(buf, generation)
    _SURFACE_META.pack_into(
        buf, 8, sequence, timestamp_ns, height, width, point_count
    )
    buf[SURFACE_HEADER_SIZE:depth_end] = depth.tobytes()
    if point_count:
        buf[depth_end:pointcloud_end] = pointcloud.tobytes()
    committed_generation = odd_generation + 1
    _store_generation(buf, committed_generation)
    return committed_generation


def read_surface_frame(
    buf,
    *,
    retries: int = DEFAULT_READ_RETRIES,
    retry_delay_s: float = DEFAULT_RETRY_DELAY_S,
) -> Optional[SurfaceFrameSnapshot]:
    def snapshot() -> Optional[SurfaceFrameSnapshot]:
        sequence, timestamp_ns, height, width, point_count = (
            _SURFACE_META.unpack_from(buf, 8)
        )
        if height == 0 or width == 0:
            return None
        depth_size = height * width * 4
        depth_end = SURFACE_HEADER_SIZE + depth_size
        pointcloud_end = depth_end + point_count * 12
        if pointcloud_end > len(buf):
            return None
        depth = np.frombuffer(
            buf[SURFACE_HEADER_SIZE:depth_end], dtype=np.float32
        ).copy().reshape(height, width)
        pointcloud = np.frombuffer(
            buf[depth_end:pointcloud_end], dtype=np.float32
        ).copy().reshape(point_count, 3)
        return SurfaceFrameSnapshot(sequence, timestamp_ns, depth, pointcloud)

    return _read_consistent(
        buf, snapshot, retries=retries, retry_delay_s=retry_delay_s
    )
