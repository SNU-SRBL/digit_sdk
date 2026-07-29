import struct
import multiprocessing as mp
from multiprocessing import shared_memory
import time

import numpy as np

from digit_sdk.shm_protocol import (
    CAMERA_HEADER_SIZE,
    SURFACE_HEADER_SIZE,
    _read_consistent,
    read_camera_frame,
    read_surface_frame,
    write_camera_frame,
    write_surface_frame,
)


def _write_camera_frames(name, shape, frame_count):
    shm = shared_memory.SharedMemory(name=name, create=False)
    generation = 0
    word_count = int(np.prod(shape)) // 8
    try:
        for sequence in range(1, frame_count + 1):
            words = np.full(word_count, sequence, dtype=np.uint64)
            image = words.view(np.uint8).reshape(shape)
            generation = write_camera_frame(
                shm.buf,
                generation=generation,
                sequence=sequence,
                timestamp_ns=sequence,
                image=image,
            )
            time.sleep(0.00005)
    finally:
        shm.close()


def test_consistent_read_retries_when_generation_changes():
    buf = bytearray(32)
    struct.pack_into("<Q", buf, 0, 2)
    calls = 0

    def snapshot():
        nonlocal calls
        calls += 1
        if calls == 1:
            struct.pack_into("<Q", buf, 0, 4)
            return "torn"
        return "stable"

    assert _read_consistent(buf, snapshot, retries=1, retry_delay_s=0) == "stable"
    assert calls == 2


def test_consistent_read_rejects_writer_in_progress():
    buf = bytearray(32)
    struct.pack_into("<Q", buf, 0, 1)

    assert _read_consistent(
        buf, lambda: "must not run", retries=0, retry_delay_s=0
    ) is None


def test_camera_frame_round_trip_uses_owned_copy():
    image = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    buf = bytearray(CAMERA_HEADER_SIZE + image.nbytes)

    generation = write_camera_frame(
        buf, generation=0, sequence=7, timestamp_ns=1234, image=image
    )
    snapshot = read_camera_frame(buf, retries=0)

    assert generation == 2
    assert snapshot is not None
    assert snapshot.sequence == 7
    assert snapshot.timestamp_ns == 1234
    np.testing.assert_array_equal(snapshot.image, image)

    image.fill(0)
    buf[CAMERA_HEADER_SIZE:] = b"\xff" * image.nbytes
    assert not np.all(snapshot.image == 0)
    assert not np.all(snapshot.image == 255)


def test_surface_frame_round_trip():
    depth = (
        np.arange(6 * 8, dtype=np.float32).reshape(6, 8) / np.float32(10)
    )
    pointcloud = np.arange(5 * 3, dtype=np.float32).reshape(5, 3)
    buf = bytearray(
        SURFACE_HEADER_SIZE + depth.nbytes + pointcloud.nbytes
    )

    generation = write_surface_frame(
        buf,
        generation=0,
        sequence=9,
        timestamp_ns=5678,
        depth=depth,
        pointcloud=pointcloud,
    )
    snapshot = read_surface_frame(buf, retries=0)

    assert generation == 2
    assert snapshot is not None
    assert snapshot.sequence == 9
    assert snapshot.timestamp_ns == 5678
    np.testing.assert_array_equal(snapshot.depth, depth)
    np.testing.assert_array_equal(snapshot.pointcloud, pointcloud)


def test_camera_protocol_never_accepts_torn_multiprocess_frame():
    shape = (48, 64, 3)
    frame_count = 500
    shm = shared_memory.SharedMemory(
        create=True, size=CAMERA_HEADER_SIZE + int(np.prod(shape))
    )
    shm.buf[:] = b"\x00" * len(shm.buf)
    process = mp.get_context("fork").Process(
        target=_write_camera_frames, args=(shm.name, shape, frame_count)
    )
    accepted = 0
    last_sequence = 0
    deadline = time.monotonic() + 5
    try:
        process.start()
        while time.monotonic() < deadline:
            snapshot = read_camera_frame(shm.buf)
            if snapshot is not None and snapshot.sequence != last_sequence:
                words = snapshot.image.reshape(-1).view(np.uint64)
                assert np.all(words == snapshot.sequence)
                last_sequence = snapshot.sequence
                accepted += 1
            if not process.is_alive() and last_sequence == frame_count:
                break
        process.join(timeout=1)
        assert process.exitcode == 0
        assert last_sequence == frame_count
        assert accepted > 10
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        shm.close()
        shm.unlink()
