#!/usr/bin/env python3
"""
Camera process: reads DIGIT frames, writes them to shared memory.

Plain Python (no rclpy, no DDS). Writes BGR→RGB frames plus metadata
to a SharedMemory block named 'tactile_{serial}'.

Layout (header + dynamic payload):
  Offset   Size  Field        Type     Description
  ------   ----  -----        ----     -----------
      0      8   generation   uint64   Odd=writing, even=stable, 0=empty
      8      8   seq          uint64   Monotonic frame counter
     16      8   timestamp_ns uint64   time.time_ns() at capture
     24      4   height       uint32   Frame height
     28      4   width        uint32   Frame width
     32    H*W*3  data         uint8[]  BGR frame (height × width × 3)

Synchronization (lock-free, single writer / multiple readers):
  1. Publish an odd generation.
  2. Write metadata and data.
  3. Publish the next even generation.
Readers accept a copied frame only when generation is unchanged and even.
"""

import argparse
import os
import signal
import sys
import time
from multiprocessing import shared_memory

from digit_sdk.camera import Camera
from digit_sdk.shm_protocol import CAMERA_HEADER_SIZE, write_camera_frame

SHM_HEADER = CAMERA_HEADER_SIZE


def _parse_affinity(spec: str):
    """Parse affinity string like '0-3' or '0,2,4' into set of ints."""
    cores = set()
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            lo, hi = part.split('-', 1)
            cores.update(range(int(lo), int(hi) + 1))
        else:
            cores.add(int(part))
    return cores


def shm_size_for(camera: Camera) -> int:
    """Compute SHM block size from camera resolution."""
    return SHM_HEADER + camera.raw_imgh * camera.raw_imgw * 3


def run(serial: str, sensors_root: str, verbose: bool = False):
    """Run the main capture loop; writes frames to shared memory."""
    shm_name = f"tactile_{serial}"

    # Create camera first — SHM size depends on resolution
    camera = Camera(serial=serial, sensors_root=sensors_root)
    camera.connect(verbose=verbose)

    # Clean up stale shm if any
    try:
        stale = shared_memory.SharedMemory(name=shm_name, create=False)
        stale.close()
        stale.unlink()
        if verbose:
            print(f"Unlinked stale shared memory '{shm_name}'")
    except FileNotFoundError:
        pass

    shm = shared_memory.SharedMemory(
        name=shm_name, create=True, size=shm_size_for(camera))

    running = True

    def _cleanup(*_args):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, _cleanup)
    signal.signal(signal.SIGINT, _cleanup)

    seq = 0
    generation = 0
    buf = shm.buf

    if verbose:
        print(f"Camera {serial}: device={camera.device} "
              f"id={camera.dev_id}", flush=True)

    TARGET_DT = 1.0 / camera.framerate

    while running:
        t0 = time.monotonic()
        frame = camera.get_image()
        if frame is None:
            # Recovery or corruption — wait a bit before retry
            time.sleep(0.005)
            continue

        # Store BGR directly (raw_bridge needs BGR, processing_engine expects BGR)
        bgr = frame
        h, w = bgr.shape[:2]
        # ROS headers require Unix-epoch time. Monotonic time is only suitable
        # for local interval measurement and must never be published as a stamp.
        ts_ns = time.time_ns()

        generation = write_camera_frame(
            buf,
            generation=generation,
            sequence=seq,
            timestamp_ns=ts_ns,
            image=bgr,
        )
        seq += 1

        # Pace at 60 Hz to avoid USB bus starvation
        elapsed = time.monotonic() - t0
        if elapsed < TARGET_DT:
            time.sleep(TARGET_DT - elapsed)

    # Clean up
    camera.release()
    shm.close()
    shm.unlink()
    if verbose:
        print(f"Camera {serial}: shut down", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="DIGIT camera → shared memory writer")
    parser.add_argument('--serial', required=True, help='Sensor serial')
    parser.add_argument('--sensors-root', required=True,
                        help='Path to sensors config directory')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('--cpu-affinity', type=str, default='',
                        help='CPU core affinity (e.g. "0-3" or "0,2,4")')
    args = parser.parse_args()

    if args.cpu_affinity:
        cores = _parse_affinity(args.cpu_affinity)
        if cores:
            os.sched_setaffinity(0, cores)

    try:
        run(serial=args.serial, sensors_root=args.sensors_root,
            verbose=args.verbose)
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr, flush=True)
        # Clean up shm on fatal error
        try:
            tmp = shared_memory.SharedMemory(
                name=f"tactile_{args.serial}", create=False)
            tmp.close()
            tmp.unlink()
        except Exception:
            pass
        sys.exit(1)


if __name__ == '__main__':
    main()
