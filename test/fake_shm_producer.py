# Copyright (c) 2026 ByungHyun Song
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
Test-only producer that writes camera and surface SHM frames at 60 Hz.

Usage:
  python test/fake_shm_producer.py --serials D1,D2 \
      --height 24 --width 32 --duration 5.0 [--restart-at 2.0]

The optional restart re-unlinks and recreates every owned segment mid-run,
mimicking a producer process restart for recovery experiments.
"""

import argparse
from multiprocessing import shared_memory
import signal
import time

import numpy as np

from digit_sdk.shm_protocol import (
    CAMERA_HEADER_SIZE,
    SURFACE_HEADER_SIZE,
    write_camera_frame,
    write_surface_frame,
)


class FakeShmProducer:
    def __init__(self, serials, height, width, rate=60.0):
        self.serials = serials
        self.height = height
        self.width = width
        self.rate = rate
        self._shms = {}
        self._generations = {}
        self._running = False

    def _camera_size(self):
        return CAMERA_HEADER_SIZE + self.height * self.width * 3

    def _surface_size(self):
        return SURFACE_HEADER_SIZE + self.height * self.width * 4

    def start(self):
        for serial in self.serials:
            self._shms[serial] = {
                "camera": shared_memory.SharedMemory(
                    name=f"tactile_{serial}", create=True,
                    size=self._camera_size(),
                ),
                "surface": shared_memory.SharedMemory(
                    name=f"tactile_{serial}_surface", create=True,
                    size=self._surface_size(),
                ),
            }
            self._generations[serial] = {
                "camera": 0,
                "surface": 0,
            }
        self._running = True

    def restart(self):
        for serial in self.serials:
            for kind, shm in self._shms[serial].items():
                shm.close()
                shm.unlink()
            self._shms[serial] = {
                "camera": shared_memory.SharedMemory(
                    name=f"tactile_{serial}", create=True,
                    size=self._camera_size(),
                ),
                "surface": shared_memory.SharedMemory(
                    name=f"tactile_{serial}_surface", create=True,
                    size=self._surface_size(),
                ),
            }
            self._generations[serial] = {
                "camera": 0,
                "surface": 0,
            }

    def stop(self):
        self._running = False
        for serial in self.serials:
            for shm in self._shms[serial].values():
                shm.close()
                shm.unlink()
        self._shms.clear()

    def run(self, duration, restart_at=None):
        start = time.monotonic()
        duration_start = start
        iteration = 0
        sequence = 0
        target_dt = 1.0 / self.rate
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        depth = np.zeros((self.height, self.width), dtype=np.float32)
        restarted = restart_at is None
        while self._running and time.monotonic() - duration_start < duration:
            if restart_at is not None and not restarted:
                if time.monotonic() - duration_start >= restart_at:
                    self.restart()
                    sequence = 0
                    start = time.monotonic()
                    iteration = 0
                    restarted = True
            timestamp_ns = time.time_ns()
            image.fill(sequence % 256)
            depth.fill(float(sequence % 100))
            for serial in self.serials:
                generation = write_camera_frame(
                    self._shms[serial]["camera"].buf,
                    generation=self._generations[serial]["camera"],
                    sequence=sequence,
                    timestamp_ns=timestamp_ns,
                    image=image,
                )
                self._generations[serial]["camera"] = generation
                generation = write_surface_frame(
                    self._shms[serial]["surface"].buf,
                    generation=self._generations[serial]["surface"],
                    sequence=sequence,
                    timestamp_ns=timestamp_ns,
                    depth=depth,
                    pointcloud=None,
                )
                self._generations[serial]["surface"] = generation
            sequence += 1
            iteration += 1
            next_t = start + iteration * target_dt
            time.sleep(max(0.0, next_t - time.monotonic()))


def _stop(signum, frame):
    producer._running = False


def main():
    parser = argparse.ArgumentParser(
        description="Write fake 60 Hz camera/surface SHM frames"
    )
    parser.add_argument("--serials", required=True)
    parser.add_argument("--height", type=int, default=24)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--restart-at", type=float, default=None)
    args = parser.parse_args()

    global producer
    producer = FakeShmProducer(
        [s.strip() for s in args.serials.split(",") if s.strip()],
        args.height,
        args.width,
    )
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    producer.start()
    try:
        producer.run(args.duration, restart_at=args.restart_at)
    finally:
        producer.stop()


if __name__ == "__main__":
    main()
