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

"""Unit tests for the fake SHM producer used by publisher tests."""

import threading
import time

import pytest

from digit_sdk.shm_protocol import read_camera_frame, read_surface_frame
from fake_shm_producer import FakeShmProducer


@pytest.mark.timeout(30)
def test_fake_producer_writes_camera_and_surface_at_60hz():
    producer = FakeShmProducer(["D1"], height=8, width=10)
    producer.start()
    thread = threading.Thread(target=producer.run, args=(3.6,))
    thread.start()
    try:
        time.sleep(0.3)
        start = time.monotonic()
        frames = 0
        last = -1
        last_camera = None
        last_surface = None
        while time.monotonic() - start < 3.0:
            camera = read_camera_frame(producer._shms["D1"]["camera"].buf)
            surface = read_surface_frame(
                producer._shms["D1"]["surface"].buf
            )
            if camera is not None and surface is not None:
                if camera.sequence != last:
                    last = camera.sequence
                    frames += 1
                last_camera = camera
                last_surface = surface
            time.sleep(0.0005)
        assert last_camera is not None, "no camera frames observed"
        assert 174 <= frames <= 186, (
            f"{frames} frames in 3.0s is outside 58-62 Hz"
        )
        assert last_surface.sequence == last_camera.sequence
        assert last_surface.depth.shape == (8, 10)
    finally:
        producer.stop()
        thread.join(timeout=10)


@pytest.mark.timeout(30)
def test_fake_producer_restart_recreates_segments():
    producer = FakeShmProducer(["D2"], height=4, width=5)
    producer.start()
    try:
        producer.run(0.02)
        before = producer._shms["D2"]["camera"].name
        producer.restart()
        after = producer._shms["D2"]["camera"].name
        assert before == after
        producer.run(0.02)
        camera = read_camera_frame(producer._shms["D2"]["camera"].buf)
        surface = read_surface_frame(producer._shms["D2"]["surface"].buf)
        assert camera is not None
        assert surface is not None
        assert camera.sequence == surface.sequence
    finally:
        producer.stop()
