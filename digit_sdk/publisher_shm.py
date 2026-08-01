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
Shared-memory connection and restart recovery helpers for publishers.

The producer processes unlink and recreate their SHM segments on restart.
Readers therefore cannot trust a once-attached mapping: after a liveness miss
they must reattach the same name and reset per-process frame state.
"""

from __future__ import annotations

from multiprocessing import shared_memory
import time

from digit_sdk.shm_protocol import open_shared_memory


class ShmConnectError(Exception):
    """Raised when a shared-memory name cannot be attached within a bound."""


def _attach(name: str) -> shared_memory.SharedMemory:
    shm = open_shared_memory(name)
    return shm


def open_shared_memory_safely(name: str):
    """Attach to ``name`` without owning cleanup; return None when absent."""
    try:
        return _attach(name)
    except FileNotFoundError:
        return None


def connect_shm_with_retry(
    name: str, timeout_s: float = 10.0, poll_s: float = 0.1
) -> shared_memory.SharedMemory:
    """Attach to ``name``, retrying until the producer creates it."""
    deadline = time.monotonic() + max(0.0, timeout_s)
    while True:
        shm = open_shared_memory_safely(name)
        if shm is not None:
            return shm
        if time.monotonic() >= deadline:
            raise ShmConnectError(
                f"shared memory '{name}' not available within "
                f"{timeout_s:.1f}s"
            )
        time.sleep(poll_s)


def reopen_shm_if_stale(
    name: str,
    current: shared_memory.SharedMemory,
    liveness_timeout_s: float = 2.0,
):
    """
    Reattach ``name`` after a liveness miss.

    Callers invoke this after observing no committed payload for
    ``liveness_timeout_s``.  Producers unlink and recreate the segment on
    restart, so a fresh open of the same name is the recovery signal.  Returns
    a fresh mapping when the name exists; returns None when the name is
    absent, in which case the caller should bounded-reconnect itself.
    """
    fresh = open_shared_memory_safely(name)
    if fresh is None:
        return None
    current.close()
    return fresh
