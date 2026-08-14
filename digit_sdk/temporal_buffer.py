"""
Temporal Buffer Utility for Tactile Force Estimation.

Manages a circular buffer of timestamped frames to support temporal-based
force estimation models that require pairs of frames separated by a stride.
"""

import time
from collections import deque
from typing import Optional, Tuple
import numpy as np

# Default buffer configuration
DEFAULT_MAX_BUFFER_SIZE = 60  # At 60 FPS with stride=5, provides ~1 second of history
DEFAULT_TEMPORAL_STRIDE = 5  # At 60 FPS, stride=5 ≈ 83ms temporal window


class TemporalBuffer:
    """
    Circular buffer for managing temporal frame pairs.

    Stores frames with timestamps in a circular buffer and provides access
    to frame pairs separated by a configurable temporal stride. Used for
    force estimation models that require temporal context.

    The buffer maintains a fixed-size deque and automatically handles:
    - Frame storage with timestamps
    - Circular buffer behavior (old frames dropped when full)
    - Temporal pair retrieval with configurable stride
    - Warmup period (returns None until sufficient frames available)

    Args:
    ----
        max_size: Maximum number of frames to store
        stride: Number of frames between temporal pairs

    Example:
    -------
        >>> buffer = TemporalBuffer(max_size=60, stride=5)
        >>>
        >>> # Add frames during warmup
        >>> for i in range(4):
        ...     buffer.add(frame)
        ...     pair = buffer.get_pair()  # Returns None (not ready)
        >>>
        >>> # After warmup (≥ stride frames)
        >>> buffer.add(frame)
        >>> frame_t, frame_t_minus_5 = buffer.get_pair()  # Returns valid pair

    """

    def __init__(
        self,
        max_size: int = DEFAULT_MAX_BUFFER_SIZE,
        stride: int = DEFAULT_TEMPORAL_STRIDE,
    ):
        """
        Initialize temporal buffer.

        Args:
        ----
            max_size: Maximum buffer size
            stride: Temporal stride for frame pairs

        """
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        if stride >= max_size:
            raise ValueError(
                f"stride ({stride}) must be < max_size ({max_size})"
            )

        self._max_size = max_size
        self._stride = stride

        # Circular buffer: stores (timestamp, frame) tuples
        self._buffer = deque(maxlen=max_size)

        # Statistics
        self._frames_added = 0
        self._last_timestamp = None

    def add(self, frame: np.ndarray, timestamp: Optional[float] = None) -> None:
        """
        Add a frame to the buffer.

        Args:
        ----
            frame: Image frame to add (typically BGR uint8, any shape)
            timestamp: Optional timestamp in seconds (uses time.time() if None)

        Note:
        ----
            When buffer is full, oldest frame is automatically dropped.

        """
        if timestamp is None:
            timestamp = time.time()

        # Store frame with timestamp
        self._buffer.append((timestamp, frame.copy()))

        self._frames_added += 1
        self._last_timestamp = timestamp

    def get_pair(
        self,
        stride: Optional[int] = None
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get a temporal pair of frames separated by stride.

        Returns the most recent frame (frame_t) and a frame from `stride`
        steps ago (frame_t_minus_stride). Returns None if insufficient
        frames are available.

        Args:
            stride: Temporal stride (uses constructor stride if None)

        Returns
        -------
        Tuple of (frame_t, frame_t_minus_stride) if available, else None
        - frame_t: Most recent frame
        - frame_t_minus_stride: Frame from `stride` indices back

        Example:
            >>> buffer.add(frame_0)
            >>> buffer.add(frame_1)
            >>> # ... (add frames 2, 3, 4)
            >>> buffer.add(frame_5)
            >>> frame_t, frame_t_5 = buffer.get_pair(stride=5)
        >>> # Returns (frame_5, frame_0)

        """
        if stride is None:
            stride = self._stride

        # Check if we have enough frames
        buffer_len = len(self._buffer)
        if buffer_len <= stride:
            # Not enough frames yet (need at least stride + 1)
            return None

        # Get current frame (most recent)
        _, frame_t = self._buffer[-1]

        # Get frame from stride steps ago
        # Index calculation: -1 is current, -(stride+1) is stride back
        _, frame_t_minus_stride = self._buffer[-(stride + 1)]

        return (frame_t, frame_t_minus_stride)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Get the most recent frame without temporal pair.

        Returns
        -------
        Most recent frame if buffer not empty, else None

        """
        if len(self._buffer) == 0:
            return None

        _, frame = self._buffer[-1]
        return frame

    def is_ready(self, stride: Optional[int] = None) -> bool:
        """
        Check if buffer has enough frames for temporal pairs.

        Args:
            stride: Temporal stride to check (uses constructor stride if None)

        Returns
        -------
        True if buffer can provide temporal pairs, False otherwise

        """
        if stride is None:
            stride = self._stride

        return len(self._buffer) > stride

    def clear(self) -> None:
        """
        Clear all frames from buffer.

        Note:
        ----
            Resets buffer but preserves max_size and stride settings.
            Does not reset statistics (_frames_added counter).

        """
        self._buffer.clear()
        self._last_timestamp = None

    def get_timestamps(self) -> list:
        """
        Get timestamps of all frames in buffer.

        Returns
        -------
        List of timestamps in chronological order (oldest to newest)

        """
        return [timestamp for timestamp, _ in self._buffer]

    def get_frame_rate(self) -> Optional[float]:
        """
        Estimate average frame rate from recent frames.

        Uses timestamps of all frames in buffer to estimate frame rate.

        Returns
        -------
        Estimated frame rate in Hz, or None if insufficient data

        """
        if len(self._buffer) < 2:
            return None

        timestamps = self.get_timestamps()
        time_span = timestamps[-1] - timestamps[0]

        if time_span <= 0:
            return None

        # Frame rate = (num_frames - 1) / time_span
        frame_rate = (len(timestamps) - 1) / time_span
        return frame_rate

    def __len__(self) -> int:
        """Get current number of frames in buffer."""
        return len(self._buffer)

    def __repr__(self) -> str:
        """Return a string representation of buffer state."""
        ready = "ready" if self.is_ready() else "warmup"
        return (
            f"TemporalBuffer(size={len(self._buffer)}/{self._max_size}, "
            f"stride={self._stride}, status={ready}, "
            f"frames_added={self._frames_added})"
        )

    @property
    def max_size(self) -> int:
        """Maximum buffer size."""
        return self._max_size

    @property
    def stride(self) -> int:
        """Temporal stride for frame pairs."""
        return self._stride

    @property
    def frames_added(self) -> int:
        """Total number of frames added since initialization."""
        return self._frames_added

    @property
    def last_timestamp(self) -> Optional[float]:
        """Timestamp of most recent frame, or None if empty."""
        return self._last_timestamp
