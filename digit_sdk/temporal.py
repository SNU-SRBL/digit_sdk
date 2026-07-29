"""Production causal temporal filtering for metric tactile depth."""

from __future__ import annotations

from collections import deque

import numpy as np


class NeuralFeelsFIR:
    """Five-frame exponentially weighted FIR used by the temporal study."""

    window = 5

    def __init__(self):
        self._history: deque[np.ndarray] = deque(maxlen=self.window)

    def reset(self) -> None:
        self._history.clear()

    def __call__(self, depth_mm: np.ndarray) -> np.ndarray:
        current = np.asarray(depth_mm, dtype=np.float32)
        if current.ndim != 2:
            raise ValueError("depth_mm must be a 2D array")
        if self._history and self._history[-1].shape != current.shape:
            self.reset()
        self._history.append(current.copy())

        count = len(self._history)
        weights = np.exp(np.arange(1, count + 1, dtype=np.float64) / count)
        weights /= weights.sum()
        output = np.zeros_like(current)
        for weight, frame in zip(weights, self._history, strict=True):
            output += np.float32(weight) * frame
        return output


class PersistenceCutoff:
    """Keep pixels at or above cutoff in at least two of three frames."""

    minimum_hits = 2
    window = 3

    def __init__(self, cutoff_mm: float):
        cutoff_mm = float(cutoff_mm)
        if not np.isfinite(cutoff_mm) or cutoff_mm <= 0:
            raise ValueError("cutoff_mm must be finite and positive")
        self.cutoff_mm = cutoff_mm
        self._history: deque[np.ndarray] = deque(maxlen=self.window)

    def reset(self) -> None:
        self._history.clear()

    def __call__(self, depth_mm: np.ndarray) -> np.ndarray:
        current = np.asarray(depth_mm, dtype=np.float32)
        if current.ndim != 2:
            raise ValueError("depth_mm must be a 2D array")
        if self._history and self._history[-1].shape != current.shape:
            self.reset()
        self._history.append(current >= self.cutoff_mm)
        hits = np.sum(np.stack(self._history), axis=0)
        return np.where(
            hits >= self.minimum_hits, current, 0.0
        ).astype(np.float32)
