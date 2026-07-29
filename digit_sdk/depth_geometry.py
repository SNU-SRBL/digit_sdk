"""Pure geometry derived from metric depth."""

import numpy as np


def depth_to_pointcloud(
    depth_mm: np.ndarray, ppmm: float, point_sample_mm: float = 0.0
) -> np.ndarray:
    """Convert indentation-positive depth to the existing optical-frame cloud."""
    if ppmm <= 0:
        raise ValueError("ppmm must be positive")
    stride = max(1, int(point_sample_mm * ppmm)) if point_sample_mm > 0 else 1
    height, width = depth_mm.shape
    ys = np.arange(0, height, stride, dtype=np.float32) - height / 2 + 0.5
    xs = np.arange(0, width, stride, dtype=np.float32) - width / 2 + 0.5
    x_grid, y_grid = np.meshgrid(xs, ys)
    depth_m = depth_mm[::stride, ::stride].astype(np.float32) / 1000.0
    return np.stack(
        (x_grid / ppmm / 1000.0, y_grid / ppmm / 1000.0, -depth_m),
        axis=-1,
    ).reshape(-1, 3)
