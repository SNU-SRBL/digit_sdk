"""Metric depth result contract for depth-estimation backends."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class DepthResult:
    """Raw and cutoff-processed indentation depth in millimetres."""

    depth_raw_mm: np.ndarray
    mask: np.ndarray
    depth_output_mm: np.ndarray


def validate_depth_raw_mm(
    depth_raw_mm: np.ndarray,
    *,
    expected_shape: Optional[Tuple[int, int]] = None,
) -> None:
    """Validate an unthresholded, indentation-positive depth array."""
    if not isinstance(depth_raw_mm, np.ndarray):
        raise TypeError("depth_raw_mm must be a numpy.ndarray")
    if depth_raw_mm.dtype != np.float32:
        raise ValueError("depth_raw_mm must have dtype float32")
    if depth_raw_mm.ndim != 2 or 0 in depth_raw_mm.shape:
        raise ValueError("depth_raw_mm must be a non-empty HxW array")
    if expected_shape is not None and depth_raw_mm.shape != expected_shape:
        raise ValueError(
            f"depth_raw_mm shape {depth_raw_mm.shape} does not match "
            f"expected shape {expected_shape}"
        )
    if not np.isfinite(depth_raw_mm).all():
        raise ValueError("depth_raw_mm must contain only finite values")
    if np.any(depth_raw_mm < 0):
        raise ValueError("depth_raw_mm must be non-negative")


def make_depth_result(
    depth_raw_mm: np.ndarray,
    *,
    depth_cutoff_mm: Real = 0.1,
    expected_shape: Optional[Tuple[int, int]] = None,
) -> DepthResult:
    """Copy raw depth and apply cutoff without modifying the raw values."""
    validate_depth_raw_mm(depth_raw_mm, expected_shape=expected_shape)
    if isinstance(depth_cutoff_mm, bool) or not isinstance(depth_cutoff_mm, Real):
        raise TypeError("depth_cutoff_mm must be a real number")
    depth_cutoff_mm = float(depth_cutoff_mm)
    if not np.isfinite(depth_cutoff_mm) or depth_cutoff_mm < 0:
        raise ValueError("depth_cutoff_mm must be finite and non-negative")

    raw = np.array(depth_raw_mm, dtype=np.float32, order="C", copy=True)
    if depth_cutoff_mm == 0:
        mask = np.ones(raw.shape, dtype=bool)
    else:
        mask = raw >= depth_cutoff_mm
    output = np.where(mask, raw, np.float32(0)).astype(np.float32, copy=False)
    return DepthResult(raw, mask, output)


def validate_depth_result(
    result: DepthResult,
    *,
    depth_cutoff_mm: Real,
    expected_shape: Optional[Tuple[int, int]] = None,
) -> None:
    """Validate a complete result against the cutoff relationship."""
    if not isinstance(result, DepthResult):
        raise TypeError("result must be a DepthResult")
    validate_depth_raw_mm(result.depth_raw_mm, expected_shape=expected_shape)
    expected = make_depth_result(
        result.depth_raw_mm,
        depth_cutoff_mm=depth_cutoff_mm,
        expected_shape=expected_shape,
    )
    if result.mask.dtype != np.bool_ or result.mask.shape != expected.mask.shape:
        raise ValueError("mask must be a bool array matching depth_raw_mm")
    if (
        result.depth_output_mm.dtype != np.float32
        or result.depth_output_mm.shape != expected.depth_output_mm.shape
    ):
        raise ValueError(
            "depth_output_mm must be a float32 array matching depth_raw_mm"
        )
    if not np.array_equal(result.mask, expected.mask):
        raise ValueError("mask does not match depth_cutoff_mm")
    if not np.array_equal(result.depth_output_mm, expected.depth_output_mm):
        raise ValueError("depth_output_mm does not match masked depth_raw_mm")
