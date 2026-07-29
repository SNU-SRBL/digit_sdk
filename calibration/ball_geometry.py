"""Validated geometry for spherical ball-indenter calibration targets."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def indentation_depth_mm(
    radius_px: float,
    ball_diameter_mm: float,
    ppmm: float,
) -> float:
    """Return spherical-cap peak depth for a labeled contact radius."""
    radius_px = float(radius_px)
    ball_diameter_mm = float(ball_diameter_mm)
    ppmm = float(ppmm)
    if not np.isfinite([radius_px, ball_diameter_mm, ppmm]).all():
        raise ValueError("ball geometry values must be finite")
    if radius_px <= 0 or ball_diameter_mm <= 0 or ppmm <= 0:
        raise ValueError("ball geometry values must be positive")
    ball_radius_mm = ball_diameter_mm / 2.0
    contact_radius_mm = radius_px / ppmm
    if contact_radius_mm >= ball_radius_mm:
        raise ValueError("ball contact must satisfy 0 < contact radius < ball radius")
    return float(
        ball_radius_mm
        - np.sqrt(ball_radius_mm**2 - contact_radius_mm**2)
    )


def spherical_cap_depth_mm(
    image_shape: Sequence[int],
    center_px: Sequence[float],
    radius_px: float,
    ball_diameter_mm: float,
    ppmm: float,
) -> np.ndarray:
    """Build a dense metric depth target clipped to the image extent."""
    if len(image_shape) < 2:
        raise ValueError("image_shape must contain height and width")
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    if len(center_px) != 2:
        raise ValueError("center_px must contain x and y")
    center_x, center_y = float(center_px[0]), float(center_px[1])
    if not np.isfinite([center_x, center_y]).all():
        raise ValueError("center_px must be finite")
    if not (0 <= center_x < width and 0 <= center_y < height):
        raise ValueError("center_px must be inside the image")

    peak_depth = indentation_depth_mm(radius_px, ball_diameter_mm, ppmm)
    ball_radius_mm = float(ball_diameter_mm) / 2.0
    boundary_height_mm = ball_radius_mm - peak_depth
    yy, xx = np.ogrid[:height, :width]
    radius_squared_px = (xx - center_x) ** 2 + (yy - center_y) ** 2
    support = radius_squared_px < float(radius_px) ** 2
    radial_squared_mm = radius_squared_px / float(ppmm) ** 2
    depth = np.zeros((height, width), dtype=np.float32)
    depth[support] = (
        np.sqrt(ball_radius_mm**2 - radial_squared_mm[support])
        - boundary_height_mm
    ).astype(np.float32)
    return depth
