import numpy as np
import pytest

from calibration.ball_geometry import (
    indentation_depth_mm,
    spherical_cap_depth_mm,
)


def test_indentation_and_dense_cap_match_sphere_geometry():
    diameter = 6.0
    ppmm = 20.0
    radius = 20.0
    expected_peak = 3.0 - np.sqrt(3.0**2 - 1.0**2)

    depth = spherical_cap_depth_mm(
        (9, 11, 3), (5.0, 4.0), radius, diameter, ppmm
    )

    assert indentation_depth_mm(radius, diameter, ppmm) == pytest.approx(
        expected_peak
    )
    assert depth.dtype == np.float32
    assert depth.shape == (9, 11)
    assert depth[4, 5] == pytest.approx(expected_peak)
    assert np.all(depth >= 0)


def test_dense_cap_is_zero_at_and_outside_contact_boundary():
    depth = spherical_cap_depth_mm((61, 61), (30, 30), 20, 6, 20)

    assert depth[30, 50] == 0
    assert depth[30, 51] == 0
    assert depth[30, 49] > 0


def test_rejects_contact_at_or_beyond_hemisphere():
    with pytest.raises(ValueError, match="contact radius < ball radius"):
        indentation_depth_mm(radius_px=30, ball_diameter_mm=3, ppmm=20)
