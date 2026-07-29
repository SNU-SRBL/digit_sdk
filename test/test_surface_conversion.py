import numpy as np

from digit_sdk.depth_geometry import depth_to_pointcloud


def test_depth_to_pointcloud_uses_metres_and_negative_indentation_z():
    depth = np.array([[0.0, 1.5], [2.0, 0.0]], dtype=np.float32)
    cloud = depth_to_pointcloud(depth, ppmm=10.0)
    assert cloud.shape == (4, 3)
    np.testing.assert_allclose(cloud[:, 2], [0.0, -0.0015, -0.002, 0.0])


def test_depth_to_pointcloud_subsampling_reduces_work():
    depth = np.ones((6, 8), dtype=np.float32)
    cloud = depth_to_pointcloud(
        depth, ppmm=10.0, point_sample_mm=0.2
    )
    assert cloud.shape == (12, 3)
