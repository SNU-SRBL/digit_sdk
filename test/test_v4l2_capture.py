import ctypes as ct

from digit_sdk.v4l2_capture import _Buffer, _Format, _RequestBuffers, _StreamParm


def test_v4l2_linux_abi_layout_matches_the_ioctl_requests():
    assert tuple(map(ct.sizeof, (_Format, _StreamParm, _RequestBuffers, _Buffer))) == (
        208, 204, 20, 88
    )
