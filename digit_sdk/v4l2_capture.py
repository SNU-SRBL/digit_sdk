"""Small V4L2 MMAP capture adapter for the DIGIT YUYV stream."""

import ctypes as ct
import fcntl
import mmap
import os
import select
import time

import cv2
import numpy as np


V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_MEMORY_MMAP = 1
V4L2_FIELD_NONE = 1
V4L2_PIX_FMT_YUYV = 0x56595559
V4L2_BUF_FLAG_ERROR = 0x40

VIDIOC_S_FMT = 0xC0D05605
VIDIOC_REQBUFS = 0xC0145608
VIDIOC_QUERYBUF = 0xC0585609
VIDIOC_QBUF = 0xC058560F
VIDIOC_DQBUF = 0xC0585611
VIDIOC_STREAMON = 0x40045612
VIDIOC_STREAMOFF = 0x40045613
VIDIOC_S_PARM = 0xC0CC5616


class _PixFormat(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in (
        "width", "height", "pixelformat", "field", "bytesperline", "sizeimage",
        "colorspace", "priv", "flags", "ycbcr_enc", "quantization", "xfer_func",
    )]


class _FormatUnion(ct.Union):
    _fields_ = [
        ("pix", _PixFormat), ("raw", ct.c_uint8 * 200),
        # v4l2_format's other union members contain pointers, so its union is
        # eight-byte aligned even though the YUYV member itself is not.
        ("_align", ct.c_uint64),
    ]


class _Format(ct.Structure):
    _fields_ = [("type", ct.c_uint32), ("fmt", _FormatUnion)]


class _Fract(ct.Structure):
    _fields_ = [("numerator", ct.c_uint32), ("denominator", ct.c_uint32)]


class _CaptureParm(ct.Structure):
    _fields_ = [
        ("capability", ct.c_uint32), ("capturemode", ct.c_uint32),
        ("timeperframe", _Fract), ("extendedmode", ct.c_uint32),
        ("readbuffers", ct.c_uint32), ("reserved", ct.c_uint32 * 4),
    ]


class _StreamParmUnion(ct.Union):
    _fields_ = [("capture", _CaptureParm), ("raw", ct.c_uint8 * 200)]


class _StreamParm(ct.Structure):
    _fields_ = [("type", ct.c_uint32), ("parm", _StreamParmUnion)]


class _RequestBuffers(ct.Structure):
    _fields_ = [
        ("count", ct.c_uint32), ("type", ct.c_uint32), ("memory", ct.c_uint32),
        ("capabilities", ct.c_uint32), ("reserved", ct.c_uint32),
    ]


class _Timecode(ct.Structure):
    _fields_ = [
        ("type", ct.c_uint32), ("flags", ct.c_uint32), ("frames", ct.c_uint8),
        ("seconds", ct.c_uint8), ("minutes", ct.c_uint8), ("hours", ct.c_uint8),
        ("userbits", ct.c_uint8 * 4),
    ]


class _Memory(ct.Union):
    _fields_ = [("offset", ct.c_uint32), ("userptr", ct.c_ulong), ("fd", ct.c_int)]


class _Request(ct.Union):
    _fields_ = [("request_fd", ct.c_int), ("reserved", ct.c_uint32)]


class _Buffer(ct.Structure):
    _fields_ = [
        ("index", ct.c_uint32), ("type", ct.c_uint32), ("bytesused", ct.c_uint32),
        ("flags", ct.c_uint32), ("field", ct.c_uint32), ("timestamp_sec", ct.c_long),
        ("timestamp_usec", ct.c_long), ("timecode", _Timecode), ("sequence", ct.c_uint32),
        ("memory", ct.c_uint32), ("m", _Memory), ("length", ct.c_uint32),
        ("reserved2", ct.c_uint32), ("request", _Request),
    ]


class V4L2Capture:
    """Open one YUYV V4L2 source with a three-buffer MMAP queue."""

    def __init__(
        self, device, width, height, fps, buffer_count=3, timeout_s=1.0,
        phase_timings=None,
    ):
        self.device = device
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.timeout_s = timeout_s
        self.phase_timings = phase_timings
        self._first_read = True
        self._fd = None
        self._maps = []
        self._streaming = False
        self._bytesperline = self.width * 2
        try:
            started = time.monotonic()
            self._fd = os.open(device, os.O_RDWR)
            self._record("open_s", started)
            self._configure(buffer_count)
        except Exception:
            self.release()
            raise

    @staticmethod
    def _ioctl(fd, request, value):
        fcntl.ioctl(fd, request, value, True)

    def _record(self, name, started):
        if self.phase_timings is not None:
            self.phase_timings[name] = time.monotonic() - started

    def _configure(self, buffer_count):
        fmt = _Format()
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        fmt.fmt.pix.width = self.width
        fmt.fmt.pix.height = self.height
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV
        fmt.fmt.pix.field = V4L2_FIELD_NONE
        started = time.monotonic()
        self._ioctl(self._fd, VIDIOC_S_FMT, fmt)
        self._record("set_format_s", started)
        if (fmt.fmt.pix.width, fmt.fmt.pix.height, fmt.fmt.pix.pixelformat) != (
            self.width, self.height, V4L2_PIX_FMT_YUYV,
        ):
            raise RuntimeError(f"V4L2 rejected QVGA/YUYV for {self.device}")
        self._bytesperline = fmt.fmt.pix.bytesperline or self.width * 2
        if self._bytesperline < self.width * 2:
            raise RuntimeError(f"V4L2 returned invalid stride for {self.device}")

        parm = _StreamParm()
        parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        parm.parm.capture.timeperframe.numerator = 1
        parm.parm.capture.timeperframe.denominator = round(self.fps)
        started = time.monotonic()
        self._ioctl(self._fd, VIDIOC_S_PARM, parm)
        self._record("set_fps_s", started)

        req = _RequestBuffers()
        req.count = buffer_count
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        req.memory = V4L2_MEMORY_MMAP
        started = time.monotonic()
        self._ioctl(self._fd, VIDIOC_REQBUFS, req)
        self._record("request_buffers_s", started)
        if req.count < 2:
            raise RuntimeError(f"V4L2 allocated too few buffers for {self.device}")

        started = time.monotonic()
        for index in range(req.count):
            buffer = _Buffer()
            buffer.index = index
            buffer.type = req.type
            buffer.memory = req.memory
            self._ioctl(self._fd, VIDIOC_QUERYBUF, buffer)
            self._maps.append(mmap.mmap(
                self._fd, buffer.length, flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE, offset=buffer.m.offset,
            ))
            self._ioctl(self._fd, VIDIOC_QBUF, buffer)
        self._record("map_queue_s", started)

        stream_type = ct.c_int(V4L2_BUF_TYPE_VIDEO_CAPTURE)
        started = time.monotonic()
        self._ioctl(self._fd, VIDIOC_STREAMON, stream_type)
        self._record("stream_on_s", started)
        self._streaming = True

    def read(self):
        if self._fd is None:
            return False, None
        try:
            first_read = self._first_read
            started = time.monotonic()
            if not select.select([self._fd], [], [], self.timeout_s)[0]:
                if first_read:
                    self._record("first_frame_wait_s", started)
                return False, None
            if first_read:
                self._record("first_frame_wait_s", started)
            buffer = _Buffer()
            buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
            buffer.memory = V4L2_MEMORY_MMAP
            started = time.monotonic()
            self._ioctl(self._fd, VIDIOC_DQBUF, buffer)
            if first_read:
                self._record("first_dequeue_s", started)
            self._first_read = False
        except OSError:
            return False, None

        try:
            expected = self._bytesperline * self.height
            if buffer.index >= len(self._maps) or buffer.bytesused < expected:
                return False, None
            yuyv = np.frombuffer(
                self._maps[buffer.index], dtype=np.uint8, count=expected
            ).reshape(self.height, self._bytesperline // 2, 2)[:, :self.width].copy()
            if buffer.flags & V4L2_BUF_FLAG_ERROR:
                return False, None
            return True, cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)
        finally:
            try:
                self._ioctl(self._fd, VIDIOC_QBUF, buffer)
            except OSError:
                pass

    def isOpened(self):
        return self._fd is not None

    def get(self, property_id):
        if property_id == cv2.CAP_PROP_FPS:
            return self.fps
        if property_id == cv2.CAP_PROP_FRAME_WIDTH:
            return self.width
        if property_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.height
        return 0.0

    def release(self):
        if self._fd is None:
            return
        if self._streaming:
            try:
                self._ioctl(self._fd, VIDIOC_STREAMOFF, ct.c_int(V4L2_BUF_TYPE_VIDEO_CAPTURE))
            except OSError:
                pass
        self._streaming = False
        for mapped in self._maps:
            mapped.close()
        self._maps = []
        os.close(self._fd)
        self._fd = None
