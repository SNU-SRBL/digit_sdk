import json
import os
from pathlib import Path
import re
import time

import cv2
import numpy as np

import pyudev

from digit_sdk.utils import load_config

'''This module provides a Camera class for low latency image acquisition using OpenCV.
It supports both DIGIT cameras and generic V4L2 cameras.
'''

DIAGNOSTIC_POST_FRAMES = 3


class FrameDiagnostics:
    """Opt-in, bounded recovery-window frame capture.

    Writes timestamped ``.npy`` frames plus a JSONL index with per-frame
    brightness statistics. Disabled when ``directory`` is falsy. Event count
    and per-event frame count are capped so a long hardware run stays small.
    """

    MAX_EVENTS = 8
    MAX_FRAMES_PER_EVENT = 20

    def __init__(
        self,
        directory,
        serial=None,
        *,
        max_events=MAX_EVENTS,
        max_frames_per_event=MAX_FRAMES_PER_EVENT,
    ):
        self.directory = Path(directory) if directory else None
        self.serial = serial
        self.max_events = max_events
        self.max_frames_per_event = max_frames_per_event
        self.events = 0
        self.current = None
        self.index_path = None
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            self.index_path = self.directory / "index.jsonl"

    @property
    def enabled(self):
        return self.directory is not None

    def start_event(self, label, reason=None):
        """Open a new capture event; flushes any still-open event first."""
        if not self.enabled:
            return False
        self.end_event()
        if self.events >= self.max_events:
            return False
        self.events += 1
        self.current = {
            "event": self.events,
            "label": label,
            "reason": reason or label,
            "serial": self.serial,
            "started_ns": time.time_ns(),
            "frames": [],
        }
        return True

    def add_frame(self, tag, frame, timestamp_ns=None):
        """Persist one frame with metadata; no-op without an open event."""
        if not self.enabled or self.current is None:
            return False
        if len(self.current["frames"]) >= self.max_frames_per_event:
            return False
        frame = np.asarray(frame)
        timestamp_ns = time.time_ns() if timestamp_ns is None else timestamp_ns
        index = len(self.current["frames"])
        name = (
            f"{self.current['event']:03d}_{index:02d}_"
            f"{timestamp_ns // 1_000_000}_{tag}.npy"
        )
        np.save(self.directory / name, frame)
        self.current["frames"].append({
            "file": name,
            "tag": tag,
            "timestamp_ns": timestamp_ns,
            "shape": list(frame.shape),
            "dtype": str(frame.dtype),
            "brightness_mean": float(frame.mean()),
            "brightness_min": int(frame.min()),
            "brightness_max": int(frame.max()),
        })
        return True

    def end_event(self, extra=None):
        """Flush the open event to the JSONL index."""
        if not self.enabled or self.current is None:
            return False
        record = dict(self.current)
        record["ended_ns"] = time.time_ns()
        if extra:
            record.update(extra)
        with open(self.index_path, "a") as stream:
            stream.write(json.dumps(record) + "\n")
        self.current = None
        return True


class Camera:
    """The camera class (synchronous, based on digit-interface pattern)."""

    def __init__(
        self,
        dev_type=None,
        imgh=None,
        imgw=None,
        raw_imgh=None,
        raw_imgw=None,
        framerate=None,
        serial=None,
        sensors_root=None,
        config_path=None,
        default_config=None,
        verbose=True,
    ):
        if serial is not None or config_path is not None:
            config = load_config(
                serial=serial,
                config_path=config_path,
                sensors_root=sensors_root,
                default_config=default_config,
            )
            dev_type = config["device_type"]
            imgh = config["imgh"]
            imgw = config["imgw"]
            raw_imgh = config["raw_imgh"]
            raw_imgw = config["raw_imgw"]
            framerate = config["framerate"] if framerate is None else framerate
            self.serial = serial
            self.device_type = dev_type
        else:
            self.serial = None
            self.device_type = dev_type

        self.raw_imgh = raw_imgh
        self.raw_imgw = raw_imgw
        self.raw_size = self.raw_imgh * self.raw_imgw * 3
        self.framerate = framerate
        self.imgh = imgh
        self.imgw = imgw

        # Configs store raw dimensions inverted (raw_imgh = width, raw_imgw = height).
        # Normalize so raw_imgh = height, raw_imgw = width for numpy reshape.
        if imgh is not None and imgw is not None and imgh > imgw:
            self.raw_imgh, self.raw_imgw = raw_imgw, raw_imgh
            self.raw_size = self.raw_imgh * self.raw_imgw * 3

        # Camera device discovery
        if self.device_type and self.device_type.upper() == "DIGIT" and self.serial:
            digit_info = DigitHandler.find_digit(
                self.serial, attempts=51, retry_interval=0.1
            )
            if digit_info is None:
                raise RuntimeError(f"Could not find DIGIT camera with serial {self.serial}")
            self.device = digit_info["dev_name"]
            self.dev_id = int(re.search(r"\d+$", self.device).group(0))
        else:
            self.dev_id = get_camera_id(self.device_type, verbose)
            self.device = f"/dev/video{self.dev_id}"

        # ── Corruption detection + recovery state ──
        self._recovery_sleep = 0.01
        self._recovery_retry_sleep = 0.1
        self._recovery_started_at = None
        self._recovery_started_ns = None
        self._recovery_reason = None
        self._recovery_failures = 0

        # ── Opt-in recovery-window diagnostics (disabled by default) ──
        self._diag = None
        self._diag_last_good = None
        self._diag_post_remaining = 0

        # ── FPS Watchdog (gap-based, derived from configured framerate) ──
        self._wd_dt = (1.0 / self.framerate) * 1.5  # 25ms@60Hz, 50ms@30Hz
        self._wd_max_slow = 10  # consecutive slow frames trigger recovery
        self._wd_slow_count = 0
        self._wd_last_time = 0.0

    @staticmethod
    def _is_corrupt(bgr: np.ndarray, threshold: float = 3.0) -> bool:
        """
        Detect STM32 DMA tear: per-column step-discontinuity detector.

        A tear is a horizontal line where pixel values jump instantaneously
        (no gradient). Each column independently scans all 239 row-pairs for
        a spike — one diff much larger than its neighbors above and below.
        If >50% of 320 columns spike at the same modal row with
        isolation ratio >10× and neighbor flatness >80%, the row is a tear.

        Uses cv2.absdiff per BGR channel → max for 3-channel sensitivity
        at cv2 speed (1.7× faster than int16 numpy, zero accuracy loss).
        """
        d0 = cv2.absdiff(bgr[:-1, :, 0], bgr[1:, :, 0]).astype(np.int16)
        d1 = cv2.absdiff(bgr[:-1, :, 1], bgr[1:, :, 1]).astype(np.int16)
        d2 = cv2.absdiff(bgr[:-1, :, 2], bgr[1:, :, 2]).astype(np.int16)
        diff = np.maximum(np.maximum(d0, d1), d2)
        spike = (diff[1:-1] > np.maximum(diff[:-2], diff[2:]) * threshold) & (diff[1:-1] > 20)

        # Only count columns that spike at the SAME (modal) row.
        # A tear has all spikes on one horizontal line; contact edges
        # spread spikes across different rows and are safely discarded.
        row_counts = np.sum(spike, axis=1)  # columns-per-row
        best_row = int(np.argmax(row_counts))
        best_count = int(np.max(row_counts))

        # Quick reject: <50% columns have a spike at the modal row
        if best_count <= bgr.shape[1] * 0.5:
            return False

        # Isolation ratio: tear is a pure step — peak must dominate neighbors by 10×
        dr = best_row + 1  # spike[best_row] → diff[best_row+1]
        peak = float(np.mean(diff[dr, :]))
        r0 = max(0, dr - 5)
        r1 = min(diff.shape[0], dr + 6)
        neigh = (float(np.mean(diff[r0:dr, :])) + float(np.mean(diff[dr + 1:r1, :]))) / 2.0
        if peak / max(neigh, 0.1) <= 10.0:
            return False

        # Neighbor flatness: tear neighbors are identical copies — diffs must be near-zero
        spike_cols = np.where(spike[best_row, :])[0]
        prev_flat = float(np.mean(diff[max(0, dr - 1), spike_cols] < 15))
        next_flat = float(np.mean(diff[min(diff.shape[0] - 1, dr + 1), spike_cols] < 15))
        return (prev_flat + next_flat) / 2.0 > 0.80

    def connect(self, verbose=True):
        """Connect to the camera using OpenCV VideoCapture (digit-interface pattern)."""
        if self.device_type and self.device_type.upper() == "DIGIT" and self.serial:
            stable = Path(
                f"/dev/v4l/by-id/"
                f"usb-Facebook_DIGIT_{self.serial}-video-index0"
            )
            if stable.exists():
                self.device = str(stable)
                self.dev_id = int(re.search(r"\d+$", os.path.realpath(stable)).group(0))
            else:
                digit_info = DigitHandler.find_digit(self.serial)
                if digit_info is None:
                    raise RuntimeError(
                        f"Could not find DIGIT camera with serial {self.serial}"
                    )
                self.device = digit_info["dev_name"]
                self.dev_id = int(re.search(r"\d+$", self.device).group(0))
        self.cap = cv2.VideoCapture(self.device)  # path string, survives index shifts
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f"Could not open camera {self.device}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.raw_imgw)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.raw_imgh)
        self.cap.set(cv2.CAP_PROP_FPS, self.framerate)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        for _ in range(10):
            ret, warmup = self.cap.read()
            if not ret:
                self.cap.release()
                raise RuntimeError(
                    f"camera warm-up failed for {self.serial or self.device}"
                )
            diag = getattr(self, "_diag", None)
            if diag is not None and diag.enabled:
                diag.add_frame("connect_discard", warmup)
        if verbose:
            print("Camera ready for use!")

    def get_image(self, flush=False):
        """
        Get the latest image with automatic corruption detection and recovery.

        Returns None if frame is corrupt and recovery was triggered.
        Caller should retry on next frame.
        """
        ret, frame = self.cap.read()
        if not ret:
            self._begin_diagnostic_event("read_failed")
            self._recover("read_failed")
            self._wd_slow_count = 0
            self._wd_last_time = 0.0
            return None
        frame_copied = frame.copy()
        now = time.monotonic()

        # ── Corruption detection + immediate recovery ──
        if self._is_corrupt(frame_copied):
            self._begin_diagnostic_event("corrupt_frame", corrupt=frame_copied)
            self._recover("corrupt_frame")
            self._wd_slow_count = 0
            self._wd_last_time = now
            return None  # skip this corrupt frame, caller will retry

        # ── FPS Watchdog (per-frame gap) ──
        if self._wd_last_time > 0:
            gap = now - self._wd_last_time
            if gap > self._wd_dt:
                self._wd_slow_count += 1
                if self._wd_slow_count >= self._wd_max_slow:
                    self._recover("sustained_low_rate")
                    self._wd_slow_count = 0
                    self._wd_last_time = 0.0  # reset after recovery
                    return None
            else:
                self._wd_slow_count = max(0, self._wd_slow_count - 1)
        self._wd_last_time = now

        diag = getattr(self, "_diag", None)
        if diag is not None and diag.enabled:
            self._diag_last_good = frame_copied
            if getattr(self, "_diag_post_remaining", 0) > 0:
                diag.add_frame("post_reconnect", frame_copied)
                self._diag_post_remaining -= 1
                if self._diag_post_remaining <= 0:
                    diag.end_event()

        return frame_copied

    def enable_diagnostics(
        self,
        directory,
        *,
        max_events=FrameDiagnostics.MAX_EVENTS,
        max_frames_per_event=FrameDiagnostics.MAX_FRAMES_PER_EVENT,
    ):
        """Opt in to bounded recovery-window diagnostics for this camera."""
        self._diag = FrameDiagnostics(
            directory,
            serial=self.serial,
            max_events=max_events,
            max_frames_per_event=max_frames_per_event,
        )
        self._diag_last_good = None
        self._diag_post_remaining = 0
        return self._diag

    def _begin_diagnostic_event(self, reason, corrupt=None):
        """Capture the pre-trigger and corrupt frames of a recovery window."""
        diag = getattr(self, "_diag", None)
        if diag is None or not diag.enabled:
            return
        if not diag.start_event(reason, reason=reason):
            return
        pre_trigger = getattr(self, "_diag_last_good", None)
        if pre_trigger is not None:
            diag.add_frame("pre_trigger", pre_trigger)
        if corrupt is not None:
            diag.add_frame("corrupt_candidate", corrupt)
        self._diag_post_remaining = DIAGNOSTIC_POST_FRAMES

    def _recover(self, reason="unspecified"):
        """
        STREAMOFF/STREAMON cycle via release()+connect().

        Resets STM32 DMA state. connect() discards startup frames.
        """
        if self._recovery_started_at is None:
            self._recovery_started_at = time.monotonic()
            self._recovery_reason = reason
            if self._diagnostics_enabled():
                # Wall clock only in diagnostics mode; monotonic elapsed is
                # enough for normal operation.
                self._recovery_started_ns = time.time_ns()
        self.release()
        time.sleep(self._recovery_sleep)
        try:
            self.connect(verbose=False)
        except RuntimeError:
            self._recovery_failures += 1
            time.sleep(self._recovery_retry_sleep)
            return False
        elapsed = time.monotonic() - self._recovery_started_at
        message = (
            f"Camera {self.serial or self.device}: recovered "
            f"reason={self._recovery_reason} elapsed={elapsed:.3f}s "
            f"failed_attempts={self._recovery_failures}"
        )
        if self._diagnostics_enabled():
            # Logged for every recovery, including ones past the image cap.
            message += (
                f" started_ns={self._recovery_started_ns}"
                f" completed_ns={time.time_ns()}"
            )
        print(message, flush=True)
        self._recovery_started_at = None
        self._recovery_started_ns = None
        self._recovery_reason = None
        self._recovery_failures = 0
        return True

    def _diagnostics_enabled(self):
        diag = getattr(self, "_diag", None)
        return diag is not None and diag.enabled

    def release(self):
        """Release camera resources."""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()


def get_camera_id(camera_name, verbose=True):
    """Find the camera ID by name."""
    cam_num = None
    for file in os.listdir("/sys/class/video4linux"):
        real_file = os.path.realpath("/sys/class/video4linux/" + file + "/name")
        with open(real_file, "rt") as name_file:
            name = name_file.read().rstrip()
        if camera_name in name:
            cam_num = int(re.search(r"\d+$", file).group(0))
            if verbose:
                found = "FOUND!"
        else:
            if verbose:
                found = "      "
        if verbose:
            print("{} {} -> {}".format(found, file, name))
    return cam_num


def resize_crop(img, imgw, imgh):
    """Resize and crop the image."""
    border_size_x, border_size_y = int(img.shape[0] * (1 / 7)), int(
        np.floor(img.shape[1] * (1 / 7))
    )
    cropped_imgh = img.shape[0] - 2 * border_size_x
    cropped_imgw = img.shape[1] - 2 * border_size_y
    extra_border_h = 0
    extra_border_w = 0
    if cropped_imgh * imgw / imgh > cropped_imgw + 1e-8:
        extra_border_h = int(cropped_imgh - cropped_imgw * imgh / imgw)
    elif cropped_imgh * imgw / imgh < cropped_imgw - 1e-8:
        extra_border_w = int(cropped_imgw - cropped_imgh * imgw / imgh)
    img = img[
        border_size_x + extra_border_h: img.shape[0] - border_size_x,
        border_size_y + extra_border_w: img.shape[1] - border_size_y,
    ]
    img = cv2.resize(img, (imgw, imgh))
    return img


class DigitHandler:
    """Utility class for DIGIT sensor discovery."""

    @staticmethod
    def _parse(digit_dev):
        return {
            "dev_name": digit_dev["DEVNAME"],
            "manufacturer": digit_dev.get("ID_VENDOR", ""),
            "model": digit_dev.get("ID_MODEL", ""),
            "revision": digit_dev.get("ID_REVISION", ""),
            "serial": digit_dev.get("ID_SERIAL_SHORT", ""),
        }

    @staticmethod
    def list_digits():
        if pyudev is None:
            raise ImportError(
                "pyudev is required for DigitHandler. "
                "Install with 'pip install pyudev'."
            )
        context = pyudev.Context()
        digits = context.list_devices(subsystem="video4linux", ID_MODEL="DIGIT")
        return [DigitHandler._parse(device) for device in digits]

    @staticmethod
    def find_digit(serial, attempts=1, retry_interval=0.0):
        for attempt in range(attempts):
            digits = DigitHandler.list_digits()
            for digit in digits:
                if digit["serial"] == serial:
                    return digit
            if attempt + 1 < attempts:
                time.sleep(retry_interval)
        return None
