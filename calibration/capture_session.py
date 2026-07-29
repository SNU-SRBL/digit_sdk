"""Shared 30 Hz camera and immutable staging support for calibration capture."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Optional

import cv2
import numpy as np

from digit_sdk.camera import Camera


COLLECTION_FPS = 30
KIND_DIRECTORIES = {
    "ball": "images",
    "touch": "touches",
}
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def session_id(value: Optional[str]) -> str:
    return value or utc_now().strftime("%Y%m%dT%H%M%SZ")


def _path_component(value: str, name: str) -> str:
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"{name} must be one non-empty path component")
    return value


class CollectionCamera:
    """Calibration camera adapter fixed to the 30 Hz collection contract."""

    def __init__(self, serial: str, sensors_root: Path, warmup_frames: int = 30):
        self.serial = serial
        self.sensors_root = Path(sensors_root)
        self.warmup_frames = int(warmup_frames)
        if self.warmup_frames < 0:
            raise ValueError("warmup_frames must be non-negative")
        self.camera = self._new_camera()
        self._connected = False

    def _new_camera(self) -> Camera:
        return Camera(
            serial=self.serial,
            sensors_root=str(self.sensors_root),
            framerate=COLLECTION_FPS,
        )

    def _prepare(self, camera: Camera, *, verbose: bool) -> None:
        camera.connect(verbose=verbose)
        if not camera.cap.isOpened():
            raise RuntimeError("camera did not open")
        accepted = camera.cap.get(cv2.CAP_PROP_FPS)
        if accepted and abs(accepted - COLLECTION_FPS) > 0.5:
            raise RuntimeError(
                f"camera rejected {COLLECTION_FPS} Hz collection rate: {accepted:g} Hz"
            )
        received_frame = self.warmup_frames == 0
        for _ in range(self.warmup_frames):
            received_frame = camera.get_image() is not None or received_frame
        if not received_frame:
            raise RuntimeError("camera produced no collection warm-up frames")

    def connect(self) -> None:
        try:
            self._prepare(self.camera, verbose=True)
        except Exception:
            self.camera.release()
            raise
        self._connected = True

    def reconnect(self) -> bool:
        """Release and rediscover this calibration camera by sensor serial."""
        self.camera.release()
        self._connected = False
        time.sleep(0.25)
        try:
            replacement = self._new_camera()
            self._prepare(replacement, verbose=False)
        except (RuntimeError, cv2.error):
            if "replacement" in locals():
                replacement.release()
            return False
        self.camera = replacement
        self._connected = True
        return True

    def read(self) -> Optional[np.ndarray]:
        if not self._connected:
            self.reconnect()
            return None
        frame = self.camera.get_image()
        if frame is None:
            self.reconnect()
        return frame

    def release(self) -> None:
        self.camera.release()
        self._connected = False


class StagedCaptureWriter:
    """Write lossless images and append checksummed staging records."""

    def __init__(
        self,
        output_root: Path,
        *,
        serial: str,
        capture_session: str,
    ):
        self.output_root = Path(output_root)
        self.serial = _path_component(serial, "serial")
        self.capture_session = _path_component(capture_session, "capture_session")

    def save(
        self,
        frame: np.ndarray,
        kind: str,
        *,
        captured_at: Optional[datetime] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> dict:
        if kind not in KIND_DIRECTORIES:
            raise ValueError(f"unsupported capture kind: {kind}")
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("frame must be a uint8 HxWx3 BGR array")
        timestamp = captured_at or utc_now()
        if timestamp.tzinfo is None:
            raise ValueError("captured_at must include a timezone")
        timestamp = timestamp.astimezone(timezone.utc)
        stamp = timestamp.strftime("%Y%m%dT%H%M%S%fZ")
        sample_id = f"{self.serial}_{kind}_{stamp}"
        relative_path = (
            Path(KIND_DIRECTORIES[kind]) / self.capture_session / f"{sample_id}.png"
        )
        path = self.output_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)

        encoded, png = cv2.imencode(".png", frame)
        if not encoded:
            raise RuntimeError(f"failed to encode {sample_id} as PNG")
        image_bytes = png.tobytes()
        try:
            with path.open("xb") as file:
                file.write(image_bytes)
        except FileExistsError as error:
            raise RuntimeError(f"refusing to overwrite staged image: {path}") from error

        record = {
            "sample_id": sample_id,
            "kind": kind,
            "serial": self.serial,
            "path": relative_path.as_posix(),
            "captured_at_utc": timestamp.isoformat(),
            "session_id": self.capture_session,
            "split_group": sample_id,
            "shape": list(frame.shape),
            "dtype": str(frame.dtype),
            "colour_space": "BGR",
            "capture_fps": COLLECTION_FPS,
            "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        }
        additions = dict(metadata or {})
        overlap = additions.keys() & record.keys()
        if overlap:
            path.unlink()
            raise ValueError(f"metadata cannot replace core fields: {sorted(overlap)}")
        record.update(additions)
        self.output_root.mkdir(parents=True, exist_ok=True)
        with (self.output_root / "metadata.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, sort_keys=True) + "\n")
            file.flush()
        return record
