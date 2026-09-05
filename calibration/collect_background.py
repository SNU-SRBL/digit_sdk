#!/usr/bin/env python3
"""Collect background training frames and their averaged reference."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

from calibration.capture_session import (
    COLLECTION_FPS,
    CollectionCamera,
    session_id,
    utc_now,
)


DEFAULT_FRAME_COUNT = 60
DEFAULT_INTERVAL_SECONDS = 1.0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--sensors-root", type=Path, default=Path("sensors"))
    parser.add_argument("--session-id")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--frame-count", type=int, default=DEFAULT_FRAME_COUNT)
    parser.add_argument(
        "--interval-seconds", type=float, default=DEFAULT_INTERVAL_SECONDS
    )
    return parser.parse_args(argv)


def average_background(frames: Iterable[np.ndarray]) -> np.ndarray:
    frames = list(frames)
    if not frames:
        raise ValueError("at least one background frame is required")
    shape = frames[0].shape
    if any(
        frame.dtype != np.uint8
        or frame.ndim != 3
        or frame.shape[2] != 3
        or frame.shape != shape
        for frame in frames
    ):
        raise ValueError("background frames must be uint8 BGR images of the same shape")
    mean = np.mean(np.stack(frames).astype(np.float32), axis=0)
    return np.rint(mean).astype(np.uint8)


def _write_png(path: Path, frame: np.ndarray) -> str:
    encoded, png = cv2.imencode(".png", frame)
    if not encoded:
        raise RuntimeError(f"failed to encode background image: {path}")
    data = png.tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as file:
        file.write(data)


def save_background_reference(
    output_root: Path,
    frames: Sequence[np.ndarray],
    *,
    serial: str,
    capture_session: str,
    interval_seconds: float,
    captured_at: Optional[Sequence[datetime]] = None,
) -> dict:
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"background reference already exists: {output_root}")
    if not capture_session or Path(capture_session).name != capture_session:
        raise ValueError("capture_session must be one non-empty path component")
    reference = average_background(frames)
    if interval_seconds < 0:
        raise ValueError("interval_seconds must be non-negative")
    timestamps = list(captured_at or [utc_now()] * len(frames))
    if len(timestamps) != len(frames):
        raise ValueError("captured_at must contain one timestamp per frame")
    if any(timestamp.tzinfo is None for timestamp in timestamps):
        raise ValueError("background timestamps must include a timezone")

    source_frames = []
    for index, (frame, timestamp) in enumerate(zip(frames, timestamps)):
        relative = Path("frames") / capture_session / f"{index:03d}.png"
        _write_png(output_root / relative, frame)
        source_frames.append({
            "path": relative.as_posix(),
            "captured_at_utc": timestamp.astimezone(timezone.utc).isoformat(),
        })
    _write_png(output_root / "reference.png", reference)
    sample_id = f"{serial}_background_reference"
    record = {
        "sample_id": sample_id,
        "kind": "background_reference",
        "serial": serial,
        "path": "reference.png",
        "captured_at_utc": timestamps[0].astimezone(timezone.utc).isoformat(),
        "session_id": capture_session,
        "split_group": sample_id,
        "shape": list(reference.shape),
        "dtype": "uint8",
        "colour_space": "BGR",
        "capture_fps": COLLECTION_FPS,
        "sample_count": len(frames),
        "interval_seconds": float(interval_seconds),
        "reduction": "float32_mean_round_uint8",
        "source_frames": source_frames,
    }
    (output_root / "metadata.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def capture_background_frames(
    camera: CollectionCamera,
    frame_count: int,
    interval_seconds: float,
) -> Tuple[list, list]:
    frames = []
    timestamps = []
    next_capture = time.monotonic()
    while len(frames) < frame_count:
        frame = camera.read()
        if frame is None:
            continue
        now = time.monotonic()
        preview = frame.copy()
        cv2.putText(
            preview,
            f"background {len(frames)}/{frame_count}",
            (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1,
        )
        cv2.imshow("DIGIT background calibration", preview)
        cv2.waitKey(1)
        if now < next_capture:
            continue
        frames.append(frame.copy())
        timestamps.append(utc_now())
        print(f"captured background frame {len(frames)}/{frame_count}")
        next_capture = now + interval_seconds
    return frames, timestamps


def main(argv=None):
    args = parse_args(argv)
    if args.frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if args.interval_seconds < 0:
        raise ValueError("interval_seconds must be non-negative")
    capture_session = session_id(args.session_id)
    output_root = args.output_root or (
        args.sensors_root / args.serial / "calibration" / "inbox" / "background"
    )
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"background reference already exists: {output_root}")

    camera = CollectionCamera(args.serial, args.sensors_root, args.warmup_frames)
    camera.connect()
    try:
        print("Remove all contact from the sensor.")
        print("b: collect averaged background | q: quit")
        while True:
            frame = camera.read()
            if frame is None:
                continue
            preview = frame.copy()
            cv2.putText(
                preview, "b collect | q quit", (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1,
            )
            cv2.imshow("DIGIT background calibration", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return 1
            if key == ord("b"):
                frames, timestamps = capture_background_frames(
                    camera, args.frame_count, args.interval_seconds
                )
                record = save_background_reference(
                    output_root,
                    frames,
                    serial=args.serial,
                    capture_session=capture_session,
                    interval_seconds=args.interval_seconds,
                    captured_at=timestamps,
                )
                print(f"saved {output_root / record['path']}")
                return 0
    finally:
        cv2.destroyAllWindows()
        camera.release()


if __name__ == "__main__":
    raise SystemExit(main())
