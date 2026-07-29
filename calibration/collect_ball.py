#!/usr/bin/env python3
"""Capture guided ball-indenter calibration data at 30 Hz."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2

from calibration.capture_session import (
    CollectionCamera,
    StagedCaptureWriter,
    session_id,
)
from digit_sdk.utils import load_config


GRID_SIZE = 5
DEPTH_LEVELS = 5


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--ball-diameter-mm", required=True, type=float)
    parser.add_argument("--sensors-root", type=Path, default=Path("sensors"))
    parser.add_argument("--session-id")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--target-row", type=int)
    parser.add_argument("--target-column", type=int)
    parser.add_argument("--target-depth", type=int)
    parser.add_argument("--replaces")
    parser.add_argument(
        "--display-difference",
        action="store_true",
        help="Start with amplified difference from the shared background",
    )
    args = parser.parse_args(argv)
    replacement_values = (
        args.target_row, args.target_column, args.target_depth, args.replaces
    )
    if any(value is not None for value in replacement_values):
        if any(value is None for value in replacement_values):
            parser.error(
                "--target-row, --target-column, --target-depth, and --replaces "
                "must be used together"
            )
        if not 0 <= args.target_row < GRID_SIZE:
            parser.error("--target-row must be between 0 and 4")
        if not 0 <= args.target_column < GRID_SIZE:
            parser.error("--target-column must be between 0 and 4")
        if not 1 <= args.target_depth <= DEPTH_LEVELS:
            parser.error("--target-depth must be between 1 and 5")
    return args


def target_for(index: int):
    cell, depth_index = divmod(index, DEPTH_LEVELS)
    row, offset = divmod(cell, GRID_SIZE)
    column = offset if row % 2 == 0 else GRID_SIZE - 1 - offset
    return row, column, depth_index + 1


def background_difference(frame, background):
    if frame.shape != background.shape:
        raise ValueError("frame and background shapes must match")
    return cv2.convertScaleAbs(cv2.absdiff(frame, background), alpha=3.0)


def load_background_reference(sensors_root: Path, serial: str):
    root = Path(sensors_root) / serial / "calibration" / "inbox" / "background"
    path = root / "reference.png"
    metadata_path = root / "metadata.json"
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"collect the shared background first: "
            f"python3 -m calibration.collect_background --serial {serial}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("kind") != "background_reference"
        or metadata.get("serial") != serial
        or metadata.get("path") != "reference.png"
        or hashlib.sha256(path.read_bytes()).hexdigest()
        != metadata.get("image_sha256")
    ):
        raise ValueError(f"invalid background reference metadata: {metadata_path}")
    background = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if background is None:
        raise ValueError(f"cannot read background reference: {path}")
    if list(background.shape) != metadata.get("shape"):
        raise ValueError(f"background shape does not match metadata: {path}")
    return background, metadata["sample_id"]


def validate_replacement_source(
    output_root: Path,
    sample_id: str,
    serial: str,
    ball_diameter_mm: float,
    target,
):
    metadata_path = Path(output_root) / "metadata.jsonl"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    records = [
        json.loads(line)
        for line in metadata_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matches = [record for record in records if record.get("sample_id") == sample_id]
    if len(matches) != 1:
        raise ValueError(f"replacement source must exist exactly once: {sample_id}")
    source = matches[0]
    if source.get("kind") != "ball" or source.get("serial") != serial:
        raise ValueError(f"replacement source does not match sensor: {sample_id}")
    if not source.get("rejected", False):
        raise ValueError(f"replacement source is not rejected: {sample_id}")
    source_target = (
        source.get("guide_grid_row"), source.get("guide_grid_column"),
        source.get("guide_depth_level"),
    )
    if source_target != tuple(target) or not abs(
        float(source.get("ball_diameter_mm", -1)) - float(ball_diameter_mm)
    ) < 1e-9:
        raise ValueError(f"replacement target does not match source: {sample_id}")
    if any(
        record.get("replaces_sample_id") == sample_id
        and not record.get("rejected", False)
        for record in records
    ):
        raise ValueError(f"rejected sample already has a replacement: {sample_id}")
    return source


def overlay_guide(
    frame, target_index: int, saved: int, show_difference=False, target=None
):
    output = frame.copy()
    height, width = output.shape[:2]
    row, column, depth_level = target or target_for(target_index)
    x0, x1 = column * width // GRID_SIZE, (column + 1) * width // GRID_SIZE
    y0, y1 = row * height // GRID_SIZE, (row + 1) * height // GRID_SIZE
    for index in range(1, GRID_SIZE):
        cv2.line(output, (index * width // GRID_SIZE, 0),
                 (index * width // GRID_SIZE, height), (0, 180, 0), 1)
        cv2.line(output, (0, index * height // GRID_SIZE),
                 (width, index * height // GRID_SIZE), (0, 180, 0), 1)
    cv2.rectangle(output, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 255), 2)
    cv2.putText(
        output,
        f"cell {row + 1},{column + 1} depth {depth_level}/5 saved {saved}",
        (5, 18), cv2.FONT_HERSHEY_SIMPLEX,
        0.38, (0, 255, 255), 1)
    cv2.putText(
        output,
        f"diff {'on' if show_difference else 'off'} | w save d toggle q quit",
        (5, 34), cv2.FONT_HERSHEY_SIMPLEX,
        0.38, (0, 255, 255), 1)
    return output


def main(argv=None):
    args = parse_args(argv)
    if args.ball_diameter_mm <= 0:
        raise ValueError("ball diameter must be positive")
    config = load_config(serial=args.serial, sensors_root=str(args.sensors_root))
    ppmm = float(config["ppmm"])
    capture_session = session_id(args.session_id)
    output_root = args.output_root or (
        args.sensors_root / args.serial / "calibration" / "inbox" / "ball"
    )
    writer = StagedCaptureWriter(
        output_root.resolve(),
        serial=args.serial,
        capture_session=capture_session,
    )
    camera = CollectionCamera(args.serial, args.sensors_root, args.warmup_frames)
    background, background_id = load_background_reference(
        args.sensors_root, args.serial
    )
    replacement_target = None
    if args.replaces is not None:
        replacement_target = (
            args.target_row, args.target_column, args.target_depth
        )
        validate_replacement_source(
            output_root, args.replaces, args.serial,
            args.ball_diameter_mm, replacement_target,
        )
    counts = Counter()
    target_index = 0
    target_count = GRID_SIZE * GRID_SIZE * DEPTH_LEVELS
    show_difference = args.display_difference
    camera.connect()
    try:
        print("w: save ball contact | d: difference | q: quit")
        while True:
            frame = camera.read()
            if frame is None:
                continue
            preview = (
                background_difference(frame, background)
                if show_difference
                else frame
            )
            cv2.imshow(
                "DIGIT ball calibration",
                overlay_guide(
                    preview,
                    target_index % target_count,
                    counts["ball"],
                    show_difference,
                    replacement_target,
                ),
            )
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                show_difference = not show_difference
                print(f"background difference {'on' if show_difference else 'off'}")
            elif key == ord("w"):
                row, column, depth_level = (
                    replacement_target
                    or target_for(target_index % target_count)
                )
                metadata = {
                    "ball_diameter_mm": args.ball_diameter_mm,
                    "ppmm": ppmm,
                    "guide_grid_row": row,
                    "guide_grid_column": column,
                    "guide_depth_level": depth_level,
                    "background_id": background_id,
                }
                if args.replaces is not None:
                    metadata["replaces_sample_id"] = args.replaces
                record = writer.save(
                    frame,
                    "ball",
                    metadata=metadata,
                )
                counts["ball"] += 1
                target_index += 1
                print(f"saved {record['path']}")
                if replacement_target is not None:
                    break
    finally:
        cv2.destroyAllWindows()
        camera.release()
    print(f"capture complete: ball={counts['ball']}")


if __name__ == "__main__":
    main()
