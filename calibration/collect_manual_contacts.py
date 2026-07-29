#!/usr/bin/env python3
"""Capture partition-free staged real-contact data at 30 Hz."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calibration.capture_session import (
    CollectionCamera,
    StagedCaptureWriter,
    session_id,
)


KINDS = ("touch",)


def default_output_root(sensors_root, serial):
    return Path(sensors_root) / serial / "calibration/inbox/manual_mask"


def staged_counts(output_root, serial):
    metadata = Path(output_root) / "metadata.jsonl"
    counts = Counter()
    if not metadata.is_file():
        return counts
    for line in metadata.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("serial") == serial and record.get("kind") in KINDS:
            counts[record["kind"]] += 1
    return counts


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True, help="DIGIT serial under sensors/")
    parser.add_argument("--sensors-root", default="sensors")
    parser.add_argument("--session-id", help="Optional capture-session identifier")
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Defaults to sensors/<serial>/calibration/inbox/manual_mask",
    )
    parser.add_argument("--warmup-frames", type=int, default=30)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    capture_session = session_id(args.session_id)
    output_root = (
        args.output_root
        or default_output_root(args.sensors_root, args.serial)
    ).resolve()
    frozen_test_root = (REPO_ROOT / "fix" / "raw").resolve()
    if output_root == frozen_test_root or frozen_test_root in output_root.parents:
        raise ValueError("output root must not be the frozen fix/raw test tree")

    writer = StagedCaptureWriter(
        output_root,
        serial=args.serial,
        capture_session=capture_session,
    )
    camera = CollectionCamera(args.serial, Path(args.sensors_root), args.warmup_frames)
    existing_counts = staged_counts(output_root, args.serial)
    counts = Counter()
    waiting_for_camera = False
    camera.connect()
    try:
        print(
            "t: contact | q: quit | "
            f"existing touch={existing_counts['touch']}"
        )
        while True:
            frame = camera.read()
            if frame is None:
                if not waiting_for_camera:
                    print("camera unavailable; waiting for serial reconnect...")
                    waiting_for_camera = True
                if cv2.waitKey(50) & 0xFF == ord("q"):
                    break
                continue
            if waiting_for_camera:
                print("camera reconnected; collection resumed")
                waiting_for_camera = False
            preview = frame.copy()
            status = "t contact | q quit"
            cv2.putText(
                preview,
                status,
                (8, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 0),
                1,
            )
            total_touch = existing_counts["touch"] + counts["touch"]
            cv2.putText(
                preview,
                (
                    f"total t:{total_touch} | session t:{counts['touch']}"
                ),
                (8, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 255),
                1,
            )
            cv2.imshow("DIGIT manual-contact capture", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            kind = {
                ord("t"): "touch",
            }.get(key)
            if kind:
                record = writer.save(frame, kind)
                counts[kind] += 1
                print(
                    f"saved {record['path']} | total touch="
                    f"{existing_counts['touch'] + counts['touch']}"
                )
    finally:
        cv2.destroyAllWindows()
        camera.release()

    print(
        "capture complete: session "
        + ", ".join(f"{kind}={counts[kind]}" for kind in KINDS)
        + " | total "
        + ", ".join(
            f"{kind}={existing_counts[kind] + counts[kind]}" for kind in KINDS
        )
    )


if __name__ == "__main__":
    main()
