"""Promote a validated calibration candidate while preserving its inbox."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from calibration.dataset_schema import validate_dataset


def promote_dataset(sensor_root: Path, candidate_root: Path) -> dict:
    """Replace the active canonical dataset and retain raw inbox data."""
    sensor_root = Path(sensor_root).resolve()
    candidate_root = Path(candidate_root).resolve()
    active_root = sensor_root / "calibration"
    backup_root = sensor_root / ".calibration_backup"

    if candidate_root.parent != sensor_root:
        raise ValueError("candidate root must be directly under the sensor root")
    if candidate_root == active_root:
        raise ValueError("candidate root is already active")
    if backup_root.exists():
        raise ValueError(f"promotion backup already exists: {backup_root}")
    if not (active_root / "inbox").is_dir():
        raise ValueError(f"calibration inbox is missing: {active_root / 'inbox'}")
    if (candidate_root / "inbox").exists():
        raise ValueError("candidate dataset must not contain an inbox")

    candidate = validate_dataset(candidate_root)
    if candidate.serial != sensor_root.name:
        raise ValueError(
            f"candidate serial {candidate.serial} does not match {sensor_root.name}"
        )
    if candidate.dataset_id != sensor_root.name:
        raise ValueError(
            f"candidate dataset_id must equal serial {sensor_root.name}"
        )

    active_root.rename(backup_root)
    try:
        candidate_root.rename(active_root)
        (backup_root / "inbox").rename(active_root / "inbox")
        promoted = validate_dataset(active_root)
    except Exception:
        if (
            (active_root / "inbox").exists()
            and backup_root.exists()
            and not (backup_root / "inbox").exists()
        ):
            (active_root / "inbox").rename(backup_root / "inbox")
        if active_root.exists() and not candidate_root.exists():
            active_root.rename(candidate_root)
        if backup_root.exists() and not active_root.exists():
            backup_root.rename(active_root)
        raise
    else:
        shutil.rmtree(backup_root)

    return {
        "serial": promoted.serial,
        "records": promoted.record_count,
        "background": promoted.modality_counts.get("background", 0),
        "ball": promoted.modality_counts.get("ball", 0),
        "manual_mask": promoted.modality_counts.get("manual_mask", 0),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--sensors-root", type=Path, default=Path("sensors"))
    parser.add_argument("--candidate-root", type=Path)
    args = parser.parse_args(argv)

    sensor_root = args.sensors_root / args.serial
    candidate_root = args.candidate_root or sensor_root / "calibration_next"
    result = promote_dataset(sensor_root, candidate_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
