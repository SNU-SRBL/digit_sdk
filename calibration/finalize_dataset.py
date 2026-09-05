"""Build one clean calibration dataset from annotated inbox captures."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Dict, Iterable, List, Mapping

import yaml

from calibration.ball_staging import (
    ball_partition,
    validate_ball_replacements,
)
from calibration.dataset_schema import validate_dataset
from digit_sdk.utils import load_config


PARTITIONS = ("train", "validation", "test")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _split_groups(records: Iterable[Mapping[str, Any]]) -> List[str]:
    return list(dict.fromkeys(record["split_group"] for record in records))


def manual_partitions(
    rows: List[Mapping[str, Any]], validation_count: int, test_count: int
) -> Dict[str, str]:
    """Assign independent contacts deterministically at finalization."""
    if validation_count < 1 or test_count < 1:
        raise ValueError("manual validation and test counts must be positive")
    groups = sorted({row["split_group"] for row in rows})
    if len(groups) <= validation_count + test_count:
        raise ValueError(
            "manual contact groups must exceed validation_count + test_count"
        )
    ranked = sorted(
        groups,
        key=lambda group: hashlib.sha256(group.encode("utf-8")).hexdigest(),
    )
    group_partitions = {}
    for index, group in enumerate(ranked):
        if index < test_count:
            partition = "test"
        elif index < test_count + validation_count:
            partition = "validation"
        else:
            partition = "train"
        group_partitions[group] = partition
    return {
        row["sample_id"]: group_partitions[row["split_group"]] for row in rows
    }


def background_partitions(
    rows: List[Mapping[str, Any]], validation_count: int, test_count: int
) -> Dict[str, str]:
    """Split only fresh no-contact training samples."""
    if validation_count < 1 or test_count < 1:
        raise ValueError("background validation and test counts must be positive")
    eligible = [
        row for row in rows if row.get("training_eligible", True)
    ]
    if len(eligible) <= validation_count + test_count:
        raise ValueError(
            "background samples must exceed validation_count + test_count"
        )
    ranked = sorted(
        eligible,
        key=lambda row: hashlib.sha256(
            row["sample_id"].encode("utf-8")
        ).hexdigest(),
    )
    assignments = {
        row["sample_id"]: (
            "test"
            if index < test_count
            else "validation"
            if index < test_count + validation_count
            else "train"
        )
        for index, row in enumerate(ranked)
    }
    for row in rows:
        assignments.setdefault(row["sample_id"], "train")
    return assignments


def _split_document(
    split_id: str,
    modality: str,
    records: List[Mapping[str, Any]],
    assignments: Mapping[str, str],
) -> Dict[str, Any]:
    document = {
        "schema_version": 1,
        "split_id": split_id,
        "modality": modality,
        "frozen": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    for partition in PARTITIONS:
        selected = [
            record for record in records
            if assignments[record["sample_id"]] == partition
        ]
        document[partition] = [record["sample_id"] for record in selected]
        document[f"{partition}_split_groups"] = _split_groups(selected)
    return document


def finalize_dataset(
    sensors_root: Path,
    serial: str,
    output_root: Path,
    *,
    manual_validation_count: int = 20,
    manual_test_count: int = 20,
    background_validation_count: int = 10,
    background_test_count: int = 10,
) -> Dict[str, int]:
    """Copy annotated inbox data into one immutable dataset."""
    sensors_root = Path(sensors_root)
    sensor_root = sensors_root / serial
    inbox = sensor_root / "calibration/inbox"
    ball_root = inbox / "ball"
    manual_root = inbox / "manual_mask"
    background_root = inbox / "background"
    output_root = Path(output_root)
    if output_root.exists():
        raise ValueError(f"output root already exists: {output_root}")

    ball_rows = validate_ball_replacements(
        _load_jsonl(ball_root / "metadata.jsonl")
    )
    manual_rows = _load_jsonl(manual_root / "metadata.jsonl")
    touches = [row for row in manual_rows if row.get("kind") == "touch"]
    if not ball_rows or not touches:
        raise ValueError("annotated ball and manual touch data are required")
    if any(not row.get("label_path") for row in ball_rows):
        raise ValueError("all accepted ball samples must be annotated")
    if any(
        not row.get("label_path")
        or not row.get("annotator")
        or int(row.get("annotation_revision", 0)) < 1
        for row in touches
    ):
        raise ValueError("all manual contacts must be annotated")

    background = json.loads(
        (background_root / "metadata.json").read_text(encoding="utf-8")
    )
    config = load_config(serial=serial, sensors_root=str(sensors_root))
    shape = background.get("shape")
    all_rows = ball_rows + touches
    partitioned = [row["sample_id"] for row in all_rows if "partition" in row]
    if partitioned:
        raise ValueError(
            "inbox records must not contain partitions: "
            + ", ".join(partitioned)
        )
    if any(
        row.get("serial") != serial
        or row.get("shape") != shape
        or row.get("dtype") != "uint8"
        or row.get("colour_space") != "BGR"
        for row in all_rows
    ):
        raise ValueError("inbox captures do not share sensor, shape, or image format")
    if background.get("serial") != serial:
        raise ValueError("background reference does not match sensor")

    manual_assignment = manual_partitions(
        touches, manual_validation_count, manual_test_count
    )
    ball_assignment = {
        row["sample_id"]: ball_partition(row) for row in ball_rows
    }

    output_root.mkdir(parents=True)
    dataset_config = {
        "schema_version": 1,
        "dataset_id": serial,
        "serial": serial,
        "image_shape": shape,
        "colour_space": "BGR",
        "dtype": "uint8",
        "ppmm": float(config["ppmm"]),
        "active_splits": {
            "background": "background",
            "ball": "ball",
            "manual_mask": "manual_mask",
        },
    }
    (output_root / "dataset.yaml").write_text(
        yaml.safe_dump(dataset_config, sort_keys=False), encoding="utf-8"
    )
    output_records = []

    reference_relative = Path("background/reference.png")
    _copy(
        background_root / background["path"],
        output_root / reference_relative,
    )
    for index, source in enumerate(background["source_frames"]):
        relative = Path("background") / source["path"]
        _copy(
            background_root / source["path"],
            output_root / relative,
        )
        sample_id = (
            f"{serial}_background_{background['session_id']}_{index:03d}"
        )
        output_records.append({
            "schema_version": 1,
            "sample_id": sample_id,
            "serial": serial,
            "modality": "background",
            "session_id": background["session_id"],
            "image_path": relative.as_posix(),
            "label_path": None,
            "captured_at_utc": source["captured_at_utc"],
            "shape": shape,
            "split_group": sample_id,
            "training_eligible": True,
        })
    (output_root / "background/metadata.json").write_text(
        json.dumps(background, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for row in ball_rows:
        image_relative = (
            Path("ball/images") / row["session_id"]
            / f"{row['sample_id']}.png"
        )
        label_relative = (
            Path("ball/labels") / row["session_id"]
            / f"{row['sample_id']}.npz"
        )
        _copy(
            ball_root / row["path"], output_root / image_relative,
        )
        _copy(
            ball_root / row["label_path"], output_root / label_relative,
        )
        record = {
            "schema_version": 1,
            "sample_id": row["sample_id"],
            "serial": serial,
            "modality": "ball",
            "session_id": row["session_id"],
            "image_path": image_relative.as_posix(),
            "label_path": label_relative.as_posix(),
            "captured_at_utc": row["captured_at_utc"],
            "shape": shape,
            "split_group": row["split_group"],
            "ball_diameter_mm": row["ball_diameter_mm"],
            "center_px": row["center_px"],
            "radius_px": row["radius_px"],
            "ppmm": row["ppmm"],
            "indentation_depth_mm": row["indentation_depth_mm"],
            "background_id": background["sample_id"],
        }
        if row.get("replaces_sample_id"):
            record["replaces_sample_id"] = row["replaces_sample_id"]
        output_records.append(record)

    manual_records = []
    for row in touches:
        image_relative = (
            Path("manual_mask/images") / row["session_id"]
            / f"{row['sample_id']}.png"
        )
        mask_relative = (
            Path("manual_mask/masks") / row["session_id"]
            / f"{row['sample_id']}.png"
        )
        _copy(
            manual_root / row["path"], output_root / image_relative,
        )
        _copy(
            manual_root / row["label_path"], output_root / mask_relative,
        )
        record = {
            "schema_version": 1,
            "sample_id": row["sample_id"],
            "serial": serial,
            "modality": "manual_mask",
            "session_id": row["session_id"],
            "image_path": image_relative.as_posix(),
            "label_path": mask_relative.as_posix(),
            "captured_at_utc": row["captured_at_utc"],
            "shape": shape,
            "split_group": row["split_group"],
            "annotator": row["annotator"],
            "annotation_revision": row["annotation_revision"],
        }
        output_records.append(record)
        manual_records.append(record)

    identifiers = [record["sample_id"] for record in output_records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("inbox sample identifiers overlap")
    (output_root / "manifest.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n"
                for record in output_records),
        encoding="utf-8",
    )
    split_root = output_root / "splits"
    split_root.mkdir()
    ball_records = [
        record for record in output_records if record["modality"] == "ball"
    ]
    background_records = [
        record for record in output_records
        if record["modality"] == "background"
    ]
    background_assignment = background_partitions(
        background_records,
        background_validation_count,
        background_test_count,
    )
    for filename, document in (
        ("background.json", _split_document(
            "background", "background", background_records,
            background_assignment,
        )),
        ("ball.json", _split_document(
            "ball", "ball", ball_records, ball_assignment
        )),
        ("manual_mask.json", _split_document(
            "manual_mask", "manual_mask", manual_records,
            manual_assignment,
        )),
    ):
        (split_root / filename).write_text(
            json.dumps(document, indent=2) + "\n", encoding="utf-8"
        )

    summary = validate_dataset(output_root)
    return {
        "records": summary.record_count,
        "background": summary.modality_counts.get("background", 0),
        "ball": summary.modality_counts.get("ball", 0),
        "manual_mask": summary.modality_counts.get("manual_mask", 0),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--sensors-root", type=Path, default=Path("sensors"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--manual-validation-count", type=int, default=20)
    parser.add_argument("--manual-test-count", type=int, default=20)
    parser.add_argument("--background-validation-count", type=int, default=10)
    parser.add_argument("--background-test-count", type=int, default=10)
    args = parser.parse_args(argv)
    output_root = args.output_root or (
        args.sensors_root / args.serial / "calibration_next"
    )
    result = finalize_dataset(
        args.sensors_root,
        args.serial,
        output_root,
        manual_validation_count=args.manual_validation_count,
        manual_test_count=args.manual_test_count,
        background_validation_count=args.background_validation_count,
        background_test_count=args.background_test_count,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
