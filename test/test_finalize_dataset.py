import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from calibration.collect_background import save_background_reference
from calibration.finalize_dataset import (
    background_partitions,
    finalize_dataset,
    manual_partitions,
)
from calibration.dataset_schema import validate_dataset
from calibration.promote_dataset import promote_dataset
from calibration.tactile_transformer.data import (
    BackgroundSupportDataset,
    ManualSupportDataset,
)


SHAPE = (8, 10, 3)


def test_manual_split_keeps_physical_groups_together():
    rows = [
        {"sample_id": "a1", "split_group": "a"},
        {"sample_id": "a2", "split_group": "a"},
        {"sample_id": "b", "split_group": "b"},
        {"sample_id": "c", "split_group": "c"},
    ]

    assignment = manual_partitions(rows, validation_count=1, test_count=1)

    assert assignment["a1"] == assignment["a2"]


def test_background_split_has_exact_holdout_counts():
    rows = [
        {"sample_id": f"background-{index}", "training_eligible": True}
        for index in range(60)
    ]

    assignment = background_partitions(
        rows, validation_count=10, test_count=10
    )

    assert list(assignment.values()).count("validation") == 10
    assert list(assignment.values()).count("test") == 10
    assert sum(
        assignment[row["sample_id"]] == "train"
        for row in rows if row["training_eligible"]
    ) == 40


def _png(path: Path, value: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full(SHAPE, value, dtype=np.uint8))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mask(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros(SHAPE[:2], dtype=np.uint8)
    image[2:6, 3:7] = 255
    cv2.imwrite(str(path), image)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _label(path: Path, diameter=10.0, radius=2.0, ppmm=2.0) -> str:
    depth = diameter / 2 - np.sqrt((diameter / 2) ** 2 - (radius / ppmm) ** 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        ball_diameter_mm=diameter,
        center_px=np.array([5.0, 4.0]),
        radius_px=radius,
        ppmm=ppmm,
        indentation_depth_mm=depth,
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_finalizes_annotated_partition_free_inboxes(tmp_path):
    sensors = tmp_path / "sensors"
    sensor = sensors / "DTEST"
    inbox = sensor / "calibration/inbox"
    sensor.mkdir(parents=True)
    (sensor / "DTEST.yaml").write_text(
        yaml.safe_dump({
            "device_type": "DIGIT", "device_serial": "DTEST",
            "ppmm": 2.0, "imgh": 10, "imgw": 8,
            "raw_imgh": 10, "raw_imgw": 8, "framerate": 60,
        }),
        encoding="utf-8",
    )
    save_background_reference(
        inbox / "background",
        [np.zeros(SHAPE, dtype=np.uint8)] * 3,
        serial="DTEST",
        capture_session="reference",
        interval_seconds=1.0,
    )

    ball_root = inbox / "ball"
    ball_rows = []
    for index, depth_level in enumerate((1, 2), start=1):
        sample_id = f"ball-{index}"
        image_path = Path("images/ball") / f"{sample_id}.png"
        label_path = Path("labels/ball") / f"{sample_id}.npz"
        ball_rows.append({
            "sample_id": sample_id, "kind": "ball", "serial": "DTEST",
            "path": image_path.as_posix(),
            "label_path": label_path.as_posix(),
            "captured_at_utc": f"2026-07-20T00:00:0{index}+00:00",
            "session_id": "ball", "split_group": sample_id,
            "shape": list(SHAPE), "dtype": "uint8", "colour_space": "BGR",
            "image_sha256": _png(ball_root / image_path, 20 + index),
            "label_sha256": _label(ball_root / label_path),
            "ball_diameter_mm": 10.0, "center_px": [5.0, 4.0],
            "radius_px": 2.0, "ppmm": 2.0,
            "indentation_depth_mm": float(
                5.0 - np.sqrt(25.0 - 1.0)
            ),
            "guide_grid_row": 0, "guide_grid_column": 0,
            "guide_depth_level": depth_level,
            "annotator": "tester", "annotation_revision": 1,
        })
    (ball_root / "metadata.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in ball_rows),
        encoding="utf-8",
    )

    manual_root = inbox / "manual_mask"
    manual_rows = []
    for index in range(3):
        sample_id = f"touch-{index}"
        image_path = Path("touches/manual") / f"{sample_id}.png"
        label_path = Path("masks/manual") / f"{sample_id}.png"
        manual_rows.append({
            "sample_id": sample_id, "kind": "touch", "serial": "DTEST",
            "path": image_path.as_posix(), "label_path": label_path.as_posix(),
            "captured_at_utc": f"2026-07-20T00:01:0{index + 1}+00:00",
            "session_id": "manual", "split_group": sample_id,
            "shape": list(SHAPE), "dtype": "uint8", "colour_space": "BGR",
            "image_sha256": _png(manual_root / image_path, 40 + index),
            "label_sha256": _mask(manual_root / label_path),
            "annotator": "tester", "annotation_revision": 1,
        })
    (manual_root / "metadata.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in manual_rows),
        encoding="utf-8",
    )

    output = sensor / "calibration_next"
    result = finalize_dataset(
        sensors, "DTEST", output,
        manual_validation_count=1, manual_test_count=1,
        background_validation_count=1, background_test_count=1,
    )

    assert result == {"records": 8, "background": 3, "ball": 2,
                      "manual_mask": 3}
    assert validate_dataset(output).dataset_id == "DTEST"
    manifest = [json.loads(line) for line in
                (output / "manifest.jsonl").read_text().splitlines()]
    assert not any("partition" in record or "capture_partition" in record
                   for record in manifest)
    manual_split = json.loads(
        (output / "splits/manual_mask.json").read_text()
    )
    assert {key: len(manual_split[key]) for key in
            ("train", "validation", "test")} == {
                "train": 1, "validation": 1, "test": 1,
            }
    background_split = json.loads(
        (output / "splits/background.json").read_text()
    )
    assert {key: len(background_split[key]) for key in
            ("validation", "test")} == {"validation": 1, "test": 1}
    assert len(ManualSupportDataset(output, "train")) == 1
    assert len(BackgroundSupportDataset(output, "train")) == 1
    background_records = [
        record for record in manifest if record["modality"] == "background"
    ]
    assert sum(
        record["training_eligible"] for record in background_records
    ) == 3
    assert len(background_records) == 3
    assert (output / "background/reference.png").is_file()
    assert not any("background_role" in record for record in background_records)

    promoted = promote_dataset(sensor, output)

    assert promoted == {
        "serial": "DTEST", "records": 8, "background": 3,
        "ball": 2, "manual_mask": 3,
    }
    assert not output.exists()
    assert (sensor / "calibration/inbox").is_dir()
    assert validate_dataset(sensor / "calibration").dataset_id == "DTEST"
