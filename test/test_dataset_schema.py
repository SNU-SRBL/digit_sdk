import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import yaml

from calibration.dataset_schema import DatasetSchemaError, validate_dataset


def _write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)



def _make_dataset(root: Path):
    shape = [8, 10, 3]
    ppmm = 20.0
    config = {
        "schema_version": 1,
        "dataset_id": "DTEST",
        "serial": "DTEST",
        "image_shape": shape,
        "colour_space": "BGR",
        "dtype": "uint8",
        "ppmm": ppmm,
    }
    (root / "dataset.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    image = np.zeros(shape, dtype=np.uint8)
    timestamp = "2026-07-16T00:00:00Z"
    common = {
        "schema_version": 1,
        "serial": "DTEST",
        "session_id": "session-1",
        "captured_at_utc": timestamp,
        "shape": shape,
        "split_group": "contact-1",
    }

    _write_png(root / "background/session-1/background.png", image)
    background = {
        **common,
        "sample_id": "background-1",
        "modality": "background",
        "image_path": "background/session-1/background.png",
        "label_path": None,
    }

    ball_image_path = root / "ball/images/session-1/ball-1.png"
    ball_label_path = root / "ball/labels/session-1/ball-1.npz"
    _write_png(ball_image_path, image)
    ball_label_path.parent.mkdir(parents=True, exist_ok=True)
    diameter = 6.0
    radius_px = 20.0
    contact_radius_mm = radius_px / ppmm
    ball_radius_mm = diameter / 2
    indentation = ball_radius_mm - np.sqrt(
        ball_radius_mm ** 2 - contact_radius_mm ** 2
    )
    np.savez(
        ball_label_path,
        ball_diameter_mm=diameter,
        center_px=np.array([5.0, 4.0]),
        radius_px=radius_px,
        ppmm=ppmm,
        indentation_depth_mm=indentation,
    )
    ball = {
        **common,
        "sample_id": "ball-1",
        "modality": "ball",
        "image_path": "ball/images/session-1/ball-1.png",
        "label_path": "ball/labels/session-1/ball-1.npz",
        "ball_diameter_mm": diameter,
        "center_px": [5.0, 4.0],
        "radius_px": radius_px,
        "ppmm": ppmm,
        "indentation_depth_mm": indentation,
    }

    mask_image_path = root / "manual_mask/images/session-1/mask-1.png"
    mask_path = root / "manual_mask/masks/session-1/mask-1.png"
    _write_png(mask_image_path, image)
    mask = np.zeros(shape[:2], dtype=np.uint8)
    mask[2:4, 3:6] = 255
    _write_png(mask_path, mask)
    manual_mask = {
        **common,
        "sample_id": "mask-1",
        "modality": "manual_mask",
        "image_path": "manual_mask/images/session-1/mask-1.png",
        "label_path": "manual_mask/masks/session-1/mask-1.png",
        "annotator": "tester",
        "annotation_revision": 1,
    }
    records = [background, ball, manual_mask]
    _write_manifest(root, records)
    splits = root / "splits"
    splits.mkdir()
    for modality, train, test in (
        ("background", ["background-1"], []),
        ("ball", ["ball-1"], []),
        ("manual_mask", [], ["mask-1"]),
    ):
        split = {
            "schema_version": 1,
            "split_id": modality,
            "modality": modality,
            "frozen": True,
            "created_at_utc": timestamp,
            "train": train,
            "train_split_groups": ["contact-1"] if train else [],
            "validation": [],
            "validation_split_groups": [],
            "test": test,
            "test_split_groups": ["contact-1"] if test else [],
        }
        (splits / f"{modality}.json").write_text(
            json.dumps(split), encoding="utf-8"
        )
    return records


def _write_manifest(root: Path, records) -> None:
    (root / "manifest.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_validate_complete_dataset(tmp_path):
    _make_dataset(tmp_path)

    summary = validate_dataset(tmp_path)

    assert summary.serial == "DTEST"
    assert summary.dataset_id == "DTEST"
    assert summary.record_count == 3
    assert summary.modality_counts == {
        "background": 1,
        "ball": 1,
        "manual_mask": 1,
    }


def test_rejects_non_binary_manual_mask(tmp_path):
    _make_dataset(tmp_path)
    mask_path = tmp_path / "manual_mask/masks/session-1/mask-1.png"
    _write_png(mask_path, np.full((8, 10), 127, dtype=np.uint8))

    with pytest.raises(DatasetSchemaError, match="mask values"):
        validate_dataset(tmp_path)


def test_rejects_ball_contact_beyond_hemisphere(tmp_path):
    records = _make_dataset(tmp_path)
    records[1]["radius_px"] = 100.0
    _write_manifest(tmp_path, records)

    with pytest.raises(DatasetSchemaError, match="0 < a < R"):
        validate_dataset(tmp_path)


def test_rejects_duplicate_sample_id(tmp_path):
    records = _make_dataset(tmp_path)
    records[1]["sample_id"] = records[0]["sample_id"]
    _write_manifest(tmp_path, records)

    with pytest.raises(DatasetSchemaError, match="duplicate sample_id"):
        validate_dataset(tmp_path)


def test_rejects_path_outside_dataset_root(tmp_path):
    records = _make_dataset(tmp_path)
    records[0]["image_path"] = "../outside.png"
    _write_manifest(tmp_path, records)

    with pytest.raises(DatasetSchemaError, match="must not leave"):
        validate_dataset(tmp_path)


def test_rejects_non_utc_capture_time(tmp_path):
    records = _make_dataset(tmp_path)
    records[0]["captured_at_utc"] = "2026-07-16T09:00:00+09:00"
    _write_manifest(tmp_path, records)

    with pytest.raises(DatasetSchemaError, match="must use UTC"):
        validate_dataset(tmp_path)



def test_rejects_unindexed_data_file(tmp_path):
    _make_dataset(tmp_path)
    _write_png(
        tmp_path / "background/session-1/orphan.png",
        np.zeros((8, 10, 3), dtype=np.uint8),
    )

    with pytest.raises(DatasetSchemaError, match="missing from manifest"):
        validate_dataset(tmp_path)


def test_rejects_sample_missing_from_split(tmp_path):
    _make_dataset(tmp_path)
    split_path = tmp_path / "splits/manual_mask.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    split["test"] = []
    split["test_split_groups"] = []
    split_path.write_text(json.dumps(split), encoding="utf-8")

    with pytest.raises(DatasetSchemaError, match="split omits samples"):
        validate_dataset(tmp_path)


def test_rejects_alternate_active_split_name(tmp_path):
    _make_dataset(tmp_path)
    split_path = tmp_path / "splits/manual_mask.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    split["split_id"] = "manual_mask_alternate"
    (tmp_path / "splits/manual_mask_alternate.json").write_text(
        json.dumps(split), encoding="utf-8"
    )
    config_path = tmp_path / "dataset.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["active_splits"] = {"manual_mask": "manual_mask_alternate"}
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(DatasetSchemaError, match="active split ID is invalid"):
        validate_dataset(tmp_path)
