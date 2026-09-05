"""Validation for calibration dataset schema version 1."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
from PIL import Image
import yaml


SCHEMA_VERSION = 1
MODALITIES = frozenset({"background", "ball", "manual_mask"})
REQUIRED_RECORD_FIELDS = frozenset({
    "schema_version",
    "sample_id",
    "serial",
    "modality",
    "session_id",
    "image_path",
    "label_path",
    "captured_at_utc",
    "shape",
    "split_group",
})
BALL_FIELDS = frozenset({
    "ball_diameter_mm",
    "center_px",
    "radius_px",
    "ppmm",
    "indentation_depth_mm",
})
MANUAL_MASK_FIELDS = frozenset({"annotator", "annotation_revision"})


class DatasetSchemaError(ValueError):
    """Raised when a calibration dataset violates schema version 1."""


@dataclass(frozen=True)
class DatasetSummary:
    serial: str
    dataset_id: str
    record_count: int
    modality_counts: Mapping[str, int]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DatasetSchemaError(message)


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    _require(
        not isinstance(value, bool) and isinstance(value, (int, float)),
        f"{name} must be a number",
    )
    value = float(value)
    _require(np.isfinite(value), f"{name} must be finite")
    if positive:
        _require(value > 0, f"{name} must be positive")
    return value


def _shape(value: Any, name: str, *, channels: Optional[int]) -> Tuple[int, ...]:
    _require(isinstance(value, list), f"{name} must be a list")
    expected_dimensions = 2 if channels is None else 3
    _require(
        len(value) == expected_dimensions
        and all(isinstance(item, int) and not isinstance(item, bool) and item > 0
                for item in value),
        f"{name} must contain {expected_dimensions} positive integers",
    )
    if channels is not None:
        _require(value[2] == channels, f"{name} must have {channels} channels")
    return tuple(value)


def _relative_path(root: Path, value: Any, name: str) -> Path:
    _require(isinstance(value, str) and value, f"{name} must be a non-empty path")
    relative = Path(value)
    _require(not relative.is_absolute(), f"{name} must be relative")
    _require(".." not in relative.parts, f"{name} must not leave the dataset root")
    path = root / relative
    _require(path.is_file(), f"{name} does not exist: {value}")
    return path


def _require_path_prefix(value: str, prefix: Tuple[str, ...], name: str) -> None:
    _require(
        Path(value).parts[:len(prefix)] == prefix,
        f"{name} must be under {'/'.join(prefix)}/",
    )


def _utc_timestamp(value: Any, name: str) -> None:
    _require(isinstance(value, str) and value, f"{name} must be an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DatasetSchemaError(f"{name} must be an ISO timestamp") from exc
    _require(
        parsed.utcoffset() == timezone.utc.utcoffset(parsed),
        f"{name} must use UTC",
    )



def _load_image(path: Path, name: str) -> np.ndarray:
    _require(path.suffix.lower() == ".png", f"{name} must be a PNG")
    try:
        with Image.open(path) as image:
            array = np.asarray(image)
    except (OSError, ValueError) as exc:
        raise DatasetSchemaError(f"cannot read {name}: {path}") from exc
    _require(array.dtype == np.uint8, f"{name} must have dtype uint8")
    return array


def _load_config(root: Path) -> Dict[str, Any]:
    path = root / "dataset.yaml"
    _require(path.is_file(), "dataset.yaml is missing")
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DatasetSchemaError("dataset.yaml is not valid YAML") from exc
    _require(isinstance(config, dict), "dataset.yaml must contain a mapping")
    required = {
        "schema_version", "dataset_id", "serial", "image_shape",
        "colour_space", "dtype", "ppmm",
    }
    missing = sorted(required - config.keys())
    _require(not missing, f"dataset.yaml missing fields: {', '.join(missing)}")
    _require(config["schema_version"] == SCHEMA_VERSION,
             f"unsupported schema_version: {config['schema_version']}")
    _require(isinstance(config["dataset_id"], str) and config["dataset_id"],
             "dataset_id must be a non-empty string")
    _require(isinstance(config["serial"], str) and config["serial"],
             "serial must be a non-empty string")
    _shape(config["image_shape"], "image_shape", channels=3)
    _require(config["colour_space"] == "BGR", "colour_space must be BGR")
    _require(config["dtype"] == "uint8", "dtype must be uint8")
    _number(config["ppmm"], "ppmm", positive=True)
    return config


def _load_manifest(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    _require(path.is_file(), "manifest.jsonl is missing")
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        _require(line.strip() != "", f"manifest line {line_number} is blank")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetSchemaError(
                f"manifest line {line_number} is not valid JSON"
            ) from exc
        _require(
            isinstance(record, dict),
            f"manifest line {line_number} must contain an object",
        )
        yield line_number, record


def _validate_split(
    root: Path,
    modality: str,
    records: Iterable[Mapping[str, Any]],
    split_id: Optional[str] = None,
) -> None:
    split_id = split_id or modality
    _require(
        split_id == modality,
        f"active split ID is invalid for {modality}: {split_id}",
    )
    path = root / "splits" / f"{split_id}.json"
    _require(path.is_file(), f"split file is missing: {path.relative_to(root)}")
    try:
        split = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetSchemaError(f"split file is not valid JSON: {path}") from exc
    _require(isinstance(split, dict), f"split file must contain an object: {path}")
    required = {
        "schema_version", "split_id", "modality", "frozen",
        "created_at_utc", "train", "train_split_groups", "validation",
        "validation_split_groups", "test", "test_split_groups",
    }
    missing = sorted(required - split.keys())
    _require(not missing, f"split file missing fields: {', '.join(missing)}")
    _require(split["schema_version"] == SCHEMA_VERSION,
             f"split file has unsupported schema_version: {path}")
    _require(split["split_id"] == split_id,
             f"split_id does not match filename: {path}")
    _require(split["modality"] == modality,
             f"split modality does not match filename: {path}")
    _require(split["frozen"] is True, f"split is not frozen: {path}")
    _utc_timestamp(split["created_at_utc"], f"{path.name}.created_at_utc")
    records_by_id = {record["sample_id"]: record for record in records}
    assigned = []
    for partition in ("train", "validation", "test"):
        sample_ids = split[partition]
        groups = split[f"{partition}_split_groups"]
        _require(
            isinstance(sample_ids, list)
            and all(isinstance(item, str) and item for item in sample_ids),
            f"{path.name}.{partition} must be a list of sample IDs",
        )
        _require(
            isinstance(groups, list)
            and all(isinstance(item, str) and item for item in groups),
            f"{path.name}.{partition}_split_groups must be a list of groups",
        )
        unknown = sorted(set(sample_ids) - records_by_id.keys())
        _require(not unknown,
                 f"{path.name}.{partition} has unknown samples: {', '.join(unknown)}")
        expected_groups = list(dict.fromkeys(
            records_by_id[sample_id]["split_group"] for sample_id in sample_ids
        ))
        _require(groups == expected_groups,
                 f"{path.name}.{partition}_split_groups do not match records")
        assigned.extend(sample_ids)
    _require(len(assigned) == len(set(assigned)),
             f"split assigns a sample more than once: {path}")
    missing_ids = sorted(records_by_id.keys() - set(assigned))
    _require(not missing_ids,
             f"split omits samples: {', '.join(missing_ids)}")


def _validate_ball(
    record: Mapping[str, Any],
    label_path: Path,
    config: Mapping[str, Any],
    prefix: str,
) -> None:
    missing = sorted(BALL_FIELDS - record.keys())
    _require(not missing, f"{prefix} missing ball fields: {', '.join(missing)}")
    diameter = _number(record["ball_diameter_mm"],
                       f"{prefix}.ball_diameter_mm", positive=True)
    ppmm = _number(record["ppmm"], f"{prefix}.ppmm", positive=True)
    _require(np.isclose(ppmm, float(config["ppmm"])),
             f"{prefix}.ppmm does not match dataset.yaml")
    center = record["center_px"]
    _require(
        isinstance(center, list) and len(center) == 2,
        f"{prefix}.center_px must contain x and y",
    )
    center = np.asarray([
        _number(center[0], f"{prefix}.center_px[0]"),
        _number(center[1], f"{prefix}.center_px[1]"),
    ], dtype=np.float64)
    height, width, _ = config["image_shape"]
    _require(0 <= center[0] < width and 0 <= center[1] < height,
             f"{prefix}.center_px is outside the image")
    radius = _number(record["radius_px"], f"{prefix}.radius_px", positive=True)
    ball_radius = diameter / 2
    contact_radius = radius / ppmm
    _require(contact_radius < ball_radius,
             f"{prefix} ball contact must satisfy 0 < a < R")
    expected_depth = ball_radius - np.sqrt(
        ball_radius ** 2 - contact_radius ** 2
    )
    indentation = _number(
        record["indentation_depth_mm"],
        f"{prefix}.indentation_depth_mm",
        positive=True,
    )
    _require(
        np.isclose(indentation, expected_depth, rtol=1e-5, atol=1e-6),
        f"{prefix}.indentation_depth_mm does not match sphere geometry",
    )
    _require(label_path.suffix.lower() == ".npz",
             f"{prefix}.label_path must be an NPZ")
    try:
        with np.load(label_path, allow_pickle=False) as label:
            required = {
                "ball_diameter_mm", "center_px", "radius_px", "ppmm",
                "indentation_depth_mm",
            }
            _require(required <= set(label.files),
                     f"{prefix} ball label is missing required arrays")
            _require(np.allclose(label["center_px"], center),
                     f"{prefix} label center_px does not match manifest")
            for key, value in (
                ("ball_diameter_mm", diameter),
                ("radius_px", radius),
                ("ppmm", ppmm),
                ("indentation_depth_mm", indentation),
            ):
                _require(np.isclose(float(label[key]), value),
                         f"{prefix} label {key} does not match manifest")
    except DatasetSchemaError:
        raise
    except (OSError, ValueError) as exc:
        raise DatasetSchemaError(f"cannot read ball label: {label_path}") from exc


def validate_dataset(root: Path) -> DatasetSummary:
    """Validate a calibration root and return record counts."""
    root = Path(root)
    _require(root.is_dir(), f"dataset root does not exist: {root}")
    config = _load_config(root)
    expected_shape = _shape(config["image_shape"], "image_shape", channels=3)
    seen_ids = set()
    seen_images = set()
    seen_labels = set()
    counts: Counter[str] = Counter()
    records_by_modality = {modality: [] for modality in MODALITIES}

    for line_number, record in _load_manifest(root / "manifest.jsonl"):
        prefix = f"manifest line {line_number}"
        missing = sorted(REQUIRED_RECORD_FIELDS - record.keys())
        _require(not missing, f"{prefix} missing fields: {', '.join(missing)}")
        _require(record["schema_version"] == SCHEMA_VERSION,
                 f"{prefix} has unsupported schema_version")
        sample_id = record["sample_id"]
        _require(isinstance(sample_id, str) and sample_id,
                 f"{prefix}.sample_id must be a non-empty string")
        _require(sample_id not in seen_ids, f"duplicate sample_id: {sample_id}")
        seen_ids.add(sample_id)
        _require(record["serial"] == config["serial"],
                 f"{prefix}.serial does not match dataset.yaml")
        modality = record["modality"]
        _require(modality in MODALITIES, f"{prefix}.modality is invalid")
        for field in ("session_id", "split_group"):
            _require(isinstance(record[field], str) and record[field],
                     f"{prefix}.{field} must be a non-empty string")
        _utc_timestamp(record["captured_at_utc"],
                       f"{prefix}.captured_at_utc")
        record_shape = _shape(record["shape"], f"{prefix}.shape", channels=3)
        _require(record_shape == expected_shape,
                 f"{prefix}.shape does not match dataset.yaml")
        image_path = _relative_path(root, record["image_path"],
                                    f"{prefix}.image_path")
        image_prefix = {
            "background": ("background",),
            "ball": ("ball", "images"),
            "manual_mask": ("manual_mask", "images"),
        }[modality]
        _require_path_prefix(record["image_path"], image_prefix,
                             f"{prefix}.image_path")
        _require(image_path not in seen_images,
                 f"duplicate image_path: {record['image_path']}")
        seen_images.add(image_path)
        image = _load_image(image_path, f"{prefix}.image_path")
        _require(image.shape == expected_shape,
                 f"{prefix}.image_path shape does not match manifest")

        label_value = record["label_path"]
        if modality == "background":
            _require(label_value is None,
                     f"{prefix}.label_path must be null for background")
        else:
            label_path = _relative_path(root, label_value,
                                        f"{prefix}.label_path")
            label_prefix = (
                ("ball", "labels")
                if modality == "ball"
                else ("manual_mask", "masks")
            )
            _require_path_prefix(label_value, label_prefix,
                                 f"{prefix}.label_path")
            _require(label_path not in seen_labels,
                     f"duplicate label_path: {label_value}")
            seen_labels.add(label_path)
            if modality == "ball":
                _validate_ball(record, label_path, config, prefix)
            else:
                missing = sorted(MANUAL_MASK_FIELDS - record.keys())
                _require(
                    not missing,
                    f"{prefix} missing manual-mask fields: {', '.join(missing)}",
                )
                _require(isinstance(record["annotator"], str)
                         and record["annotator"],
                         f"{prefix}.annotator must be a non-empty string")
                revision = record["annotation_revision"]
                _require(isinstance(revision, int) and not isinstance(revision, bool)
                         and revision >= 1,
                         f"{prefix}.annotation_revision must be >= 1")
                mask = _load_image(label_path, f"{prefix}.label_path")
                _require(mask.shape == expected_shape[:2],
                         f"{prefix} mask shape does not match its image")
                _require(set(np.unique(mask)).issubset({0, 255}),
                         f"{prefix} mask values must be 0 or 255")
        counts[modality] += 1
        records_by_modality[modality].append(record)

    _require(seen_ids, "manifest.jsonl must contain at least one record")
    indexed_files = seen_images | seen_labels
    data_files = set()
    for pattern in (
        "background/**/*.png",
        "ball/images/**/*.png",
        "ball/labels/**/*.npz",
        "manual_mask/images/**/*.png",
        "manual_mask/masks/**/*.png",
    ):
        data_files.update(path for path in root.glob(pattern) if path.is_file())
    data_files.discard(root / "background/reference.png")
    orphaned = sorted(str(path.relative_to(root))
                      for path in data_files - indexed_files)
    _require(not orphaned,
             f"data files missing from manifest: {', '.join(orphaned)}")
    for modality in ("background", "ball", "manual_mask"):
        if records_by_modality[modality]:
            active_splits = config.get("active_splits", {})
            _require(isinstance(active_splits, dict),
                     "dataset.yaml.active_splits must be a mapping")
            _validate_split(
                root,
                modality,
                records_by_modality[modality],
                active_splits.get(modality),
            )
    return DatasetSummary(
        serial=config["serial"],
        dataset_id=config["dataset_id"],
        record_count=len(seen_ids),
        modality_counts=dict(counts),
    )
