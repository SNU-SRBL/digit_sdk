"""Evaluate a selected decoder once on its held-out test split."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import ConcatDataset, DataLoader

from calibration.dataset_schema import validate_dataset
from calibration.tactile_transformer.data import (
    BallDepthDataset,
    BackgroundSupportDataset,
    ManualSupportDataset,
)
from calibration.tactile_transformer.model import (
    BASE_REVISION,
    TactileDPT,
    freeze_encoder,
    load_base,
    resolve_base,
)
from calibration.tactile_transformer.train import (
    METHOD,
    _load_decoder,
    evaluate_ball,
    evaluate_background,
    evaluate_manual,
)



def _write_json_once(path: Path, document: dict) -> None:
    """Atomically create a result without allowing test-set reevaluation."""
    if path.exists():
        raise FileExistsError(f"frozen test result already exists: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"incomplete test result exists: {temporary}")
    with temporary.open("x") as stream:
        json.dump(document, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _load_selection(path: Path, serial: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"select the validation seed first: {path}")
    selection = json.loads(path.read_text())
    if selection.get("serial") != serial:
        raise ValueError("selection serial does not match requested sensor")
    if selection.get("method") != METHOD:
        raise ValueError("selection method is not the canonical method")
    if not selection.get("configuration_frozen"):
        raise ValueError("selection configuration is not frozen")
    if selection.get("test_evaluated"):
        raise RuntimeError("held-out test split was already evaluated")
    return selection


def _verify_selection(
    selection: dict,
    model_root: Path,
    dataset_id: str,
) -> Path:
    if selection.get("dataset_id") != dataset_id:
        raise ValueError("selection dataset does not match active calibration")
    if selection.get("base", {}).get("revision") != BASE_REVISION:
        raise ValueError("selection uses an unsupported encoder revision")
    decoder = model_root / selection["selected"]["decoder"]
    if not decoder.is_file():
        raise FileNotFoundError(decoder)
    return decoder


def _mark_selection_tested(
    selection_path: Path,
    selection: dict,
    model_root: Path,
    test_path: Path,
) -> None:
    selection["test_evaluated"] = True
    selection["test_result"] = test_path.relative_to(model_root).as_posix()
    temporary = selection_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, selection_path)


def _is_eligible(ball: dict, manual: dict, background: dict) -> bool:
    return (
        manual["missed_contacts"] == 0
        and ball["contact_mae_mm"] <= 0.15
        and background["max_depth_mm"] < 0.1
    )


def _decoder_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _test_identity(selection: dict, model_root: Path) -> dict:
    selected = selection["selected"]
    decoder = model_root / selected["decoder"]
    if not decoder.is_file():
        raise FileNotFoundError(decoder)
    return {
        "serial": selection["serial"],
        "method": selection["method"],
        "dataset_id": selection["dataset_id"],
        "selected_seed": int(selected["seed"]),
        "selected_decoder": selected["decoder"],
        "decoder_sha256": _decoder_sha256(decoder),
        "ball_split_id": selection["ball_split_id"],
        "manual_mask_split_id": selection["manual_mask_split_id"],
        "maximum_depth_mm": float(selection["maximum_depth_mm"]),
        "contact_threshold_mm": float(selection["contact_threshold_mm"]),
    }


def evaluate_or_load(
    serial: str,
    sensors_root: Path,
    device: torch.device,
    batch_size: int,
    workers: int,
    base_cache_dir: Path | None,
) -> dict:
    model_root = sensors_root / serial / "model" / "tactile_transformer"
    method_root = model_root / METHOD
    selection_path = method_root / "selection.json"
    test_path = method_root / "test.json"
    if not test_path.is_file():
        return evaluate(
            serial=serial,
            sensors_root=sensors_root,
            device=device,
            batch_size=batch_size,
            workers=workers,
            base_cache_dir=base_cache_dir,
        )

    selection = json.loads(selection_path.read_text())
    result = json.loads(test_path.read_text())
    expected = _test_identity(selection, model_root)
    if any(result.get(key) != value for key, value in expected.items()):
        raise RuntimeError("existing test result does not match selected decoder")
    if not selection.get("test_evaluated"):
        _mark_selection_tested(selection_path, selection, model_root, test_path)
    return result


def evaluate(
    serial: str,
    sensors_root: Path,
    device: torch.device,
    batch_size: int,
    workers: int,
    base_cache_dir: Path | None,
) -> dict:
    dataset_root = sensors_root / serial / "calibration"
    summary = validate_dataset(dataset_root)
    if summary.serial != serial or summary.dataset_id != serial:
        raise ValueError("active calibration must use the sensor serial as dataset ID")

    model_root = sensors_root / serial / "model" / "tactile_transformer"
    method_root = model_root / METHOD
    selection_path = method_root / "selection.json"
    test_path = method_root / "test.json"
    if test_path.exists():
        raise FileExistsError(f"frozen test result already exists: {test_path}")
    selection = _load_selection(selection_path, serial)
    decoder = _verify_selection(selection, model_root, summary.dataset_id)

    model = TactileDPT()
    load_base(model, resolve_base(base_cache_dir))
    freeze_encoder(model)
    _load_decoder(model, decoder)
    model.to(device).eval()

    common = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "shuffle": False,
    }
    ball_loader = DataLoader(BallDepthDataset(dataset_root, "test"), **common)
    manual_loader = DataLoader(
        ConcatDataset([
            ManualSupportDataset(dataset_root, "test"),
            BackgroundSupportDataset(dataset_root, "test"),
        ]),
        **common,
    )
    background_loader = DataLoader(
        BackgroundSupportDataset(dataset_root, "test"),
        **common,
    )
    evaluation_args = SimpleNamespace(
        device=device,
        maximum_depth_mm=float(selection["maximum_depth_mm"]),
        contact_threshold_mm=float(selection["contact_threshold_mm"]),
    )
    ball = evaluate_ball(model, ball_loader, evaluation_args)
    manual = evaluate_manual(model, manual_loader, evaluation_args)
    background = evaluate_background(model, background_loader, evaluation_args)
    eligible = _is_eligible(ball, manual, background)
    result = {
        "schema_version": 2,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        **_test_identity(selection, model_root),
        "ball": ball,
        "manual": manual,
        "background": background,
        "eligible": eligible,
    }
    _write_json_once(test_path, result)

    _mark_selection_tested(selection_path, selection, model_root, test_path)
    return result
