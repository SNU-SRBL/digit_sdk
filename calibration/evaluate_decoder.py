"""Evaluate one frozen decoder once on its held-out test split."""

from __future__ import annotations

import argparse
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
    BASE_SHA256,
    TactileDPT,
    freeze_encoder,
    load_base,
    resolve_base,
)
from calibration.tactile_transformer.train import (
    METHOD,
    _load_decoder,
    dataset_fingerprint,
    evaluate_ball,
    evaluate_manual,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    dataset_root: Path,
    dataset_id: str,
) -> Path:
    if selection.get("dataset_id") != dataset_id:
        raise ValueError("selection dataset does not match active calibration")
    if selection.get("base", {}).get("revision") != BASE_REVISION:
        raise ValueError("selection uses an unsupported encoder revision")
    if selection.get("base", {}).get("sha256") != BASE_SHA256:
        raise ValueError("selection uses an unsupported encoder checksum")

    decoder = model_root / selection["selected"]["decoder"]
    if not decoder.is_file():
        raise FileNotFoundError(decoder)
    if _sha256(decoder) != selection["selected"]["decoder_sha256"]:
        raise ValueError("selected decoder checksum mismatch")

    seed = int(selection["selected"]["seed"])
    training_path = (
        model_root / selection["method"] / f"seed_{seed}" / "training.json"
    )
    training = json.loads(training_path.read_text())
    if training.get("dataset_fingerprint") != dataset_fingerprint(dataset_root):
        raise ValueError("selected decoder was trained on a different dataset")
    return decoder


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
    decoder = _verify_selection(
        selection, model_root, dataset_root, summary.dataset_id
    )

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
    evaluation_args = SimpleNamespace(
        device=device,
        maximum_depth_mm=float(selection["maximum_depth_mm"]),
        contact_threshold_mm=float(selection["contact_threshold_mm"]),
    )
    ball = evaluate_ball(model, ball_loader, evaluation_args)
    manual = evaluate_manual(model, manual_loader, evaluation_args)
    eligible = manual["missed_contacts"] == 0 and ball["contact_mae_mm"] <= 0.15
    result = {
        "schema_version": 1,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "serial": serial,
        "method": METHOD,
        "selected_seed": int(selection["selected"]["seed"]),
        "dataset_id": summary.dataset_id,
        "dataset_fingerprint": dataset_fingerprint(dataset_root),
        "ball_split_id": selection["ball_split_id"],
        "manual_mask_split_id": selection["manual_mask_split_id"],
        "decoder_sha256": selection["selected"]["decoder_sha256"],
        "contact_threshold_mm": float(selection["contact_threshold_mm"]),
        "ball": ball,
        "manual": manual,
        "eligible": eligible,
    }
    _write_json_once(test_path, result)

    selection["test_evaluated"] = True
    selection["test_result"] = test_path.relative_to(model_root).as_posix()
    temporary = selection_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, selection_path)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--sensors-root", type=Path, default=Path("sensors"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--base-cache-dir", type=Path)
    args = parser.parse_args(argv)
    args.device = torch.device(args.device)
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    result = evaluate(
        serial=args.serial,
        sensors_root=args.sensors_root,
        device=args.device,
        batch_size=args.batch_size,
        workers=args.workers,
        base_cache_dir=args.base_cache_dir,
    )
    print(
        f"{args.serial} {METHOD}: "
        f"ball_mae={result['ball']['contact_mae_mm']:.4f} mm "
        f"dice={result['manual']['mean_dice']:.4f} "
        f"fpr={result['manual']['false_positive_rate']:.4f} "
        f"missed={result['manual']['missed_contacts']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
