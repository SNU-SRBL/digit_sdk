"""Promote the frozen, tested mixed decoder to production."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

from calibration.dataset_schema import validate_dataset
from calibration.tactile_transformer.model import BASE_REVISION, BASE_SHA256
from calibration.tactile_transformer.train import dataset_fingerprint


PRODUCTION_METHOD = "mixed"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def promote(serial: str, sensors_root: Path) -> Path:
    sensor_root = sensors_root / serial
    dataset_root = sensor_root / "calibration"
    summary = validate_dataset(dataset_root)
    if summary.serial != serial or summary.dataset_id != serial:
        raise ValueError("active calibration must use the sensor serial as dataset ID")

    candidate_root = (
        sensor_root / "model" / "tactile_transformer" / PRODUCTION_METHOD
    )
    selection_path = candidate_root / "selection.json"
    test_path = candidate_root / "test.json"
    decoder_path = candidate_root / "decoder.pth"
    selection = json.loads(selection_path.read_text())
    test = json.loads(test_path.read_text())

    if selection.get("serial") != serial:
        raise ValueError("selection serial mismatch")
    if selection.get("method") != PRODUCTION_METHOD:
        raise ValueError("only the globally frozen mixed method can be promoted")
    if not selection.get("configuration_frozen"):
        raise ValueError("selection configuration is not frozen")
    if not selection.get("test_evaluated"):
        raise ValueError("held-out test was not evaluated")
    if test.get("serial") != serial or test.get("method") != PRODUCTION_METHOD:
        raise ValueError("test result does not match the selected decoder")
    if not test.get("eligible"):
        raise ValueError("selected decoder failed the frozen test gates")
    if test.get("dataset_fingerprint") != dataset_fingerprint(dataset_root):
        raise ValueError("test result does not match the active calibration")

    decoder_sha256 = _sha256(decoder_path)
    if decoder_sha256 != selection["selected"]["decoder_sha256"]:
        raise ValueError("selected decoder checksum mismatch")
    if decoder_sha256 != test["decoder_sha256"]:
        raise ValueError("tested decoder checksum mismatch")
    base = selection.get("base", {})
    if (
        base.get("revision") != BASE_REVISION
        or base.get("sha256") != BASE_SHA256
    ):
        raise ValueError("selected decoder uses an unsupported encoder")

    production_root = sensor_root / "model" / "depth"
    if production_root.exists():
        raise FileExistsError(f"production depth model already exists: {production_root}")
    temporary = production_root.with_name("depth.promoting")
    if temporary.exists():
        raise FileExistsError(f"incomplete promotion exists: {temporary}")
    temporary.mkdir(parents=True)
    shutil.copy2(decoder_path, temporary / "decoder.pth")
    metadata = {
        "schema_version": 1,
        "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
        "serial": serial,
        "dataset_id": summary.dataset_id,
        "dataset_fingerprint": test["dataset_fingerprint"],
        "method": PRODUCTION_METHOD,
        "encoder_frozen": True,
        "decoder_scope": "per_sensor",
        "selected_seed": int(selection["selected"]["seed"]),
        "decoder_sha256": decoder_sha256,
        "base": base,
        "maximum_depth_mm": float(selection["maximum_depth_mm"]),
        "contact_threshold_mm": float(selection["contact_threshold_mm"]),
        "validation": selection["selected"]["validation"],
        "test": test,
    }
    (temporary / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary, production_root)
    return production_root


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--sensors-root", type=Path, default=Path("sensors"))
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    destination = promote(args.serial, args.sensors_root)
    print(f"promoted {PRODUCTION_METHOD} decoder -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
