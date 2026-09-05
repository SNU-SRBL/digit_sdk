"""Promote the frozen, tested mixed decoder to production."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil

from calibration.dataset_schema import validate_dataset


PRODUCTION_METHOD = "mixed"


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
    production_root = sensor_root / "model" / "depth"
    if production_root.exists():
        raise FileExistsError(f"production depth model already exists: {production_root}")
    temporary = production_root.with_name("depth.promoting")
    if temporary.exists():
        raise FileExistsError(f"incomplete promotion exists: {temporary}")
    temporary.mkdir(parents=True)
    shutil.copy2(decoder_path, temporary / "decoder.pth")
    os.replace(temporary, production_root)
    config_path = sensor_root / f"{serial}.yaml"
    maximum = float(selection["maximum_depth_mm"])
    config = config_path.read_text()
    line = f"maximum_depth_mm: {maximum}\n"
    if re.search(r"^maximum_depth_mm:.*$", config, flags=re.MULTILINE):
        config = re.sub(r"^maximum_depth_mm:.*$", line.rstrip(), config, flags=re.MULTILINE)
    else:
        config += "\n# Metric depth calibration\n" + line
    config_path.write_text(config)
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
