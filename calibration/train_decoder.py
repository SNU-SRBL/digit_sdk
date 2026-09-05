"""Train and select one fixed decoder method across the standard seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from calibration.dataset_schema import validate_dataset
from calibration.tactile_transformer.select_model import (
    DEFAULT_DICE_TOLERANCE,
    DEFAULT_SEEDS,
    select,
)
from calibration.tactile_transformer.train import METHOD, TRAINING_PROTOCOL


def completed_run_matches(
    run_root: Path,
    *,
    serial: str,
    seed: int,
) -> bool:
    metadata_path = run_root / "training.json"
    decoder_path = run_root / "decoder.pth"
    if not metadata_path.is_file() or not decoder_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metadata.get("training_protocol") == TRAINING_PROTOCOL
        and metadata.get("serial") == serial
        and metadata.get("objective") == METHOD
        and int(metadata.get("seed", -1)) == seed
    )


def train_sensor(
    sensors_root: Path,
    serial: str,
    dice_tolerance: float = DEFAULT_DICE_TOLERANCE,
) -> dict:
    sensors_root = Path(sensors_root)
    dataset_root = sensors_root / serial / "calibration"
    summary = validate_dataset(dataset_root)
    if summary.serial != serial or summary.dataset_id != serial:
        raise ValueError("active dataset identity must equal the sensor serial")
    model_root = sensors_root / serial / "model/tactile_transformer"

    for seed in DEFAULT_SEEDS:
        run_root = model_root / METHOD / f"seed_{seed}"
        if completed_run_matches(
            run_root,
            serial=serial,
            seed=seed,
        ):
            print(f"reuse {METHOD} seed={seed}", flush=True)
            continue
        if run_root.exists():
            raise ValueError(
                f"stale or incomplete training run exists: {run_root}"
            )

        temporary = model_root / METHOD / f".seed_{seed}_training"
        if temporary.exists():
            shutil.rmtree(temporary)
        command = [
            sys.executable,
            "-m",
            "calibration.tactile_transformer.train",
            "--serial",
            serial,
            "--sensors-root",
            str(sensors_root),
            "--seed",
            str(seed),
            "--output-root",
            str(temporary),
        ]
        print(f"train {METHOD} seed={seed}", flush=True)
        try:
            subprocess.run(command, check=True)
            if not completed_run_matches(
                temporary,
                serial=serial,
                seed=seed,
            ):
                raise RuntimeError(
                    f"training run did not produce valid artifacts: {temporary}"
                )
            temporary.rename(run_root)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise

    return select(model_root, serial, dice_tolerance)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--sensors-root", type=Path, default=Path("sensors"))
    parser.add_argument(
        "--dice-tolerance", type=float, default=DEFAULT_DICE_TOLERANCE
    )
    args = parser.parse_args(argv)
    selection = train_sensor(
        args.sensors_root,
        args.serial,
        args.dice_tolerance,
    )
    selected = selection["selected"]
    print(
        f"selected {METHOD} seed={selected['seed']} -> "
        f"{args.sensors_root / args.serial / 'model/tactile_transformer' / selected['decoder']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
