"""Select one seed within one fixed per-sensor training method."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import statistics

from calibration.tactile_transformer.train import METHOD

DEFAULT_SEEDS = (17, 29, 43)
DEFAULT_DICE_TOLERANCE = 0.005


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flatten(path: Path, document: dict) -> dict:
    selected = document["selected_validation"]
    ball = selected["ball"]
    manual = selected["manual"]
    return {
        "objective": document["objective"],
        "seed": int(document["seed"]),
        "path": path,
        "document": document,
        "eligible": bool(selected["eligible"]),
        "ball_contact_mae_mm": float(ball["contact_mae_mm"]),
        "ball_contact_rmse_mm": float(ball["contact_rmse_mm"]),
        "manual_dice": float(manual["mean_dice"]),
        "manual_false_positive_rate": float(manual["false_positive_rate"]),
        "manual_missed_contacts": int(manual["missed_contacts"]),
    }


def load_runs(model_root: Path, seeds=DEFAULT_SEEDS) -> list[dict]:
    runs = []
    missing = []
    for seed in seeds:
        path = model_root / METHOD / f"seed_{seed}" / "training.json"
        if not path.is_file():
            missing.append(path)
            continue
        runs.append(_flatten(path, json.loads(path.read_text())))
    if missing:
        listing = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"missing validation runs:\n{listing}")
    return runs


def _mean(runs: list[dict], key: str) -> float:
    return statistics.mean(run[key] for run in runs)


def _stdev(runs: list[dict], key: str) -> float:
    values = [run[key] for run in runs]
    return statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate_runs(runs: list[dict]) -> dict:
    return {
        "eligible_runs": sum(run["eligible"] for run in runs),
        "run_count": len(runs),
        "manual_missed_contacts": sum(
            run["manual_missed_contacts"] for run in runs
        ),
        **{
            key: {
                "mean": _mean(runs, key),
                "sample_stdev": _stdev(runs, key),
            }
            for key in (
                "ball_contact_mae_mm",
                "ball_contact_rmse_mm",
                "manual_dice",
                "manual_false_positive_rate",
            )
        },
    }


def choose_run(runs: list[dict], dice_tolerance: float) -> dict:
    if not runs or any(not run["eligible"] for run in runs):
        raise RuntimeError("method must pass validation for every seed")
    best_dice = max(run["manual_dice"] for run in runs)
    close = [
        run for run in runs
        if best_dice - run["manual_dice"] <= dice_tolerance
    ]
    return min(
        close,
        key=lambda run: (
            run["manual_false_positive_rate"],
            run["ball_contact_mae_mm"],
            run["ball_contact_rmse_mm"],
            run["seed"],
        ),
    )


def select(
    model_root: Path,
    serial: str,
    dice_tolerance: float,
) -> dict:
    runs = load_runs(model_root)
    aggregate = aggregate_runs(runs)
    run = choose_run(runs, dice_tolerance)
    source = run["path"].parent / "decoder.pth"
    destination = model_root / METHOD / "decoder.pth"
    shutil.copyfile(source, destination)

    training = run["document"]
    selection = {
        "schema_version": 1,
        "selected_at_utc": datetime.now(timezone.utc).isoformat(),
        "serial": serial,
        "method": METHOD,
        "dataset_id": training["dataset_id"],
        "ball_split_id": training["ball_split_id"],
        "manual_mask_split_id": training["manual_mask_split_id"],
        "selection_protocol": {
            "seeds": list(DEFAULT_SEEDS),
            "all_seeds_must_be_eligible": True,
            "manual_dice_close_tie_tolerance": dice_tolerance,
            "close_tie_order": [
                "manual_false_positive_rate",
                "ball_contact_mae_mm",
                "ball_contact_rmse_mm",
            ],
        },
        "method_aggregate": aggregate,
        "selected": {
            "seed": run["seed"],
            "source_decoder": source.relative_to(model_root).as_posix(),
            "decoder": destination.relative_to(model_root).as_posix(),
            "decoder_sha256": _sha256(destination),
            "validation": training["selected_validation"],
        },
        "base": training["base"],
        "maximum_depth_mm": training["maximum_depth_mm"],
        "contact_threshold_mm": training["contact_threshold_mm"],
        "configuration_frozen": True,
        "test_evaluated": False,
    }
    (model_root / METHOD / "selection.json").write_text(
        json.dumps(selection, indent=2) + "\n"
    )
    return selection


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", required=True)
    parser.add_argument("--sensors-root", type=Path, default=Path("sensors"))
    parser.add_argument(
        "--dice-tolerance", type=float, default=DEFAULT_DICE_TOLERANCE
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    model_root = (
        args.sensors_root / args.serial / "model" / "tactile_transformer"
    )
    result = select(model_root, args.serial, args.dice_tolerance)
    chosen = result["selected"]
    print(
        f"selected {METHOD} seed={chosen['seed']} -> "
        f"{model_root / chosen['decoder']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
