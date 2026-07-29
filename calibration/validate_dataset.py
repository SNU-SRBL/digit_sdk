"""Command-line calibration dataset validator."""

import argparse
from pathlib import Path

from calibration.dataset_schema import DatasetSchemaError, validate_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a DIGIT calibration dataset root"
    )
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    try:
        summary = validate_dataset(args.root)
    except DatasetSchemaError as exc:
        parser.error(str(exc))
    counts = ", ".join(
        f"{name}={count}"
        for name, count in sorted(summary.modality_counts.items())
    )
    print(
        f"valid dataset {summary.dataset_id} for {summary.serial}: "
        f"{summary.record_count} records ({counts})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
