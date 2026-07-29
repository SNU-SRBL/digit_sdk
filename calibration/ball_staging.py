"""Validation and split policy for staged ball captures."""

from __future__ import annotations

from typing import Any, Dict, List


def ball_partition(row: Dict[str, Any]) -> str:
    """Resolve a guided capture into a balanced train/validation/test split."""
    if "partition" in row:
        raise ValueError(f"staged ball sample has a partition: {row['sample_id']}")
    try:
        grid_row = int(row["guide_grid_row"])
        grid_column = int(row["guide_grid_column"])
        depth_level = int(row["guide_depth_level"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"unassigned ball sample lacks guide fields: {row['sample_id']}"
        ) from error
    if not (0 <= grid_row < 5 and 0 <= grid_column < 5 and 1 <= depth_level <= 5):
        raise ValueError(f"invalid guide fields for {row['sample_id']}")
    validation_depth = (grid_row * 5 + grid_column) % 5 + 1
    test_depth = validation_depth % 5 + 1
    if depth_level == validation_depth:
        return "validation"
    if depth_level == test_depth:
        return "test"
    return "train"


def validate_ball_replacements(rows: List[Dict[str, Any]]):
    """Return accepted balls after validating optional replacements."""
    ball_rows = [row for row in rows if row.get("kind") == "ball"]
    rejected = [row for row in ball_rows if row.get("rejected", False)]
    active = [row for row in ball_rows if not row.get("rejected", False)]
    rejected_ids = {row["sample_id"] for row in rejected}
    unknown_sources = sorted({
        row["replaces_sample_id"] for row in active
        if row.get("replaces_sample_id")
        and row["replaces_sample_id"] not in rejected_ids
    })
    if unknown_sources:
        raise ValueError(
            "replacement source is not rejected: " + ", ".join(unknown_sources)
        )
    fields = (
        "ball_diameter_mm", "guide_grid_row", "guide_grid_column",
        "guide_depth_level",
    )
    for source in rejected:
        replacements = [
            row for row in active
            if row.get("replaces_sample_id") == source["sample_id"]
        ]
        if len(replacements) > 1:
            raise ValueError(
                "rejected sample has multiple active replacements: "
                f"{source['sample_id']}"
            )
        if replacements and any(
            replacements[0].get(field) != source.get(field) for field in fields
        ):
            raise ValueError(
                "replacement does not match rejected guide target: "
                f"{source['sample_id']}"
            )
    return active
