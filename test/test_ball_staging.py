import pytest

from calibration.ball_staging import ball_partition, validate_ball_replacements


def test_guided_ball_split_is_balanced_by_cell_and_depth():
    partitions = []
    for row in range(5):
        for column in range(5):
            cell = [
                ball_partition({
                    "sample_id": f"ball-{row}-{column}-{depth}",
                    "guide_grid_row": row,
                    "guide_grid_column": column,
                    "guide_depth_level": depth,
                })
                for depth in range(1, 6)
            ]
            assert cell.count("train") == 3
            assert cell.count("validation") == 1
            assert cell.count("test") == 1
            partitions.extend(cell)
    assert partitions.count("train") == 75
    assert partitions.count("validation") == 25
    assert partitions.count("test") == 25


def test_staged_ball_partition_is_rejected():
    with pytest.raises(ValueError, match="has a partition"):
        ball_partition({"sample_id": "ball", "partition": "train"})


def test_replacement_must_match_rejected_guide_target():
    rejected = {
        "sample_id": "bad", "kind": "ball", "rejected": True,
        "ball_diameter_mm": 3.0, "guide_grid_row": 0,
        "guide_grid_column": 4, "guide_depth_level": 5,
    }
    replacement = {
        "sample_id": "new", "kind": "ball", "rejected": False,
        "ball_diameter_mm": 3.0, "guide_grid_row": 0,
        "guide_grid_column": 4, "guide_depth_level": 5,
        "replaces_sample_id": "bad",
    }

    assert validate_ball_replacements([rejected, replacement]) == [replacement]
    with pytest.raises(ValueError, match="does not match rejected guide target"):
        validate_ball_replacements([
            rejected, {**replacement, "guide_depth_level": 4}
        ])
