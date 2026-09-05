import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch import nn
import yaml

from calibration.tactile_transformer.data import (
    BallDepthDataset,
    BackgroundSupportDataset,
    ManualSupportDataset,
)
from calibration.tactile_transformer.metrics import (
    ball_metrics,
    binary_metrics,
)
from calibration.tactile_transformer.losses import (
    outside_zero_loss,
    region_balanced_depth_loss,
    support_loss,
)
from calibration.tactile_transformer.model import (
    decoder_state,
    freeze_encoder,
)
from calibration.tactile_transformer.select_model import (
    aggregate_runs,
    choose_run,
    select,
)
from calibration.tactile_transformer.train import (
    TRAINING_PROTOCOL,
    checkpoint_rank,
    parse_args,
)
from calibration.train_decoder import completed_run_matches


def _write_image(path: Path, value: int = 0):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((24, 32, 3), value, dtype=np.uint8))


def _make_dataset(root: Path):
    config = {
        "schema_version": 1,
        "dataset_id": "DTEST",
        "serial": "DTEST",
        "image_shape": [24, 32, 3],
        "colour_space": "BGR",
        "dtype": "uint8",
        "ppmm": 10.0,
        "active_splits": {
            "background": "background",
            "manual_mask": "manual_mask",
        },
    }
    (root / "dataset.yaml").write_text(yaml.safe_dump(config))
    ball_image = Path("ball/images/s/ball.png")
    ball_label = Path("ball/labels/s/ball.npz")
    manual_image = Path("manual_mask/images/s/touch.png")
    manual_label = Path("manual_mask/masks/s/touch.png")
    background_image = Path("background/s/background.png")
    for path, value in (
        (ball_image, 20), (manual_image, 40), (background_image, 0)
    ):
        _write_image(root / path, value)
    (root / ball_label).parent.mkdir(parents=True, exist_ok=True)
    np.savez(root / ball_label, unused=np.array(0))
    mask = np.zeros((24, 32), dtype=np.uint8)
    mask[8:16, 10:22] = 255
    (root / manual_label).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(root / manual_label), mask)
    common = {
        "schema_version": 1,
        "serial": "DTEST",
        "session_id": "s",
        "captured_at_utc": "2026-07-20T00:00:00+00:00",
        "shape": [24, 32, 3],
    }
    records = [
        {
            **common,
            "sample_id": "ball",
            "modality": "ball",
            "image_path": ball_image.as_posix(),
            "label_path": ball_label.as_posix(),
            "split_group": "ball",
            "ball_diameter_mm": 4.0,
            "center_px": [16.0, 12.0],
            "radius_px": 10.0,
            "ppmm": 10.0,
            "indentation_depth_mm": 2.0 - np.sqrt(3.0),
        },
        {
            **common,
            "sample_id": "touch",
            "modality": "manual_mask",
            "image_path": manual_image.as_posix(),
            "label_path": manual_label.as_posix(),
            "split_group": "touch",
            "annotator": "tester",
            "annotation_revision": 1,
            "background_path": background_image.as_posix(),
        },
        {
            **common,
            "sample_id": "background",
            "modality": "background",
            "image_path": background_image.as_posix(),
            "label_path": None,
            "split_group": "background",
            "training_eligible": True,
        },
    ]
    (root / "manifest.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    (root / "splits").mkdir()
    for split_id, modality, train in (
        ("background", "background", ["background"]),
        ("ball", "ball", ["ball"]),
        ("manual_mask", "manual_mask", ["touch"]),
    ):
        split = {
            "split_id": split_id,
            "modality": modality,
            "train": train,
            "validation": train,
            "test": [],
        }
        (root / "splits" / f"{split_id}.json").write_text(json.dumps(split))


def test_schema_datasets_return_metric_depth_and_background_negatives(tmp_path):
    _make_dataset(tmp_path)

    ball = BallDepthDataset(tmp_path, "train")[0]
    manual = ManualSupportDataset(tmp_path, "train")
    background = BackgroundSupportDataset(tmp_path, "train")

    assert ball["image"].shape == (3, 224, 224)
    assert ball["depth_mm"].shape == (1, 224, 224)
    assert float(ball["depth_mm"].max()) == pytest.approx(
        2.0 - np.sqrt(3.0), rel=0.02
    )
    assert len(manual) == 1
    assert manual[0]["has_contact"] is True
    assert len(background) == 1
    assert background[0]["has_contact"] is False


def test_region_balanced_loss_does_not_dilute_contact_region():
    target = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]])
    prediction = torch.zeros_like(target)

    loss = region_balanced_depth_loss(prediction, target)

    assert float(loss) == pytest.approx(1.0)


def test_support_and_zero_losses_prefer_correct_contact():
    mask = torch.tensor([[[[True, True, False, False]]]])
    correct = torch.tensor([[[[0.3, 0.3, 0.0, 0.0]]]])
    inverse = torch.tensor([[[[0.0, 0.0, 0.3, 0.3]]]])

    assert support_loss(correct, mask) < support_loss(inverse, mask)
    assert outside_zero_loss(correct, mask) < outside_zero_loss(inverse, mask)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_encoders = nn.Linear(2, 2)
        self.decoder = nn.Linear(2, 1)


def test_encoder_is_frozen_and_omitted_from_decoder_state():
    model = _TinyModel()

    freeze_encoder(model)
    state = decoder_state(model)

    assert all(not parameter.requires_grad
               for parameter in model.transformer_encoders.parameters())
    assert set(state) == {"decoder.weight", "decoder.bias"}


def test_checkpoint_selection_prefers_eligible_validation_result():
    eligible = checkpoint_rank(
        {"contact_mae_mm": 0.14, "contact_rmse_mm": 0.2},
        {"missed_contacts": 0, "mean_dice": 0.7,
         "false_positive_rate": 0.1},
    )
    ineligible = checkpoint_rank(
        {"contact_mae_mm": 0.05, "contact_rmse_mm": 0.1},
        {"missed_contacts": 1, "mean_dice": 0.99,
         "false_positive_rate": 0.0},
    )

    assert eligible < ineligible
    assert parse_args(["--serial", "DTEST"]).maximum_depth_mm == 2.2


def _selection_run(objective, seed, dice, fpr, mae, eligible=True):
    return {
        "objective": objective,
        "seed": seed,
        "eligible": eligible,
        "ball_contact_mae_mm": mae,
        "ball_contact_rmse_mm": mae + 0.02,
        "manual_dice": dice,
        "manual_false_positive_rate": fpr,
        "manual_missed_contacts": 0 if eligible else 1,
    }


def test_method_aggregation_keeps_seed_stability_visible():
    runs = [
        _selection_run("mixed", 17, 0.825, 0.015, 0.081),
        _selection_run("mixed", 29, 0.833, 0.017, 0.085),
        _selection_run("mixed", 43, 0.834, 0.023, 0.081),
    ]

    aggregate = aggregate_runs(runs)

    assert aggregate["eligible_runs"] == 3
    assert aggregate["run_count"] == 3
    assert aggregate["manual_dice"]["mean"] == pytest.approx(0.8306667)


def test_seed_selection_uses_false_positives_for_close_dice():
    runs = [
        _selection_run("mixed", 17, 0.825, 0.015, 0.081),
        _selection_run("mixed", 29, 0.833, 0.017, 0.085),
        _selection_run("mixed", 43, 0.834, 0.023, 0.081),
    ]

    assert choose_run(runs, dice_tolerance=0.005)["seed"] == 29


def test_seed_selection_requires_every_seed_to_be_eligible():
    runs = [
        _selection_run("mixed", 17, 0.825, 0.015, 0.081),
        _selection_run("mixed", 29, 0.833, 0.017, 0.085, False),
        _selection_run("mixed", 43, 0.834, 0.023, 0.081),
    ]

    with pytest.raises(RuntimeError, match="every seed"):
        choose_run(runs, dice_tolerance=0.005)


def test_completed_run_requires_matching_protocol(tmp_path):
    run_root = tmp_path / "seed_17"
    run_root.mkdir()
    (run_root / "decoder.pth").write_bytes(b"decoder")
    (run_root / "training.json").write_text(json.dumps({
        "training_protocol": TRAINING_PROTOCOL,
        "serial": "DTEST",
        "objective": "mixed",
        "seed": 17,
    }))

    assert completed_run_matches(
        run_root,
        serial="DTEST",
        seed=17,
    )


def test_selection_writes_only_method_scoped_artifacts(tmp_path):
    model_root = tmp_path / "tactile_transformer"
    for seed, dice, fpr, mae in (
        (17, 0.825, 0.015, 0.081),
        (29, 0.833, 0.017, 0.085),
        (43, 0.834, 0.023, 0.081),
    ):
        run_root = model_root / "mixed" / f"seed_{seed}"
        run_root.mkdir(parents=True)
        (run_root / "decoder.pth").write_bytes(f"seed-{seed}".encode())
        (run_root / "training.json").write_text(json.dumps({
            "objective": "mixed",
            "seed": seed,
            "dataset_id": "DTEST",
            "ball_split_id": "ball",
            "manual_mask_split_id": "manual_mask",
            "selected_validation": {
                "eligible": True,
                "ball": {
                    "contact_mae_mm": mae,
                    "contact_rmse_mm": mae + 0.02,
                },
                "manual": {
                    "mean_dice": dice,
                    "false_positive_rate": fpr,
                    "missed_contacts": 0,
                },
            },
            "base": {"revision": "base"},
            "maximum_depth_mm": 2.2,
            "contact_threshold_mm": 0.1,
        }))

    result = select(model_root, "DTEST", 0.005)

    assert result["method"] == "mixed"
    assert result["selected"]["seed"] == 29
    assert (model_root / "mixed/decoder.pth").read_bytes() == b"seed-29"
    assert (model_root / "mixed/selection.json").is_file()
    assert not (model_root / "decoder.pth").exists()
    assert not (model_root / "selection.json").exists()


def test_frozen_test_metrics_compare_raw_arrays():
    truth_depth = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    predicted_depth = np.array([[0.8, 0.1], [1.2, 0.0]], dtype=np.float32)

    depth = ball_metrics(truth_depth, predicted_depth)
    mask = binary_metrics(truth_depth > 0, predicted_depth >= 0.5)

    assert depth["contact_mae_mm"] == pytest.approx(0.2)
    assert depth["background_mae_mm"] == pytest.approx(0.05)
    assert mask["dice"] == 1.0
    assert mask["missed_contact"] is False
