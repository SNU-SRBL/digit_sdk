import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml

from calibration.tactile_transformer.data import image_tensor
from digit_sdk.depth import DepthEstimator, _image_bytes, _image_tensor


MODEL_ROOT = Path("sensors/D21275/model/depth")
SERIALS = ("D21275", "D21119", "D21242", "D21273")


def _ball_image_path(serial):
    """Return the first ball calibration image for a serial, or None."""
    manifest = Path("sensors") / serial / "calibration" / "manifest.jsonl"
    if not manifest.is_file():
        return None
    record = next(
        (
            json.loads(line)
            for line in manifest.read_text().splitlines()
            if '"modality": "ball"' in line
        ),
        None,
    )
    if record is None:
        return None
    image = Path("sensors") / serial / "calibration" / record["image_path"]
    return image if image.is_file() else None


def test_production_preprocessing_matches_training_preprocessing():
    image = np.random.default_rng(4).integers(
        0, 256, size=(240, 320, 3), dtype=np.uint8
    )
    torch.testing.assert_close(_image_tensor(image), image_tensor(image))
    reconstructed = _image_bytes(image).float().div(255.0).sub(0.5).div(0.5)
    torch.testing.assert_close(reconstructed, image_tensor(image))


def test_production_depth_limit_is_sensor_configured():
    config = yaml.safe_load(Path("sensors/D21275/D21275.yaml").read_text())
    assert config["maximum_depth_mm"] == 2.2


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not (MODEL_ROOT / "decoder.pth").is_file()
    or _ball_image_path("D21275") is None,
    reason=(
        "CUDA, a trained clean D21275 decoder, and calibration ball images "
        "are required; ball images are gitignored data, download separately"
    ),
)
def test_production_estimator_smoke_uses_selected_decoder():
    root = Path("sensors")
    image = cv2.imread(str(_ball_image_path("D21275")))
    estimator = DepthEstimator(["D21275"], root, "cuda")
    depth = estimator.estimate("D21275", image)
    assert depth.shape == image.shape[:2]
    assert depth.dtype == np.float32
    assert np.isfinite(depth).all()
    assert float(depth.min()) >= 0.0
    assert float(depth.max()) <= 2.2


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not all(
        (Path("sensors") / serial / "model/depth/decoder.pth").is_file()
        for serial in SERIALS
    )
    or any(_ball_image_path(serial) is None for serial in SERIALS),
    reason=(
        "CUDA, all four trained decoders, and calibration ball images are "
        "required; ball images are gitignored data, download separately"
    ),
)
def test_compiled_four_sensor_batch_matches_eager_path():
    root = Path("sensors")
    frames = {
        serial: cv2.imread(str(_ball_image_path(serial)))
        for serial in SERIALS
    }

    estimator = DepthEstimator(SERIALS, root, "cuda", backend="torch")
    compiled = estimator.estimate_batch(frames)
    compiled_function = estimator._backend._compiled_full_batch
    estimator._backend._compiled_full_batch = None
    eager = estimator.estimate_batch(frames)
    estimator._backend._compiled_full_batch = compiled_function

    for serial in SERIALS:
        difference = np.abs(compiled[serial] - eager[serial])
        assert float(difference.max()) < 0.01
        assert float(difference.mean()) < 0.001
