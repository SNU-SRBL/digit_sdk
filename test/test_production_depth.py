import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from calibration.tactile_transformer.data import image_tensor
from digit_sdk.depth import DepthEstimator, _image_bytes, _image_tensor


MODEL_ROOT = Path("sensors/D21275/model/depth")
SERIALS = ("D21275", "D21119", "D21242", "D21273")


def test_production_preprocessing_matches_training_preprocessing():
    image = np.random.default_rng(4).integers(
        0, 256, size=(240, 320, 3), dtype=np.uint8
    )
    torch.testing.assert_close(_image_tensor(image), image_tensor(image))
    reconstructed = _image_bytes(image).float().div(255.0).sub(0.5).div(0.5)
    torch.testing.assert_close(reconstructed, image_tensor(image))


@pytest.mark.skipif(
    not (MODEL_ROOT / "metadata.json").is_file(),
    reason="clean D21275 decoder has not been promoted",
)
def test_production_model_metadata_is_serial_bound():
    metadata = json.loads((MODEL_ROOT / "metadata.json").read_text())
    assert metadata["serial"] == "D21275"
    assert metadata["encoder_frozen"] is True
    assert metadata["decoder_scope"] == "per_sensor"
    assert metadata["maximum_depth_mm"] == 2.2


@pytest.mark.skipif(
    not torch.cuda.is_available() or not (MODEL_ROOT / "decoder.pth").is_file(),
    reason="CUDA and a promoted clean D21275 decoder are required",
)
def test_production_estimator_smoke_uses_selected_decoder():
    root = Path("sensors")
    record = next(
        json.loads(line)
        for line in (root / "D21275/calibration/manifest.jsonl").read_text().splitlines()
        if '"modality": "ball"' in line
    )
    image = cv2.imread(str(root / "D21275/calibration" / record["image_path"]))
    estimator = DepthEstimator(["D21275"], root, "cuda")
    depth = estimator.estimate("D21275", image)
    assert depth.shape == image.shape[:2]
    assert depth.dtype == np.float32
    assert np.isfinite(depth).all()
    assert float(depth.min()) >= 0.0
    assert float(depth.max()) <= 2.2


@pytest.mark.skipif(
    not torch.cuda.is_available() or not all(
        (Path("sensors") / serial / "model/depth/decoder.pth").is_file()
        for serial in SERIALS
    ),
    reason="CUDA and all four promoted decoders are required",
)
def test_compiled_four_sensor_batch_matches_eager_path():
    root = Path("sensors")
    frames = {}
    for serial in SERIALS:
        record = next(
            json.loads(line)
            for line in (
                root / serial / "calibration/manifest.jsonl"
            ).read_text().splitlines()
            if '"modality": "ball"' in line
        )
        frames[serial] = cv2.imread(
            str(root / serial / "calibration" / record["image_path"])
        )

    estimator = DepthEstimator(SERIALS, root, "cuda")
    compiled = estimator.estimate_batch(frames)
    compiled_function = estimator._compiled_full_batch
    estimator._compiled_full_batch = None
    eager = estimator.estimate_batch(frames)
    estimator._compiled_full_batch = compiled_function

    for serial in SERIALS:
        difference = np.abs(compiled[serial] - eager[serial])
        assert float(difference.max()) < 0.01
        assert float(difference.mean()) < 0.001
