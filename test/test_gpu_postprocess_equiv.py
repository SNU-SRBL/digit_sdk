"""CPU-only equivalence checks for GPU postprocess tensor paths."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from digit_sdk.depth import _resize_depth, _torch_estimate_prepared_batch_tensors
from digit_sdk.temporal import NeuralFeelsFIR, PersistenceCutoff


class _FakeEncoder:
    def __call__(self, _images):
        activations = torch.rand(1, 14, 14, 384, device="cuda")
        return tuple(activations for _ in range(4))


class _FakeDecoder:
    def __call__(self, _activations):
        return torch.full((1, 1, 224, 224), 2.0, device="cuda")


def _frames(count: int, shape=(320, 240)):
    rng = np.random.default_rng(7)
    return [
        rng.random(shape, dtype=np.float32) for _ in range(count)
    ]


def test_neuralfeels_fir_tensor_matches_numpy():
    numpy_fir = NeuralFeelsFIR()
    tensor_fir = NeuralFeelsFIR()

    for frame in _frames(6):
        numpy_output = numpy_fir(frame)
        tensor_output = tensor_fir(torch.from_numpy(frame))
        np.testing.assert_allclose(
            tensor_output.cpu().numpy(),
            numpy_output,
            rtol=1e-5,
            atol=1e-6,
        )


def test_persistence_tensor_matches_numpy():
    numpy_persistence = PersistenceCutoff(0.1)
    tensor_persistence = PersistenceCutoff(0.1)

    for frame in _frames(6):
        numpy_output = numpy_persistence(frame)
        tensor_output = tensor_persistence(torch.from_numpy(frame))
        np.testing.assert_allclose(
            tensor_output.cpu().numpy(),
            numpy_output,
            rtol=1e-5,
            atol=1e-6,
        )


def test_gpu_resize_matches_cv2_within_tolerance():
    """GPU resize is close but not bit-identical to cv2 INTER_LINEAR."""
    rng = np.random.default_rng(11)
    prediction = rng.random((224, 224), dtype=np.float32)
    height, width = 240, 320

    torch_resized = torch.nn.functional.interpolate(
        torch.from_numpy(prediction).unsqueeze(0).unsqueeze(0),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()
    cv2_resized = _resize_depth(
        (prediction, width, height, 1.0)
    )
    max_difference = float(np.abs(torch_resized - cv2_resized).max())
    assert max_difference <= 1e-4
    np.testing.assert_allclose(
        torch_resized, cv2_resized, rtol=1e-4, atol=1e-4
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason=(
        "stub estimator still needs real CUDA stream APIs in "
        "estimate_prepared_batch_tensors; CPU-only run skips the "
        "shape/range check"
    ),
)
def test_estimate_prepared_batch_tensors_shape_and_range():
    estimator = SimpleNamespace()
    estimator.device = torch.device("cuda")
    estimator.serials = ("left",)
    estimator._maximum_depth_mm = {"left": 2.2}
    estimator._compiled_full_batch = None
    estimator._decoder_streams = {
        "left": torch.cuda.Stream(device=estimator.device)
    }
    estimator._encoder = _FakeEncoder()
    estimator._decoders = {"left": _FakeDecoder()}
    prepared = {
        "left": SimpleNamespace(
            tensor=torch.zeros((3, 224, 224), dtype=torch.uint8),
            output_shape=(240, 320),
        )
    }

    results = _torch_estimate_prepared_batch_tensors(estimator, prepared)
    depth = results["left"]
    assert tuple(depth.shape) == (240, 320)
    assert depth.dtype == torch.float32
    assert float(depth.min()) >= 0.0
    assert float(depth.max()) == pytest.approx(2.2)
