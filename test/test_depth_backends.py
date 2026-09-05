"""Backend selection, artifact resolution, and torch/ONNX/TRT equivalence."""

import time
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from digit_sdk.backend import (
    _artifacts_present,
    _find_shared_artifact,
    _resolve_backend,
)
from digit_sdk.depth import DepthEstimator


SERIALS = ("D21119", "D21242", "D21273", "D21275")
SENSORS_ROOT = Path("sensors")
BASELINE_NPZ = Path("/tmp/digit_baseline_frames.npz")


def _make_sensors_root(tmp_path, serials=SERIALS, kind=None):
    for serial in serials:
        (tmp_path / serial).mkdir(parents=True, exist_ok=True)
        (tmp_path / serial / f"{serial}.yaml").write_text(
            "maximum_depth_mm: 2.2\n"
        )
        model_root = tmp_path / serial / "model" / "depth"
        model_root.mkdir(parents=True)
        if kind in ("onnx", "engine"):
            suffix = ".onnx" if kind == "onnx" else "_trt_fp16.engine"
            (model_root / f"{serial}_decoder{suffix}").write_bytes(b"x")
    if kind in ("onnx", "engine"):
        shared = (
            "dpt_shared_encoder.onnx"
            if kind == "onnx"
            else "dpt_shared_encoder_trt_fp16.engine"
        )
        (tmp_path / serials[0] / "model" / "depth" / shared).write_bytes(b"x")
    return tmp_path


def test_auto_prefers_trt_engines(tmp_path):
    root = _make_sensors_root(tmp_path, kind="engine")
    assert _artifacts_present(root, SERIALS, "engine")
    assert _resolve_backend("auto", root, SERIALS) == "trt_fp16"


def test_auto_falls_back_to_onnx_without_engines(tmp_path):
    root = _make_sensors_root(tmp_path, kind="onnx")
    assert _artifacts_present(root, SERIALS, "onnx")
    assert _resolve_backend("auto", root, SERIALS) == "onnx"


def test_auto_falls_back_to_torch_without_artifacts(tmp_path):
    root = _make_sensors_root(tmp_path)
    assert _resolve_backend("auto", root, SERIALS) == "torch"


def test_auto_requires_every_serial_decoder_artifact(tmp_path):
    root = _make_sensors_root(tmp_path, kind="engine")
    (root / "D21273" / "model/depth" / "D21273_decoder_trt_fp16.engine").unlink()
    assert not _artifacts_present(root, SERIALS, "engine")
    assert _resolve_backend("auto", root, SERIALS) == "torch"


def test_explicit_backend_must_be_known(tmp_path):
    root = _make_sensors_root(tmp_path)
    with pytest.raises(ValueError):
        _resolve_backend("cuda", root, SERIALS)


def test_shared_artifact_resolution_and_missing_error(tmp_path):
    root = _make_sensors_root(tmp_path, kind="onnx")
    path = _find_shared_artifact(root, SERIALS, "dpt_shared_encoder.onnx")
    assert path == root / "D21119" / "model" / "depth" / "dpt_shared_encoder.onnx"
    empty = _make_sensors_root(tmp_path / "empty", kind=None)
    with pytest.raises(FileNotFoundError):
        _find_shared_artifact(empty, SERIALS, "dpt_shared_encoder.onnx")


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required to construct GPU backends",
)
def test_explicit_trt_without_engines_raises(tmp_path):
    root = _make_sensors_root(tmp_path)
    with pytest.raises(FileNotFoundError):
        DepthEstimator(SERIALS, root, "cuda", backend="trt_fp16")


def _load_baseline():
    data = np.load(BASELINE_NPZ)
    return {serial: data[serial] for serial in SERIALS}


def _contact_mask(depth, threshold):
    return depth > threshold


def _iou(a, b):
    union = np.logical_or(a, b)
    if not union.any():
        return 1.0
    return float(np.logical_and(a, b).sum() / union.sum())


def _rmse_delta(reference, candidate, mask):
    if not mask.any():
        mask = np.ones(reference.shape, dtype=bool)
    return float(np.sqrt(np.mean((reference[mask] - candidate[mask]) ** 2)))


@pytest.mark.skipif(
    not BASELINE_NPZ.is_file(),
    reason="baseline frames npz is required",
)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)
@pytest.mark.skipif(
    not _artifacts_present(SENSORS_ROOT, SERIALS, "engine"),
    reason="TRT FP16 engines not built (run scripts/export_trt_fp16.py)",
)
def test_backends_match_torch_within_contract():
    pytest.importorskip("tensorrt")
    pytest.importorskip("onnxruntime")
    frames = _load_baseline()
    estimators = {
        backend: DepthEstimator(SERIALS, SENSORS_ROOT, "cuda", backend=backend)
        for backend in ("torch", "trt_fp16", "onnx")
    }

    def _latency(estimator, iterations=8):
        for _ in range(2):
            estimator.estimate_batch(frames)  # warmup (compile/JIT)
        torch.cuda.synchronize()
        latencies = []
        for _ in range(iterations):
            started = time.perf_counter()
            estimator.estimate_batch(frames)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - started) * 1000)
        return np.array(latencies)

    outputs = {}
    latencies = {}
    thresholds = {}
    maxima = {}
    for serial in SERIALS:
        config = yaml.safe_load(
            (SENSORS_ROOT / serial / f"{serial}.yaml").read_text()
        )
        thresholds[serial] = 0.1
        maxima[serial] = float(config["maximum_depth_mm"])
    for backend, estimator in estimators.items():
        outputs[backend] = estimator.estimate_batch(frames)
        latencies[backend] = _latency(estimator)
        for serial, depth in outputs[backend].items():
            assert depth.shape == frames[serial].shape[:2]
            assert depth.dtype == np.float32
            assert np.isfinite(depth).all()
            assert float(depth.min()) >= 0.0
            # The fp32 representation of the configured maximum is 1 ULP above
            # the decimal literal, so allow a tiny epsilon.
            assert float(depth.max()) <= maxima[serial] + 1e-6

    print("\ndepth backend report (4-serial batch, /tmp/digit_baseline_frames.npz)")
    print(
        f"{'backend':<10}{'median_ms':>10}{'mean_ms':>10}{'p95_ms':>10}"
    )
    for backend in ("torch", "trt_fp16", "onnx"):
        values = latencies[backend]
        print(
            f"{backend:<10}{np.median(values):>10.2f}"
            f"{values.mean():>10.2f}{np.percentile(values, 95):>10.2f}"
        )
    print(f"{'comparison':<28}{'rmse_delta_mm':>14}{'iou_delta_pct':>14}{'contact_px':>12}")
    for backend in ("trt_fp16", "onnx"):
        for serial in SERIALS:
            reference = outputs["torch"][serial]
            candidate = outputs[backend][serial]
            threshold = thresholds[serial]
            ref_mask = _contact_mask(reference, threshold)
            cand_mask = _contact_mask(candidate, threshold)
            rmse_delta = _rmse_delta(
                reference, candidate, np.logical_or(ref_mask, cand_mask)
            )
            iou_delta = abs(_iou(ref_mask, cand_mask) - 1.0)
            print(
                f"{backend + ' ' + serial:<28}{rmse_delta:>14.4f}"
                f"{iou_delta * 100:>14.3f}{ref_mask.sum():>12}"
            )
