"""
Production metric-depth estimator for calibrated DIGIT sensors.

The estimator owns one frozen encoder and one decoder per sensor serial.  Its
public contract is batch-first and algorithm-neutral: BGR images in, raw
indentation-positive depth in millimetres out.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import cv2
from einops.layers.torch import Rearrange
from huggingface_hub import hf_hub_download
import numpy as np
import timm
import torch
from torch import nn

from .depth_contract import validate_depth_raw_mm


_MODEL_SIZE = (224, 224)
_HOOKS = (2, 5, 8, 11)
_BASE_REPOSITORY = "suddhu/tactile_transformer"
_BASE_FILENAME = "dpt_real.p"
_BASE_REVISION = "b05cfe1df2c90d3d91f8378633173b26de5a2d2c"
_BASE_SHA256 = "7ab6864c03af38def576e165fe4b1d44646e5dad2b66bf4b62ea47d894007752"
_BASE_SIZE = 310_969_033


def _resize_depth(item) -> np.ndarray:
    prediction, width, height, maximum_depth_mm = item
    depth = cv2.resize(
        prediction, (width, height), interpolation=cv2.INTER_LINEAR
    )
    return np.clip(depth, 0.0, maximum_depth_mm).astype(np.float32)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_base(cache_dir: Path | None = None) -> Path:
    path = Path(hf_hub_download(
        repo_id=_BASE_REPOSITORY,
        filename=_BASE_FILENAME,
        revision=_BASE_REVISION,
        cache_dir=str(cache_dir) if cache_dir else None,
    ))
    if path.stat().st_size != _BASE_SIZE or _sha256(path) != _BASE_SHA256:
        raise ValueError(f"base depth checkpoint verification failed: {path}")
    return path


class _ReadProjection(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(2 * features, features), nn.GELU()
        )

    def forward(self, values):
        readout = values[:, 0].unsqueeze(1).expand_as(values[:, 1:])
        return self.project(torch.cat((values[:, 1:], readout), -1))


class _Resample(nn.Module):
    def __init__(self, scale: int, embedding: int = 384, features: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(embedding, features, 1)
        if scale == 4:
            self.conv2 = nn.ConvTranspose2d(features, features, 4, stride=4)
        elif scale == 8:
            self.conv2 = nn.ConvTranspose2d(features, features, 2, stride=2)
        elif scale == 16:
            self.conv2 = nn.Identity()
        elif scale == 32:
            self.conv2 = nn.Conv2d(features, features, 2, stride=2)
        else:
            raise ValueError(f"unsupported decoder scale: {scale}")

    def forward(self, values):
        return self.conv2(self.conv1(values))


class _Reassemble(nn.Module):
    def __init__(self, scale: int, embedding: int = 384, features: int = 128):
        super().__init__()
        self.read = _ReadProjection(embedding)
        self.tokens_to_image = Rearrange(
            "b (h w) c -> b c h w", h=14, w=14
        )
        self.resample = _Resample(scale, embedding, features)

    def forward(self, values):
        return self.resample(self.tokens_to_image(self.read(values)))


class _ResidualConvUnit(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(features, features, 3, padding=1)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1)

    def forward(self, values):
        output = self.conv1(self.relu(values))
        output = self.conv2(self.relu(output))
        return output + values


class _Fusion(nn.Module):
    def __init__(self, features: int = 128):
        super().__init__()
        self.res_conv1 = _ResidualConvUnit(features)
        self.res_conv2 = _ResidualConvUnit(features)

    def forward(self, values, previous=None):
        output = self.res_conv1(values)
        if previous is not None:
            output = output + previous
        output = self.res_conv2(output)
        return nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=True
        )


class _DepthHead(nn.Module):
    def __init__(self, features: int = 128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(features, features // 2, 3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, values):
        return self.head(values)


class _SharedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            "vit_small_patch16_224.dino", pretrained=False
        )
        self.activations = {}
        for index in _HOOKS:
            self.model.blocks[index].register_forward_hook(self._capture(index))

    def _capture(self, index: int):
        def hook(_module, _inputs, output):
            self.activations[index] = output
        return hook

    def forward(self, images):
        self.activations.clear()
        self.model(images)
        return tuple(self.activations[index] for index in _HOOKS)


class _SensorDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.reassembles = nn.ModuleList(
            [_Reassemble(scale) for scale in (4, 8, 16, 32)]
        )
        self.fusions = nn.ModuleList([_Fusion() for _ in range(4)])
        self.head_depth = _DepthHead()

    def forward(self, activations):
        previous = None
        for index in range(3, -1, -1):
            feature = self.reassembles[index](activations[index])
            previous = self.fusions[index](feature, previous)
        return self.head_depth(previous)


def _image_bytes(image: np.ndarray) -> torch.Tensor:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise TypeError("depth input must be a uint8 numpy.ndarray")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("depth input must have shape (height, width, 3)")
    rgb = cv2.cvtColor(
        cv2.resize(image, _MODEL_SIZE, interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2RGB,
    )
    return torch.from_numpy(rgb).permute(2, 0, 1)


def _image_tensor(image: np.ndarray) -> torch.Tensor:
    tensor = _image_bytes(image).float().div(255.0)
    return (tensor - 0.5) / 0.5


@dataclass(frozen=True)
class DepthInput:
    """Resized RGB bytes prepared on CPU while the prior GPU batch runs."""

    tensor: torch.Tensor
    output_shape: tuple[int, int]


@torch.inference_mode()
def _torch_estimate_prepared_batch_tensors(
    estimator, inputs: Mapping[str, DepthInput]
) -> Dict[str, torch.Tensor]:
    """Estimate a batch on GPU and return float32 depth tensors."""
    unknown = set(inputs) - set(estimator.serials)
    if unknown:
        raise KeyError(f"unconfigured sensor serials: {sorted(unknown)}")
    ordered = [serial for serial in estimator.serials if serial in inputs]
    if not ordered:
        return {}
    shapes = {serial: inputs[serial].output_shape for serial in ordered}
    images = (
        torch.stack([inputs[s].tensor for s in ordered])
        .to(estimator.device, non_blocking=True)
        .float()
        .div_(255.0)
        .sub_(0.5)
        .div_(0.5)
    )
    predictions = []
    if (
        estimator._compiled_full_batch is not None
        and tuple(ordered) == estimator.serials
    ):
        torch.compiler.cudagraph_mark_step_begin()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            batch = estimator._compiled_full_batch(images)
        predictions = [
            batch[index] * estimator._maximum_depth_mm[serial]
            for index, serial in enumerate(ordered)
        ]
    else:
        with torch.autocast(
            device_type=estimator.device.type,
            dtype=torch.float16,
            enabled=estimator.device.type == "cuda",
        ):
            activations = estimator._encoder(images)
        if not predictions:
            encoder_complete = torch.cuda.Event()
            encoder_complete.record(
                torch.cuda.current_stream(estimator.device)
            )
            for index, serial in enumerate(ordered):
                stream = estimator._decoder_streams[serial]
                stream.wait_event(encoder_complete)
                with torch.cuda.stream(stream), torch.autocast(
                    device_type="cuda", dtype=torch.float16
                ):
                    prediction = estimator._decoders[serial](
                        tuple(
                            value[index:index + 1]
                            for value in activations
                        )
                    )[0, 0] * estimator._maximum_depth_mm[serial]
                    predictions.append(prediction)
            current = torch.cuda.current_stream(estimator.device)
            for serial in ordered:
                current.wait_stream(estimator._decoder_streams[serial])

    results = {}
    for serial, prediction in zip(ordered, predictions):
        height, width = shapes[serial]
        resized = torch.nn.functional.interpolate(
            prediction.unsqueeze(0).unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        results[serial] = torch.clamp(
            resized, 0.0, estimator._maximum_depth_mm[serial]
        )
    return results


@torch.inference_mode()
def _torch_estimate_prepared_batch(
    estimator, inputs: Mapping[str, DepthInput]
) -> Dict[str, np.ndarray]:
    """Estimate a prepared batch using parallel serial-bound decoders."""
    unknown = set(inputs) - set(estimator.serials)
    if unknown:
        raise KeyError(f"unconfigured sensor serials: {sorted(unknown)}")
    ordered = [serial for serial in estimator.serials if serial in inputs]
    if not ordered:
        return {}
    shapes = {serial: inputs[serial].output_shape for serial in ordered}
    if estimator.device.type == "cuda":
        tensors = _torch_estimate_prepared_batch_tensors(estimator, inputs)
        results = {}
        for serial in ordered:
            depth = (
                tensors[serial].float().cpu().numpy().astype(np.float32)
            )
            height, width = shapes[serial]
            validate_depth_raw_mm(
                depth, expected_shape=(height, width)
            )
            results[serial] = depth
        return results

    images = (
        torch.stack([inputs[s].tensor for s in ordered])
        .to(estimator.device, non_blocking=estimator.device.type == "cuda")
        .float()
        .div_(255.0)
        .sub_(0.5)
        .div_(0.5)
    )
    predictions = []
    if (
        estimator._compiled_full_batch is not None
        and tuple(ordered) == estimator.serials
    ):
        torch.compiler.cudagraph_mark_step_begin()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            batch = estimator._compiled_full_batch(images)
        predictions = [
            batch[index] * estimator._maximum_depth_mm[serial]
            for index, serial in enumerate(ordered)
        ]
    else:
        with torch.autocast(
            device_type=estimator.device.type,
            dtype=torch.float16,
            enabled=estimator.device.type == "cuda",
        ):
            activations = estimator._encoder(images)
    if (
        estimator.device.type == "cuda"
        and not predictions
    ):
        encoder_complete = torch.cuda.Event()
        encoder_complete.record(torch.cuda.current_stream(estimator.device))
        for index, serial in enumerate(ordered):
            stream = estimator._decoder_streams[serial]
            stream.wait_event(encoder_complete)
            with torch.cuda.stream(stream), torch.autocast(
                device_type="cuda", dtype=torch.float16
            ):
                prediction = estimator._decoders[serial](
                    tuple(value[index:index + 1] for value in activations)
                )[0, 0] * estimator._maximum_depth_mm[serial]
                predictions.append(prediction)
        current = torch.cuda.current_stream(estimator.device)
        for serial in ordered:
            current.wait_stream(estimator._decoder_streams[serial])
    elif estimator.device.type != "cuda":
        predictions = [
            estimator._decoders[serial](
                tuple(value[index:index + 1] for value in activations)
            )[0, 0] * estimator._maximum_depth_mm[serial]
            for index, serial in enumerate(ordered)
        ]

    resize_inputs = []
    for serial, prediction in zip(ordered, predictions):
        height, width = shapes[serial]
        resize_inputs.append((
            prediction.float().cpu().numpy(),
            width,
            height,
            estimator._maximum_depth_mm[serial],
        ))
    if estimator._postprocess_pool is None or len(resize_inputs) == 1:
        resized = map(_resize_depth, resize_inputs)
    else:
        resized = estimator._postprocess_pool.map(_resize_depth, resize_inputs)

    results = {}
    for serial, depth in zip(ordered, resized):
        height, width = shapes[serial]
        validate_depth_raw_mm(depth, expected_shape=(height, width))
        results[serial] = depth
    return results


class DepthEstimator:
    """Batch-first production estimator with a selectable inference backend.

    `backend="auto"` selects TensorRT FP16 when engines exist, then ONNX
    Runtime GPU, then the PyTorch path.  The public API and the metric-depth
    contract are identical across backends.
    """

    def __init__(
        self,
        serials: Sequence[str],
        sensors_root: str | Path,
        device: str | torch.device = "cuda",
        base_cache_dir: str | Path | None = None,
        backend: str = "auto",
    ):
        if not serials:
            raise ValueError("at least one sensor serial is required")
        if len(set(serials)) != len(serials):
            raise ValueError("sensor serials must be unique")
        self.serials = tuple(serials)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA depth inference requested but unavailable")

        from .backend import (
            _resolve_backend,
            OnnxDepthBackend,
            TorchDepthBackend,
            TrtFP16DepthBackend,
        )

        backend_name = _resolve_backend(backend, sensors_root, serials)
        self._backend_name = backend_name
        if backend_name == "torch":
            self._backend = TorchDepthBackend(
                serials, sensors_root, device, base_cache_dir
            )
            # Mirror the torch internals so the legacy torch-path helpers and
            # tests that toggle `_compiled_full_batch` keep working.
            self._encoder = self._backend._encoder
            self._decoders = self._backend._decoders
            self._maximum_depth_mm = self._backend._maximum_depth_mm
            self._decoder_streams = self._backend._decoder_streams
            self._postprocess_pool = self._backend._postprocess_pool
            self._compiled_full_batch = self._backend._compiled_full_batch
        elif backend_name == "onnx":
            self._backend = OnnxDepthBackend(serials, sensors_root, device)
        else:
            self._backend = TrtFP16DepthBackend(serials, sensors_root, device)

    def prepare(self, image: np.ndarray) -> DepthInput:
        """Prepare one BGR frame without touching the GPU."""
        return self._backend.prepare(image)

    @torch.inference_mode()
    def estimate_batch(
        self, frames: Mapping[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Estimate raw millimetre depth for the supplied sensor frames."""
        return self.estimate_prepared_batch({
            serial: self.prepare(frame) for serial, frame in frames.items()
        })

    @torch.inference_mode()
    def estimate_prepared_batch(
        self, inputs: Mapping[str, DepthInput]
    ) -> Dict[str, np.ndarray]:
        """Estimate a prepared batch using the active backend."""
        backend = getattr(self, "_backend", None)
        if backend is not None and backend.kind != "torch":
            return backend.estimate_prepared_batch(inputs)
        return _torch_estimate_prepared_batch(self, inputs)

    @torch.inference_mode()
    def estimate_prepared_batch_tensors(
        self, inputs: Mapping[str, DepthInput]
    ) -> Dict[str, torch.Tensor]:
        """Estimate a batch and return GPU float32 depth tensors."""
        backend = getattr(self, "_backend", None)
        if backend is not None and backend.kind != "torch":
            return backend.estimate_prepared_batch_tensors(inputs)
        return _torch_estimate_prepared_batch_tensors(self, inputs)

    def estimate(self, serial: str, frame: np.ndarray) -> np.ndarray:
        """Provide a convenience wrapper for one frame; scheduling is batched."""
        return self.estimate_batch({serial: frame})[serial]
