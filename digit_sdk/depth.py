"""Production metric-depth estimator for calibrated DIGIT sensors.

The estimator owns one frozen encoder and one decoder per sensor serial.  Its
public contract is batch-first and algorithm-neutral: BGR images in, raw
indentation-positive depth in millimetres out.
"""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
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
from torch.func import functional_call, stack_module_state, vmap

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


class DepthEstimator:
    """Batch-first production estimator with serial-bound calibration."""

    def __init__(
        self,
        serials: Sequence[str],
        sensors_root: str | Path,
        device: str | torch.device = "cuda",
        base_cache_dir: str | Path | None = None,
    ):
        if not serials:
            raise ValueError("at least one sensor serial is required")
        if len(set(serials)) != len(serials):
            raise ValueError("sensor serials must be unique")
        self.serials = tuple(serials)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA depth inference requested but unavailable")

        root = Path(sensors_root)
        self._encoder = _SharedEncoder()
        base = _resolve_base(
            Path(base_cache_dir) if base_cache_dir is not None else None
        )
        base_payload = torch.load(base, map_location="cpu", weights_only=True)
        encoder_state = {
            key.removeprefix("transformer_encoders."): value
            for key, value in base_payload["model_state_dict"].items()
            if key.startswith("transformer_encoders.")
        }
        self._encoder.model.load_state_dict(encoder_state, strict=True)
        for parameter in self._encoder.parameters():
            parameter.requires_grad = False
        self._encoder.to(self.device).eval()
        del base_payload, encoder_state

        self._decoders: Dict[str, _SensorDecoder] = {}
        self._maximum_depth_mm: Dict[str, float] = {}
        for serial in self.serials:
            model_root = root / serial / "model" / "depth"
            metadata_path = model_root / "metadata.json"
            decoder_path = model_root / "decoder.pth"
            if not metadata_path.is_file() or not decoder_path.is_file():
                raise FileNotFoundError(
                    f"production depth model missing for {serial}: {model_root}"
                )
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("serial") != serial:
                raise ValueError(f"depth model serial mismatch for {serial}")
            expected_sha = metadata.get("decoder_sha256")
            if expected_sha and _sha256(decoder_path) != expected_sha:
                raise ValueError(f"depth decoder verification failed: {decoder_path}")
            base_metadata = metadata.get("base", {})
            if (
                base_metadata.get("revision") != _BASE_REVISION
                or base_metadata.get("sha256") != _BASE_SHA256
            ):
                raise ValueError(f"unsupported depth encoder metadata for {serial}")

            payload = torch.load(
                decoder_path, map_location="cpu", weights_only=True
            )
            decoder = _SensorDecoder()
            decoder.load_state_dict(payload["model_state_dict"], strict=True)
            for parameter in decoder.parameters():
                parameter.requires_grad = False
            self._decoders[serial] = decoder.to(self.device).eval()
            maximum = float(metadata["maximum_depth_mm"])
            if not np.isfinite(maximum) or maximum <= 0:
                raise ValueError(f"invalid maximum_depth_mm for {serial}")
            self._maximum_depth_mm[serial] = maximum
        self._decoder_streams = {
            serial: torch.cuda.Stream(device=self.device)
            for serial in self.serials
        } if self.device.type == "cuda" else {}
        self._postprocess_pool = (
            ThreadPoolExecutor(max_workers=min(4, len(self.serials)))
            if len(self.serials) > 1 else None
        )
        self._compiled_full_batch = None
        if self.device.type == "cuda" and len(self.serials) > 1:
            torch.backends.cudnn.benchmark = True
            decoder_parameters, decoder_buffers = stack_module_state(
                [self._decoders[serial] for serial in self.serials]
            )
            base_decoder = copy.deepcopy(
                self._decoders[self.serials[0]]
            ).to("meta")

            def call_decoder(parameters, buffers, activations):
                return functional_call(
                    base_decoder, (parameters, buffers), (activations,)
                )

            batched_decoder = vmap(
                call_decoder, in_dims=(0, 0, 0)
            )
            def full_batch(images):
                activations = self._encoder(images)
                batched_activations = tuple(
                    value.unsqueeze(1) for value in activations
                )
                return batched_decoder(
                    decoder_parameters,
                    decoder_buffers,
                    batched_activations,
                )[:, 0, 0]

            self._compiled_full_batch = torch.compile(
                full_batch, mode="reduce-overhead", fullgraph=True
            )

    def prepare(self, image: np.ndarray) -> DepthInput:
        """Prepare one BGR frame without touching the GPU."""
        return DepthInput(_image_bytes(image), image.shape[:2])

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
        """Estimate a prepared batch using parallel serial-bound decoders."""
        unknown = set(inputs) - set(self.serials)
        if unknown:
            raise KeyError(f"unconfigured sensor serials: {sorted(unknown)}")
        ordered = [serial for serial in self.serials if serial in inputs]
        if not ordered:
            return {}
        shapes = {serial: inputs[serial].output_shape for serial in ordered}
        images = (
            torch.stack([inputs[s].tensor for s in ordered])
            .to(self.device, non_blocking=self.device.type == "cuda")
            .float()
            .div_(255.0)
            .sub_(0.5)
            .div_(0.5)
        )
        predictions = []
        if (
            self._compiled_full_batch is not None
            and tuple(ordered) == self.serials
        ):
            torch.compiler.cudagraph_mark_step_begin()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                batch = self._compiled_full_batch(images)
            predictions = [
                batch[index] * self._maximum_depth_mm[serial]
                for index, serial in enumerate(ordered)
            ]
        else:
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.device.type == "cuda",
            ):
                activations = self._encoder(images)
        if (
            self.device.type == "cuda"
            and not predictions
        ):
            encoder_complete = torch.cuda.Event()
            encoder_complete.record(torch.cuda.current_stream(self.device))
            for index, serial in enumerate(ordered):
                stream = self._decoder_streams[serial]
                stream.wait_event(encoder_complete)
                with torch.cuda.stream(stream), torch.autocast(
                    device_type="cuda", dtype=torch.float16
                ):
                    prediction = self._decoders[serial](
                        tuple(value[index:index + 1] for value in activations)
                    )[0, 0] * self._maximum_depth_mm[serial]
                    predictions.append(prediction)
            current = torch.cuda.current_stream(self.device)
            for serial in ordered:
                current.wait_stream(self._decoder_streams[serial])
        elif self.device.type != "cuda":
            predictions = [
                self._decoders[serial](
                    tuple(value[index:index + 1] for value in activations)
                )[0, 0] * self._maximum_depth_mm[serial]
                for index, serial in enumerate(ordered)
            ]

        resize_inputs = []
        for serial, prediction in zip(ordered, predictions):
            height, width = shapes[serial]
            resize_inputs.append((
                prediction.float().cpu().numpy(),
                width,
                height,
                self._maximum_depth_mm[serial],
            ))
        if self._postprocess_pool is None or len(resize_inputs) == 1:
            resized = map(_resize_depth, resize_inputs)
        else:
            resized = self._postprocess_pool.map(_resize_depth, resize_inputs)

        results = {}
        for serial, depth in zip(ordered, resized):
            height, width = shapes[serial]
            validate_depth_raw_mm(depth, expected_shape=(height, width))
            results[serial] = depth
        return results

    def estimate(self, serial: str, frame: np.ndarray) -> np.ndarray:
        """Convenience wrapper for one frame; production scheduling is batched."""
        return self.estimate_batch({serial: frame})[serial]
