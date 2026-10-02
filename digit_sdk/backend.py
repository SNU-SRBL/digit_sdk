"""Depth-estimation backends: torch, ONNX Runtime GPU, and TensorRT FP16.

`DepthEstimator` picks one backend at construction time.  Every backend
shares the `DepthInput` and metric-depth contract, so the pipeline and the
public estimator API do not change.
"""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch.func import functional_call, stack_module_state, vmap

from .depth import (
    _SharedEncoder,
    _SensorDecoder,
    _image_bytes,
    _resolve_base,
    _torch_estimate_prepared_batch,
    _torch_estimate_prepared_batch_tensors,
    DepthInput,
)
from .depth_contract import validate_depth_raw_mm


_SHARED_ONNX_NAME = "dpt_shared_encoder.onnx"
_SHARED_TRT_NAME = "dpt_shared_encoder_trt_fp16.engine"
_DECODER_ONNX_NAME = "{serial}_decoder.onnx"
_DECODER_TRT_NAME = "{serial}_decoder_trt_fp16.engine"
_ACTIVATION_NAMES = ("act0", "act1", "act2", "act3")


class DepthBackend:
    """Common surface implemented by every depth backend."""

    kind = "backend"

    def prepare(self, image: np.ndarray) -> DepthInput:
        """Prepare one BGR frame without touching the GPU."""
        return DepthInput(_image_bytes(image), image.shape[:2])


def _registered_shared_artifacts(sensors_root, name: str) -> list[Path]:
    """Locate shared encoder artifacts anywhere in the sensor registry.

    The encoder is stored once for all serials (currently under D21119), so
    discovery spans every registered serial rather than only the requested
    ones.
    """
    return sorted(Path(sensors_root).glob(f"*/model/depth/{name}"))


def _artifacts_present(
    sensors_root,
    serials: Sequence[str],
    kind: str,
    *,
    require_current_decoder: bool = False,
) -> bool:
    """True when every serial has a decoder artifact and a shared encoder exists."""
    shared_name = _SHARED_TRT_NAME if kind == "engine" else _SHARED_ONNX_NAME
    decoder_name = _DECODER_TRT_NAME if kind == "engine" else _DECODER_ONNX_NAME
    root = Path(sensors_root)
    shared = bool(_registered_shared_artifacts(root, shared_name))
    decoders = True
    for serial in serials:
        model_root = root / serial / "model" / "depth"
        artifact = model_root / decoder_name.format(serial=serial)
        if not artifact.is_file():
            decoders = False
            break
        if require_current_decoder:
            production_decoder = model_root / "decoder.pth"
            if (
                not production_decoder.is_file()
                or artifact.stat().st_mtime_ns
                < production_decoder.stat().st_mtime_ns
            ):
                decoders = False
                break
    return shared and decoders


def _find_shared_artifact(sensors_root, name: str) -> Path:
    """Locate the one shared encoder artifact anywhere in the sensor registry."""
    matches = _registered_shared_artifacts(sensors_root, name)
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"{name} not found under {sensors_root}; "
        "run scripts/export_trt_fp16.py first"
    )


def _resolve_backend(backend: str, sensors_root, serials: Sequence[str]) -> str:
    """Resolve the requested backend name, applying auto selection."""
    if backend == "auto":
        if _artifacts_present(
            sensors_root, serials, "engine", require_current_decoder=True
        ):
            return "trt_fp16"
        if _artifacts_present(
            sensors_root, serials, "onnx", require_current_decoder=True
        ):
            return "onnx"
        return "torch"
    if backend not in ("trt_fp16", "onnx", "torch"):
        raise ValueError(
            f"unknown depth backend {backend!r}; expected auto|trt_fp16|onnx|torch"
        )
    return backend


def _load_depth_limits(sensors_root, serials: Sequence[str]) -> Dict[str, float]:
    """Read the metric-depth scale from each registered sensor config."""
    root = Path(sensors_root)
    maximum_depth_mm: Dict[str, float] = {}
    for serial in serials:
        config_path = root / serial / f"{serial}.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"sensor config missing for {serial}: {config_path}")
        config = yaml.safe_load(config_path.read_text()) or {}
        maximum = float(config["maximum_depth_mm"])
        if not np.isfinite(maximum) or maximum <= 0:
            raise ValueError(f"invalid maximum_depth_mm for {serial}")
        maximum_depth_mm[serial] = maximum
    return maximum_depth_mm


class TorchDepthBackend(DepthBackend):
    """Current PyTorch path: frozen shared encoder plus per-serial decoders."""

    kind = "torch"

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
        self._maximum_depth_mm = _load_depth_limits(root, self.serials)
        for serial in self.serials:
            model_root = root / serial / "model" / "depth"
            decoder_path = model_root / "decoder.pth"
            if not decoder_path.is_file():
                raise FileNotFoundError(
                    f"production depth model missing for {serial}: {model_root}"
                )
            payload = torch.load(
                decoder_path, map_location="cpu", weights_only=True
            )
            decoder = _SensorDecoder()
            decoder.load_state_dict(payload["model_state_dict"], strict=True)
            for parameter in decoder.parameters():
                parameter.requires_grad = False
            self._decoders[serial] = decoder.to(self.device).eval()
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

    @torch.inference_mode()
    def estimate_prepared_batch(
        self, inputs: Mapping[str, DepthInput]
    ) -> Dict[str, np.ndarray]:
        return _torch_estimate_prepared_batch(self, inputs)

    @torch.inference_mode()
    def estimate_prepared_batch_tensors(
        self, inputs: Mapping[str, DepthInput]
    ) -> Dict[str, torch.Tensor]:
        return _torch_estimate_prepared_batch_tensors(self, inputs)


class OnnxDepthBackend(DepthBackend):
    """ONNX Runtime GPU (FP32) path over the same exported graphs."""

    kind = "onnx"

    def __init__(
        self,
        serials: Sequence[str],
        sensors_root: str | Path,
        device: str | torch.device = "cuda",
    ):
        if not serials:
            raise ValueError("at least one sensor serial is required")
        if len(set(serials)) != len(serials):
            raise ValueError("sensor serials must be unique")
        self.serials = tuple(serials)
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("onnx depth backend requires CUDA")

        root = Path(sensors_root)
        self._maximum_depth_mm = _load_depth_limits(root, serials)
        shared = _find_shared_artifact(root, _SHARED_ONNX_NAME)
        decoder_paths = {
            serial: root / serial / "model" / "depth"
            / _DECODER_ONNX_NAME.format(serial=serial)
            for serial in serials
        }
        missing = [
            str(path) for path in decoder_paths.values() if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"missing ONNX decoder files: {missing}; "
                "run scripts/export_trt_fp16.py first"
            )
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._encoder_session = ort.InferenceSession(str(shared), providers=providers)
        self._decoder_sessions = {
            serial: ort.InferenceSession(str(path), providers=providers)
            for serial, path in decoder_paths.items()
        }

    @torch.inference_mode()
    def estimate_prepared_batch_tensors(
        self, inputs: Mapping[str, DepthInput]
    ) -> Dict[str, torch.Tensor]:
        unknown = set(inputs) - set(self.serials)
        if unknown:
            raise KeyError(f"unconfigured sensor serials: {sorted(unknown)}")
        ordered = [serial for serial in self.serials if serial in inputs]
        if not ordered:
            return {}
        shapes = {serial: inputs[serial].output_shape for serial in ordered}
        image = (
            torch.stack([inputs[serial].tensor for serial in ordered])
            .float()
            .div_(255.0)
            .sub_(0.5)
            .div_(0.5)
            .numpy()
        )
        activations = self._encoder_session.run(None, {"image": image})
        results = {}
        for index, serial in enumerate(ordered):
            decoder_inputs = {
                name: activations[i][index:index + 1]
                for i, name in enumerate(_ACTIVATION_NAMES)
            }
            depth = self._decoder_sessions[serial].run(None, decoder_inputs)[0]
            tensor = (
                torch.from_numpy(depth)[0, 0].to(self.device)
                * self._maximum_depth_mm[serial]
            )
            height, width = shapes[serial]
            resized = torch.nn.functional.interpolate(
                tensor.unsqueeze(0).unsqueeze(0),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            results[serial] = torch.clamp(
                resized, 0.0, self._maximum_depth_mm[serial]
            )
        return results

    @torch.inference_mode()
    def estimate_prepared_batch(
        self, inputs: Mapping[str, DepthInput]
    ) -> Dict[str, np.ndarray]:
        tensors = self.estimate_prepared_batch_tensors(inputs)
        results = {}
        for serial, tensor in tensors.items():
            depth = tensor.float().cpu().numpy().astype(np.float32)
            validate_depth_raw_mm(
                depth, expected_shape=inputs[serial].output_shape
            )
            results[serial] = depth
        return results


class TrtFP16DepthBackend(DepthBackend):
    """TensorRT FP16 path over exported encoder and per-serial decoder engines."""

    kind = "trt_fp16"

    def __init__(
        self,
        serials: Sequence[str],
        sensors_root: str | Path,
        device: str | torch.device = "cuda",
    ):
        if not serials:
            raise ValueError("at least one sensor serial is required")
        if len(set(serials)) != len(serials):
            raise ValueError("sensor serials must be unique")
        self.serials = tuple(serials)
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("trt_fp16 depth backend requires CUDA")
        self._stream = torch.cuda.Stream(device=self.device)

        import tensorrt as trt

        self._trt = trt
        root = Path(sensors_root)
        self._maximum_depth_mm = _load_depth_limits(root, serials)
        shared = _find_shared_artifact(root, _SHARED_TRT_NAME)
        decoder_paths = {
            serial: root / serial / "model" / "depth"
            / _DECODER_TRT_NAME.format(serial=serial)
            for serial in serials
        }
        missing = [
            str(path) for path in decoder_paths.values() if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"missing TensorRT engine files: {missing}; "
                "run scripts/export_trt_fp16.py first"
            )
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self._engines: Dict[str, object] = {}
        self._encoder_engine = self._deserialize(runtime, shared)
        self._engines["_shared"] = self._encoder_engine
        self._encoder_context = self._encoder_engine.create_execution_context()
        self._decoder_contexts = {}
        for serial in serials:
            engine = self._deserialize(runtime, decoder_paths[serial])
            self._engines[serial] = engine
            self._decoder_contexts[serial] = engine.create_execution_context()

    def _deserialize(self, runtime, path: Path):
        engine = runtime.deserialize_cuda_engine(path.read_bytes())
        if engine is None:
            raise RuntimeError(f"failed to load TensorRT engine: {path}")
        return engine

    def _execute(self, context, feeds: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run one engine on the dedicated CUDA stream with shape-matched buffers."""
        engine = context.engine
        caller_stream = torch.cuda.current_stream(self.device)
        with torch.cuda.stream(self._stream):
            # Feeds may have been produced on the caller's stream; order the
            # dedicated stream after it before casting/copying and enqueueing.
            self._stream.wait_stream(caller_stream)
            for name, tensor in feeds.items():
                expected = engine.get_tensor_dtype(name)
                if expected == self._trt.float16 and tensor.dtype != torch.float16:
                    tensor = tensor.half()
                elif expected == self._trt.float32 and tensor.dtype != torch.float32:
                    tensor = tensor.float()
                tensor = tensor.contiguous()
                context.set_input_shape(name, tuple(tensor.shape))
                context.set_tensor_address(name, tensor.data_ptr())
                feeds[name] = tensor
            outputs = {}
            for index in range(engine.num_io_tensors):
                name = engine.get_tensor_name(index)
                if name in feeds:
                    continue
                shape = tuple(context.get_tensor_shape(name))
                dtype = engine.get_tensor_dtype(name)
                if dtype == self._trt.float16:
                    torch_dtype = torch.float16
                elif dtype == self._trt.float32:
                    torch_dtype = torch.float32
                else:
                    raise RuntimeError(f"unsupported engine dtype for {name}: {dtype}")
                tensor = torch.empty(shape, dtype=torch_dtype, device=self.device)
                context.set_tensor_address(name, tensor.data_ptr())
                outputs[name] = tensor
            if not context.execute_async_v3(self._stream.cuda_stream):
                raise RuntimeError("TensorRT engine execution failed")
            for tensor in outputs.values():
                tensor.record_stream(self._stream)
        self._stream.synchronize()
        return outputs

    @torch.inference_mode()
    def estimate_prepared_batch_tensors(
        self, inputs: Mapping[str, DepthInput]
    ) -> Dict[str, torch.Tensor]:
        unknown = set(inputs) - set(self.serials)
        if unknown:
            raise KeyError(f"unconfigured sensor serials: {sorted(unknown)}")
        ordered = [serial for serial in self.serials if serial in inputs]
        if not ordered:
            return {}
        shapes = {serial: inputs[serial].output_shape for serial in ordered}
        image = (
            torch.stack([inputs[serial].tensor for serial in ordered])
            .to(self.device)
            .float()
            .div_(255.0)
            .sub_(0.5)
            .div_(0.5)
        )
        activations = self._execute(self._encoder_context, {"image": image})
        activation_tensors = [
            activations[name].float() for name in _ACTIVATION_NAMES
        ]
        results = {}
        for index, serial in enumerate(ordered):
            depth = self._execute(
                self._decoder_contexts[serial],
                {
                    name: activation_tensors[i][index:index + 1]
                    for i, name in enumerate(_ACTIVATION_NAMES)
                },
            )["depth"].float()[0, 0] * self._maximum_depth_mm[serial]
            height, width = shapes[serial]
            resized = torch.nn.functional.interpolate(
                depth.unsqueeze(0).unsqueeze(0),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            results[serial] = torch.clamp(
                resized, 0.0, self._maximum_depth_mm[serial]
            )
        return results

    @torch.inference_mode()
    def estimate_prepared_batch(
        self, inputs: Mapping[str, DepthInput]
    ) -> Dict[str, np.ndarray]:
        tensors = self.estimate_prepared_batch_tensors(inputs)
        results = {}
        for serial, tensor in tensors.items():
            depth = tensor.float().cpu().numpy().astype(np.float32)
            validate_depth_raw_mm(
                depth, expected_shape=inputs[serial].output_shape
            )
            results[serial] = depth
        return results
