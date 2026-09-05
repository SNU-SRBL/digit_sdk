#!/usr/bin/env python3
"""Export the DIGIT depth encoder and per-serial decoders to TensorRT FP16.

Writes one shared encoder ONNX + engine (under the first --serials entry) and
one per-serial decoder ONNX + engine under sensors/<serial>/model/depth/.
ONNX outputs are verified against torch (max abs diff < 1e-3) before engines
are built.  Requires tensorrt and onnxruntime-gpu in the running Python.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import timm
import torch
from torch import nn

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from digit_sdk.depth import (  # noqa: E402
    _HOOKS,
    _SensorDecoder,
    _resolve_base,
)


def _network_creation_flags(trt) -> int:
    explicit_batch = getattr(
        trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None
    )
    return 0 if explicit_batch is None else 1 << int(explicit_batch)


class _TraceableEncoder(nn.Module):
    """Hook-free replica of _SharedEncoder returning the same activation tuple."""

    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            "vit_small_patch16_224.dino", pretrained=False
        )

    def forward(self, images):
        x = self.model.patch_embed(images)
        x = self.model._pos_embed(x)
        activations = []
        for index, block in enumerate(self.model.blocks):
            x = block(x)
            if index in _HOOKS:
                activations.append(x)
        return tuple(activations)


def _load_base_encoder() -> _TraceableEncoder:
    encoder = _TraceableEncoder()
    base = _resolve_base()
    payload = torch.load(base, map_location="cpu", weights_only=True)
    state = {
        key.removeprefix("transformer_encoders."): value
        for key, value in payload["model_state_dict"].items()
        if key.startswith("transformer_encoders.")
    }
    encoder.model.load_state_dict(state, strict=True)
    encoder.eval()
    return encoder


def _verify_traceable_encoder(encoder: _TraceableEncoder) -> None:
    from digit_sdk.depth import _SharedEncoder

    hooked = _SharedEncoder()
    hooked.model.load_state_dict(encoder.model.state_dict())
    hooked.eval()
    sample = torch.rand(2, 3, 224, 224)
    with torch.inference_mode():
        reference = hooked(sample)
        traced = encoder(sample)
    for index, (expected, actual) in enumerate(zip(reference, traced)):
        delta = float((expected - actual).abs().max())
        if delta != 0.0:
            raise AssertionError(
                f"traceable encoder activation {index} differs from hooked: {delta}"
            )
    print("traceable encoder matches hooked forward (max diff 0.0)")


def _export_onnx(module, sample_inputs, path, input_names, output_names) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    args = (
        (tuple(sample_inputs),) if len(sample_inputs) > 1 else tuple(sample_inputs)
    )
    torch.onnx.export(
        module,
        args,
        str(path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            name: {0: "batch"} for name in (*input_names, *output_names)
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )


def _verify_onnx(path, module, sample_inputs, input_names, output_names) -> None:
    import onnxruntime as ort

    # CPU-vs-CPU keeps both sides TF32-free so the comparison is a graph
    # equivalence check, not a precision-policy comparison.
    session = ort.InferenceSession(
        str(path),
        providers=["CPUExecutionProvider"],
    )
    module = module.to("cpu")
    sample_inputs = [tensor.detach().cpu() for tensor in sample_inputs]
    args = (
        (tuple(sample_inputs),) if len(sample_inputs) > 1 else tuple(sample_inputs)
    )
    with torch.inference_mode():
        outputs = module(*args)
        expected = outputs if isinstance(outputs, tuple) else (outputs,)
        feeds = {
            name: tensor.numpy()
            for name, tensor in zip(input_names, sample_inputs)
        }
        actual = session.run(None, feeds)
    for name, ref, got in zip(output_names, expected, actual):
        delta = float(np.abs(ref.numpy() - got).max())
        if delta >= 1e-3:
            raise AssertionError(
                f"ONNX {path.name} output {name} max abs diff {delta} >= 1e-3"
            )
        print(f"  {name}: max abs diff {delta:.2e}")


def _to_fp16_onnx(onnx_path: Path, fp16_path: Path) -> None:
    """Create and validate the strongly typed FP16 graph required by TRT 11."""
    import onnx
    from modelopt.onnx import autocast

    model = autocast.convert_to_f16(onnx.load(str(onnx_path)), keep_io_types=True)
    onnx.checker.check_model(model)
    onnx.save(model, str(fp16_path))
    onnx.checker.check_model(str(fp16_path))


def _build_engine(
    onnx_path: Path, engine_path: Path, max_batch: int, fp16_onnx_path: Path
) -> float:
    import tensorrt as trt

    trt11 = not hasattr(trt.BuilderFlag, "FP16")
    engine_onnx = onnx_path
    if trt11:
        _to_fp16_onnx(onnx_path, fp16_onnx_path)
        engine_onnx = fp16_onnx_path

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(_network_creation_flags(trt))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(engine_onnx)):
        errors = [parser.get_error(i) for i in range(parser.num_errors)]
        raise RuntimeError(f"ONNX parse failed for {engine_onnx}: {errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if not trt11:
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("TensorRT platform does not support fast FP16")
        config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        shape = tuple(tensor.shape)

        def _shape(batch, shape=shape):
            return tuple(
                batch if axis == 0 else (dim if dim > 0 else 1)
                for axis, dim in enumerate(shape)
            )

        profile.set_shape(
            tensor.name, _shape(1), _shape(min(max_batch, 4)), _shape(max_batch)
        )
    config.add_optimization_profile(profile)
    started = time.monotonic()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT engine build failed for {engine_onnx}")
    runtime = trt.Runtime(logger)
    if runtime.deserialize_cuda_engine(serialized) is None:
        raise RuntimeError(f"TensorRT engine failed to deserialize: {engine_path}")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)
    return time.monotonic() - started


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Export DIGIT depth models to ONNX and TensorRT FP16 engines"
    )
    parser.add_argument("--serials", required=True, help="comma-separated serials")
    parser.add_argument(
        "--sensors-root",
        type=Path,
        default=_PACKAGE_ROOT / "sensors",
        help="sensors root (default: repo sensors/)",
    )
    args = parser.parse_args(argv)
    serials = [serial.strip() for serial in args.serials.split(",") if serial.strip()]
    if not serials:
        parser.error("--serials requires at least one serial")
    root = args.sensors_root
    max_batch = max(4, len(serials))

    encoder = _load_base_encoder()
    _verify_traceable_encoder(encoder)
    sample = torch.rand(1, 3, 224, 224)
    encoder_onnx = root / serials[0] / "model" / "depth" / "dpt_shared_encoder.onnx"
    print(f"[1/3] exporting shared encoder ONNX -> {encoder_onnx}")
    _export_onnx(
        encoder,
        [sample],
        encoder_onnx,
        ["image"],
        [f"act{index}" for index in range(4)],
    )
    _verify_onnx(
        encoder_onnx,
        encoder,
        [sample],
        ["image"],
        [f"act{index}" for index in range(4)],
    )

    sample_activations = [torch.rand(1, 197, 384) for _ in range(4)]
    for serial in serials:
        model_root = root / serial / "model" / "depth"
        decoder_path = model_root / "decoder.pth"
        if not decoder_path.is_file():
            raise FileNotFoundError(f"depth decoder missing for {serial}: {decoder_path}")
        payload = torch.load(decoder_path, map_location="cpu", weights_only=True)
        decoder = _SensorDecoder()
        decoder.load_state_dict(payload["model_state_dict"], strict=True)
        decoder.eval()
        decoder_onnx = model_root / f"{serial}_decoder.onnx"
        print(f"[2/3] exporting decoder ONNX {serial} -> {decoder_onnx}")
        _export_onnx(
            decoder,
            sample_activations,
            decoder_onnx,
            [f"act{index}" for index in range(4)],
            ["depth"],
        )
        _verify_onnx(
            decoder_onnx,
            decoder,
            sample_activations,
            [f"act{index}" for index in range(4)],
            ["depth"],
        )

    encoder_engine = (
        root / serials[0] / "model" / "depth" / "dpt_shared_encoder_trt_fp16.engine"
    )
    encoder_fp16 = (
        root / serials[0] / "model" / "depth" / "dpt_shared_encoder_fp16.onnx"
    )
    print(f"[3/3] building shared encoder TRT FP16 engine (batch 1..{max_batch})")
    elapsed = _build_engine(encoder_onnx, encoder_engine, max_batch, encoder_fp16)
    print(
        f"  built in {elapsed:.1f}s -> {encoder_engine} "
        f"({encoder_engine.stat().st_size / 1e6:.1f} MB)"
    )
    for serial in serials:
        decoder_onnx = root / serial / "model" / "depth" / f"{serial}_decoder.onnx"
        decoder_engine = (
            root / serial / "model" / "depth" / f"{serial}_decoder_trt_fp16.engine"
        )
        decoder_fp16 = (
            root / serial / "model" / "depth" / f"{serial}_decoder_fp16.onnx"
        )
        print(f"[3/3] building decoder TRT FP16 engine {serial}")
        elapsed = _build_engine(
            decoder_onnx, decoder_engine, max_batch, decoder_fp16
        )
        print(
            f"  built in {elapsed:.1f}s -> {decoder_engine} "
            f"({decoder_engine.stat().st_size / 1e6:.1f} MB)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
