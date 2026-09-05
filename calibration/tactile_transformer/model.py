"""Pinned NeuralFeels DPT model and decoder-only checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from einops.layers.torch import Rearrange
from huggingface_hub import hf_hub_download
import timm
import torch
from torch import nn


BASE_REPOSITORY = "suddhu/tactile_transformer"
BASE_FILENAME = "dpt_real.p"
BASE_REVISION = "b05cfe1df2c90d3d91f8378633173b26de5a2d2c"


class ReadProjection(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(2 * features, features), nn.GELU()
        )

    def forward(self, values):
        readout = values[:, 0].unsqueeze(1).expand_as(values[:, 1:])
        return self.project(torch.cat((values[:, 1:], readout), -1))


class Resample(nn.Module):
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
            raise ValueError(f"unsupported DPT scale: {scale}")

    def forward(self, values):
        return self.conv2(self.conv1(values))


class Reassemble(nn.Module):
    def __init__(self, scale: int, embedding: int = 384, features: int = 128):
        super().__init__()
        self.read = ReadProjection(embedding)
        self.tokens_to_image = Rearrange(
            "b (h w) c -> b c h w", h=14, w=14
        )
        self.resample = Resample(scale, embedding, features)

    def forward(self, values):
        return self.resample(self.tokens_to_image(self.read(values)))


class ResidualConvUnit(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(features, features, 3, padding=1)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1)

    def forward(self, values):
        output = self.conv1(self.relu(values))
        output = self.conv2(self.relu(output))
        return output + values


class Fusion(nn.Module):
    def __init__(self, features: int = 128):
        super().__init__()
        self.res_conv1 = ResidualConvUnit(features)
        self.res_conv2 = ResidualConvUnit(features)

    def forward(self, values, previous=None):
        output = self.res_conv1(values)
        if previous is not None:
            output = output + previous
        output = self.res_conv2(output)
        return nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=True
        )


class DepthHead(nn.Module):
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


class TactileDPT(nn.Module):
    """Exact real-DIGIT NeuralFeels model configuration."""

    hooks = (2, 5, 8, 11)

    def __init__(self):
        super().__init__()
        self.transformer_encoders = timm.create_model(
            "vit_small_patch16_224.dino", pretrained=False
        )
        self.activations = {}
        for block_index in self.hooks:
            self.transformer_encoders.blocks[block_index].register_forward_hook(
                self._capture(block_index)
            )
        self.reassembles = nn.ModuleList(
            [Reassemble(scale) for scale in (4, 8, 16, 32)]
        )
        self.fusions = nn.ModuleList([Fusion() for _ in range(4)])
        self.head_depth = DepthHead()

    def _capture(self, index: int):
        def hook(_module, _inputs, output):
            self.activations[index] = output
        return hook

    def forward(self, image):
        self.transformer_encoders(image)
        previous = None
        for index in range(3, -1, -1):
            feature = self.reassembles[index](
                self.activations[self.hooks[index]]
            )
            previous = self.fusions[index](feature, previous)
        return self.head_depth(previous)


def resolve_base(cache_dir: Path = None) -> Path:
    """Resolve the pinned official Hugging Face artifact."""
    path = Path(hf_hub_download(
        repo_id=BASE_REPOSITORY,
        filename=BASE_FILENAME,
        revision=BASE_REVISION,
        cache_dir=str(cache_dir) if cache_dir else None,
    ))
    return path


def load_base(model: TactileDPT, checkpoint: Path) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model_state_dict"], strict=True)


def freeze_encoder(model: TactileDPT) -> None:
    for parameter in model.transformer_encoders.parameters():
        parameter.requires_grad = False
    model.transformer_encoders.eval()


def decoder_parameters(model: TactileDPT) -> Iterable[nn.Parameter]:
    return (parameter for parameter in model.parameters()
            if parameter.requires_grad)


def decoder_state(model: TactileDPT):
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith("transformer_encoders.")
    }
