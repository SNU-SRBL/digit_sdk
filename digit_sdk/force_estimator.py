"""
Force estimation using Sparsh vision-based tactile sensing models.

This module implements force field and force vector estimation using pretrained
Sparsh models (ViT encoder + DPT decoder) with temporal frame pairs.
"""

import os
import pathlib
import importlib.util
import warnings
from typing import Dict, Optional, Tuple, Union

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

from .temporal_buffer import TemporalBuffer


def _load_encoder_checkpoint(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """
    Load encoder weights from Lightning checkpoint.

    Args:
        checkpoint_path: Path to .ckpt file

    Returns
    -------
    OrderedDict of model weights with 'student_encoder.backbone.' prefix removed

    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model']

    # Remove 'student_encoder.backbone.' prefix
    cleaned_state_dict = {}
    prefix = 'student_encoder.backbone.'
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            cleaned_state_dict[new_key] = value

    return cleaned_state_dict


def _ensure_sparsh_path() -> str:
    """Ensure Sparsh source root is available on sys.path."""
    import sys

    sdk_root = pathlib.Path(__file__).resolve().parents[1]
    sparsh_root_path = sdk_root / 'sparsh-main'
    if not sparsh_root_path.exists():
        cwd_candidate = pathlib.Path.cwd() / 'sparsh-main'
        if cwd_candidate.exists():
            sparsh_root_path = cwd_candidate
    sparsh_root = str(sparsh_root_path)
    if sparsh_root not in sys.path:
        sys.path.insert(0, sparsh_root)
    return sparsh_root


_SPARSH_MODULE_CACHE: Dict[str, object] = {}


def _load_symbol_from_file(module_key: str, file_path: str, symbol_name: str):
    """Load a symbol from a Python file without importing parent packages."""
    if module_key not in _SPARSH_MODULE_CACHE:
        spec = importlib.util.spec_from_file_location(module_key, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to create import spec for {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _SPARSH_MODULE_CACHE[module_key] = module
    module = _SPARSH_MODULE_CACHE[module_key]
    return getattr(module, symbol_name)


class SparshEncoder(nn.Module):
    """Sparsh-native ViT-base encoder wrapper for force estimation."""

    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_chans: int = 6,  # Temporal pair (2 RGB frames)
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0):
        super().__init__()

        _ensure_sparsh_path()
        from tactile_ssl.model.vision_transformer import vit_base

        if any(v != default for v, default in [
            (embed_dim, 768),
            (depth, 12),
            (num_heads, 12),
            (mlp_ratio, 4.0),
        ]):
            warnings.warn(
                "SparshEncoder wrapper uses Sparsh vit_base canonical settings; "
                "non-canonical constructor arguments are ignored."
            )

        self.model = vit_base(
            img_size=(img_size, img_size),
            patch_size=patch_size,
            in_chans=in_chans,
            num_register_tokens=1,
            pos_embed_fn='sinusoidal',
        )

        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = self.model.patch_embed.num_patches
        self.register_tokens = self.model.register_tokens
        self.blocks = self.model.blocks

        self.feature_layers = [2, 5, 8, 11]
        self.intermediate_features: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with intermediate feature extraction.

        Args:
            x: [B, 6, 224, 224] input tensor (temporal pair)

        Returns
        -------
        [B, N, C] final features

        """
        x = self.model.prepare_tokens_with_masks(x)

        self.intermediate_features = []
        for i, blk in enumerate(self.model.blocks):
            x = blk(x)
            if i in self.feature_layers:
                self.intermediate_features.append(x)

        return x

    def get_intermediate_features(self) -> list:
        """Get intermediate features from specified layers."""
        return self.intermediate_features


class ForceFieldDecoder(nn.Module):
    """Sparsh-equivalent force-field decoder for inference."""

    def __init__(
        self,
        image_size: Tuple[int, int, int] = (3, 224, 224),
        embed_dim: int = 768,
        patch_size: int = 16,
        out_dim: int = 128,
        hooks: list[int] = [2, 5, 8, 11],
        reassemble_s: list[int] = [4, 8, 16, 32],
    ):
        super().__init__()

        sparsh_root = _ensure_sparsh_path()
        reassemble_cls = _load_symbol_from_file(
            'sparsh_reassemble_mod',
            os.path.join(
                sparsh_root,
                'tactile_ssl',
                'downstream_task',
                'utils_forcefield',
                'layers',
                'Reassemble.py'
            ),
            'Reassemble',
        )
        fusion_cls = _load_symbol_from_file(
            'sparsh_fusion_mod',
            os.path.join(
                sparsh_root,
                'tactile_ssl',
                'downstream_task',
                'utils_forcefield',
                'layers',
                'Fusion.py'
            ),
            'Fusion',
        )
        head_cls = _load_symbol_from_file(
            'sparsh_head_mod',
            os.path.join(
                sparsh_root,
                'tactile_ssl',
                'downstream_task',
                'utils_forcefield',
                'layers',
                'Head.py'
            ),
            'NormalShearHead',
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.hooks = hooks
        self.n_patches = (image_size[1] // patch_size) ** 2
        self.reassembles = nn.ModuleList([
            reassemble_cls(image_size, 'ignore', patch_size, s, embed_dim, out_dim)
            for s in reassemble_s
        ])
        self.fusions = nn.ModuleList([fusion_cls(out_dim) for _ in reassemble_s])
        self.probe = head_cls(features=out_dim)

    def forward(
        self,
        intermediate_features: Union[list, Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode intermediate features to force fields.

        Args:
            intermediate_features: Dict keyed by {'t2','t5','t8','t11'} or
                list ordered as [2, 5, 8, 11] with [B, N+1, C] tensors.

        Returns
        -------
        normal: [B, 1, 224, 224]
        shear: [B, 2, 224, 224]

        """
        if isinstance(intermediate_features, dict):
            encoder_activations = {k: v for k, v in intermediate_features.items()}
        else:
            if len(intermediate_features) != 4:
                raise ValueError("Expected 4 intermediate feature tensors for layers [2,5,8,11]")
            encoder_activations = {
                't2': intermediate_features[0],
                't5': intermediate_features[1],
                't8': intermediate_features[2],
                't11': intermediate_features[3],
            }

        sample_key = list(encoder_activations.keys())[0]
        start_idx = encoder_activations[sample_key].shape[1] - self.n_patches
        for key in encoder_activations.keys():
            encoder_activations[key] = self.norm(encoder_activations[key][:, start_idx:, :])

        previous_stage = None
        for i in np.arange(len(self.fusions) - 1, -1, -1, dtype=int):
            hook_to_take = 't' + str(self.hooks[int(i)])
            activation_result = encoder_activations[hook_to_take]
            reassemble_result = self.reassembles[i](activation_result)
            fusion_result = self.fusions[i](reassemble_result, previous_stage)
            previous_stage = fusion_result

        y = self.probe(previous_stage, mode='normal_shear')
        normal = y[:, 0, :, :].unsqueeze(1)
        shear = y[:, 1:, :, :]
        return normal, shear


class ForceEstimator:
    """
    Main interface for force estimation using Sparsh models.

    Supports both force field (dense heatmaps) and force vector (aggregated) outputs.
    """

    def __init__(self,
                 encoder_path: str,
                 decoder_path: str,
                 temporal_stride: int = 5,
                 bg_offset: float = 0.5,
                 device: str = 'cuda',
                 force_field_baseline: bool = False,
                 force_vector_scale: Optional[Union[list, tuple]] = None):
        """
        Initialize force estimator.

        Args:
        ----
        encoder_path: Path to Sparsh encoder checkpoint (.ckpt)
            decoder_path: Path to force field decoder weights (.pth)
            temporal_stride: Frames between temporal pair (default: 5)
            bg_offset: Background subtraction offset (default: 0.5)
            device: 'cuda' or 'cpu'
            force_field_baseline: If True, compute/save a per-pixel background
                template during `load_background()` and subtract it from
                subsequent `force_field` outputs at runtime. Default: False.
        force_vector_scale: Per-axis scale to convert normalized
            `force_vector` -> physical units (N).

        """
        # Validate paths
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(
                f"Sparsh encoder not found at {encoder_path}. "
                f"Run: python scripts/download_models.py"
            )
        if not os.path.exists(decoder_path):
            raise FileNotFoundError(
                f"Sparsh decoder not found at {decoder_path}. "
                f"Run: python scripts/download_models.py"
            )

        # Setup device
        self.device = device
        if device == 'cuda' and not torch.cuda.is_available():
            warnings.warn("CUDA not available, falling back to CPU")
            self.device = 'cpu'

        # Initialize models
        self.encoder = SparshEncoder()
        self.decoder = ForceFieldDecoder()

        # Load pretrained weights
        print(f"Loading encoder from {encoder_path}...")
        encoder_weights = _load_encoder_checkpoint(encoder_path)
        encoder_probe = self.encoder.model.load_state_dict(encoder_weights, strict=False)
        if encoder_probe.missing_keys or encoder_probe.unexpected_keys:
            raise RuntimeError(
                "Strict Sparsh encoder load failed: "
                f"missing_keys={encoder_probe.missing_keys}, "
                f"unexpected_keys={encoder_probe.unexpected_keys}"
            )
        self.encoder.model.load_state_dict(encoder_weights, strict=True)

        print(f"Loading decoder from {decoder_path}...")
        decoder_weights = torch.load(decoder_path, map_location='cpu', weights_only=True)
        # Remove 'model_task.' prefix if present
        cleaned_decoder = {}
        prefix = 'model_task.'
        for key, value in decoder_weights.items():
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                cleaned_decoder[new_key] = value
            else:
                cleaned_decoder[key] = value
        decoder_probe = self.decoder.load_state_dict(cleaned_decoder, strict=False)
        if decoder_probe.missing_keys or decoder_probe.unexpected_keys:
            raise RuntimeError(
                "Strict Sparsh decoder load failed: "
                f"missing_keys={decoder_probe.missing_keys}, "
                f"unexpected_keys={decoder_probe.unexpected_keys}"
            )
        self.decoder.load_state_dict(cleaned_decoder, strict=True)

        # Move to device
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        self.encoder.eval()
        self.decoder.eval()

        # Preprocessing
        self.bg_offset = bg_offset
        self.background = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),  # Converts to [0, 1]
        ])

        # Per-axis force scale (normalized model units -> physical units (N)).
        # Per-axis scale mapping normalized model units -> physical units (N)
        self.force_vector_scale = np.array(
            force_vector_scale if force_vector_scale is not None
            else [1.0, 1.0, 1.0],
            dtype=float,
        )

        # Temporal buffer
        self.temporal_stride = temporal_stride
        self.temporal_buffer = TemporalBuffer(max_size=temporal_stride + 1)

        # Runtime force_field baseline subtraction (per-pixel template)
        # Disabled by default to preserve Sparsh demo behavior.
        self.force_field_baseline_enabled = bool(force_field_baseline)
        self.force_field_baseline_template = None

        print(f"Force estimator initialized on {self.device}")

    def load_background(self, background: np.ndarray):
        """
        Load background image for subtraction and compute no-contact baseline.

        The baseline is computed by running the model on a background pair and
        saved to `self.force_vector_baseline`. This baseline is subtracted from
        `force_vector` outputs to remove sensor/model bias.

        Args:
        ----
        background: [H, W, 3] BGR background image (uint8)

        """
        self.background = background.copy()

        # Compute baseline force vector from background (no-contact) frames.
        # We run a single forward pass with bg / bg to estimate steady-state bias.
        try:
            # Prepare input and run encoder+decoder on CPU to be safe
            input_tensor = self._preprocess(self.background, self.background).to(self.device)
            with torch.no_grad():
                _ = self.encoder(input_tensor)
                intermediate_features = self.encoder.get_intermediate_features()
                normal, shear = self.decoder(intermediate_features)
                normal = normal.squeeze(0).squeeze(0).cpu().numpy()
                shear = shear.squeeze(0).permute(1, 2, 0).cpu().numpy()

            # Aggregate baseline vector (mean over spatial dims)
            baseline_fz = float(np.mean(normal))
            baseline_fx = float(np.mean(shear[:, :, 0]))
            baseline_fy = float(np.mean(shear[:, :, 1]))

            self.force_vector_baseline = {'fx': baseline_fx, 'fy': baseline_fy, 'fz': baseline_fz}

            # Optionally save per-pixel force_field baseline template for runtime subtraction
            if getattr(self, 'force_field_baseline_enabled', False):
                try:
                    self.force_field_baseline_template = {
                        'normal': normal.copy(),
                        'shear': shear.copy()
                    }
                except Exception:
                    self.force_field_baseline_template = None
        except Exception:
            # If baseline computation fails for any reason, default to zero baseline
            self.force_vector_baseline = {'fx': 0.0, 'fy': 0.0, 'fz': 0.0}
            self.force_field_baseline_template = None

    def _preprocess(self, img_t: np.ndarray, img_t_minus: np.ndarray) -> torch.Tensor:
        """
        Preprocess temporal pair for force estimation.

        Args:
            img_t: [H, W, 3] BGR image at time t (uint8)
            img_t_minus: [H, W, 3] BGR image at time t-stride (uint8)

        Returns
        -------
        [1, 6, 224, 224] preprocessed tensor

        """
        if self.background is None:
            raise ValueError("Background not loaded. Call load_background() first.")

        # Background subtraction with offset
        def subtract_bg(img, bg, offset):
            diff = img.astype(np.int32) - bg.astype(np.int32)
            diff = diff / 255.0 + offset
            diff = np.clip(diff, 0.0, 1.0)
            diff = (diff * 255.0).astype(np.uint8)
            return diff

        img_t_diff = subtract_bg(img_t, self.background, self.bg_offset)
        img_t_minus_diff = subtract_bg(img_t_minus, self.background, self.bg_offset)

        # Convert to RGB PIL images (input frames from camera are BGR)
        # Sparsh expects RGB images — convert from BGR -> RGB first.
        img_t_rgb = cv2.cvtColor(img_t_diff, cv2.COLOR_BGR2RGB)
        img_t_minus_rgb = cv2.cvtColor(img_t_minus_diff, cv2.COLOR_BGR2RGB)
        img_t_pil = Image.fromarray(img_t_rgb).convert("RGB")
        img_t_minus_pil = Image.fromarray(img_t_minus_rgb).convert("RGB")

        # Resize and convert to tensor
        tensor_t = self.transform(img_t_pil)         # [3, 224, 224]
        tensor_t_minus = self.transform(img_t_minus_pil)  # [3, 224, 224]

        # Temporal concatenation
        input_tensor = torch.cat([tensor_t, tensor_t_minus], dim=0)  # [6, 224, 224]

        return input_tensor.unsqueeze(0)  # [1, 6, 224, 224]

    def estimate(self,
                 image: np.ndarray,
                 timestamp: Optional[float] = None
                 ) -> Optional[Dict[str, Union[np.ndarray, float]]]:
        """
        Estimate force field and vector from image.

        Args:
            image: [H, W, 3] BGR image (uint8)
            timestamp: Optional timestamp for temporal tracking

        Returns
        -------
        Dictionary with:
        - 'force_field': {'normal': [224, 224], 'shear': [224, 224, 2]}
          or None if buffer not ready
        - 'force_vector': {'fx': float, 'fy': float, 'fz': float}
          or None if buffer not ready

        """
        # Add to temporal buffer
        self.temporal_buffer.add(image, timestamp)

        # Check if buffer is ready
        if not self.temporal_buffer.is_ready():
            return None

        # Get temporal pair
        frame_pair = self.temporal_buffer.get_pair(stride=self.temporal_stride)
        if frame_pair is None:
            return None

        img_t, img_t_minus = frame_pair

        # Preprocess
        input_tensor = self._preprocess(img_t, img_t_minus)
        input_tensor = input_tensor.to(self.device)

        # Inference
        with torch.no_grad():
            try:
                # Encoder
                _ = self.encoder(input_tensor)
                intermediate_features = self.encoder.get_intermediate_features()

                # Decoder
                normal, shear = self.decoder(intermediate_features)

                # Move to CPU and convert to numpy
                normal = normal.squeeze(0).squeeze(0).cpu().numpy()  # [224, 224]
                shear = shear.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [224, 224, 2]

            except RuntimeError as e:
                if "out of memory" in str(e):
                    warnings.warn("CUDA OOM, falling back to CPU")
                    self.encoder = self.encoder.cpu()
                    self.decoder = self.decoder.cpu()
                    self.device = "cpu"
                    # Retry on CPU
                    return self.estimate(image, timestamp)
                else:
                    raise

        # Optionally subtract per-pixel force_field baseline template
        if (
            getattr(self, 'force_field_baseline_enabled', False)
            and self.force_field_baseline_template is not None
        ):
            try:
                normal = normal - self.force_field_baseline_template['normal']
                shear = shear - self.force_field_baseline_template['shear']
            except Exception:
                # If subtraction fails, leave original arrays
                pass

        # Aggregate force vector
        fz = float(np.mean(normal))
        fx = float(np.mean(shear[:, :, 0]))
        fy = float(np.mean(shear[:, :, 1]))

        # Subtract baseline (if available) to remove sensor/model bias
        baseline = getattr(self, 'force_vector_baseline', None)
        if baseline is not None:
            fx = fx - float(baseline.get('fx', 0.0))
            fy = fy - float(baseline.get('fy', 0.0))
            fz = fz - float(baseline.get('fz', 0.0))

        # Compute physical (Newton) values by applying per-axis force_vector_scale
        try:
            sx, sy, sz = tuple(self.force_vector_scale.tolist())
        except Exception:
            sx, sy, sz = 1.0, 1.0, 1.0
        force_vector_physical = {
            'fx': float(fx * sx),
            'fy': float(fy * sy),
            'fz': float(fz * sz),
        }

        return {
            'force_field': {
                'normal': normal,
                'shear': shear
            },
            'force_vector': {
                'fx': fx,
                'fy': fy,
                'fz': fz
            },
            'force_vector_physical': force_vector_physical
        }
