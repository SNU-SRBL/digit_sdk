"""Mixed metric-depth and binary-support losses in millimetres."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = values[mask]
    return selected.mean() if selected.numel() else values.sum() * 0.0


def region_balanced_depth_loss(
    prediction_mm: torch.Tensor,
    target_mm: torch.Tensor,
    *,
    background_weight: float = 1.0,
) -> torch.Tensor:
    contact = target_mm > 0
    error = (prediction_mm - target_mm).square()
    return _masked_mean(error, contact) + background_weight * _masked_mean(
        error, ~contact
    )


def support_loss(
    prediction_mm: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    threshold_mm: float = 0.1,
    temperature_mm: float = 0.02,
) -> torch.Tensor:
    target = target_mask.bool()
    logits = (prediction_mm - threshold_mm) / temperature_mm
    probabilities = torch.sigmoid(logits)
    losses = []
    for sample_logits, sample_probability, sample_target in zip(
        logits, probabilities, target
    ):
        positive = sample_target
        negative = ~positive
        if positive.any():
            positive_bce = F.binary_cross_entropy_with_logits(
                sample_logits[positive],
                torch.ones_like(sample_logits[positive]),
            )
            if negative.any():
                negative_bce = F.binary_cross_entropy_with_logits(
                    sample_logits[negative],
                    torch.zeros_like(sample_logits[negative]),
                )
                bce = 0.5 * (positive_bce + negative_bce)
            else:
                bce = positive_bce
            intersection = (sample_probability * sample_target.float()).sum()
            dice = (2 * intersection + 1e-6) / (
                sample_probability.sum() + sample_target.sum() + 1e-6
            )
            losses.append(bce + 1.0 - dice)
        else:
            losses.append(F.binary_cross_entropy_with_logits(
                sample_logits,
                torch.zeros_like(sample_logits),
            ))
    return torch.stack(losses).mean()


def outside_zero_loss(
    prediction_mm: torch.Tensor, target_mask: torch.Tensor
) -> torch.Tensor:
    return _masked_mean(prediction_mm.square(), ~target_mask.bool())
