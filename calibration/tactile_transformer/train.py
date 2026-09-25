"""Fine-tune a per-sensor decoder with ball depth and manual support.

Beyond the default mixed protocol, the CLI can replay fixed no-contact
background frames with generated per-pixel Gaussian noise
(``--fixed-background-weight``, ``--background-noise-sigma``) and train an
independent per-pixel contact-probability head
(``--contact-head-weight``, ``--contact-gate-threshold``).  Any non-default
setting is a different objective: write it to a distinct ``--output-root``
instead of the canonical method directory.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from calibration.dataset_schema import validate_dataset
from calibration.tactile_transformer.data import (
    BallDepthDataset,
    BackgroundSupportDataset,
    ManualSupportDataset,
    active_split,
)
from calibration.tactile_transformer.losses import (
    contact_bce_loss,
    outside_zero_loss,
    region_balanced_depth_loss,
    support_loss,
)
from calibration.tactile_transformer.model import (
    BASE_FILENAME,
    BASE_REPOSITORY,
    BASE_REVISION,
    TactileDPT,
    contact_head_state,
    decoder_parameters,
    decoder_state,
    freeze_encoder,
    load_base,
    resolve_base,
)


METHOD = "mixed"
TRAINING_PROTOCOL = "mixed_ball_manual"
EXPERIMENTAL_FLAGS = (
    "fixed_background_weight",
    "background_noise_sigma",
    "contact_head_weight",
    "contact_gate_threshold",
)


def experimental_objective(args) -> bool:
    """True when any non-default objective flag is set."""
    return any(
        getattr(args, name, 0.0) > 0.0 for name in EXPERIMENTAL_FLAGS
    )


def guard_output_root(args) -> None:
    """Keep experimental objectives out of the canonical method directory."""
    if experimental_objective(args) and args.output_root is None:
        raise ValueError(
            "experimental objective flags "
            f"({', '.join(EXPERIMENTAL_FLAGS)}) require an explicit "
            "--output-root; the canonical method directory is reserved for "
            f"the frozen {METHOD} protocol"
        )


def objective_name(args) -> str:
    if not experimental_objective(args):
        return METHOD
    return METHOD + "_fixed_background"



def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _forward_mm(model, images, args):
    """Metric depth in millimetres, optionally gated by the contact head.

    The gate is applied only in eval mode; training keeps raw depth so the
    depth loss still shapes suppressed regions.
    """
    threshold = getattr(args, "contact_gate_threshold", 0.0)
    wants_contact = (
        threshold > 0.0
        or getattr(args, "contact_head_weight", 0.0) > 0.0
    )
    if wants_contact:
        depth, logits = model(images, return_contact=True)
    else:
        depth, logits = model(images), None
    depth = depth * args.maximum_depth_mm
    if threshold > 0.0 and not model.training:
        gate = (torch.sigmoid(logits) >= threshold).to(depth.dtype)
        depth = depth * gate
    return depth, logits


def _contact_term(contact_logits, target_mask, args):
    weight = getattr(args, "contact_head_weight", 0.0)
    if contact_logits is None or weight <= 0.0:
        return None
    return weight * contact_bce_loss(contact_logits, target_mask)


def ball_loss(prediction, target, args, contact_logits=None):
    loss = region_balanced_depth_loss(
        prediction,
        target,
        background_weight=args.background_weight,
    )
    loss = loss + args.ball_support_weight * support_loss(
        prediction,
        target >= args.contact_threshold_mm,
        threshold_mm=args.contact_threshold_mm,
        temperature_mm=args.support_temperature_mm,
    )
    term = _contact_term(
        contact_logits, target >= args.contact_threshold_mm, args
    )
    return loss if term is None else loss + term


def manual_loss(prediction, mask, args, contact_logits=None):
    loss = args.manual_weight * (
        support_loss(
            prediction,
            mask,
            threshold_mm=args.contact_threshold_mm,
            temperature_mm=args.support_temperature_mm,
        )
        + args.manual_zero_weight * outside_zero_loss(prediction, mask)
    )
    term = _contact_term(contact_logits, mask, args)
    return loss if term is None else loss + term


def background_loss(prediction, args, contact_logits=None):
    """Zero-depth and zero-contact supervision on fixed no-contact frames."""
    loss = args.fixed_background_weight * outside_zero_loss(
        prediction, torch.zeros_like(prediction, dtype=torch.bool)
    )
    term = _contact_term(
        contact_logits, torch.zeros_like(prediction, dtype=torch.bool), args
    )
    return loss if term is None else loss + term


def _scheduler(optimizer, total_steps: int, warmup_fraction: float):
    warmup_steps = max(1, round(total_steps * warmup_fraction))

    def scale(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _infinite(loader):
    while True:
        yield from loader


def mixed_steps(ball_loader, manual_loader, background_loader=None):
    """Steps a mixed epoch actually runs, shared by loop and scheduler.

    Returns ``(steps_per_epoch, steps_per_type)`` for the included streams, so
    the scheduler horizon always matches the executed optimizer steps.
    """
    loaders = [ball_loader, manual_loader]
    if background_loader is not None:
        loaders.append(background_loader)
    steps_per_type = max(len(loader) for loader in loaders)
    return len(loaders) * steps_per_type, steps_per_type


def _optimizer_step(loss, optimizer, scheduler, scaler, parameters, args):
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(parameters, args.gradient_clip)
    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    if scaler.get_scale() >= scale_before:
        scheduler.step()


def train_ball_epoch(model, loader, optimizer, scheduler, scaler, parameters, args):
    model.train()
    model.transformer_encoders.eval()
    losses = []
    for batch in loader:
        images = batch["image"].to(args.device, non_blocking=True)
        targets = batch["depth_mm"].to(args.device, non_blocking=True)
        with torch.autocast(
            device_type=args.device.type,
            dtype=torch.float16,
            enabled=args.amp,
        ):
            prediction, contact_logits = _forward_mm(model, images, args)
            loss = ball_loss(prediction, targets, args, contact_logits)
        _optimizer_step(
            loss, optimizer, scheduler, scaler, parameters, args
        )
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def train_mixed_epoch(
    model, ball_loader, manual_loader, optimizer, scheduler, scaler,
    parameters, args, background_loader=None,
):
    model.train()
    model.transformer_encoders.eval()
    iterators = {
        "ball": _infinite(ball_loader),
        "manual": _infinite(manual_loader),
    }
    losses = {"ball": [], "manual": []}
    kinds = ["ball", "manual"]
    if background_loader is not None:
        iterators["background"] = _infinite(background_loader)
        losses["background"] = []
        kinds.append("background")
    _, steps_per_type = mixed_steps(
        ball_loader, manual_loader, background_loader
    )
    for _ in range(steps_per_type):
        for kind in kinds:
            batch = next(iterators[kind])
            images = batch["image"].to(args.device, non_blocking=True)
            with torch.autocast(
                device_type=args.device.type,
                dtype=torch.float16,
                enabled=args.amp,
            ):
                prediction, contact_logits = _forward_mm(model, images, args)
                if kind == "ball":
                    target = batch["depth_mm"].to(args.device, non_blocking=True)
                    loss = ball_loss(prediction, target, args, contact_logits)
                elif kind == "manual":
                    target = batch["mask"].to(args.device, non_blocking=True)
                    loss = manual_loss(prediction, target, args, contact_logits)
                else:
                    loss = background_loss(prediction, args, contact_logits)
            _optimizer_step(
                loss, optimizer, scheduler, scaler, parameters, args
            )
            losses[kind].append(float(loss.detach().cpu()))
    return {kind: float(np.mean(values)) for kind, values in losses.items()}


@torch.no_grad()
def evaluate_ball(model, loader, args):
    model.eval()
    contact_absolute = []
    contact_squared = []
    background_absolute = []
    for batch in loader:
        images = batch["image"].to(args.device, non_blocking=True)
        targets = batch["depth_mm"].to(args.device, non_blocking=True)
        prediction, _ = _forward_mm(model, images, args)
        error = prediction - targets
        contact = targets > 0
        contact_absolute.append(error[contact].abs().cpu())
        contact_squared.append(error[contact].square().cpu())
        background_absolute.append(error[~contact].abs().cpu())
    contact_absolute = torch.cat(contact_absolute)
    contact_squared = torch.cat(contact_squared)
    background_absolute = torch.cat(background_absolute)
    contact_mae = float(contact_absolute.mean())
    background_mae = float(background_absolute.mean())
    return {
        "contact_mae_mm": contact_mae,
        "contact_rmse_mm": float(contact_squared.mean().sqrt()),
        "background_mae_mm": background_mae,
        "region_balanced_mae_mm": 0.5 * (contact_mae + background_mae),
    }


@torch.no_grad()
def evaluate_manual(model, loader, args):
    model.eval()
    dice = []
    iou = []
    missed = 0
    false_positive = 0
    negative = 0
    contacts = 0
    for batch in loader:
        images = batch["image"].to(args.device, non_blocking=True)
        targets = batch["mask"].to(args.device, non_blocking=True).bool()
        predictions = _forward_mm(model, images, args)[0] >= (
            args.contact_threshold_mm
        )
        for prediction, target in zip(predictions, targets):
            false_positive += int((prediction & ~target).sum())
            negative += int((~target).sum())
            if target.any():
                contacts += 1
                intersection = int((prediction & target).sum())
                predicted = int(prediction.sum())
                actual = int(target.sum())
                union = int((prediction | target).sum())
                dice.append(2 * intersection / max(predicted + actual, 1))
                iou.append(intersection / max(union, 1))
                missed += predicted == 0
    return {
        "mean_dice": float(np.mean(dice)),
        "mean_iou": float(np.mean(iou)),
        "false_positive_rate": false_positive / max(negative, 1),
        "missed_contacts": int(missed),
        "contact_samples": int(contacts),
    }


@torch.no_grad()
def evaluate_background(model, loader, args):
    """No-contact depth statistics; the target is exactly zero everywhere."""
    model.eval()
    maximum = 0.0
    positive_pixels = 0
    pixels = 0
    absolute_sum = 0.0
    for batch in loader:
        images = batch["image"].to(args.device, non_blocking=True)
        prediction = _forward_mm(model, images, args)[0]
        maximum = max(maximum, float(prediction.max()))
        positive_pixels += int((prediction > 0.0).sum())
        pixels += prediction.numel()
        absolute_sum += float(prediction.abs().sum())
    return {
        "max_depth_mm": maximum,
        "strict_positive_pixels": positive_pixels,
        "pixels": pixels,
        "mean_depth_mm": absolute_sum / max(pixels, 1),
        "samples": len(loader.dataset),
    }


def checkpoint_rank(ball_metrics, manual_metrics, background_metrics=None):
    """Lexicographic validation rank; lower is better.

    When background replay is active, a candidate is eligible only if the
    held-out no-contact max depth is below the 0.1 mm contract as well.
    """
    ball_mae = ball_metrics["contact_mae_mm"]
    missed = manual_metrics["missed_contacts"]
    background_max = (
        background_metrics["max_depth_mm"]
        if background_metrics is not None else 0.0
    )
    eligible = (
        missed == 0 and ball_mae <= 0.15 and background_max < 0.1
    )
    background_penalty = max(background_max - 0.1, 0.0)
    return (
        0 if eligible else 1,
        0 if eligible else missed,
        0.0 if eligible else background_penalty,
        0.0 if eligible else max(ball_mae - 0.15, 0.0),
        background_max,
        -manual_metrics["mean_dice"],
        manual_metrics["false_positive_rate"],
        ball_mae,
        ball_metrics["contact_rmse_mm"],
    )


def _save_checkpoint(path, model, metadata):
    """Write a production-compatible decoder plus optional experimental head.

    Production only ever reads ``model_state_dict``, so the contact-probability
    head is stored in a separate field and never leaks into that dict.
    """
    state = decoder_state(model)
    if any(key.startswith("transformer_encoders.") for key in state):
        raise RuntimeError("decoder checkpoint contains encoder weights")
    if any(key.startswith("head_contact.") for key in state):
        raise RuntimeError("decoder checkpoint contains contact head weights")
    payload = {"model_state_dict": state, "metadata": metadata}
    head = contact_head_state(model)
    if head:
        payload["contact_head_state_dict"] = head
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _load_decoder(model, path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(
        payload["model_state_dict"], strict=False
    )
    if unexpected or any(
        not key.startswith(("transformer_encoders.", "head_contact."))
        for key in missing
    ):
        raise ValueError("decoder checkpoint is incompatible with base model")
    head = payload.get("contact_head_state_dict")
    if head:
        model.load_state_dict(head, strict=False)


def _loaders(root, args):
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": args.device.type == "cuda",
    }
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        "ball_train": DataLoader(
            BallDepthDataset(root, "train"), shuffle=True,
            generator=generator, **common,
        ),
        "ball_validation": DataLoader(
            BallDepthDataset(root, "validation"), shuffle=False, **common,
        ),
        "manual_train": DataLoader(
            ManualSupportDataset(root, "train"), shuffle=True,
            generator=generator, **common,
        ),
        "manual_validation": DataLoader(
            ManualSupportDataset(root, "validation"),
            shuffle=False, **common,
        ),
    }
    if args.fixed_background_weight > 0.0:
        loaders["background_train"] = DataLoader(
            BackgroundSupportDataset(
                root, "train", noise_sigma=args.background_noise_sigma,
                seed=args.seed,
            ),
            shuffle=True, generator=generator, **common,
        )
        loaders["background_validation"] = DataLoader(
            BackgroundSupportDataset(root, "validation"),
            shuffle=False, **common,
        )
    return loaders


def _base_metadata(args, dataset_id, root):
    return {
        "training_protocol": TRAINING_PROTOCOL,
        "serial": args.serial,
        "dataset_id": dataset_id,
        "ball_split_id": active_split(root, "ball")["split_id"],
        "manual_mask_split_id": active_split(root, "manual_mask")["split_id"],
        "objective": objective_name(args),
        "objective_parameters": {
            name: getattr(args, name) for name in EXPERIMENTAL_FLAGS
        },
        "seed": args.seed,
        "maximum_depth_mm": args.maximum_depth_mm,
        "contact_threshold_mm": args.contact_threshold_mm,
        "support_temperature_mm": args.support_temperature_mm,
        "base": {
            "repository": BASE_REPOSITORY,
            "filename": BASE_FILENAME,
            "revision": BASE_REVISION,
        },
        "fine_tuned_parameters": (
            "reassembly, fusion, depth head"
            + (", contact head" if args.contact_head_weight > 0.0 else "")
        ),
        "encoder_frozen": True,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--sensors-root", type=Path, default=Path("sensors"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--stage1-epochs", type=int, default=80)
    parser.add_argument("--stage2-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--stage1-learning-rate", type=float, default=1e-4)
    parser.add_argument("--stage2-learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--maximum-depth-mm", type=float, default=2.2)
    parser.add_argument("--background-weight", type=float, default=1.0)
    parser.add_argument("--ball-support-weight", type=float, default=0.1)
    parser.add_argument("--manual-weight", type=float, default=0.1)
    parser.add_argument("--manual-zero-weight", type=float, default=1.0)
    parser.add_argument(
        "--fixed-background-weight", type=float, default=0.0,
        help="weight of zero-depth replay on fixed no-contact frames",
    )
    parser.add_argument(
        "--background-noise-sigma", type=float, default=0.0,
        help="per-pixel Gaussian sigma (8-bit units) for background replay",
    )
    parser.add_argument(
        "--contact-head-weight", type=float, default=0.0,
        help="weight of the independent contact-probability BCE",
    )
    parser.add_argument(
        "--contact-gate-threshold", type=float, default=0.0,
        help="eval-only contact-probability gate; 0 disables the gate",
    )
    parser.add_argument("--contact-threshold-mm", type=float, default=0.1)
    parser.add_argument("--support-temperature-mm", type=float, default=0.02)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--base-cache-dir", type=Path)
    args = parser.parse_args(argv)
    args.device = torch.device(args.device)
    args.amp = args.device.type == "cuda" and not args.no_amp
    return args


def main(argv=None):
    args = parse_args(argv)
    guard_output_root(args)
    seed_everything(args.seed)
    root = args.sensors_root / args.serial / "calibration"
    summary = validate_dataset(root)
    loaders = _loaders(root, args)
    output = args.output_root or (
        args.sensors_root / args.serial / "model/tactile_transformer"
        / METHOD / f"seed_{args.seed}"
    )
    output.mkdir(parents=True, exist_ok=True)
    base_path = resolve_base(args.base_cache_dir)
    model = TactileDPT()
    load_base(model, base_path)
    freeze_encoder(model)
    model.to(args.device)
    parameters = list(decoder_parameters(model))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    metadata = _base_metadata(args, summary.dataset_id, root)
    history = {"stage1": [], "stage2": []}
    started = time.time()

    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.stage1_learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = _scheduler(
        optimizer,
        args.stage1_epochs * len(loaders["ball_train"]),
        args.warmup_fraction,
    )
    stage1_path = output / "stage1_decoder.pth"
    best_rank = None
    stale = 0
    for epoch in range(1, args.stage1_epochs + 1):
        train_loss = train_ball_epoch(
            model, loaders["ball_train"], optimizer, scheduler, scaler,
            parameters, args,
        )
        metrics = evaluate_ball(model, loaders["ball_validation"], args)
        rank = (metrics["contact_mae_mm"], metrics["background_mae_mm"])
        row = {"epoch": epoch, "train_loss": train_loss, **metrics}
        history["stage1"].append(row)
        if best_rank is None or rank < best_rank:
            best_rank = rank
            stale = 0
            _save_checkpoint(stage1_path, model, {**metadata, "stage": 1, **row})
        else:
            stale += 1
        print(
            f"stage1 epoch={epoch:03d} loss={train_loss:.6f} "
            f"contact_mae={metrics['contact_mae_mm']:.4f} "
            f"bg_mae={metrics['background_mae_mm']:.4f}",
            flush=True,
        )
        if stale >= args.patience:
            break
    _load_decoder(model, stage1_path)

    final_path = output / "decoder.pth"
    background_loader = loaders.get("background_train")
    steps_per_epoch, _ = mixed_steps(
        loaders["ball_train"], loaders["manual_train"], background_loader
    )
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.stage2_learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = _scheduler(
        optimizer,
        args.stage2_epochs * steps_per_epoch,
        args.warmup_fraction,
    )
    best_rank = None
    stale = 0
    for epoch in range(1, args.stage2_epochs + 1):
        train_losses = train_mixed_epoch(
            model, loaders["ball_train"], loaders["manual_train"],
            optimizer, scheduler, scaler, parameters, args,
            background_loader=background_loader,
        )
        ball_metrics = evaluate_ball(
            model, loaders["ball_validation"], args
        )
        manual_metrics = evaluate_manual(
            model, loaders["manual_validation"], args
        )
        background_metrics = (
            evaluate_background(
                model, loaders["background_validation"], args
            )
            if "background_validation" in loaders else None
        )
        rank = checkpoint_rank(
            ball_metrics, manual_metrics, background_metrics
        )
        row = {
            "epoch": epoch,
            "train": train_losses,
            "ball": ball_metrics,
            "manual": manual_metrics,
            "eligible": rank[0] == 0,
        }
        if background_metrics is not None:
            row["background"] = background_metrics
        history["stage2"].append(row)
        if best_rank is None or rank < best_rank:
            best_rank = rank
            stale = 0
            _save_checkpoint(
                final_path, model, {**metadata, "stage": 2, **row}
            )
        else:
            stale += 1
        print(
            f"stage2 epoch={epoch:03d} "
            f"ball_mae={ball_metrics['contact_mae_mm']:.4f} "
            f"dice={manual_metrics['mean_dice']:.4f} "
            f"fpr={manual_metrics['false_positive_rate']:.4f} "
            f"missed={manual_metrics['missed_contacts']}"
            + (
                f" bg_max={background_metrics['max_depth_mm']:.4f}"
                if background_metrics is not None else ""
            ),
            flush=True,
        )
        if stale >= args.patience:
            break

    _load_decoder(model, final_path)
    selected_validation = {
        "ball": evaluate_ball(model, loaders["ball_validation"], args),
        "manual": evaluate_manual(model, loaders["manual_validation"], args),
    }
    if "background_validation" in loaders:
        selected_validation["background"] = evaluate_background(
            model, loaders["background_validation"], args
        )
    selected_validation["eligible"] = checkpoint_rank(
        selected_validation["ball"],
        selected_validation["manual"],
        selected_validation.get("background"),
    )[0] == 0
    metadata.update({
        "arguments": {
            key: str(value) if isinstance(value, (Path, torch.device)) else value
            for key, value in vars(args).items()
        },
        "history": history,
        "selected_validation": selected_validation,
        "elapsed_seconds": time.time() - started,
        "test_evaluated": False,
    })
    (output / "training.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
