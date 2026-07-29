"""Model-independent depth and binary-contact evaluation metrics."""

import numpy as np


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 1.0


def binary_metrics(truth: np.ndarray, predicted: np.ndarray) -> dict:
    tp = int(np.logical_and(truth, predicted).sum())
    fp = int(np.logical_and(~truth, predicted).sum())
    fn = int(np.logical_and(truth, ~predicted).sum())
    tn = int(np.logical_and(~truth, ~predicted).sum())
    return {
        "iou": safe_ratio(tp, tp + fp + fn),
        "dice": safe_ratio(2 * tp, 2 * tp + fp + fn),
        "precision": safe_ratio(tp, tp + fp),
        "recall": safe_ratio(tp, tp + fn),
        "false_positive_rate": safe_ratio(fp, fp + tn),
        "missed_contact": tp == 0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def ball_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    contact = truth > 0
    error = prediction - truth
    contact_absolute = np.abs(error[contact])
    contact_squared = np.square(error[contact])
    background_absolute = np.abs(error[~contact])
    contact_mae = float(contact_absolute.mean())
    background_mae = float(background_absolute.mean())
    return {
        "contact_mae_mm": contact_mae,
        "contact_rmse_mm": float(np.sqrt(contact_squared.mean())),
        "background_mae_mm": background_mae,
        "region_balanced_mae_mm": 0.5 * (contact_mae + background_mae),
    }
