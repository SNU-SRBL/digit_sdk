"""Ball-depth, manual-contact, and background datasets."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import yaml

from calibration.ball_geometry import spherical_cap_depth_mm


MODEL_SIZE = (224, 224)


def image_tensor(image: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(
        cv2.resize(image, MODEL_SIZE, interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2RGB,
    )
    tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
    return (tensor - 0.5) / 0.5


def _load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def active_split(root: Path, modality: str):
    config = yaml.safe_load((root / "dataset.yaml").read_text())
    split_id = config.get("active_splits", {}).get(modality, modality)
    return json.loads((root / "splits" / f"{split_id}.json").read_text())


class BallDepthDataset(Dataset):
    def __init__(self, root: Path, partition: str):
        self.root = Path(root)
        records = _load_jsonl(self.root / "manifest.jsonl")
        by_id = {record["sample_id"]: record for record in records}
        split = active_split(self.root, "ball")
        self.records = [by_id[sample_id] for sample_id in split[partition]]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = cv2.imread(str(self.root / record["image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(self.root / record["image_path"])
        depth = spherical_cap_depth_mm(
            image.shape,
            record["center_px"],
            record["radius_px"],
            record["ball_diameter_mm"],
            record["ppmm"],
        )
        depth = cv2.resize(depth, MODEL_SIZE, interpolation=cv2.INTER_AREA)
        return {
            "image": image_tensor(image),
            "depth_mm": torch.from_numpy(depth).unsqueeze(0).float(),
            "sample_id": record["sample_id"],
        }


class ManualSupportDataset(Dataset):
    def __init__(self, root: Path, partition: str):
        self.root = Path(root)
        records = _load_jsonl(self.root / "manifest.jsonl")
        by_id = {record["sample_id"]: record for record in records}
        split = active_split(self.root, "manual_mask")
        self.records = [by_id[sample_id] for sample_id in split[partition]]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = cv2.imread(str(self.root / record["image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(self.root / record["image_path"])
        mask = cv2.imread(
            str(self.root / record["label_path"]), cv2.IMREAD_GRAYSCALE
        )
        if mask is None:
            raise FileNotFoundError(self.root / record["label_path"])
        mask = cv2.resize(mask, MODEL_SIZE, interpolation=cv2.INTER_NEAREST) > 0
        return {
            "image": image_tensor(image),
            "mask": torch.from_numpy(mask).unsqueeze(0).bool(),
            "sample_id": record["sample_id"],
            "has_contact": bool(mask.any()),
        }


class BackgroundSupportDataset(Dataset):
    def __init__(self, root: Path, partition: str):
        self.root = Path(root)
        records = _load_jsonl(self.root / "manifest.jsonl")
        by_id = {record["sample_id"]: record for record in records}
        split = active_split(self.root, "background")
        self.records = [
            by_id[sample_id]
            for sample_id in split[partition]
            if by_id[sample_id].get("training_eligible", False)
        ]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = cv2.imread(str(self.root / record["image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(self.root / record["image_path"])
        mask = np.zeros(MODEL_SIZE[::-1], dtype=bool)
        return {
            "image": image_tensor(image),
            "mask": torch.from_numpy(mask).unsqueeze(0).bool(),
            "sample_id": record["sample_id"],
            "has_contact": False,
        }
