from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from calibration.annotate_contact.server import AnnotationDataset


def _save_image(path: Path, value: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 10, 3), value, dtype=np.uint8)).save(path)


def _mask_png(width=10, height=8) -> bytes:
    mask = np.zeros((height, width, 4), dtype=np.uint8)
    mask[2:5, 3:7] = [255, 255, 255, 255]
    buffer = BytesIO()
    Image.fromarray(mask, mode="RGBA").save(buffer, format="PNG")
    return buffer.getvalue()


def _staged_queue(root: Path) -> None:
    background = root.parent / "background/reference.png"
    touch = root / "touches/session/touch.png"
    _save_image(background, 0)
    _save_image(touch, 40)
    rows = [
        {
            "sample_id": "touch", "kind": "touch", "serial": "DTEST",
            "path": "touches/session/touch.png",
            "captured_at_utc": "2026-07-17T00:00:01+00:00",
            "session_id": "session", "shape": [8, 10, 3],
            "dtype": "uint8", "colour_space": "BGR",
            "split_group": "touch",
        },
    ]
    (root / "metadata.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_annotates_staged_contacts_without_dataset_preparation(tmp_path):
    root = tmp_path / "inbox/manual_mask"
    _staged_queue(root)
    dataset = AnnotationDataset(root)

    assert dataset.samples()[0]["background_url"] is not None
    first = dataset.save_mask("touch", _mask_png(), "tester")

    saved = [json.loads(line) for line in
             (root / "metadata.jsonl").read_text().splitlines()][0]
    assert saved["label_path"] == "masks/session/touch.png"
    assert first["annotation_revision"] == 1
    assert dataset.status() == {"total": 1, "completed": 1, "remaining": 0}
    with Image.open(dataset.file_path("touch", "mask")) as image:
        assert image.mode == "L"
        assert set(np.unique(np.asarray(image))) == {0, 255}


def test_staged_annotation_preserves_mask_revision(tmp_path):
    root = tmp_path / "inbox/manual_mask"
    _staged_queue(root)
    dataset = AnnotationDataset(root)
    dataset.save_mask("touch", _mask_png(), "first")

    reloaded = AnnotationDataset(root)
    result = reloaded.save_mask("touch", _mask_png(), "second")

    assert result["annotation_revision"] == 2
    assert result["annotator"] == "second"


def test_save_mask_requires_annotator_and_matching_dimensions(tmp_path):
    root = tmp_path / "inbox/manual_mask"
    _staged_queue(root)
    dataset = AnnotationDataset(root)

    with pytest.raises(ValueError, match="annotator"):
        dataset.save_mask("touch", _mask_png(), "")
    with pytest.raises(ValueError, match="mask size"):
        dataset.save_mask("touch", _mask_png(width=9), "tester")


def test_existing_canonical_dataset_remains_annotatable(tmp_path):
    root = tmp_path / "calibration"
    image = root / "manual_mask/images/session/touch.png"
    _save_image(image, 40)
    record = {
        "sample_id": "touch", "modality": "manual_mask",
        "serial": "DTEST", "image_path": "manual_mask/images/session/touch.png",
        "label_path": "manual_mask/masks/session/touch.png",
        "captured_at_utc": "2026-07-17T00:00:01+00:00",
        "session_id": "session", "shape": [8, 10, 3],
        "split_group": "touch", "annotator": "",
        "annotation_revision": 0,
    }
    (root / "manifest.jsonl").write_text(json.dumps(record) + "\n")

    dataset = AnnotationDataset(root)
    dataset.save_mask("touch", _mask_png(), "tester")

    assert dataset.status()["completed"] == 1
