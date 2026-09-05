from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.request import urlopen

import numpy as np
from PIL import Image
import pytest

from calibration.annotate_ball.server import BallAnnotationDataset, make_handler


def _make_staging(root: Path):
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    image_path = root / "images/session-1/ball-1.png"
    background_path = root / "shared-background.png"
    image_path.parent.mkdir(parents=True)
    Image.fromarray(image).save(image_path)
    Image.fromarray(image).save(background_path)
    common = {
        "serial": "DTEST",
        "partition": "train",
        "session_id": "session-1",
        "shape": [8, 10, 3],
        "dtype": "uint8",
        "colour_space": "BGR",
        "capture_fps": 30,
    }
    records = [{
            **common,
            "sample_id": "ball-1",
            "kind": "ball",
            "path": "images/session-1/ball-1.png",
            "captured_at_utc": "2026-07-16T00:00:01+00:00",
            "ball_diameter_mm": 6.0,
            "ppmm": 20.0,
        }]
    (root / "metadata.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return background_path


def test_save_ball_annotation_is_atomic_and_geometrically_complete(tmp_path):
    background_path = _make_staging(tmp_path)
    dataset = BallAnnotationDataset(tmp_path, background_path)

    result = dataset.save_annotation("ball-1", [5.0, 4.0], 20.0, "tester")

    assert result["annotation_revision"] == 1
    assert result["indentation_depth_mm"] > 0
    label_path = tmp_path / "labels/session-1/ball-1.npz"
    with np.load(label_path, allow_pickle=False) as label:
        assert set(label.files) == {
            "ball_diameter_mm",
            "center_px",
            "radius_px",
            "ppmm",
            "indentation_depth_mm",
        }
        assert np.allclose(label["center_px"], [5.0, 4.0])
        assert float(label["radius_px"]) == 20.0
    assert dataset.status() == {
        "total": 1, "completed": 1, "rejected": 0, "remaining": 0,
    }
    record = json.loads((tmp_path / "metadata.jsonl").read_text().splitlines()[0])
    assert record["label_path"] == "labels/session-1/ball-1.npz"
    assert record["annotator"] == "tester"


def test_annotation_rejects_invalid_geometry(tmp_path):
    background_path = _make_staging(tmp_path)
    dataset = BallAnnotationDataset(tmp_path, background_path)

    with pytest.raises(ValueError, match="inside the image"):
        dataset.save_annotation("ball-1", [10.0, 4.0], 20.0, "tester")
    with pytest.raises(ValueError, match="contact radius < ball radius"):
        dataset.save_annotation("ball-1", [5.0, 4.0], 60.0, "tester")
    with pytest.raises(ValueError, match="annotator"):
        dataset.save_annotation("ball-1", [5.0, 4.0], 20.0, "")


def test_samples_expose_shared_background(tmp_path):
    background_path = _make_staging(tmp_path)
    dataset = BallAnnotationDataset(tmp_path, background_path)
    sample = dataset.samples()[0]

    assert sample["background_url"].endswith("/background")
    assert sample["saved"] is False
    assert dataset.file_path("ball-1", "background") == background_path


def test_rejection_is_reversible_and_excludes_annotation_completion(tmp_path):
    background_path = _make_staging(tmp_path)
    dataset = BallAnnotationDataset(tmp_path, background_path)
    dataset.save_annotation("ball-1", [5.0, 4.0], 20.0, "tester")

    rejected = dataset.set_rejection("ball-1", True, "tester")

    assert rejected == {
        "sample_id": "ball-1",
        "rejected": True,
        "rejected_by": "tester",
        "rejection_reason": "visual quality control",
    }
    assert dataset.status() == {
        "total": 1, "completed": 0, "rejected": 1, "remaining": 0,
    }
    assert dataset.samples()[0]["rejected"] is True
    with pytest.raises(ValueError, match="rejected"):
        dataset.save_annotation("ball-1", [5.0, 4.0], 20.0, "tester")

    restored = dataset.set_rejection("ball-1", False, "tester")

    assert restored == {"sample_id": "ball-1", "rejected": False}
    assert dataset.status() == {
        "total": 1, "completed": 1, "rejected": 0, "remaining": 0,
    }


def test_static_annotation_assets_are_not_cached(tmp_path):
    background_path = _make_staging(tmp_path)
    dataset = BallAnnotationDataset(tmp_path, background_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(dataset))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(
            f"http://127.0.0.1:{server.server_port}/app.js"
        ) as response:
            body = response.read()
            assert response.headers["Cache-Control"] == "no-store"
        assert b"ctx.fill(" not in body
        assert b'"/rejection"' in body
        with urlopen(
            f"http://127.0.0.1:{server.server_port}/?v=replacement-v1"
        ) as response:
            index = response.read()
        assert b'app.js?v=replacement-v1' in index
        assert b'id="reject"' in index
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
