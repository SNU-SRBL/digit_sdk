import hashlib
import json

import pytest

from calibration.promote_decoder import PRODUCTION_METHOD, promote
from calibration.tactile_transformer.model import BASE_REVISION, BASE_SHA256


def _write_candidate(sensor_root, fingerprint, *, eligible=True):
    root = sensor_root / "model/tactile_transformer/mixed"
    root.mkdir(parents=True)
    decoder = root / "decoder.pth"
    decoder.write_bytes(b"decoder")
    checksum = hashlib.sha256(decoder.read_bytes()).hexdigest()
    selection = {
        "serial": "DTEST",
        "method": "mixed",
        "configuration_frozen": True,
        "test_evaluated": True,
        "selected": {
            "seed": 29,
            "decoder_sha256": checksum,
            "validation": {"eligible": True},
        },
        "base": {"revision": BASE_REVISION, "sha256": BASE_SHA256},
        "maximum_depth_mm": 2.2,
        "contact_threshold_mm": 0.1,
    }
    test = {
        "serial": "DTEST",
        "method": "mixed",
        "eligible": eligible,
        "dataset_fingerprint": fingerprint,
        "decoder_sha256": checksum,
    }
    (root / "selection.json").write_text(json.dumps(selection))
    (root / "test.json").write_text(json.dumps(test))


def test_only_tested_eligible_decoder_is_promoted(
    tmp_path, monkeypatch
):
    sensor_root = tmp_path / "DTEST"
    calibration = sensor_root / "calibration"
    calibration.mkdir(parents=True)
    fingerprint = "dataset-fingerprint"
    _write_candidate(sensor_root, fingerprint)
    summary = type("Summary", (), {"serial": "DTEST", "dataset_id": "DTEST"})
    monkeypatch.setattr(
        "calibration.promote_decoder.validate_dataset", lambda _: summary
    )
    monkeypatch.setattr(
        "calibration.promote_decoder.dataset_fingerprint",
        lambda _: fingerprint,
    )

    destination = promote("DTEST", tmp_path)

    assert destination == sensor_root / "model/depth"
    metadata = json.loads((destination / "metadata.json").read_text())
    assert metadata["method"] == PRODUCTION_METHOD
    assert metadata["encoder_frozen"] is True
    assert metadata["decoder_scope"] == "per_sensor"
    assert metadata["selected_seed"] == 29
    assert (destination / "decoder.pth").read_bytes() == b"decoder"
    with pytest.raises(FileExistsError, match="already exists"):
        promote("DTEST", tmp_path)


def test_failed_frozen_test_cannot_be_promoted(tmp_path, monkeypatch):
    sensor_root = tmp_path / "DTEST"
    calibration = sensor_root / "calibration"
    calibration.mkdir(parents=True)
    _write_candidate(sensor_root, "fingerprint", eligible=False)
    summary = type("Summary", (), {"serial": "DTEST", "dataset_id": "DTEST"})
    monkeypatch.setattr(
        "calibration.promote_decoder.validate_dataset", lambda _: summary
    )
    monkeypatch.setattr(
        "calibration.promote_decoder.dataset_fingerprint",
        lambda _: "fingerprint",
    )

    with pytest.raises(ValueError, match="failed"):
        promote("DTEST", tmp_path)
