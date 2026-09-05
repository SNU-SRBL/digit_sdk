import json

import pytest

from calibration.promote_decoder import PRODUCTION_METHOD, promote


def _write_candidate(sensor_root, *, eligible=True):
    root = sensor_root / "model/tactile_transformer/mixed"
    root.mkdir(parents=True)
    decoder = root / "decoder.pth"
    decoder.write_bytes(b"decoder")
    selection = {
        "serial": "DTEST",
        "method": "mixed",
        "configuration_frozen": True,
        "test_evaluated": True,
        "selected": {
            "seed": 29,
            "validation": {"eligible": True},
        },
        "maximum_depth_mm": 2.2,
        "contact_threshold_mm": 0.1,
    }
    test = {
        "serial": "DTEST",
        "method": "mixed",
        "eligible": eligible,
    }
    (root / "selection.json").write_text(json.dumps(selection))
    (root / "test.json").write_text(json.dumps(test))


def test_only_tested_eligible_decoder_is_promoted(
    tmp_path, monkeypatch
):
    sensor_root = tmp_path / "DTEST"
    calibration = sensor_root / "calibration"
    calibration.mkdir(parents=True)
    (sensor_root / "DTEST.yaml").write_text("device_type: DIGIT\n")
    _write_candidate(sensor_root)
    summary = type("Summary", (), {"serial": "DTEST", "dataset_id": "DTEST"})
    monkeypatch.setattr(
        "calibration.promote_decoder.validate_dataset", lambda _: summary
    )

    destination = promote("DTEST", tmp_path)

    assert destination == sensor_root / "model/depth"
    assert (destination / "decoder.pth").read_bytes() == b"decoder"
    assert "maximum_depth_mm: 2.2" in (sensor_root / "DTEST.yaml").read_text()
    with pytest.raises(FileExistsError, match="already exists"):
        promote("DTEST", tmp_path)


def test_failed_frozen_test_cannot_be_promoted(tmp_path, monkeypatch):
    sensor_root = tmp_path / "DTEST"
    calibration = sensor_root / "calibration"
    calibration.mkdir(parents=True)
    (sensor_root / "DTEST.yaml").write_text("device_type: DIGIT\n")
    _write_candidate(sensor_root, eligible=False)
    summary = type("Summary", (), {"serial": "DTEST", "dataset_id": "DTEST"})
    monkeypatch.setattr(
        "calibration.promote_decoder.validate_dataset", lambda _: summary
    )

    with pytest.raises(ValueError, match="failed"):
        promote("DTEST", tmp_path)
