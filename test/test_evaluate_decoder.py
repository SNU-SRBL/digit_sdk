import json

import pytest

from calibration.evaluate_decoder import (
    _test_identity,
    _is_eligible,
    _load_selection,
    _write_json_once,
    evaluate_or_load,
)


def test_frozen_test_result_cannot_be_overwritten(tmp_path):
    path = tmp_path / "test.json"

    _write_json_once(path, {"eligible": True})

    assert json.loads(path.read_text()) == {"eligible": True}
    with pytest.raises(FileExistsError, match="already exists"):
        _write_json_once(path, {"eligible": False})


def test_selection_must_be_frozen_and_untested(tmp_path):
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({
        "serial": "DTEST",
        "method": "mixed_background",
        "configuration_frozen": True,
        "test_evaluated": False,
    }))

    assert _load_selection(path, "DTEST")["configuration_frozen"]

    document = json.loads(path.read_text())
    document["test_evaluated"] = True
    path.write_text(json.dumps(document))
    with pytest.raises(RuntimeError, match="already evaluated"):
        _load_selection(path, "DTEST")


def _cached_selection(root):
    decoder = root / ".seed_17_training" / "decoder.pth"
    decoder.parent.mkdir()
    decoder.write_bytes(b"decoder-v1")
    return {
        "serial": "DTEST",
        "method": "mixed_background",
        "dataset_id": "DTEST",
        "selected": {
            "seed": 17,
            "decoder": "mixed_background/.seed_17_training/decoder.pth",
        },
        "ball_split_id": "ball-split",
        "manual_mask_split_id": "manual-split",
        "maximum_depth_mm": 2.2,
        "contact_threshold_mm": 0.1,
        "test_evaluated": False,
    }


def test_existing_matching_test_is_reused(tmp_path):
    root = tmp_path / "DTEST" / "model/tactile_transformer/mixed_background"
    root.mkdir(parents=True)
    selection = _cached_selection(root)
    (root / "selection.json").write_text(json.dumps(selection))
    result = {
        **_test_identity(selection, root.parent),
        "eligible": True,
    }
    (root / "test.json").write_text(json.dumps(result))

    actual = evaluate_or_load(
        "DTEST", tmp_path, device=None, batch_size=4, workers=0,
        base_cache_dir=None,
    )

    assert actual == result
    assert json.loads((root / "selection.json").read_text())["test_evaluated"]


def test_cached_test_rejects_changed_selected_decoder(tmp_path):
    root = tmp_path / "DTEST" / "model/tactile_transformer/mixed_background"
    root.mkdir(parents=True)
    selection = _cached_selection(root)
    (root / "selection.json").write_text(json.dumps(selection))
    (root / "test.json").write_text(json.dumps({
        **_test_identity(selection, root.parent),
        "eligible": True,
    }))
    (root / ".seed_17_training" / "decoder.pth").write_bytes(b"decoder-v2")

    with pytest.raises(RuntimeError, match="does not match selected decoder"):
        evaluate_or_load(
            "DTEST", tmp_path, device=None, batch_size=4, workers=0,
            base_cache_dir=None,
        )


def test_test_eligibility_requires_no_contact_depth_below_contract():
    ball = {"contact_mae_mm": 0.15}
    manual = {"missed_contacts": 0}

    assert _is_eligible(ball, manual, {"max_depth_mm": 0.099})
    assert not _is_eligible(ball, manual, {"max_depth_mm": 0.1})
