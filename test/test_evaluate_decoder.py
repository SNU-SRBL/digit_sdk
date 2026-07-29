import json

import pytest

from calibration.evaluate_decoder import (
    _load_selection,
    _write_json_once,
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
        "method": "mixed",
        "configuration_frozen": True,
        "test_evaluated": False,
    }))

    assert _load_selection(path, "DTEST")["configuration_frozen"]

    document = json.loads(path.read_text())
    document["test_evaluated"] = True
    path.write_text(json.dumps(document))
    with pytest.raises(RuntimeError, match="already evaluated"):
        _load_selection(path, "DTEST")
