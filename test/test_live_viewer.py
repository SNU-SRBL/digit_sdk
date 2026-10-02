"""Non-hardware checks for the standalone live viewer."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


APP = Path(__file__).resolve().parents[1] / "apps" / "live_viewer.py"


def _load():
    spec = importlib.util.spec_from_file_location("live_viewer", APP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_render_depth_clips_and_shapes():
    viewer = _load()
    depth = np.array([[0.0, 1.0], [2.0, 4.0]], dtype=np.float32)
    rendered = viewer.render_depth(depth, cutoff_mm=2.0)
    assert rendered.shape == (2, 2, 3)
    assert rendered.dtype == np.uint8


def test_render_depth_rejects_nonpositive_cutoff():
    viewer = _load()
    with pytest.raises(ValueError):
        viewer.render_depth(np.zeros((2, 2), dtype=np.float32), cutoff_mm=0.0)


def test_production_depth_filter_matches_ros_persistence_and_gap_reset():
    viewer = _load()
    depth = np.full((2, 2), 0.3, dtype=np.float32)
    depth_filter = viewer.ProductionDepthFilter(0.2)
    assert not depth_filter(depth, timestamp_ns=1).any()
    assert depth_filter(depth, timestamp_ns=2).min() > 0.2
    assert not depth_filter(depth, timestamp_ns=100_000_003).any()


def test_production_depth_filter_allows_zero_cutoff():
    viewer = _load()
    depth = np.full((2, 2), 0.05, dtype=np.float32)
    assert np.allclose(
        viewer.ProductionDepthFilter(0)(depth, timestamp_ns=1), depth
    )


def test_to_bgr_promotes_grayscale():
    viewer = _load()
    gray = np.zeros((2, 2), dtype=np.uint8)
    assert viewer.to_bgr(gray).shape == (2, 2, 3)


def test_render_pointcloud_projection_draws_real_points():
    viewer = _load()
    points = np.array(
        [[0.0, 0.0, -0.002], [10.0, 10.0, -0.002]], dtype=np.float32
    )
    image = viewer.render_pointcloud_projection(points, 32, 24, ppmm=18.5)
    assert image.shape == (24, 32, 3)
    assert image.dtype == np.uint8
    assert image[12, 16].any()  # centre point lands on the canvas


def test_render_pointcloud_projection_empty_is_blank():
    viewer = _load()
    empty = np.zeros((0, 3), dtype=np.float32)
    image = viewer.render_pointcloud_projection(empty, 32, 24, ppmm=18.5)
    assert image.shape == (24, 32, 3)
    assert not image.any()


def test_render_pointcloud_projection_rejects_nonpositive_ppmm():
    viewer = _load()
    points = np.zeros((1, 3), dtype=np.float32)
    assert not viewer.render_pointcloud_projection(points, 8, 8, ppmm=0.0).any()


def test_resolve_force_model_paths_maps_logical_names(tmp_path):
    viewer = _load()
    encoder = tmp_path / "sparsh_dino_base_encoder.ckpt"
    decoder = tmp_path / "sparsh_digit_forcefield_decoder.pth"
    encoder.write_bytes(b"e")
    decoder.write_bytes(b"d")
    config = {
        "force": {
            "encoder": "sparsh-dino-base",
            "decoder": "sparsh-digit-forcefield",
        }
    }
    enc, dec = viewer.resolve_force_model_paths(config, str(tmp_path))
    assert Path(enc) == encoder
    assert Path(dec) == decoder


def test_resolve_force_model_paths_rejects_unknown_name(tmp_path):
    viewer = _load()
    config = {
        "force": {
            "encoder": "unknown-encoder",
            "decoder": "sparsh-digit-forcefield",
        }
    }
    with pytest.raises(ValueError):
        viewer.resolve_force_model_paths(config, str(tmp_path))


def test_force_mode_requires_sensor_enablement(monkeypatch):
    viewer = _load()
    monkeypatch.setattr(
        viewer,
        "load_sensor_config",
        lambda *_args: {"force": {"enable_force": False}},
    )
    args = SimpleNamespace(
        serial="D1",
        sensors_root="/sensors",
        models_root="/models",
        background=None,
        framerate=None,
        device="cpu",
    )

    assert viewer.run_force(args) == 2


def test_default_background_path_targets_sensor_reference():
    viewer = _load()
    path = viewer.default_background_path("D1", "/sensors")
    assert path == "/sensors/D1/calibration/background/reference.png"


def test_load_image_missing_raises(tmp_path):
    viewer = _load()
    with pytest.raises(FileNotFoundError):
        viewer.load_image(str(tmp_path / "missing.png"), "force background")


def test_main_requires_serial():
    viewer = _load()
    with pytest.raises(SystemExit):
        viewer.main([])


def test_main_rejects_unknown_mode():
    viewer = _load()
    with pytest.raises(SystemExit):
        viewer.main(["--serial", "D1", "--mode", "nope"])


@pytest.mark.parametrize(
    "mode,runner",
    [("depth", "run_depth"), ("pointcloud", "run_pointcloud"),
     ("force", "run_force")],
)
def test_main_dispatches_selected_mode(monkeypatch, mode, runner):
    viewer = _load()
    seen = {}

    def fake(args):
        seen["mode"] = args.mode
        seen["serial"] = args.serial
        return 7

    monkeypatch.setattr(viewer, runner, fake)
    code = viewer.main(["--serial", "D1", "--mode", mode])
    assert code == 7
    assert seen == {"mode": mode, "serial": "D1"}


def test_main_defaults_to_depth_mode(monkeypatch):
    viewer = _load()
    seen = {}
    monkeypatch.setattr(
        viewer, "run_depth",
        lambda args: seen.update(mode=args.mode) or 0,
    )
    assert viewer.main(["--serial", "D1"]) == 0
    assert seen["mode"] == "depth"


def test_main_forwards_mode_options(monkeypatch):
    viewer = _load()
    seen = {}
    monkeypatch.setattr(
        viewer, "run_pointcloud",
        lambda args: seen.update(
            point_sample_mm=args.point_sample_mm, device=args.device,
            framerate=args.framerate,
        ) or 0,
    )
    code = viewer.main([
        "--serial", "D1", "--mode", "pointcloud",
        "--point-sample-mm", "0.5", "--device", "cpu", "--framerate", "30",
    ])
    assert code == 0
    assert seen == {"point_sample_mm": 0.5, "device": "cpu", "framerate": 30.0}


@pytest.mark.parametrize("value", ("0", "-1", "nan", "inf"))
def test_main_rejects_invalid_framerate(capsys, value):
    viewer = _load()
    assert viewer.main(["--serial", "D1", "--framerate", value]) == 2
    assert "--framerate must be a finite positive value" in capsys.readouterr().err
