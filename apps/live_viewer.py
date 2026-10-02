"""Standalone live DIGIT viewer with selectable direct-SDK modes.

Modes:
    depth       raw frame beside production metric depth (default)
    pointcloud  production depth-to-points geometry in an Open3D window
    force       production Sparsh force field and vector overlay

Usage:
    python3 apps/live_viewer.py --serial D21275
    python3 apps/live_viewer.py --serial D21275 --mode pointcloud
    python3 apps/live_viewer.py --serial D21275 --mode force --device cpu
"""
import argparse
import math
import os
import sys
import time

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from digit_sdk.depth_geometry import depth_to_pointcloud  # noqa: E402
from digit_sdk.temporal import NeuralFeelsFIR, PersistenceCutoff  # noqa: E402
from digit_sdk.viz_utils import (  # noqa: E402
    force_field_to_rgb,
    visualize_force_vector,
)

DEFAULT_SENSORS_ROOT = os.path.join(PROJECT_ROOT, "sensors")
DEFAULT_MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")

# Logical force names from sensors/<serial>/<serial>.yaml -> files in models/.
_FORCE_MODEL_FILES = {
    "sparsh-dino-base": "sparsh_dino_base_encoder.ckpt",
    "sparsh-digit-forcefield": "sparsh_digit_forcefield_decoder.pth",
}

WINDOW_DEPTH = "DIGIT live viewer (raw | depth mm)"
WINDOW_POINTCLOUD = "DIGIT live viewer (pointcloud)"
WINDOW_POINTCLOUD_PROJECTED = "DIGIT live viewer (pointcloud, projected)"
WINDOW_FORCE = "DIGIT live viewer (raw | force)"
_TEMPORAL_RESET_GAP_NS = 100_000_000


class ProductionDepthFilter:
    """Mirror the ROS depth temporal and persistence post-processing."""

    def __init__(self, cutoff_mm):
        self._fir = NeuralFeelsFIR()
        self._persistence = (
            PersistenceCutoff(cutoff_mm) if cutoff_mm > 0 else None
        )
        self._last_timestamp_ns = None

    def __call__(self, depth_raw_mm, timestamp_ns=None):
        timestamp_ns = time.monotonic_ns() if timestamp_ns is None else timestamp_ns
        if (
            self._last_timestamp_ns is not None
            and (
                timestamp_ns <= self._last_timestamp_ns
                or timestamp_ns - self._last_timestamp_ns > _TEMPORAL_RESET_GAP_NS
            )
        ):
            self._fir.reset()
            if self._persistence is not None:
                self._persistence.reset()
        self._last_timestamp_ns = timestamp_ns
        depth_filtered_mm = self._fir(depth_raw_mm)
        return (
            self._persistence(depth_filtered_mm)
            if self._persistence is not None
            else depth_filtered_mm
        )


def to_bgr(frame):
    """Return a 3-channel BGR view of a grayscale or colour frame."""
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def render_depth(depth_mm, cutoff_mm):
    """Colour-map raw millimetre depth, clipped to [0, cutoff_mm]."""
    if cutoff_mm <= 0:
        raise ValueError("cutoff_mm must be positive")
    depth = np.asarray(depth_mm, dtype=np.float32)
    scaled = np.clip(depth / cutoff_mm, 0.0, 1.0)
    gray = (scaled * 255.0).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)


def render_pointcloud_projection(points, width, height, ppmm):
    """Orthographically project real point coordinates to a BGR point image.

    Fallback display when no 3D visualizer is available. Draws the actual
    ``(N, 3)`` optical-frame points, coloured by depth, instead of a depth map.
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if points.size == 0 or ppmm <= 0:
        return canvas
    depth_mm = np.clip(-points[:, 2] * 1000.0, 0.0, None)
    peak = float(depth_mm.max())
    if peak <= 0:
        return canvas
    gray = np.clip(depth_mm / peak, 0.0, 1.0)
    colors = cv2.applyColorMap(
        (gray * 255.0).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO
    ).reshape(-1, 3)
    cols = np.rint(points[:, 0] * ppmm * 1000.0 + width / 2 - 0.5).astype(int)
    rows = np.rint(points[:, 1] * ppmm * 1000.0 + height / 2 - 0.5).astype(int)
    inside = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
    canvas[rows[inside], cols[inside]] = colors[inside]
    return canvas


def resolve_force_model_paths(config, models_root):
    """Map logical force model names in the sensor config to model files."""
    force = (config or {}).get("force") or {}
    encoder_name = force.get("encoder", "sparsh-dino-base")
    decoder_name = force.get("decoder", "sparsh-digit-forcefield")
    try:
        encoder_file = _FORCE_MODEL_FILES[encoder_name]
        decoder_file = _FORCE_MODEL_FILES[decoder_name]
    except KeyError as error:
        raise ValueError(
            f"unsupported force model name {error.args[0]!r}; "
            f"known names: {sorted(_FORCE_MODEL_FILES)}"
        ) from error
    return (
        os.path.join(models_root, encoder_file),
        os.path.join(models_root, decoder_file),
    )


def default_background_path(serial, sensors_root):
    """Return the standard no-contact background reference for a sensor."""
    return os.path.join(
        sensors_root, serial, "calibration", "background", "reference.png"
    )


def load_image(path, description="image"):
    """Load a BGR uint8 image, raising a clear error when unavailable."""
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"{description} not found or unreadable: {path}")
    return image


def load_sensor_config(serial, sensors_root):
    from digit_sdk.utils import load_config

    return load_config(serial=serial, sensors_root=sensors_root)


def effective_framerate(requested, config=None):
    """Effective capture rate: validated CLI override else sensor config."""
    if requested is None:
        return None if config is None else config.get("framerate")
    value = float(requested)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("--framerate must be a finite positive value")
    return value


def start_camera(serial, sensors_root, framerate=None):
    """Open and start the camera; return None with a clear error on failure."""
    from digit_sdk.camera import Camera

    camera = Camera(serial=serial, sensors_root=sensors_root,
                    framerate=framerate)
    try:
        camera.connect(verbose=True)
    except RuntimeError as error:
        print(f"error: could not start camera: {error}", file=sys.stderr)
        return None
    return camera


def start_depth_estimator(serial, sensors_root, device):
    """Build the production depth estimator; return None on failure."""
    try:
        from digit_sdk.depth import DepthEstimator
    except ImportError as error:
        print(f"error: depth stack unavailable: {error}", file=sys.stderr)
        return None
    try:
        return DepthEstimator([serial], sensors_root, device)
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        print(f"error: could not start depth estimator: {error}", file=sys.stderr)
        return None


def wait_or_quit(window_name):
    """Return False when the user pressed q/Esc or closed the window."""
    key = cv2.waitKey(1)
    if key in (ord("q"), 27):
        return False
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        return False
    return True


def camera_size(camera):
    """Return the expected (width, height) for the configured camera."""
    return int(camera.raw_imgw), int(camera.raw_imgh)


class _ProjectedPointCloudDisplay:
    """cv2 window showing projected points beside the raw frame."""

    def __init__(self, width, height, ppmm):
        self._name = WINDOW_POINTCLOUD_PROJECTED
        self._width = width
        self._height = height
        self._ppmm = ppmm

    def update(self, frame, points):
        view = render_pointcloud_projection(
            points, self._width, self._height, self._ppmm
        )
        cv2.imshow(self._name, np.hstack([to_bgr(frame), view]))
        return wait_or_quit(self._name)

    def close(self):
        cv2.destroyAllWindows()


class _Open3DPointCloudDisplay:
    """Open3D window showing the production point-cloud geometry."""

    def __init__(self):
        import open3d as o3d

        self._o3d = o3d
        self._vis = o3d.visualization.Visualizer()
        if not self._vis.create_window(
            window_name=WINDOW_POINTCLOUD, width=640, height=480
        ):
            raise RuntimeError("could not create Open3D window")
        self._pcd = o3d.geometry.PointCloud()
        if not self._vis.add_geometry(self._pcd):
            raise RuntimeError("could not add Open3D geometry")
        option = self._vis.get_render_option()
        if option is not None:
            option.point_size = 4.0

    def update(self, frame, points):
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        self._pcd.points = self._o3d.utility.Vector3dVector(pts)
        depth_mm = np.clip(-pts[:, 2] * 1000.0, 0.0, None)
        peak = float(depth_mm.max()) if depth_mm.size else 0.0
        if peak > 0:
            gray = (depth_mm / peak).reshape(-1, 1)
            colors = np.repeat(gray, 3, axis=1)
        else:
            colors = np.zeros((0, 3))
        self._pcd.colors = self._o3d.utility.Vector3dVector(colors)
        self._vis.update_geometry(self._pcd)
        if not self._vis.poll_events():
            return False
        self._vis.update_renderer()
        return True

    def close(self):
        try:
            self._vis.destroy_window()
        except Exception:  # window already gone
            pass


def try_import_open3d():
    """Return the Open3D module, or None when it is not installed."""
    try:
        import open3d
    except ImportError:
        return None
    return open3d


def _depth_loop(camera, estimator, serial, depth_filter, display_max_mm):
    while True:
        frame = camera.get_image()
        if frame is None:
            continue
        depth_mm = depth_filter(estimator.estimate(serial, frame))
        if depth_mm.shape != frame.shape[:2]:
            depth_mm = cv2.resize(
                depth_mm, (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        cv2.imshow(
            WINDOW_DEPTH,
            np.hstack([to_bgr(frame), render_depth(depth_mm, display_max_mm)]),
        )
        if not wait_or_quit(WINDOW_DEPTH):
            break
    return 0


def _pointcloud_loop(
    camera, estimator, serial, ppmm, point_sample_mm, display, depth_filter
):
    while True:
        frame = camera.get_image()
        if frame is None:
            continue
        depth_mm = depth_filter(estimator.estimate(serial, frame))
        if depth_mm.shape != frame.shape[:2]:
            depth_mm = cv2.resize(
                depth_mm, (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        points = depth_to_pointcloud(depth_mm, ppmm, point_sample_mm)
        if not display.update(frame, points):
            break
    return 0


def _force_loop(camera, estimator):
    while True:
        frame = camera.get_image()
        if frame is None:
            continue
        result = estimator.estimate(frame)
        if result is None:
            view = to_bgr(frame)
        else:
            vector = result["force_vector_physical"]
            left = visualize_force_vector(
                vector["fx"], vector["fy"], vector["fz"], frame
            )
            field = result["force_field"]
            heat = force_field_to_rgb(field["normal"], field["shear"])
            heat = cv2.resize(
                heat, (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            view = np.hstack([to_bgr(left), cv2.cvtColor(heat, cv2.COLOR_RGB2BGR)])
        cv2.imshow(WINDOW_FORCE, view)
        if not wait_or_quit(WINDOW_FORCE):
            break
    return 0


def run_depth(args):
    """Raw frame beside the production metric depth map."""
    serial, sensors_root, device = args.serial, args.sensors_root, args.device
    try:
        config = load_sensor_config(serial, sensors_root)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    cutoff_mm = float(args.depth_cutoff_mm)
    if not math.isfinite(cutoff_mm) or cutoff_mm < 0:
        print("error: --depth-cutoff-mm must be finite and non-negative",
              file=sys.stderr)
        return 2
    if args.depth_display_max_mm is not None:
        display_max_mm = float(args.depth_display_max_mm)
    else:
        try:
            display_max_mm = float(config["maximum_depth_mm"])
        except (KeyError, TypeError, ValueError):
            print("error: config missing numeric maximum_depth_mm",
                  file=sys.stderr)
            return 2
    if not math.isfinite(display_max_mm) or display_max_mm <= 0:
        print("error: --depth-display-max-mm must be finite and positive",
              file=sys.stderr)
        return 2

    camera = start_camera(serial, sensors_root, args.framerate)
    if camera is None:
        return 2
    try:
        estimator = start_depth_estimator(serial, sensors_root, device)
        if estimator is None:
            return 2
        depth_filter = ProductionDepthFilter(cutoff_mm)
        print(
            f"Depth viewer ready for {serial} (capture {camera.framerate} Hz, "
            f"ROS cutoff {cutoff_mm} mm, display range {display_max_mm} mm, "
            f"device {device}). "
            "Press q or Esc to quit.",
            flush=True,
        )
        return _depth_loop(
            camera, estimator, serial, depth_filter, display_max_mm
        )
    finally:
        camera.release()
        cv2.destroyAllWindows()


def run_pointcloud(args):
    """Production depth-to-points geometry in an installed visualizer."""
    serial, sensors_root, device = args.serial, args.sensors_root, args.device
    try:
        config = load_sensor_config(serial, sensors_root)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    try:
        ppmm = float(config["ppmm"])
    except (KeyError, TypeError, ValueError):
        print("error: config missing numeric ppmm", file=sys.stderr)
        return 2
    if ppmm <= 0:
        print("error: positive ppmm is required for pointcloud mode",
              file=sys.stderr)
        return 2
    point_sample_mm = float(args.point_sample_mm)
    if point_sample_mm < 0:
        print("error: --point-sample-mm must be non-negative", file=sys.stderr)
        return 2
    cutoff_mm = float(args.depth_cutoff_mm)
    if not math.isfinite(cutoff_mm) or cutoff_mm < 0:
        print("error: --depth-cutoff-mm must be finite and non-negative",
              file=sys.stderr)
        return 2

    camera = start_camera(serial, sensors_root, args.framerate)
    if camera is None:
        return 2
    display = None
    try:
        estimator = start_depth_estimator(serial, sensors_root, device)
        if estimator is None:
            return 2
        depth_filter = ProductionDepthFilter(cutoff_mm)
        if try_import_open3d() is None:
            print(
                "note: Open3D not installed; using projected point view",
                file=sys.stderr,
            )
            display = _ProjectedPointCloudDisplay(*camera_size(camera), ppmm)
        else:
            try:
                display = _Open3DPointCloudDisplay()
            except Exception as error:
                print(
                    f"note: Open3D window unavailable ({error}); "
                    "using projected point view",
                    file=sys.stderr,
                )
                display = _ProjectedPointCloudDisplay(*camera_size(camera), ppmm)
        print(
            f"Point-cloud viewer ready for {serial} (capture {camera.framerate} Hz, "
            f"ROS cutoff {cutoff_mm} mm, ppmm {ppmm}, "
            f"sample {point_sample_mm} mm, device {device}). "
            "Press q or Esc to quit.",
            flush=True,
        )
        return _pointcloud_loop(
            camera, estimator, serial, ppmm, point_sample_mm, display, depth_filter
        )
    finally:
        if display is not None:
            display.close()
        camera.release()
        cv2.destroyAllWindows()


def run_force(args):
    """Production Sparsh force field and vector overlay."""
    serial, sensors_root, device = args.serial, args.sensors_root, args.device
    try:
        config = load_sensor_config(serial, sensors_root)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    force_cfg = config.get("force") or {}
    if not force_cfg.get("enable_force", False):
        print(
            f"error: force mode is disabled for {serial} "
            "(set force.enable_force: true in its sensor YAML)",
            file=sys.stderr,
        )
        return 2
    try:
        encoder_path, decoder_path = resolve_force_model_paths(
            config, args.models_root
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    missing = [
        path for path in (encoder_path, decoder_path)
        if not os.path.isfile(path)
    ]
    if missing:
        print(
            "error: force model files missing: "
            + ", ".join(missing)
            + " (run: python3 scripts/download_models.py)",
            file=sys.stderr,
        )
        return 2
    background_path = args.background or default_background_path(
        serial, sensors_root
    )
    try:
        background = load_image(background_path, "force background")
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    temporal_stride = int(force_cfg.get("temporal_stride", 5))
    bg_offset = float(force_cfg.get("bg_offset", 0.5))
    force_scale = force_cfg.get("force_vector_scale")

    camera = start_camera(serial, sensors_root, args.framerate)
    if camera is None:
        return 2
    try:
        try:
            from digit_sdk.force_estimator import ForceEstimator
        except ImportError as error:
            print(f"error: force stack unavailable: {error}", file=sys.stderr)
            return 2
        try:
            estimator = ForceEstimator(
                encoder_path,
                decoder_path,
                temporal_stride=temporal_stride,
                bg_offset=bg_offset,
                device=device,
                force_vector_scale=force_scale,
            )
            estimator.load_background(background)
        except (RuntimeError, ValueError, FileNotFoundError, OSError) as error:
            print(
                f"error: could not start force estimator: {error}",
                file=sys.stderr,
            )
            return 2
        print(
            f"Force viewer ready for {serial} (capture {camera.framerate} Hz, "
            f"stride {temporal_stride}, device {device}, "
            f"background {background_path}). "
            "Press q or Esc to quit.",
            flush=True,
        )
        return _force_loop(camera, estimator)
    finally:
        camera.release()
        cv2.destroyAllWindows()


def build_parser():
    parser = argparse.ArgumentParser(
        description="Live view of raw DIGIT frames with a selectable "
                    "production mode (depth, pointcloud, force)."
    )
    parser.add_argument("--serial", required=True, help="sensor serial number")
    parser.add_argument(
        "--mode", choices=("depth", "pointcloud", "force"), default="depth",
        help="viewer mode (default: depth)",
    )
    parser.add_argument("--sensors-root", default=DEFAULT_SENSORS_ROOT,
                        help="root directory containing sensors/<serial>/")
    parser.add_argument("--device", default="cuda",
                        help="depth/force inference device (cuda or cpu)")
    parser.add_argument(
        "--framerate", type=float, default=None,
        help="requested camera capture rate in Hz (default: sensor config)",
    )
    parser.add_argument("--depth-cutoff-mm", type=float, default=0.2,
                        help="ROS-equivalent post-FIR cutoff in mm; 0 disables it")
    parser.add_argument("--depth-display-max-mm", type=float, default=None,
                        help="depth colour-map maximum in mm "
                             "(default: sensor maximum_depth_mm)")
    parser.add_argument("--point-sample-mm", type=float, default=0.2,
                        help="point-cloud sampling step in mm (pointcloud mode)")
    parser.add_argument("--background", default=None,
                        help="no-contact background image for force mode "
                             "(default: sensors/<serial>/calibration/"
                             "background/reference.png)")
    parser.add_argument("--models-root", default=DEFAULT_MODELS_ROOT,
                        help="directory holding Sparsh force checkpoints")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.framerate = effective_framerate(args.framerate)
    except (TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    runners = {
        "depth": run_depth,
        "pointcloud": run_pointcloud,
        "force": run_force,
    }
    return runners[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
