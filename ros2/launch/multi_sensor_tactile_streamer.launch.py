"""Launch isolated capture/publish processes and one batched depth pipeline."""

import os
import subprocess
from typing import List

import yaml
from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

try:
    from core.config.config_mapper import map_tactile_cpu_affinity
except ImportError:
    map_tactile_cpu_affinity = None


_DEFAULT_TACTILE_AFFINITY = {
    "camera_shm": [0, 1, 2, 3],
    "pipeline_node": [4, 5, 10, 11],
    "raw_publisher": [6, 7, 8, 9],
    "surface_publisher": [6, 7, 8, 9],
    "pointcloud_publisher": [6, 7, 8, 9],
}


def _list_to_affinity_string(cores: List[int]) -> str:
    if not cores:
        return ""
    sorted_cores = sorted(set(cores))
    parts = []
    start = end = sorted_cores[0]
    for core in sorted_cores[1:]:
        if core == end + 1:
            end = core
        else:
            parts.append(f"{start}-{end}" if start != end else str(start))
            start = end = core
    parts.append(f"{start}-{end}" if start != end else str(start))
    return ",".join(parts)


def _get_tactile_affinity() -> dict:
    if map_tactile_cpu_affinity is not None:
        try:
            configured = map_tactile_cpu_affinity()
            if configured:
                return configured
        except Exception:
            pass
    return dict(_DEFAULT_TACTILE_AFFINITY)


def _sensor_config(sensors_root: str, serial: str) -> dict:
    path = os.path.join(sensors_root, serial, f"{serial}.yaml")
    try:
        with open(path) as stream:
            return yaml.safe_load(stream) or {}
    except (OSError, yaml.YAMLError):
        return {}


def _owned_serials(context) -> List[str]:
    sensors_root = LaunchConfiguration("sensors_root").perform(context)
    serials_argument = LaunchConfiguration("serials").perform(context)
    sensors = [
        value.strip() for value in serials_argument.split(",") if value.strip()
    ]
    if not sensors and os.path.isdir(sensors_root):
        for item in sorted(os.listdir(sensors_root)):
            config = os.path.join(sensors_root, item, f"{item}.yaml")
            if os.path.isfile(config):
                sensors.append(item)
    if not sensors:
        sensors = ["D21275", "D21273", "D21242", "D21119"]
    return sensors


def _serial_scoped_cleanup(serial: str) -> None:
    """Stop only processes and SHM files owned by this launch instance."""
    for pattern in (
        rf"camera_shm .*--serial {serial}( |$)",
        rf"raw_publisher .*serial.{serial}( |$)",
        rf"surface_publisher .*serial.{serial}( |$)",
    ):
        subprocess.run(
            ["pkill", "-2", "-f", pattern],
            check=False,
            timeout=3,
        )
    shm_names = [
        f"/dev/shm/tactile_{serial}{suffix}"
        for suffix in ("", "_surface", "_force")
    ]
    subprocess.run(
        ["rm", "-f", *shm_names],
        check=False,
        timeout=3,
    )


def launch_setup(context, *_args, **_kwargs):
    sensors = _owned_serials(context)
    for serial in sensors:
        _serial_scoped_cleanup(serial)

    sensors_root = LaunchConfiguration("sensors_root").perform(context)
    model_device = LaunchConfiguration("model_device").perform(context)
    rate = float(LaunchConfiguration("rate").perform(context))
    depth_cutoff = float(LaunchConfiguration("depth_cutoff").perform(context))
    publish_raw = LaunchConfiguration("publish_raw").perform(context) == "true"
    publish_depth = (
        LaunchConfiguration("publish_depth").perform(context) == "true"
    )
    publish_pointcloud = (
        LaunchConfiguration("publish_pointcloud").perform(context) == "true"
    )
    point_sample_mm = float(
        LaunchConfiguration("point_sample_mm").perform(context)
    )
    shm_connect_timeout = float(
        LaunchConfiguration("shm_connect_timeout").perform(context)
    )

    affinity = _get_tactile_affinity()
    camera_cores = affinity.get("camera_shm", [0, 1, 2, 3])
    raw_cores = _list_to_affinity_string(
        affinity.get("raw_publisher", [6, 7, 8, 9])
    )
    pipeline_cores = _list_to_affinity_string(
        affinity.get("pipeline_node", [4, 5, 10, 11])
    )
    surface_cores = _list_to_affinity_string(
        affinity.get("surface_publisher", [6, 7, 8, 9])
    )
    pointcloud_cores = _list_to_affinity_string(
        affinity.get("pointcloud_publisher", [6, 7, 8, 9])
    )
    camera_executable = os.path.join(
        get_package_prefix("digit_sdk"), "lib", "digit_sdk", "camera_shm"
    )
    nodes = []

    for index, serial in enumerate(sensors):
        core = camera_cores[index] if index < len(camera_cores) else index
        nodes.append(ExecuteProcess(
            cmd=[
                camera_executable,
                "--serial", serial,
                "--sensors-root", sensors_root,
                "--cpu-affinity", str(core),
            ],
            name=f"camera_{serial}",
            output="screen",
        ))

    if publish_depth or publish_pointcloud:
        nodes.append(Node(
            package="digit_sdk",
            executable="pipeline_node",
            name="pipeline_node",
            output="screen",
            parameters=[{
                "serials": sensors,
                "sensors_root": sensors_root,
                "model_device": model_device,
                "depth_cutoff": depth_cutoff,
                "rate": rate,
                "cpu_affinity": pipeline_cores,
            }],
        ))

    for serial in sensors:
        if publish_raw:
            nodes.append(Node(
                package="digit_sdk",
                executable="raw_publisher",
                name=f"raw_pub_{serial}",
                output="screen",
                parameters=[{
                    "serial": serial,
                    "rate": rate,
                    "cpu_affinity": raw_cores,
                }],
            ))

        config = _sensor_config(sensors_root, serial)
        surface_outputs = []
        if publish_depth:
            surface_outputs.append(("depth", True, False))
        if publish_pointcloud:
            surface_outputs.append(("pointcloud", False, True))
        for suffix, enable_depth, enable_pointcloud in surface_outputs:
            nodes.append(Node(
                package="digit_sdk",
                executable="surface_publisher",
                name=f"surface_{suffix}_pub_{serial}",
                output="screen",
                parameters=[{
                    "serial": serial,
                    "rate": rate,
                    "cpu_affinity": (
                        pointcloud_cores if enable_pointcloud else surface_cores
                    ),
                    "publish_depth": enable_depth,
                    "publish_pointcloud": enable_pointcloud,
                    "shm_connect_timeout": shm_connect_timeout,
                    "ppmm": float(config.get("ppmm", 0.0)),
                    "point_sample_mm": point_sample_mm,
                }],
            ))

    return nodes


def generate_launch_description():
    launch_dir = os.path.dirname(os.path.abspath(__file__))
    source_path = os.path.abspath(os.path.join(launch_dir, "..", "..", "sensors"))
    installed_path = os.path.abspath(os.path.join(launch_dir, "..", "sensors"))
    default_root = source_path if os.path.exists(source_path) else installed_path
    return LaunchDescription([
        DeclareLaunchArgument(
            "sensors_root",
            default_value=default_root,
            description="Root directory for sensor configuration and depth models",
        ),
        DeclareLaunchArgument(
            "serials",
            default_value="",
            description="Comma-separated sensor serials; empty auto-discovers configs",
        ),
        DeclareLaunchArgument(
            "model_device", default_value="cuda", description="cuda or cpu"
        ),
        DeclareLaunchArgument(
            "rate", default_value="60.0", description="Capture/publish rate in Hz"
        ),
        DeclareLaunchArgument(
            "publish_raw",
            default_value="true",
            description="Publish per-sensor BGR camera images",
        ),
        DeclareLaunchArgument(
            "publish_depth",
            default_value="true",
            description="Publish per-sensor 32FC1 depth images in metres",
        ),
        DeclareLaunchArgument(
            "publish_pointcloud",
            default_value="false",
            description="Derive and publish per-sensor point clouds",
        ),
        DeclareLaunchArgument(
            "depth_cutoff",
            default_value="0.2",
            description="Post-process depth cutoff in millimetres; 0 disables it",
        ),
        DeclareLaunchArgument(
            "point_sample_mm",
            default_value="0.2",
            description="Point-cloud sample spacing in mm; 0 requests every pixel",
        ),
        DeclareLaunchArgument(
            "shm_connect_timeout",
            default_value="30.0",
            description="Seconds to wait for surface SHM to appear at startup",
        ),
        OpaqueFunction(function=launch_setup),
    ])
