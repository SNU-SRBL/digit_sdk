import os
import yaml


def load_config(serial=None, config_path=None, sensors_root=None, default_config=None):
    """
    Load sensor configuration by serial or config_path.

    Priority: config_path > serial > default_config

    Args:
        serial (str): Sensor serial number.
        config_path (str): Path to config yaml.
        sensors_root (str): Root directory containing sensors/<serial>/config.yaml.
        default_config (str): Path to fallback config.

    Returns
    -------
    dict: Loaded configuration.

    """
    if config_path is not None:
        path = config_path
    elif serial is not None and sensors_root is not None:
        path = os.path.join(sensors_root, serial, f"{serial}.yaml")
    elif default_config is not None:
        path = default_config
    else:
        raise ValueError(
            "Must provide config_path, or serial and sensors_root, "
            "or default_config."
        )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config
