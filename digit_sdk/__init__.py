"""
DIGIT camera capture and calibrated metric-depth runtime.

Public exports are lazy so camera and ROS publisher processes do not import the
Torch/TIMM model stack owned exclusively by the pipeline process.
"""

from importlib import import_module


__version__ = "2.0.0"
__all__ = [
    "Camera",
    "DepthEstimator",
    "DepthFrame",
    "DepthInput",
    "DepthResult",
    "ProcessingEngine",
    "make_depth_result",
    "validate_depth_raw_mm",
    "validate_depth_result",
]

_EXPORTS = {
    "Camera": (".camera", "Camera"),
    "DepthEstimator": (".depth", "DepthEstimator"),
    "DepthInput": (".depth", "DepthInput"),
    "DepthFrame": (".processing_engine", "DepthFrame"),
    "ProcessingEngine": (".processing_engine", "ProcessingEngine"),
    "DepthResult": (".depth_contract", "DepthResult"),
    "make_depth_result": (".depth_contract", "make_depth_result"),
    "validate_depth_raw_mm": (".depth_contract", "validate_depth_raw_mm"),
    "validate_depth_result": (".depth_contract", "validate_depth_result"),
}


def __getattr__(name):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
