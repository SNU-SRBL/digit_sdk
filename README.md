# DIGIT SDK

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE.txt)

`digit_sdk` is a multi-sensor tactile perception and ROS 2 stack for
[DIGIT](https://digit.ml/) sensors. It provides metric depth estimation,
optional force estimation, isolated camera capture, shared batched GPU
inference, camera-corruption recovery, and independent raw-image, depth, and
point-cloud publication.

Depth uses the Tactile Transformer released with
[NeuralFeels](https://github.com/facebookresearch/neuralfeels). One frozen
encoder is shared by the active sensor batch, while each sensor uses its own
fine-tuned decoder. The custom calibration data combines sphere-indentation
geometry inspired by [GS-SDK](https://github.com/joehjhuang/gs_sdk) with
manually annotated real-world contacts. The calibration approach is also
informed by [digit-depth](https://github.com/vocdex/digit-depth).

Force estimation is a separate, optional
[Sparsh](https://github.com/facebookresearch/sparsh) pipeline.

**Author:** [Byung-Hyun Song](https://github.com/bhsong1011)
(bh.song@snu.ac.kr)

## Method and attribution

| Component | Source or influence | This project |
|---|---|---|
| Depth backbone | NeuralFeels Tactile Transformer | Shared frozen encoder |
| Sensor adaptation | Custom calibration | One fine-tuned decoder per sensor |
| Geometric supervision | GS-SDK-inspired calibration | 3, 5, 7, and 9 mm sphere contacts |
| Real-contact supervision | Custom data | Binary manual contact masks |
| Temporal refinement | NeuralFeels finite weighted blend | Fixed five-frame FIR per sensor |
| Contact suppression | Runtime postprocessing | `0.1 mm` cutoff by default |
| Force estimation | Sparsh | Optional, separate from depth |
| Additional depth influence | digit-depth | Per-sensor DIGIT calibration |

GS-SDK and digit-depth are methodological influences, not production runtime
backends.

## Production depth pipeline

```text
DIGIT RGB
  → Tactile Transformer shared encoder
  → sensor-specific decoder
  → raw metric depth in mm
  → five-frame NeuralFeels FIR
  → depth_cutoff (default 0.1 mm)
  → ROS depth in metres / point cloud in metres
```

The FIR operates on unthresholded model output and maintains independent
history for every sensor. Its history resets after a non-monotonic timestamp
or a gap greater than `100 ms`. The cutoff is applied afterward and does not
affect model inference, FIR history, or training. Set `depth_cutoff:=0.0` to
disable thresholding.

Production decoders use the `mixed_background` objective, combining ball-depth
supervision with manual real-contact support.

## Features

- Multi-DIGIT automatic serial discovery and registration.
- One isolated camera process per sensor.
- One shared batched encoder and sensor-specific decoders.
- Generation-checked camera and surface shared memory.
- Latest-frame scheduling without an unbounded queue.
- Row-discontinuity detection, stream recovery, and frame-rate watchdog.
- Independent raw, depth, and point-cloud outputs.
- Metric `32FC1` ROS depth and metric point clouds.
- Optional Sparsh force estimation.

## Requirements

- Ubuntu 22.04
- ROS 2 Humble
- Python 3.10 or newer
- DIGIT tactile sensors at QVGA (`320×240`); the default capture rate is 30 Hz
- CUDA-capable GPU recommended for production depth and force estimation

CPU depth inference is supported for development, but the multi-sensor rate
target is evaluated on CUDA.

## Installation

```bash
git clone git@github.com:SNU-SRBL/digit_sdk.git
cd digit_sdk

python3 -m pip install -e .
colcon build --packages-select digit_sdk --symlink-install
source install/setup.bash
```

### Optional force dependencies

Force estimation is optional and is not required for tactile depth. Its
checkpoints alone are insufficient: the runtime also needs a compatible Sparsh
source checkout at `sparsh-main/` in the `digit_sdk` root (preferred) or the
launch working directory. Install the [upstream Sparsh dependencies](https://github.com/facebookresearch/sparsh#-installation-and-setup)
in the Python environment that runs force estimation, then fetch the force
checkpoints:

```bash
git clone https://github.com/facebookresearch/sparsh.git sparsh-main
python3 scripts/download_models.py
python3 -m pip install -e ".[gpu]"
```

The download is approximately `1.7 GB` and stores Sparsh models under
`models/`. Package installation places that directory at
`share/digit_sdk/models`; `models_root:=/absolute/path/to/models` overrides it
when checkpoints are stored elsewhere.

### Tactile depth base weights

The Tactile Transformer base checkpoint is deliberately user-acquired: it is
not included in this repository, and root setup does not clone a model
repository or use Git LFS. On the first tactile training, evaluation, or
PyTorch depth-runtime use, `huggingface_hub` resolves the pinned
`suddhu/tactile_transformer` `dpt_real.p` artifact in its cache. Provide
access to that artifact before using depth.

The current interfaces do not define a repository-local manual placement path
for the base checkpoint.

## Sensor registration

Each sensor requires:

```text
sensors/<serial>/<serial>.yaml
sensors/<serial>/model/depth/decoder.pth
```

The YAML file defines the camera stream and sensor geometry. The calibration
workflow produces the decoder and records `maximum_depth_mm` in that sensor's
YAML. The decoder is a local, non-versioned production artifact. Calibrate and
train one decoder for each physical sensor.

## ROS 2 launch

Launch all registered sensors:

```bash
source install/setup.bash
ros2 launch digit_sdk multi_sensor_tactile_streamer.launch.py
```

Launch selected sensors:

```bash
ros2 launch digit_sdk multi_sensor_tactile_streamer.launch.py \
  serials:=D21119,D21242,D21273,D21275 \
  publish_raw:=true \
  publish_depth:=true \
  publish_pointcloud:=false \
  capture_fps:=30 \
  depth_cutoff:=0.1 \
  model_device:=cuda \
  rate:=30.0
```

### Published topics

| Topic | Type | Description |
|---|---|---|
| `/tactile/{serial}/raw` | `sensor_msgs/Image` (`bgr8`) | Raw camera image |
| `/tactile/{serial}/depth` | `sensor_msgs/Image` (`32FC1`) | FIR-filtered, cutoff-processed depth in metres |
| `/tactile/{serial}/pointcloud` | `sensor_msgs/PointCloud2` | XYZ in metres from the latest depth generation |

### Launch parameters

| Parameter | Default | Description |
|---|---:|---|
| `serials` | auto | Comma-separated serials; empty discovers registered sensors |
| `rate` | `30.0` | Requested inference and publication rate in Hz |
| `capture_fps` | `30.0` | Camera capture rate in Hz |
| `model_device` | `cuda` | `cuda` or `cpu` |
| `publish_raw` | `true` | Launch raw-image publishers |
| `publish_depth` | `true` | Publish `32FC1` metric depth |
| `publish_pointcloud` | `false` | Derive and publish point clouds |
| `depth_cutoff` | `0.1` | Cutoff in millimetres; `0` disables it |
| `point_sample_mm` | `0.2` | Point-cloud spacing; `0` retains every pixel |
| `sensors_root` | auto | Sensor configuration and model root |

Raw, depth, and point cloud are independent. Disabled outputs do not launch
their publisher or derived compute path.

## Runtime architecture

```text
camera_shm ×N ── camera SHM ──┬── raw_publisher ×N ── bgr8
                              └── pipeline_node ×1
                                     │ shared encoder batch
                                     │ per-sensor decoders
                                     │ per-sensor FIR
                                     │ depth cutoff
                                     ▼
                               surface SHM ×N (float32 mm)
                                     ├── depth publisher ── 32FC1 m
                                     └── point-cloud publisher ── XYZ m
```

Camera capture, raw DDS publication, depth inference, and surface publication
run in separate processes. If depth and point cloud are both enabled, separate
surface-publisher processes prevent point-cloud serialization from blocking
depth publication.

Camera and surface SHM use an odd/even generation protocol. Writers mark a
generation odd while updating and even after commit; readers accept only a
stable matching generation. Each surface result retains its source sequence
and capture timestamp, so rates cannot be inflated by republishing the same
depth.

## Camera reliability

Some QVGA streams have shown row discontinuities without a USB/V4L2 error.
The cause has not been identified. `Camera` detects the discontinuity,
performs a stream recovery, warms up, and resumes with the next committed
source frame. A frame-rate watchdog triggers the same recovery after sustained
abnormal capture gaps.

FIR state resets automatically when recovery creates a timestamp discontinuity
greater than `100 ms`.

## Calibration

The per-sensor workflow is:

1. Capture one averaged shared background.
2. Collect 3, 5, 7, and 9 mm sphere contacts.
3. Annotate sphere position and indentation depth.
4. Collect representative real-world contacts.
5. Annotate binary contact masks.
6. Finalize deterministic train, validation, and test splits.
7. Fine-tune decoders from the frozen Tactile Transformer.
8. Train and install the selected `mixed_background` decoder.

Commands and data contracts are documented in the
[calibration README](calibration/README.md).

Calibration is sensor-specific. Recalibrate after changes to the gel, optical
surface, illumination, camera geometry, or sensor hardware.

## Python camera API

```python
from digit_sdk import Camera

camera = Camera(serial="D21273", sensors_root="sensors")
camera.connect()

try:
    while True:
        frame = camera.get_image()
        if frame is not None:
            process(frame)
finally:
    camera.disconnect()
```

`Camera.get_image()` includes corruption detection, recovery, and watchdog
handling. The production depth path is provided by the ROS batch pipeline.

## Repository layout

| Path | Purpose |
|---|---|
| `digit_sdk/` | Camera, depth model, temporal filter, geometry, and SHM protocols |
| `ros2/` | Capture, batch inference, publishers, and launch files |
| `calibration/` | Data collection, annotation, finalization, and decoder training |
| `sensors/` | Per-sensor configuration, calibration datasets, and decoders |
| `test/` | Production and regression tests |

## License

The repository is distributed under the
[GNU General Public License v3](LICENSE.txt). Applicable attribution and
license notices for code derived from GS-SDK must be preserved.

Sparsh model assets are licensed under
[CC BY-NC 4.0](https://github.com/facebookresearch/sparsh/blob/main/LICENSE). NeuralFeels, Tactile Transformer, and
other third-party model assets remain subject to their respective upstream
licenses.

## References

1. Suresh et al., “NeuralFeels with neural fields: Visuotactile perception for in-hand manipulation,”
   *Science Robotics*, 2024.
   [Paper](https://www.science.org/doi/10.1126/scirobotics.adl0628) ·
   [Code](https://github.com/facebookresearch/neuralfeels) ·
   [Tactile Transformer](https://huggingface.co/suddhu/tactile_transformer)
2. Huang et al., [GS-SDK](https://github.com/joehjhuang/gs_sdk).
3. vocdex, [digit-depth](https://github.com/vocdex/digit-depth).
4. Akhter et al., “Sparsh: Self-supervised touch representations for vision-based
   tactile sensing,” CoRL 2024.
   [Code](https://github.com/facebookresearch/sparsh)
5. Lambeta et al., “DIGIT: A Novel Design for a Low-Cost Compact
   High-Resolution Tactile Sensor with Application to In-Hand Manipulation,”
   *IEEE RA-L*, 2020. [Project](https://digit.ml/)
