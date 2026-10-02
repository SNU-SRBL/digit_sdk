# Sensor Calibration

Calibration trains one metric-depth decoder per DIGIT sensor. The shared
Tactile Transformer encoder is frozen; ball indentation provides metric depth
and manual masks provide real-contact support.

## Pixel-per-millimetre measurement

Borrowed from
[digit-depth/scripts/mm_to_pix.py](https://github.com/vocdex/digit-depth/blob/main/scripts/mm_to_pix.py).

Press the sensor with a known length, such as a caliper. Click the two points
for five measurements. Copy the averaged output and update `ppmm` in the
corresponding `sensors/<serial>/<serial>.yaml` before collecting calibration
data.

**DO NOT PRESS THE SENSOR WITH THE TIP OF THE CALIPER. It will tear the gel.
Press the caliper from the side.**

<img src="../assets/caliper.jpg" width="150"/>

```bash
python3 -m calibration.mm_to_ppmm \
  --serial D21242 \
  --distance_mm DISTANCE_MM \
  --frames 5
```

## Data layout

Collection and annotation operate only inside `calibration/inbox/`:

```text
sensors/<serial>/calibration/inbox/
├── background/
│   ├── metadata.json
│   ├── reference.png
│   └── frames/<session_id>/*.png
├── ball/
│   ├── metadata.jsonl
│   ├── images/<session_id>/*.png
│   └── labels/<session_id>/*.npz
└── manual_mask/
    ├── metadata.jsonl
    ├── touches/<session_id>/*.png
    └── masks/<session_id>/*.png
```

Finalization produces the canonical dataset:

```text
sensors/<serial>/calibration/
├── dataset.yaml
├── manifest.jsonl
├── background/
├── ball/
│   ├── images/<session_id>/*.png
│   └── labels/<session_id>/*.npz
├── manual_mask/
│   ├── images/<session_id>/*.png
│   └── masks/<session_id>/*.png
└── splits/
    ├── background.json
    ├── ball.json
    └── manual_mask.json
```

Inbox records never contain train/validation/test assignments. Finalization is
the only operation that creates frozen splits. Captured PNG files are never
modified.

## Collection

All calibration capture runs at 30 Hz. Run commands from the repository root.

### 1. Shared ball background

Remove all contact, run the command, then press `b`:

```bash
python3 -m calibration.collect_background --serial D21242
```

The collector saves 60 individual no-contact frames one second apart and a
checked, averaged `reference.png`. Finalization uses exactly those 60 fresh
frames: 40 train, 10 validation, and 10 test. The average is the single
reference used by the ball and manual-contact tools; it is not a training
sample. The collector refuses to overwrite an existing reference.

### 2. Ball indentation

Collect the four ball diameters:

```bash
python3 -m calibration.collect_ball --serial D21242 --ball-diameter-mm 3 --display-difference
python3 -m calibration.collect_ball --serial D21242 --ball-diameter-mm 5 --display-difference
python3 -m calibration.collect_ball --serial D21242 --ball-diameter-mm 7 --display-difference
python3 -m calibration.collect_ball --serial D21242 --ball-diameter-mm 9 --display-difference
```

Controls:

- `w`: save one independent indentation;
- `d`: toggle amplified background difference;
- `q`: quit.

The guide covers a 5×5 zigzag grid and five indentation levels per cell. The
level is only capture guidance; metric ground truth comes from the annotated
circle, known ball diameter, and sensor `ppmm`.

Use a targeted replacement only for an explicitly rejected sample:

```bash
python3 -m calibration.collect_ball \
  --serial D21242 \
  --ball-diameter-mm 3 \
  --target-row 0 \
  --target-column 4 \
  --target-depth 5 \
  --replaces SAMPLE_ID
```

### 3. Manual contacts

```bash
python3 -m calibration.collect_manual_contacts --serial D21242
```

Collect 100 independent contacts: 60 training, 20 validation, and 20 test.

Controls:

- `t`: save one contact;
- `q`: quit.

Release and remake contact before every `t`. The preview and console show both
cumulative inbox counts and counts added during the current session. Camera
reconnection preserves the current session and previously saved data.

## Annotation

### Ball circles

```bash
python3 -m calibration.annotate_ball.server --serial D21242
```

Move the circle by dragging or with arrow keys. Change its radius with the
slider, mouse wheel, or `+`/`-`. The interface rejects a circle at or beyond the
ball hemisphere. `Reject sample` excludes an acquisition artifact without
deleting it; `Undo rejection` restores it.

### Manual contact masks

```bash
python3 -m calibration.annotate_contact.server --serial D21242
```

Masks are binary PNG files: `0` is non-contact and `255` is contact.

- `B`, `P`, `E`: brush, polygon, eraser;
- `Ctrl+Z`, `Ctrl+Y`: undo and redo;
- `Enter`: finish polygon;
- `Escape`: cancel polygon;
- left/right arrows: previous/next sample.

Every save records the annotator, increments `annotation_revision`, and updates
the mask geometry.

## Finalize calibration

After all accepted ball samples and manual contacts are annotated:

```bash
python3 -m calibration.finalize_dataset --serial D21242
python3 -m calibration.promote_dataset --serial D21242
```

`finalize_dataset` creates and validates `calibration_next`; it does not touch
the active dataset. `promote_dataset` validates that candidate again, swaps it
into `calibration/`, and preserves the immutable `inbox/`. A failed promotion
rolls back automatically.

Ball split policy per complete five-depth grid cell:

- three depths: training;
- one rotating depth: validation;
- one rotating depth: test.

Manual contacts are assigned deterministically by `split_group`: groups are
ranked by the SHA-256 hash of their `split_group`, the lowest ranks become
test, the next become validation, and the rest train. The default split is
60% train, 20% validation, and 20% test. Counts are derived from the group
total with any remainder going to training, and every partition gets at least
one group. Override the fractions when necessary:

```bash
python3 -m calibration.finalize_dataset \
  --serial D21242 \
  --manual-validation-fraction 0.2 \
  --manual-test-fraction 0.2
```

The separate `validate_dataset` command is only needed when investigating a
failed finalization or promotion.

## Dataset contract

`dataset.yaml` fixes the serial, decoded image shape, BGR `uint8` colour
format, and `ppmm`. `manifest.jsonl` records relative paths, UTC timestamps,
physical `split_group`, and calibration geometry.

Ball labels contain:

- `ball_diameter_mm`;
- `center_px` in `[x, y]` order;
- `radius_px`;
- `ppmm`;
- `indentation_depth_mm`.

The label NPZ contains the same fields. Valid geometry requires
`0 < radius_px / ppmm < ball_diameter_mm / 2`.

Manual-mask records contain a non-empty annotator and a positive annotation
revision. Masks must match the source image dimensions and contain only `0`
and `255`.

## Decoder training

Train each sensor independently after its calibration is active.

The sole method is `mixed_background`: stage 1 uses ball metric depth; stage 2
alternates ball depth, manual contact masks, and full-frame no-contact
backgrounds. Background targets are zero depth and no contact.

The default schedule uses 80 stage-1 epochs and 20 stage-2 epochs. It keeps
the frozen encoder and fine-tunes the decoder with the standard three seeds.

Train one sensor:

```bash
python3 -m calibration.train_decoder --serial D21242
```

This trains and selects one validation candidate, evaluates it once on the
held-out test split, and prints both result sets. A rerun reuses the matching
immutable test result. A passing decoder is written to `model/depth/`, and its
maximum depth is recorded in the sensor YAML.

Then check settled live no-contact output and light, medium, and deep ball
contact.

## Depth units

Training targets and raw model output use `float32` millimetres. The runtime
`depth_cutoff` is also configured in millimetres and is applied only after
inference. ROS depth images and point clouds use metres.
