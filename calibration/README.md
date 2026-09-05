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

All calibration capture runs at 30 Hz. Run commands from the repository root:

```bash
cd ~/ros2/digit_sdk
```

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
for DIAMETER in 3 5 7 9; do
  python3 -m calibration.collect_ball \
    --serial D21242 \
    --ball-diameter-mm "$DIAMETER" \
    --display-difference
done
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

## Finalization

After all accepted ball samples and manual contacts are annotated:

```bash
python3 -m calibration.finalize_dataset --serial D21242
```

The default review output is `sensors/<serial>/calibration_next`. The command
copies checked data, writes the manifest and frozen splits, and runs strict
schema validation. It never overwrites the active calibration root.

Ball split policy per complete five-depth grid cell:

- three depths: training;
- one rotating depth: validation;
- one rotating depth: test.

Manual contacts are assigned deterministically by `split_group`. Defaults are
20 validation contacts, 20 test contacts, and all remaining contacts for
training. Change them when necessary:

```bash
python3 -m calibration.finalize_dataset \
  --serial D21242 \
  --manual-validation-count 20 \
  --manual-test-count 20
```

Validate any canonical dataset with:

```bash
python3 -m calibration.validate_dataset \
  sensors/D21242/calibration_next
```

Promote the validated candidate while preserving its inbox:

```bash
python3 -m calibration.promote_dataset --serial D21242
```

The active dataset is then `sensors/D21242/calibration`. Its inbox remains the
immutable source for future review and relabeling.

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

The frozen base is deliberately user-acquired. It is not version-controlled,
and setup does not clone a model repository or use Git LFS. Training and
evaluation resolve `dpt_real.p` from `suddhu/tactile_transformer` at pinned
revision `b05cfe1df2c90d3d91f8378633173b26de5a2d2c` through
`huggingface_hub`'s cache, then verify its size and SHA-256. Provide access to
that artifact before training or evaluation; the current CLI has no
repository-local manual placement path.

Each physical DIGIT is fine-tuned separately. Sensor checkpoints contain only
decoder/head weights and remain local, not version-controlled.

Finish collection, annotation, and finalization for all four sensors before
starting decoder training.

The sole training method is `mixed`: stage 1 learns metric depth from sphere
contacts, then stage 2 alternates sphere-depth and manual contact-mask updates.
Background records are held out for evaluation; they are not training
negatives.

For one sensor, a single invocation runs seeds 17, 29, and 43 in separate
processes and selects the best eligible validation seed:

```bash
python3 -m calibration.train_decoder --serial D21242
```

Completed runs matching the active dataset and frozen training protocol are
reused. Partial or stale published runs are rejected. Seed selection is part
of training; it is not method selection.

The selected artifacts remain under the `mixed` directory:

```text
sensors/<serial>/model/tactile_transformer/mixed/
├── seed_17/
├── seed_29/
├── seed_43/
├── decoder.pth
└── selection.json
```

Test partitions remain unopened until the global method and per-sensor seed
choices are frozen. Production promotion remains a separate acceptance step.

Open each frozen test split once:

```bash
python3 -m calibration.evaluate_decoder --serial D21242
```

The evaluator refuses to overwrite an existing `test.json`. After its test
gates pass, promote the exact tested decoder:

```bash
python3 -m calibration.promote_decoder --serial D21242
```

Promotion is fixed to `mixed`, writes the depth limit to the sensor YAML, and
refuses to replace an existing
production model. Runtime artifacts are written locally to
`sensors/D21242/model/depth/`; they are not version-controlled.

## Depth units

Training targets and raw model output use `float32` millimetres. The runtime
`depth_cutoff` is also configured in millimetres and is applied only after
inference. ROS depth images and point clouds use metres.
