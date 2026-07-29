# Next-session handoff

## Accepted baseline

- Production depth: frozen `suddhu/tactile_transformer` `dpt_real.p` encoder
  with one per-sensor `mixed` decoder.
- Training: sphere metric depth, then alternating sphere-depth and manual
  contact-mask updates. Backgrounds are evaluation-only.
- Runtime: four-sensor batched inference, FIR length 5, 2-of-3 persistence,
  and a default `depth_cutoff` of 0.1 mm.
- SDK depth uses millimetres; ROS depth images and point clouds use metres.

## Camera startup and reconnect

Production currently discards 10 frames after opening or reconnecting a
camera. Recorded startup tests found a strong green intensity ramp: 40
discarded frames removed measured false-depth output, while 50 frames were
needed for strict raw-image stabilization. **40 discarded frames is high for
real-time management.** Do not increase the production value without first
designing how reconnect latency, last-valid-frame publication, and readiness
are exposed.

Continue from:

- `fix/results/camera_startup_inspection/report.md`
- `fix/results/camera_startup_inspection/depth_metrics.json`

## Depth false positives

The accepted model is stable on ordinary no-touch frames, but shadows and
sparkles outside a real contact can still produce false positive depth. A
candidate trained with fresh no-contact backgrounds improved easy no-touch
frames but worsened false positives outside real contacts, especially on
D21275, so it was rejected.

The next study should use the completed prerecorded contact sessions, preserve
the current production output as the baseline, and test hard negatives taken
from outside manual masks while a real contact is present. Keep camera startup
artifacts separate from steady-state contact-shadow errors.

Continue from:

- `fix/results/tactile_transformer_training/background_regularization.md`
- `fix/results/temporal_study/final_report.md`
