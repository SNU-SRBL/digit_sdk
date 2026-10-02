# Domain context

## Runtime terms

- **Depth frame:** one metric depth estimate for one captured image from one
  sensor.
- **Per-sensor rate:** processed depth frames per second for each sensor, not
  the sum across sensors.
- **Production profile:** QVGA capture and depth publication default to 30 Hz
  per sensor. Actual throughput depends on the connected cameras and host.
- **ROS process topology:** one `camera_shm` per sensor, an optional
  `raw_publisher` per sensor, one central `pipeline_node`, and one
  `surface_publisher` instance per enabled surface output. When depth and point
  cloud are both enabled, separate instances consume the same latest-depth SHM
  so point-cloud serialization cannot block depth.
- **Force publisher:** optional per-sensor process that reads camera SHM,
  runs `ForceEstimator`, and publishes DDS force outputs. It starts only when
  both launch `publish_force:=true` and that sensor's `force.enable_force: true`;
  otherwise no force model is loaded into RAM or VRAM.
- **Force-coloured point cloud:** optional `rgb` field on the standard
  point-cloud topic. It resizes the latest force RGB visualization to the
  depth grid; it is for visualization and is not frame-synchronized or a
  calibrated force-to-3D mapping.
- **GPU inference scheduling:** ownership and scheduling of Transformer model
  execution. It is independent of the retained per-sensor ROS process topology.
- **Sensor slot:** a stable batch position bound explicitly to one sensor
  serial and its decoder. Batch position alone is not calibration identity.
- **Shared encoder:** one frozen ViT encoder used for a batch containing the
  four sensor slots.
- **Sensor decoder:** reassembly, fusion, and metric-depth head calibrated only
  for one sensor. Encoder fine-tuning is outside the production design.
- **Batch deadline:** the configured scheduling boundary (30 Hz by default) at
  which the coordinator consumes each sensor's latest unseen frame. A missing
  sensor never stalls the others.
- **ProcessingEngine:** public multi-sensor runtime. It owns slot scheduling,
  batched depth inference, post-processing, and typed results; it no longer
  creates one processor/model per sensor.
- **DepthEstimator:** the only production depth model component. It contains
  one shared frozen encoder and serial-bound decoders and exposes a batch-first
  API. Architecture names belong in model metadata, not runtime class names.
- **Latest depth:** the newest completed depth result identified by its source
  sequence. A derived consumer may skip older sequences but must never process
  or publish the same sequence twice.
- **Point-cloud path:** optional per-sensor work performed by
  `surface_publisher` from latest unseen depth. It is outside the inference
  critical path and cannot increase the depth batch latency.
- **Frame identity:** `(sensor_serial, sequence)`. Timestamps are never used as
  unique identifiers because different sensors may legitimately share them.
- **Capture monotonic time:** monotonic nanoseconds used for scheduling and
  latency measurement, never as a ROS epoch stamp.
- **Capture system time:** epoch nanoseconds propagated to ROS raw, depth, and
  point-cloud headers using integer `divmod` conversion.
- **Raw publisher:** optional per-sensor process that publishes only camera
  images from camera SHM.
- **Surface publisher:** reusable per-sensor executable for either depth or
  point-cloud publication. Both modes consume latest depth from surface SHM;
  launch isolates them into separate processes when both are enabled.
- **SHM generation:** aligned `uint64` synchronization counter. Zero is empty,
  odd means a writer is active, and an unchanged even value before/after an
  owned copy proves a stable read.
- **Completion rate:** surface generation commits per second divided by two.
  Source sequence span is not a completion count because latest-frame
  scheduling may skip camera sequences.
- **SHM contention policy:** retry a generation-checked read twice with a
  `100 us` pause, then skip it. The system uses one payload per channel; a
  multi-slot ring is deferred unless production measurements justify it.

## Depth terms

- **Raw depth:** unthresholded, finite, non-negative `float32` depth in
  millimetres.
- **Temporally filtered depth:** raw depth processed by the causal five-frame
  NeuralFeels FIR independently for each sensor.
- **Mask:** a binary map derived from temporally filtered depth and
  `depth_cutoff`. All pixels are valid when the cutoff is `0.0`; otherwise a
  pixel must meet the cutoff in at least two of the latest three frames. It is
  not predicted separately or published as a ROS topic.
- **Filtered depth:** the current temporally filtered depth where the
  persistence mask is true and zero elsewhere. It is a post-processing
  product, never a training target or stored raw result.
- **`depth_cutoff`:** launcher/runtime parameter in millimetres. `0.0`
  disables suppression; `0.1` applies the persistence mask at `0.1 mm`. It
  never changes model inference or saved raw evaluation arrays.
- **ROS depth:** the cutoff-processed depth image published as `32FC1` in
  metres. Conversion from the SDK's millimetres occurs only at the ROS
  publication boundary.
- **Depth visualization:** presentation-specific scaling or colour mapping. It
  is not a core SDK result or ROS transport topic.
