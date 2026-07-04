# Event Analysis Import Candidates

This note captures analysis material that should inform event-window,
accumulation, and event-only experiment design.

## Source Scope

- `external_hybrid_package/docs/prj/20260330_170419_02_event_binning_and_accumulation_flow.md`
- `external_hybrid_package/docs/prj/20260410_140000_v2e_event_generation_experiment_workflow.md`
- `hgtxr_software/anlaysis/xr-eye-tracking/DETAILED_CODEBASE_ANALYSIS.md`
- `hgtxr_software/anlaysis/xr-eye-tracking/DETAILED_PAPER_ANALYSIS.md`

## P0 Analysis To Keep

### Event builder contract

Source:

- `external_hybrid_package/docs/prj/20260330_170419_02_event_binning_and_accumulation_flow.md`

Useful facts:

- Default event policy: `fixed_count`.
- Default event count target: `5000`.
- Default time bin: `5000 us`.
- Default accumulation: `fast_causal_linear`.
- Default polarity split: true.
- The event tensor is accumulated first at sensor resolution as
  `[2, H_sensor, W_sensor]`, then transformed to model input
  `[2, 256, 256]`.
- Active policies:
  `fixed_count`, `time_bin`, `interval_all`.
- Active accumulation modes:
  `plain`, `causal_linear`, `causal_linear_ori`, `fast_causal_linear`.

Recommended action:

- Preserve the two-phase contract:
  manifest fixes the event-window choice, dataset/runtime performs the actual
  event slice and accumulation.
- Keep event-window settings in experiment manifests so runs are reproducible.

### Event tensor ABI

Useful facts:

- Raw event arrays:
  `t`, `x`, `y`, `p`.
- Selected window:
  `t_sel`, `x_sel`, `y_sel`, `p_sel`.
- Accumulated sensor voxel:
  `[2, H_sensor, W_sensor]`.
- Transformed event input:
  `[2, 256, 256]`.
- Batch ABI:
  `event: [B, 2, 256, 256]`,
  `frame: [B, 1, 256, 256]`,
  `prev_state: [B, 6]`.

Recommended action:

- Use this ABI as the event track's documentation baseline.
- Add tests around channel order, polarity semantics, and transform order before
  adding new event accumulation modes.

## P1 Analysis To Keep

### Reference implementation families

Source:

- `hgtxr_software/anlaysis/xr-eye-tracking/DETAILED_CODEBASE_ANALYSIS.md`

Useful families:

- Geometry-first online trackers:
  E-Track, EX-Gaze, EV-Eye, Event-Based Near-Eye Gaze, EyeLoRiN.
- Dense event-frame neural trackers:
  FACET, 3ET/CB-ConvLSTM, AIS challenge models, Mamba/EventMamba.
- Segmentation teachers:
  RITnet, EllSeg, EV-Eye U-Net, Swift-Eye detector.
- Hardware-facing sparse systems:
  Retina, SEE/ESDA, lightweight spatiotemporal designs.

Recommended action:

- For event-only code, start with event-count slicing and ROI/failure
  diagnostics.
- Defer sparse/SNN deployment until the dense software baseline is credible.

### Paper-analysis implications

Source:

- `hgtxr_software/anlaysis/xr-eye-tracking/DETAILED_PAPER_ANALYSIS.md`

Useful ideas:

- Event quantity slicing from Retina.
- ROI ellipse tracking and failure recovery from E-Track.
- Heatmap/KL temporal decoding from TDTracker.
- Event segmentation and ellipse-state supervision from FACET/E-Track style
  systems.

Recommended action:

- Treat paper-reported numbers as qualitative guidance only. Dataset, split,
  resolution, label rate, and metric definitions are not normalized.

## P2 Analysis To Keep

- v2e workflow:
  useful for synthetic event generation and ablation, not yet an event baseline.
- Hardware-facing sparse event paths:
  useful for later quantization/hardware co-design after software accuracy
  stabilizes.
