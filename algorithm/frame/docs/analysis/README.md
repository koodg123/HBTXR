# Frame Analysis Import Candidates

This note captures analysis material that should inform the frame-only track.

## Source Scope

- `external_hybrid_package/docs/others/analysis/*`
- `external_hybrid_package/docs/prj/*`
- `hgtxr_software/docs/resources/stage1_frame_search_*`
- `hgtxr_software/anlaysis/xr-eye-tracking/*`

## P0 Analysis To Keep

### EV-Eye dataset inventory

Source:

- `external_hybrid_package/docs/others/analysis/20260330_170419_07_ev_eye_dataset_analysis_results.md`

Useful facts:

- Total sessions: `388`, total frames: `1,506,387`.
- Valid standard-layout sessions: `384`, frames: `1,490,548`.
- Clean sessions after timestamp-quality filtering: `366`, frames: `1,411,579`.
- Valid 384 synthetic frame counts:
  `29,803,664` at 500 fps,
  `59,606,944` at 1000 fps,
  `119,213,504` at 2000 fps.
- Clean 366 synthetic frame counts:
  `28,224,626` at 500 fps,
  `56,448,886` at 1000 fps,
  `112,897,406` at 2000 fps.

Recommended action:

- Use valid 384 for raw storage/inventory estimates.
- Use clean 366 for interpolation, event interval statistics, and time-axis
  preprocessing.
- Store this as the canonical frame dataset accounting baseline.

### Frame preprocessing and transform contract

Sources:

- `external_hybrid_package/docs/prj/*data_pipeline*`
- `external_hybrid_package/docs/prj/*mode0*`

Useful conclusion:

- The frame branch should treat spatial transform policy as part of the metric
  contract. Metrics from direct resize, ROI crop, and full-sensor letterbox are
  not interchangeable.

Recommended action:

- Add transform policy fields to any future frame experiment manifest:
  `resize_policy`, sensor size, model input size, ROI source, and label-space
  conversion.

### HGTXR frame baseline promotion logic

Sources:

- `hgtxr_software/docs/resources/current_stage1_frame_search_baseline_manifest.json`
- `hgtxr_software/docs/resources/stage1_frame_search_baseline_plan_2026_06_21.md`
- `hgtxr_software/docs/resources/stage1_frame_search_midrun_results_2026_06_21.md`

Useful conclusion:

- Promotion should use a compound rule: P10/P5 can improve, but center error
  must not regress beyond the acceptance guard.

Recommended action:

- Reuse this promotion pattern for frame search selection:
  primary metric, center guard, min-epoch condition, checkpoint existence, and
  source manifest.

## P1 Analysis To Keep

### Grounded-SAM label preparation

Source:

- `external_hybrid_package/docs/exps/20260330_170419_14_groundedsam_roi_crop_prompted_experiment_summary.md`

Useful facts:

- Bbox-only ROI recovery reduced near-full-frame eye boxes from `14/16` to
  `0/16` in the small checked set.
- Prompted crop-local pupil segmentation reduced large dark-mask failures.
- Partial all-session preview processed `80/388` sessions, `1280` sampled
  frames, with `1254` completed, `12` eye failures, `14` pupil failures, and
  `35` fail-like samples.

Recommended action:

- Keep as a label-generation reference for frame masks and eye boxes.
- Default fallback should remain off unless manually audited.

## P2 Analysis To Keep

- TimeLens/frame interpolation workflow:
  useful for dense supervision generation, but should not be a frame baseline
  until interpolation artifacts are measured.
- Paper-reference analysis:
  useful for method ideas, but reported numbers are not directly comparable
  without dataset, resolution, and metric normalization.
