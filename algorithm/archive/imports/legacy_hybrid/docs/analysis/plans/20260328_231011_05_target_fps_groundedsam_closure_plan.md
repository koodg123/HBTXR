# Target-FPS + Grounded-SAM Closure Plan

Last updated: 2026-03-28 KST

## Summary

This plan closes the remaining `target_fps + Grounded-SAM` work based on the **current checked-out code**, not on older plan snapshots alone.

Main conclusion:

- the `target_fps` builder / canonical bridge / manifest bridge / dataset runtime path is already materially implemented
- the main remaining gap is **annotation-source closure**, not the target-FPS storage bridge itself
- the most important architectural decision is whether live Grounded-SAM inference should be embedded inside `build_target_fps_dataset()` or kept as an explicit upstream producer stage

Recommended direction:

- keep `target_fps_build.py` as a **consumer/fuser** of precomputed annotation stores
- add explicit producer steps that materialize the Grounded-SAM stores the builder already knows how to consume

## 1. Confirmed Current Implementation State

The following are already present in source and tests:

- target timestamp grid construction
- target-FPS frame rendering
- rebinned existing-event packet storage
- `npz` and `h5` session-store writing code
- frame-level label rows with status / reason fields
- target-FPS canonical bridge
- target-FPS manifest bridge
- dataset loading from session stores
- synthetic `mode1` / `mode2` regression coverage for the `npz` path

Primary code surface:

- `src/hbtxr/preprocess/target_fps_build.py`
- `src/hbtxr/preprocess/target_fps_canonical.py`
- `src/hbtxr/preprocess/build_manifests.py`
- `src/hbtxr/data/dataset.py`

Primary regression surface:

- `tests/test_target_fps_build_v3.py`
- `tests/test_target_fps_canonical_v3.py`
- `tests/test_target_fps_manifest_v3.py`
- `tests/test_target_fps_dataset_v3.py`

## 2. Code-Level Remaining TODO Signals

The remaining open work is visible in the explicit row statuses and reasons produced by `target_fps_build.py`.

### A. Eye ROI source missing

Observed status/reason:

- `label_status = pending_annotation_sources`
- `annotation_reason = groundedsam_eye_roi_missing`

Condition:

- manual CSV pupil state exists
- but no Grounded-SAM eye ROI source is available

Interpretation:

- pupil state alone is not allowed to backfill eye ROI
- this is an intentional contract choice, not a bug
- closure therefore requires a real eye-ROI producer, not a quick row-level patch

### B. No-CSV eye pipeline still pending

Observed status/reason:

- `label_status = pending_annotation_sources`
- `annotation_reason = groundedsam_no_csv_eye_mask_pipeline_pending`

Condition:

- no manual CSV
- no usable Grounded-SAM eye source

Interpretation:

- this is the clearest remaining Grounded-SAM production gap

### C. No-CSV pupil pipeline still pending

Observed status/reason:

- `label_status = annotated_eye_only`
- `annotation_reason = groundedsam_no_csv_pupil_pipeline_pending`

Condition:

- eye ROI exists from `groundedsam_eye_bbox_only` or fallback full store
- but no completed ROI-local pupil source exists

Interpretation:

- this is the exact point where the all-user ROI-crop `v2` work needs to be connected back into the production label path

### D. H5 path is implemented but not fully closed by tests

Confirmed in code:

- `_write_h5_session_store()` exists in `target_fps_build.py`
- H5 reading exists in `data/dataset.py`
- `requirements.txt` already includes `h5py`

Gap:

- current tests mostly exercise `npz`
- H5 bridge/runtime coverage is still thin or absent

## 3. Key Architecture Decision

### Recommendation

Do **not** fold live Grounded-SAM runtime directly into `build_target_fps_dataset()` as the default path.

Instead:

1. keep `build_target_fps_dataset()` responsible for:
   - target grid generation
   - frame/event materialization
   - source fusion
   - explicit pending/failure bookkeeping
2. add or extract reusable Grounded-SAM producer modules that write into:
   - `annotation_root/sessions/.../frame_annotations.jsonl`
   - `annotation_root/eye_region_bbox_only/sessions/.../eye_region_bboxes.jsonl`
3. let `target_fps_build.py` consume those stores without knowing how they were produced

### Why this is the better boundary

- it preserves reproducibility
- it keeps expensive vision inference separable from dataset materialization
- it matches the current code structure
- it lets the same annotation stores be reused across repeated target-FPS builds

## 4. Closure Work Packages

## P0. Documentation Reality Sync

Goal:

- update active tracking so it reflects the current code state

Tasks:

- mark the implemented `target_fps` bridge items as done in progress docs
- move remaining open items under explicit annotation-production and validation buckets
- keep the all-48 ROI-crop handoff linked as the production pupil-path reference

Files:

- `docs/chat/20260326_011438_01_v3_update_history.md`
- `docs/chat/20260326_011438_02_v3_progress_checklist.md`
- `docs/chat/20260326_011438_03_v3_conversation_and_decision_log.md`

Closure criterion:

- docs no longer describe the entire target-FPS stack as “still pending”

## P1. Eye ROI Producer Integration

Goal:

- eliminate `groundedsam_eye_roi_missing`
- eliminate `groundedsam_no_csv_eye_mask_pipeline_pending` when Grounded-SAM runtime is available

Current reusable surface:

- `src/hbtxr/preprocess/groundedsam_eye_region_bbox.py`

Required work:

- extract a reusable, non-preview producer API from the bbox-only preview path
- ensure it writes canonical bbox-only stores under:
  - `annotation_root/eye_region_bbox_only/sessions/.../eye_region_bboxes.jsonl`
- add a CLI entrypoint for batch materialization, not only preview generation
- make the output schema match what `target_fps_build.py` already consumes

Recommended new surface:

- `src/hbtxr/preprocess/groundedsam_eye_bbox_store.py`
- `scripts/build_groundedsam_eye_bbox_store.py`

Closure criterion:

- with valid Grounded-SAM runtime and checkpoints, target-FPS builds no longer emit eye-missing rows for sessions that are otherwise runnable

## P2. ROI-Local Pupil Producer Integration

Goal:

- eliminate `groundedsam_no_csv_pupil_pipeline_pending`
- connect the validated ROI-crop `v2` operating point to the production target-FPS label path

Current source material:

- `scripts/run_roi_crop_prompted_preview_batch.py`
- `docs/exps/20260328_110310_07_groundedsam_roi_crop_prompted_experiment_summary.md`
- `docs/exps/20260328_110310_08_all48_v2_baseline_failure_review.md`
- `docs/exps/20260328_111235_09_all48_v2_handoff.md`

Required work:

- extract the crop-local pupil inference logic from the preview/batch script into a reusable module
- make it consume:
  - frame image
  - eye ROI bbox
  - Grounded-SAM runtime
  - `v2` option config by default
- make it write session-local production outputs in a stable store format that `target_fps_build.py` can treat as `groundedsam_full`
- preserve geometry flags and blink metadata propagation

Recommended new surface:

- `src/hbtxr/preprocess/groundedsam_roi_crop_pupil.py`
- `scripts/build_groundedsam_roi_crop_pupil_store.py`

Important policy:

- keep `v2` as the default strict production path
- do not enable fallback by default
- keep geometry warnings explicit rather than silently accepting large masks

Closure criterion:

- no-CSV sessions with valid eye ROI and valid runtime/checkpoints can progress from `annotated_eye_only` to `annotated_complete`

## P3. Target-FPS Builder Orchestration

Goal:

- make the annotation production stages callable in a clean sequence before fusion

Required work:

- add an orchestration script that runs:
  1. eye bbox store build
  2. ROI-local pupil store build where needed
  3. target-FPS dataset build
  4. target-FPS canonical bridge
  5. manifest build
- keep each step restartable and cache-friendly

Recommended new surface:

- `scripts/build_target_fps_groundedsam_pipeline.py`

Non-goal:

- do not turn `build_target_fps_dataset.py` into a monolithic live-inference script by default

Closure criterion:

- one documented command sequence can regenerate target-FPS labels from raw data plus Grounded-SAM runtime dependencies

## P4. H5 Validation Closure

Goal:

- close the remaining storage-format validation gap

Required work:

- add H5-focused tests for:
  - target-FPS session store materialization
  - canonical bridge metadata
  - manifest rows
  - dataset loading from `session.h5`
- verify `session_h5_path` propagation
- ensure the H5 layout matches dataset reader expectations

Primary files:

- `tests/test_target_fps_build_v3.py`
- `tests/test_target_fps_canonical_v3.py`
- `tests/test_target_fps_manifest_v3.py`
- `tests/test_target_fps_dataset_v3.py`

Closure criterion:

- the same synthetic target-FPS path passes for both `npz` and `h5`

## P5. Real-Data Validation

Goal:

- confirm that the integrated path works beyond synthetic fixtures

Required work:

- run at least one small real-session target-FPS materialization
- run with:
  - manual CSV present
  - no CSV + bbox-only
  - no CSV + full ROI-local pupil path
- inspect resulting `annotation_failures.jsonl`
- confirm that remaining failures reflect true source absence or deliberate geometry rejection

Closure criterion:

- real-session validation produces interpretable remaining failure rows with no unexplained pipeline-pending cases in the supported path

## 5. Recommended Execution Order

1. `P0` documentation sync
2. `P1` eye ROI producer extraction
3. `P2` ROI-local pupil producer extraction
4. `P3` orchestration script
5. `P4` H5 regression coverage
6. `P5` real-data validation

Reason:

- producer closure is the real blocker
- H5 validation is important but secondary to annotation-source completion
- real-data validation should come after the production path is structurally closed

## 6. Definition Of Done

This `target_fps + Grounded-SAM` closure track is complete when all of the following are true:

- `target_fps` docs match the actual implementation state
- eye ROI production is available as a reusable store-building step
- ROI-local pupil production is available as a reusable store-building step
- target-FPS fusion can consume those stores without ad hoc manual intervention
- H5 and NPZ both have synthetic regression coverage
- remaining non-complete rows are true data/runtime absence cases, not missing integration work

## 7. Immediate Next Move

Start with `P1`, not with more target-FPS bridge refactoring.

Why:

- the bridge/fusion path already exists
- the biggest remaining functional gap is missing Grounded-SAM producer integration
- solving eye ROI production first unlocks both manual-CSV and no-CSV target-FPS closure paths
