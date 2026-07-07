# All-48 Sampled Dataset Construction Experiment Plan

Last updated: 2026-03-29 KST

## Summary

This plan defines a new **sampled dataset-construction experiment** built on the same all-48 comparison slice already used by the recent ROI-crop follow-up runs.

Target scope:

- `48 users`
- `4 official sessions`
- `2 eyes`
- `8 sampled frames per eye-session`
- total sampled frames:
  - `3072`

Requested experiment goals:

1. annotate **Eye Region ROI Bounding Box** and **Eye Region ROI Mask** for all sampled frames using Grounded-SAM
2. use **Raw CSV** to annotate all CSV-labeled frames with:
   - pupil state
   - pupil mask
   - pupil bounding box
3. track only frames that still have **no pupil-related annotation** and analyze them explicitly

Main recommendation:

- freeze the exact same stable sampled frame set used by the recent `v9` / `v10` / `v11` / `v12` comparisons
- reuse the already validated **bbox-only eye ROI path** for the first construction pass
- reuse the existing **raw CSV ellipse -> pupil mask / bbox / state** path for geometry-positive CSV rows
- split the final analysis into:
  - fully annotated pupil geometry
  - state-only pupil labeling
  - missing pupil annotation

## 1. Scope And Constraints

This plan assumes the following execution contract:

- server default environment:
  - `conda` env `hbtxr`
- GPU rule:
  - use only `GPU0`, `GPU1`
- sample freeze:
  - keep the same stable seed prefix already used in the all-48 preview comparisons
  - this preserves direct comparability with the current `v2` / `v9` / `v10` / `v11` / `v12` results

Important code-grounded interpretation:

- **raw CSV positive ellipse rows** can provide full pupil geometry
- **raw CSV `region_count == 0` rows** do not provide ellipse geometry directly
- therefore those frames can support:
  - state-only labeling
  - but not direct pupil mask / bbox generation unless an extra policy is added

Because of this, the experiment should distinguish:

- `pupil_geometry_complete`
- `pupil_state_only`
- `pupil_annotation_missing`

## 2. Current Reusable Code Surface

The current repository already provides most of the required building blocks.

### A. Stable sampled-frame selection

Relevant surface:

- `scripts/run_roi_crop_prompted_preview_batch.py`

Reason:

- it already freezes the all-48 sampled frame subset with a stable seed
- it already proved reproducible across `v2`, `v8`, `v9`, `v10`, `v11`, and `v12`

### B. Grounded-SAM eye ROI source

Relevant surface:

- `src/hbtxr/preprocess/groundedsam_eye_region_bbox.py`
- `scripts/groundedsam_eye_region_bbox_preview.py`

What already exists:

- validated bbox-only eye ROI detection
- explicit rejection reasons:
  - `no_eye_detections`
  - `rejected_large_box(...)`
  - `rejected_border_touch(...)`
  - `no_valid_eye_box`
- stored per-frame bbox rows under:
  - `annotation_root/eye_region_bbox_only/sessions/.../eye_region_bboxes.jsonl`

Important limitation:

- the currently validated reusable path is **bbox-only**
- the current `target_fps_build.py` contract materializes the eye ROI mask by **rasterizing the resolved eye ROI bbox**
- a first-class dedicated "true eye contour mask" producer is not yet the strongest validated path in the checked-out code

### C. Raw CSV pupil source

Relevant surface:

- `src/hbtxr/preprocess/io_utils.py`
- `src/hbtxr/preprocess/raw_ellipse_blink.py`
- `src/hbtxr/preprocess/target_fps_build.py`

What already exists:

- `parse_via_csv_with_report(...)`
  - parses ellipse-positive CSV rows
- `build_raw_ellipse_blink_metadata(...)`
  - derives closed-eye / blink metadata
- `_resolve_pupil_annotation(...)`
  - converts raw CSV ellipse rows into:
    - pupil mask
    - pupil bbox
    - pupil ellipse state

### D. Failure / missing analysis patterns

Relevant surface:

- `scripts/review_roi_crop_prompted_batch.py`
- `src/hbtxr/preprocess/target_fps_build.py`

What already exists:

- CSV-positive vs no-CSV breakdown
- blink-labeled grouping
- failure status / reason fields
- frame-level missing-row collection via:
  - `label_status`
  - `annotation_reason`
  - `roi_status`
  - `pupil_status`

## 3. Key Design Choice For This Experiment

### Recommended eye-ROI contract for the first pass

Use:

- Grounded-SAM **eye ROI bbox** as the primary eye source
- rasterized eye ROI mask derived from the stored eye bbox

Why this is the recommended first-pass choice:

- it is the already validated path in:
  - `groundedsam_eye_region_bbox.py`
  - `target_fps_build.py`
- it matches the current production-facing label contract
- it avoids reopening the unstable older direct eye-mask route described in:
  - `docs/prj/20260330_170419_03_groundedsam_dataset_pipeline.md`

Implication:

- the initial experiment will produce an **Eye Region ROI Mask** artifact
- but that mask will be the **dataset-contract mask derived from the resolved eye ROI bbox**
- not necessarily a new dense eye contour segmentation mask

If a true contour-style eye mask is required later, treat that as a second experimental branch, not the first construction baseline.

## 4. Proposed Experiment Outputs

Recommended experiment root naming:

- raw sampled frames:
  - `dataset_session_samples_10/raw_samples/all48_dataset_construction_same8samples`
- workspace:
  - `workspace_session_samples_10/all48_dataset_construction_same8samples`
- analysis:
  - `workspace_session_samples_10/all48_dataset_construction_same8samples_analysis`

Recommended logical outputs:

1. sampled frame manifest
   - fixed list of the exact `3072` frames used in the experiment
2. eye ROI source store
   - per-frame eye ROI bbox rows
   - rasterized eye ROI masks
3. raw-CSV pupil geometry store
   - pupil state
   - pupil mask
   - pupil bbox
   - blink / closed-eye metadata
4. merged frame-level annotation table
   - one row per sampled frame
5. missing-only analysis report
   - only frames without final pupil geometry

Recommended merged-row fields:

- sample identity:
  - `session_key`
  - `frame_filename`
  - `frame_idx`
  - `timestamp_us`
- eye ROI:
  - `eye_region_bbox_xywh_sensor`
  - `eye_region_mask_path`
  - `eye_region_mask_valid`
  - `eye_region_mask_source`
- pupil:
  - `pupil_ellipse_xywht_sensor`
  - `pupil_region_bbox_xywh_sensor`
  - `pupil_mask_path`
  - `mask_valid`
  - `closed_eye_flag`
  - `eye_state_flag`
- bookkeeping:
  - `label_status`
  - `annotation_reason`
  - `roi_status`
  - `pupil_status`
  - `annotation_sources_used`

## 5. Work Packages

## P0. Freeze The Exact Sample Set

Goal:

- guarantee that the dataset-construction experiment uses the exact same `3072` frames already used in the recent all-48 ROI-crop comparisons

Tasks:

- reuse the stable sample seed prefix:
  - `roi-crop-prompted-preview-v1`
- materialize a sample manifest per session or globally
- record the manifest in the experiment root

Success criterion:

- every later annotation and analysis step can be keyed against the same sampled frame list

## P1. Eye Region ROI Annotation For All Sampled Frames

Goal:

- annotate all sampled frames with eye ROI bbox
- materialize the eye ROI mask contract for all frames

Recommended implementation:

- run the validated bbox-only Grounded-SAM eye detector on the sampled frames
- save:
  - `eye_region_bboxes.jsonl`
  - rasterized eye ROI masks derived from those bboxes

Recommended output interpretation:

- `eye_region_bbox_xywh_sensor` comes from Grounded-SAM eye ROI detection
- `eye_region_mask_path` is created by rasterizing the resolved eye ROI bbox for the dataset contract

Recommended metrics:

- total sampled frames
- eye ROI detected frames
- eye ROI missing frames
- rejection-status counts by:
  - `no_eye_detections`
  - `rejected_large_box`
  - `rejected_border_touch`
  - `no_valid_eye_box`

Success criterion:

- all `3072` sampled frames have one eye ROI result row
- detection coverage and missing reasons are summarized explicitly

## P2. Raw-CSV Pupil Annotation For CSV-Labeled Frames

Goal:

- for all frames with usable raw CSV pupil geometry, generate:
  - pupil state
  - pupil mask
  - pupil bbox

Recommended implementation:

- parse raw CSV with:
  - `parse_via_csv_with_report(...)`
- derive blink / closed-eye metadata with:
  - `build_raw_ellipse_blink_metadata(...)`
- convert ellipse geometry into:
  - rasterized pupil mask
  - pupil bbox
  - pupil state representation

Important split:

### `csv_positive`

Condition:

- `region_count > 0`

Expected output:

- full pupil geometry:
  - mask
  - bbox
  - ellipse state

### `csv_region_zero`

Condition:

- CSV row exists but no positive ellipse region is present

Expected output:

- no pupil geometry
- state-only interpretation may still be recorded, for example:
  - `closed_eye_flag = True`
  - `eye_state_flag = closed`

This distinction is important because these frames should not be mixed with:

- parser failure
- missing CSV file
- unknown-frame CSV rows

Success criterion:

- every CSV-positive sampled frame has pupil geometry
- every CSV-nonpositive sampled frame is explicitly classified rather than silently dropped

## P3. Build A Merged Sampled Annotation Table

Goal:

- produce a single experiment-facing frame table for the `3072` sampled frames

Recommended row statuses:

- `annotated_complete`
  - eye ROI + pupil geometry present
- `annotated_state_only`
  - eye ROI present
  - pupil state present
  - pupil geometry absent
- `annotated_eye_only`
  - eye ROI present
  - no pupil annotation
- `pending_annotation_sources`
  - eye ROI missing

Why `annotated_state_only` is needed:

- it separates `csv_region_zero` or similar state-only cases from true missing rows
- it keeps the final analysis from overcounting closed-eye state-only frames as total annotation failure

Success criterion:

- the merged table fully partitions the `3072` frames into explicit status buckets

## P4. Missing-Only Tracking And Analysis

Goal:

- analyze only frames that still do not have final pupil geometry

Primary analysis target:

- rows where pupil geometry is absent after the merge step

Recommended analysis buckets:

1. `state_only`
   - state exists
   - geometry missing by design
2. `csv_missing`
   - no CSV source available for that frame/session
3. `csv_region_zero`
   - CSV exists but has no positive ellipse region
4. `csv_parse_or_mapping_skip`
   - CSV rows exist but could not be matched or parsed
5. `eye_roi_missing`
   - eye ROI annotation failed

Recommended outputs:

- aggregate JSON summary
- markdown report
- top sessions by missing pupil geometry
- top reasons by count
- contact sheets or grouped overlay pages for the unresolved subset only

Success criterion:

- every frame without pupil geometry is assigned to an explicit reason bucket

## 6. Recommended Execution Order

1. freeze the exact sampled frame manifest
2. run eye ROI bbox annotation on only those sampled frames
3. materialize rasterized eye ROI masks from the stored eye bboxes
4. parse raw CSV and build pupil geometry for CSV-positive frames
5. explicitly mark `csv_region_zero` frames as state-only rather than silently missing
6. merge into a single frame-level table
7. analyze only frames without final pupil geometry

## 7. Expected Deliverables

At the end of this experiment, the repository should have:

- a reproducible `3072`-frame sampled manifest
- per-frame eye ROI bbox annotation for all sampled frames
- eye ROI mask artifacts for all sampled frames
- raw-CSV-derived pupil geometry for all CSV-positive sampled frames
- a merged sampled annotation table
- a missing-only analysis report for frames without pupil geometry

## 8. Open Questions To Keep Explicit

1. Do we accept **bbox-rasterized eye ROI masks** as the official mask contract for this experiment?

- recommended answer for the first pass:
  - yes

2. Should `csv_region_zero` be counted as:

- missing pupil annotation
- or state-only annotation

- recommended answer:
  - state-only annotation

3. Do we want the final merged output to reuse the existing `frame_annotations.jsonl` contract directly, or keep a sampled-experiment-specific merged table first?

- recommended answer:
  - start with a sampled-experiment-specific merged table
  - then bridge to the production contract if the experiment validates cleanly

## 9. Immediate Next Step

Before any new model-tuning experiment such as `v13`, the most useful next execution step is:

- build the exact sampled-frame manifest
- then run the eye ROI annotation producer and the raw-CSV pupil merge on that fixed `3072`-frame slice

This keeps the next experiment focused on **dataset construction quality**, not on another prompt sweep.

## 10. Execution Result Snapshot On 2026-03-29

The planned sampled dataset-construction flow was executed in two layers:

1. all-48 sampled slice
2. focused `session_101` repeat

### All-48 sampled slice

Construction root:

- `workspace_session_samples_10/all48_dataset_construction_same8samples`

Result:

- `sampled_frames=3072`
- `eye_roi_present=3071`
- `eye_roi_missing=1`
- `pupil_geometry_complete=16`
- `pupil_state_only=2288`
- `pupil_related_missing=768`

Missing-only analysis:

- all `768` unresolved rows were:
  - `raw_csv_file_missing`

No-CSV completion follow-up root:

- `workspace_session_samples_11/all48_dataset_construction_same8samples_no_csv_completion_v10`

Completion result:

- `attempted=768`
- `completed=768`
- `completed_clean=753`
- `completed_fail_like=15`
- `unresolved=0`

### Focused `session_101` repeat

Preview root:

- `workspace_session_samples_12/roi_crop_prompted_session101_v10_same8samples`

Construction root:

- `workspace_session_samples_13/session101_dataset_construction_same8samples`

Completion root:

- `workspace_session_samples_14/session101_dataset_construction_same8samples_no_csv_completion_v10`

Result:

- preview:
  - `sampled_frames=768`
  - `completed=768`
  - `fail_like=15`
- construction:
  - `eye_roi_present=768`
  - `pupil_related_missing=768`
  - `csv_file_missing=768`
- completion:
  - `attempted=768`
  - `completed=768`
  - `completed_clean=753`
  - `completed_fail_like=15`
  - `unresolved=0`

## 11. Plan-Vs-Progress

### `P0` Freeze The Exact Sample Set

Status:

- completed

Evidence:

- the all-48 construction root reused the fixed sampled slice from the earlier `v10` preview batch
- the focused `session_101` repeat also used a stable sampled `8`-frame-per-eye-session contract

### `P1` Eye Region ROI Annotation For All Sampled Frames

Status:

- completed on the sampled experiment roots

Evidence:

- all-48:
  - `3071 / 3072` eye ROI present
- `session_101`:
  - `768 / 768` eye ROI present

### `P2` Raw-CSV Pupil Annotation For CSV-Labeled Frames

Status:

- partially completed

Evidence:

- all-48:
  - `csv_positive=16`
  - `csv_region_zero=2288`
- `session_101`:
  - no raw CSV geometry source existed

Interpretation:

- the code path worked where the source existed
- the main limitation is source coverage, not pipeline collapse

### `P3` Build A Merged Sampled Annotation Table

Status:

- completed

Evidence:

- all-48 merged root built
- `session_101` merged root built

### `P4` Missing-Only Tracking And Analysis

Status:

- completed

Evidence:

- all-48 missing-only analysis explicitly assigned every unresolved row to:
  - `raw_csv_file_missing`
- `session_101` repeat confirmed the same structure on the focused subset

### Beyond The Original Plan

Executed follow-up:

- no-CSV completion using the current `v10` ROI-crop pupil path

Result:

- the missing bucket closed completely on both the all-48 and `session_101` roots
- residual trust issue reduced to:
  - `fail_like=15`

### Current Recommended Next Step

- keep the current sampled dataset-construction outputs
- treat the clean no-CSV completion rows as usable pseudo-label candidates
- exclude or separately review the `fail_like=15` subset
- only after that, bridge the chosen policy into the production `target_fps` label path

## 12. Post-Run Policy Clarification

After the completion runs and follow-up discussion, the sampled-plan reading is now:

- do not hard-delete fail-like rows from the stored dataset artifacts
- keep event continuity intact by retaining the row-level sample metadata
- gate supervision instead of deleting data:
  - lower trust through `annotation_quality`
  - disable direct mask/track supervision through `mask_valid` and `valid_track`
- when pairing frames for tracking:
  - do not use fail-like rows as trusted previous anchors
  - if a pair crosses a fail-like or skipped gap, set `valid_track=false`

Storage clarification:

- if this logic is packed into H5 later, the heavy storage burden comes from frame/event arrays and optional bitmap masks
- fail-like flags themselves are negligible
- a practical production policy is:
  - keep annotation metadata lightweight
  - archive heavy bitmap masks primarily for accepted supervision rows
