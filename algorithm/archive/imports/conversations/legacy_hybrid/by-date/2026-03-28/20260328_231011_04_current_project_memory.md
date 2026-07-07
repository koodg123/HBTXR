# Current Project Memory Snapshot

Last updated: 2026-03-29 KST

This document is a compact working-memory snapshot for the current `HBTXR_v3_0` repository state.

It is intended to reduce repeated re-discovery across future sessions by keeping the most important active context in one place.

## 1. Canonical Reference Set

Use these documents first when reloading project context:

- update history:
  - `docs/chat/20260326_011438_01_v3_update_history.md`
- progress checklist:
  - `docs/chat/20260326_011438_02_v3_progress_checklist.md`
- conversation and decision log:
  - `docs/chat/20260326_011438_03_v3_conversation_and_decision_log.md`
- active target-FPS transition plan:
  - `docs/plan/20260327_163915_04_v3_2_update_plan.md`
- code architecture analysis:
  - `docs/prj/20260330_170419_01_v3_code_architecture_analysis.md`
- repository tree and dependency map:
  - `docs/hbtxr/06_v3_repository_tree_and_module_dependency_map.md`
- Grounded-SAM ROI experiment summary:
  - `docs/exps/20260328_110310_07_groundedsam_roi_crop_prompted_experiment_summary.md`
- all-user failure review:
  - `docs/exps/20260328_110310_08_all48_v2_baseline_failure_review.md`
- cross-server handoff:
  - `docs/exps/20260328_111235_09_all48_v2_handoff.md`

## 2. Current Project Identity

`HBTXR_v3_0` is not only a model repository.

It is an execution-oriented research codebase covering:

- raw dataset discovery
- Grounded-SAM annotation and ROI tooling
- canonicalization
- manifest generation
- dataset assembly
- stage1 / stage2 training
- evaluation / inference / visualization
- target-FPS dataset transition work

The main active code surface is:

- `src/hbtxr/preprocess/`
- `src/hbtxr/data/`
- `src/hbtxr/models/`
- `src/hbtxr/training/`
- `src/hbtxr/runtime/`
- `scripts/`
- `tests/`

## 3. Two Major Data Paths Coexist

### A. Existing canonical/manifests path

This path is centered on:

- `src/hbtxr/preprocess/canonicalize.py`
- `src/hbtxr/preprocess/build_manifests.py`
- `src/hbtxr/data/dataset.py`

Flow:

`raw -> canonical1/canonical2 -> manifest1/manifest2 -> dataset -> train/eval/infer`

### B. New target-FPS session-store path

This path is centered on:

- `src/hbtxr/preprocess/target_fps_build.py`
- `src/hbtxr/preprocess/target_fps_canonical.py`
- `src/hbtxr/preprocess/build_manifests.py`
- `src/hbtxr/data/dataset.py`

Flow:

`raw -> fps_<target>/session store -> canonical bridge -> manifest bridge -> same dataset/training runtime`

Interpretation:

- the repository is in a transition phase
- the older canonical flow still matters
- the newer target-FPS path is already materially implemented

## 4. Important Code-vs-Docs Reality Check

Some tracking docs still read as if `target_fps` work is mostly pending.

That is no longer the full truth of the checked-out code.

Based on source and tests, the following are already present in the repository:

- target-FPS session scan and grid planning
- target-FPS materialization into session stores
- canonical bridge from target-FPS session stores
- manifest rows that reference session-store-backed frames/events
- dataset loading from session stores
- synthetic mode1/mode2 coverage for the target-FPS bridge path

Practical rule:

- trust `docs/plan/20260327_163915_04_v3_2_update_plan.md` for design intent
- trust source files and tests for actual current implementation state

## 5. What Still Looks Genuinely Open

The following still appear to be live gaps or at least not fully settled:

- real dataset / real checkpoint end-to-end validation
- actual CLI-level use of the runtime FSM
  - `RuntimeHBTXRTracker` exists
  - main `eval/infer` scripts still use direct batch forward
- cases where only partial annotation sources exist
  - manual CSV pupil without Grounded-SAM eye ROI remains pending
  - no-CSV sessions with only eye-bbox preview can remain eye-only without full pupil completion
- some documentation is behind the code and should be re-synced

## 6. Grounded-SAM ROI-Crop Handoff Context

The `09_all48_v2_handoff.md` document is specifically about continuing the all-user ROI-crop Grounded-SAM baseline on another server.

It is not a full-project handoff for the entire training stack.

Current recommended operating point in that handoff:

- eye stage:
  - stronger `v2` eye prompt
- pupil stage:
  - strict prompted crop-local pupil stage
- fallback:
  - disabled by default

Broader sweep result recorded in the handoff and linked review:

- sampled_frames = `1536`
- completed = `1506`
- eye_failed = `0`
- pupil_failed = `30`
- fail_like = `28`

Interpretation:

- stage-1 eye ROI is stable at sweep level
- remaining issues concentrate in blink / near-close misses, oversized pupil geometry, and small ROI escapes

## 7. Current Working Conclusion

The best current high-level reading of the repository is:

1. the execution surface is broader and more complete than some older tracking docs suggest
2. the target-FPS path is already a real implementation track, not only a plan
3. the Grounded-SAM ROI-crop work is an active adjacent stream with its own handoff and failure-analysis documents
4. the most meaningful remaining uncertainty is real-environment validation, not synthetic contract coverage

## 8. Local Environment Note

At the time this memory snapshot was written:

- this server should use the conda environment named `hbtxr` as the default runtime environment
- repository-local `.venv` was not present
- `conda run -n hbtxr python` was the reliable execution entrypoint on this server
- imports for `pytest`, `torch`, and `h5py` were confirmed inside the `hbtxr` environment
- for future GPU work on this server, only `GPU0` and `GPU1` should be used unless the user explicitly changes that rule

Implication:

- future Python/package execution on this server should prefer the `hbtxr` conda environment
- local source/test inspection is possible
- package/test/script execution should be routed through `conda run -n hbtxr ...`

## 9. 2026-03-28 All-48 ROI-Crop v2 Rerun Snapshot

This run resumed the prior handoff task on the current server and re-executed the all-user ROI-crop prompted pupil baseline with random sampling.

Configuration and operational notes:

- paths config was updated to the current server for:
  - `configs/paths/ev_eye_groundedsam_paths.json`
- Grounded-SAM package content was synced into:
  - `packages/Grounded-Segment-Anything`
- the local vendored GroundingDINO runtime required a defensive guard for empty merged phrases in:
  - `packages/Grounded-Segment-Anything/GroundingDINO/groundingdino/util/inference.py`
- the full run used the `roi_crop_prompted_pupil_v2_eye_prompted.json` options config
- initial single-GPU execution was restarted as 4 GPU shards to finish in reasonable time

Scope:

- users: `48`
- official sessions per eye: `4` (`101`, `102`, `201`, `202`)
- eyes: `left + right`
- random samples per session slice: `8`
- total session slices: `384`
- total sampled frames: `3072`

Output roots:

- sampled raw-frame links:
  - `dataset_session_samples_2/raw_samples/roi_crop_prompted_all48_v2_baseline_8samples`
- per-session annotation summaries and overlays:
  - `workspace_session_samples_2/roi_crop_prompted_all48_v2_baseline_8samples`
- grouped review outputs:
  - `workspace_session_samples_2/roi_crop_prompted_all48_v2_baseline_8samples_review_live`

Aggregate result:

- completed: `3019 / 3072`
- eye_failed: `1`
- pupil_failed: `52`
- fail_like: `60`
- predicted class counts:
  - `small dark circular pupil`: `2983`
  - `black pupil`: `36`

Review / raw-CSV comparison result:

- csv_positive: `16`
- no_raw_csv_supervision: `3056`
- blink_labeled: `2290`
- csv_status_counts:
  - `csv_file_missing`: `768`
  - `csv_positive`: `16`
  - `csv_region_zero`: `2288`

Grouped review page counts:

- csv_positive_compare_pages: `2`
- no_raw_csv_pages: `191`
- blink_labeled_pages: `144`
- pupil_failed_pages: `4`
- fail_like_pages: `4`

Useful summary files:

- `workspace_session_samples_2/roi_crop_prompted_all48_v2_baseline_8samples_review_live/review_summary.json`
- `workspace_session_samples_2/roi_crop_prompted_all48_v2_baseline_8samples_review_live/review_summary.md`
- `dataset_session_samples_2/logs/shard_cuda0/2026-03-28_roi_crop_prompted_preview_all_sessions_summary.json`
- `dataset_session_samples_2/logs/shard_cuda1/2026-03-28_roi_crop_prompted_preview_all_sessions_summary.json`
- `dataset_session_samples_2/logs/shard_cuda2/2026-03-28_roi_crop_prompted_preview_all_sessions_summary.json`
- `dataset_session_samples_2/logs/shard_cuda3/2026-03-28_roi_crop_prompted_preview_all_sessions_summary.json`

Interpretation:

- the rerun reproduced the intended all-48 sampling scope on this server
- raw CSV supervision is still extremely sparse in the sampled subset
- most sampled frames are currently no-CSV and are therefore being judged mainly through the generated Grounded-SAM review artifacts
- the most concerning sessions for follow-up are concentrated in:
  - `user21/left/session_201`
  - `user33/right/session_102`
  - `user17/left/session_201`

## 10. 2026-03-28 ROI-Crop v2 Crop-128 Follow-up

After the baseline all-48 rerun, a follow-up experiment was run with the same sampled frames but with the pupil-stage crop input resized to `128x128`.

Implementation notes:

- `scripts/run_roi_crop_prompted_preview_batch.py` now supports `crop_resize_wh`
- the pupil-stage crop is resized before inference
- the predicted mask is resized back to the original ROI size before computing bbox / ellipse / full-frame backprojection
- the new options config is:
  - `configs/groundedsam/roi_crop_prompted_pupil_v2_eye_prompted_crop128.json`
- this server rule still applies:
  - use only `GPU0` and `GPU1`

Run scope:

- same stable seed:
  - `roi-crop-prompted-preview-v1`
- same total sampled frames:
  - `3072`
- same session slices:
  - `384`
- same raw-CSV availability profile as the baseline rerun

Output roots:

- batch workspace:
  - `workspace_session_samples_3/roi_crop_prompted_all48_v2_crop128_same8samples`
- review root:
  - `workspace_session_samples_3/roi_crop_prompted_all48_v2_crop128_same8samples_review_live`

Aggregate result:

- completed: `3048 / 3072`
- eye_failed: `1`
- pupil_failed: `23`
- fail_like: `463`
- predicted class counts:
  - `small dark circular pupil`: `2925`
  - `black pupil`: `123`

Direct comparison vs previous v2 baseline rerun:

- completed: `+29`
- pupil_failed: `-29`
- fail_like: `+403`

Interpretation:

- `128x128` crop input recovered many previously missed pupil cases
- but it also caused a major increase in fail-like geometry, so the run is not a clean improvement
- the current reading is:
  - recall improved
  - compactness / shape quality regressed sharply

Priority review targets in the crop-128 run:

- top pupil-failed:
  - `user17/left/session_201`
  - `user29/left/session_102`
  - `user07/left/session_102`
  - `user38/right/session_102`
- top fail-like:
  - `user45/right/session_202`
  - `user29/left/session_102`
  - `user29/left/session_202`
  - `user41/right/session_102`

## 11. 2026-03-28 Crop-128 Main-Gated Follow-up

To suppress pupil masks that visibly consume iris / interior regions, a stricter main-stage gated config was tested on the same `3072` sampled frames:

- config:
  - `configs/groundedsam/roi_crop_prompted_pupil_v6_eye_prompted_crop128_main_gated.json`
- batch workspace:
  - `workspace_session_samples_4/roi_crop_prompted_all48_v6_crop128_main_gated_same8samples`
- review root:
  - `workspace_session_samples_4/roi_crop_prompted_all48_v6_crop128_main_gated_same8samples_review_live`

Key config idea:

- keep the `128x128` crop input
- add `acceptance_filters` directly to the main pupil stage
- use the same geometry limits as the prior fail-like warning thresholds
- disable retry on fail-like for this variant

Aggregate result:

- completed: `2585 / 3072`
- eye_failed: `1`
- pupil_failed: `486`
- fail_like: `0`

Direct comparison:

- vs `crop128` ungated:
  - completed: `-463`
  - pupil_failed: `+463`
  - fail_like: `-463`
- vs baseline `v2`:
  - completed: `-434`
  - pupil_failed: `+434`
  - fail_like: `-60`

Interpretation:

- the gate did exactly what it was supposed to do
  - iris-overlap / oversized pupil candidates were almost entirely removed
- but the threshold is too aggressive for production use
  - it converts a very large fraction of previously accepted outputs into `pupil_failed`
- current conclusion:
  - directionally correct
  - not yet balanced
  - the next experiment should be a softer gate, not this hard gate as-is

Priority sessions after the hard-gated run:

- `user17/left/session_201`
- `user29/left/session_102`
- `user45/right/session_202`
- `user29/left/session_202`
- `user41/right/session_102`

## 12. 2026-03-28 Crop-128 Fail-Like Retry Smoke

Before launching another full all-48 run, a lighter config-only idea was tested on the worst fail-like session:

- config:
  - `configs/groundedsam/roi_crop_prompted_pupil_v7_eye_prompted_crop128_faillike_retry_gated_fallback.json`
- smoke target:
  - `user45/right/session_202`
- workspace:
  - `workspace_session_samples_smoke/roi_crop_prompted_user45_right202_v7_crop128_smoke`

Key idea:

- keep `128x128`
- keep the strict main prompt
- on fail-like, retry with a compact fallback prompt and compact gating

Smoke result:

- completed: `8 / 8`
- pupil_failed: `0`
- fail_like: `7`

Interpretation:

- the fallback stage was attempted on the problematic frames
- but it did not produce a better accepted candidate than the main stage
- because the smoke result was effectively unchanged from the ungated crop-128 run, the full all-48 `v7` run was intentionally skipped

## 13. 2026-03-28 Crop-128 Multi-Box Follow-up

After the config-only retry idea underperformed, the next change moved into code:

- `scripts/run_roi_crop_prompted_preview_batch.py` now supports multi-candidate pupil box ranking
- a stage can expose more than one candidate via `max_candidate_boxes`
- the runtime now prefers the first non-fail-like candidate and otherwise keeps the least-bad fail-like candidate

Experiment config:

- `configs/groundedsam/roi_crop_prompted_pupil_v8_eye_prompted_crop128_multibox.json`

Output roots:

- batch workspace:
  - `workspace_session_samples_5/roi_crop_prompted_all48_v8_crop128_multibox_same8samples`
- review root:
  - `workspace_session_samples_5/roi_crop_prompted_all48_v8_crop128_multibox_same8samples_review_live`

Aggregate result:

- completed: `3051 / 3072`
- eye_failed: `1`
- pupil_failed: `20`
- fail_like: `223`
- predicted class counts:
  - `small dark circular pupil`: `2908`
  - `black pupil`: `143`

Direct comparison:

- vs `crop128` ungated:
  - completed: `+3`
  - pupil_failed: `-3`
  - fail_like: `-240`
- vs baseline `v2`:
  - completed: `+32`
  - pupil_failed: `-32`
  - fail_like: `+163`
- vs `crop128` hard-gated:
  - completed: `+466`
  - pupil_failed: `-466`
  - fail_like: `+223`

Multi-box usage:

- selected candidate rank counts:
  - rank `1`: `2798`
  - rank `2`: `251`
  - rank `3`: `2`
- fail-like among selected ranks:
  - rank `1`: `213`
  - rank `2`: `10`
  - rank `3`: `0`

Frame-level comparison vs `crop128` ungated:

- `242` frames changed from `completed + fail_like` to `completed + clean`
- `3` frames changed from `pupil_failed` to `completed`
- no regression case was observed in the sampled subset

Interpretation:

- this is the first crop-128 follow-up that materially reduces iris-overlap style fail-like cases without sacrificing recall
- multi-box selection is doing real work, not only moving counts around
- the remaining gap is still significant, but the direction is much healthier than hard gating

Priority sessions after the multibox run:

- remaining fail-like heavy:
  - `user45/right/session_202`
  - `user23/right/session_202`
  - `user23/left/session_202`
  - `user18/right/session_201`
- strongest improvements:
  - `user41/right/session_102`
  - `user29/left/session_202`
  - `user29/left/session_102`
  - `user33/right/session_202`

## 14. 2026-03-28 Native / Rescue / Source-Seg Follow-up

After `v8`, three new all-48 runs were executed on the same `3072` sampled frames.

### A. `v9` Native Multibox

Config:

- `configs/groundedsam/roi_crop_prompted_pupil_v9_eye_prompted_native_multibox.json`

Output roots:

- `workspace_session_samples_6/roi_crop_prompted_all48_v9_native_multibox_same8samples`
- `workspace_session_samples_6/roi_crop_prompted_all48_v9_native_multibox_same8samples_review_live`

Aggregate result:

- completed: `3046 / 3072`
- eye_failed: `1`
- pupil_failed: `25`
- fail_like: `65`

Interpretation:

- native-resolution multibox is a low-risk improvement over the original baseline
- compared with baseline `v2`, the key net effect was:
  - many prior `pupil_failed` frames were recovered
  - fail-like stayed close to the original baseline scale

### B. `v10` Native Primary + Crop128 Rescue

Config:

- `configs/groundedsam/roi_crop_prompted_pupil_v10_eye_prompted_native_then_crop128_rescue.json`

Output roots:

- `workspace_session_samples_7/roi_crop_prompted_all48_v10_native_then_crop128_rescue_same8samples`
- `workspace_session_samples_7/roi_crop_prompted_all48_v10_native_then_crop128_rescue_same8samples_review_live`

Aggregate result:

- completed: `3068 / 3072`
- eye_failed: `1`
- pupil_failed: `3`
- fail_like: `65`

Stage usage:

- native primary accepted:
  - `2984` frames
- crop128 rescue accepted:
  - `84` frames

Frame-level comparison vs `v9`:

- `19` frames changed from `completed + fail_like` to `completed + clean`
- `23` frames changed from `pupil_failed` to `completed`
- `7` clean frames regressed to fail-like
- `1` fail-like frame regressed to `pupil_failed`

Interpretation:

- this is the strongest current operating point in the sampled subset
- the rescue stage is used sparingly, which is desirable
- most outputs still come from the cleaner native path

### C. `v11` Crop128 Proposal + Source-Resolution Segmentation

Config:

- `configs/groundedsam/roi_crop_prompted_pupil_v11_eye_prompted_crop128_multibox_source_segment_softgate.json`

Output roots:

- `workspace_session_samples_8/roi_crop_prompted_all48_v11_crop128_source_segment_same8samples`
- `workspace_session_samples_8/roi_crop_prompted_all48_v11_crop128_source_segment_same8samples_review_live`

Aggregate result:

- completed: `3048 / 3072`
- eye_failed: `1`
- pupil_failed: `23`
- fail_like: `239`

Interpretation:

- source-resolution segmentation works technically
- but by itself it does not solve the crop128 geometry problem
- relative to `v8`, it slightly regressed on the sampled subset:
  - more fail-like
  - more pupil failures

## 15. Current Best Read

The strongest current ranking is:

1. `v10` native-primary + crop128-rescue
2. `v12` native-primary + crop128-rescue-softgate
3. `v9` native multibox
4. original `v2` baseline
5. `v8` crop128 multibox
6. `v11` crop128 source-segmentation

Practical meaning:

- for the current preview-style annotation path, `v10` is the best candidate
- `v12` is the cleaner but more conservative comparison point
- for production closure work, the next important step is not another option-only sweep
- the next important step is extracting the winning `v10` logic out of the preview runner and into a reusable producer module / store-builder path

## 16. Failure Status Definitions To Preserve

These definitions come directly from the current preview runner and should be used consistently when reading review summaries.

### `eye_failed`

- `eye_failed` is assigned before the crop-local pupil stage runs
- it happens when `_detect_eye_region_bbox(...)` returns `None`
- the recorded `eye_detection_status` can currently be:
  - `no_eye_detections`
  - `rejected_large_box(area_ratio=...)`
  - `rejected_border_touch(border_touches=...)`
  - `no_valid_eye_box`

Current `v2` eye-stage rejection thresholds:

- `max_box_area_ratio=0.85`
- `border_margin_px=3`
- `max_border_touches=3`

Practical meaning:

- this is primarily a stage-1 eye-box failure / rejection bucket
- it is not the same as "eye was found but pupil segmentation later failed"

### `pupil_failed`

- `pupil_failed` is assigned only after an eye ROI already exists
- it happens when the full pupil stage chain ends with no surviving `pupil_result`
- common underlying causes are:
  - no detection candidate
  - mask area below `min_mask_area`
  - invalid bbox / ellipse fit from the mask
  - stage acceptance-filter rejection
  - no clean result and no retained fail-like fallback

Important boundary:

- `fail_like` is not a separate terminal status
- a frame with `fail_like=True` is still recorded as `completed`
- therefore `pupil_failed` means "no final pupil result kept", while `fail_like` means "result kept, but geometry warning present"

## 17. 2026-03-29 `v12` Rescue-Softgate Follow-up

Config:

- `configs/groundedsam/roi_crop_prompted_pupil_v12_eye_prompted_native_then_crop128_rescue_softgate.json`

Output roots:

- `workspace_session_samples_9/roi_crop_prompted_all48_v12_native_then_crop128_rescue_softgate_same8samples`
- `workspace_session_samples_9/roi_crop_prompted_all48_v12_native_then_crop128_rescue_softgate_same8samples_review_live`

Aggregate result:

- completed: `3049 / 3072`
- eye_failed: `1`
- pupil_failed: `22`
- fail_like: `46`

Stage usage:

- native primary accepted:
  - `3005` frames
- crop128 rescue softgate accepted:
  - `44` frames

Frame-level comparison vs `v10`:

- `19` frames changed from `completed + fail_like` to `pupil_failed`
- no offsetting `pupil_failed -> completed` recovery appeared in this follow-up

Interpretation:

- the rescue-only soft gate successfully suppressed rescue-stage fail-like acceptance
- but it did so mostly by discarding borderline rescue outputs
- remaining `v12` fail-like frames came from the native primary path
- this means `v12` is a cleaner but more conservative comparison point, not the new best operating point

## 18. 2026-03-29 All-48 Sampled Dataset Construction And No-CSV Completion

Construction roots:

- base construction:
  - `workspace_session_samples_10/all48_dataset_construction_same8samples`
- no-CSV completion:
  - `workspace_session_samples_11/all48_dataset_construction_same8samples_no_csv_completion_v10`
- no-CSV overlay preview:
  - `workspace_session_samples_11/all48_dataset_construction_same8samples_no_csv_completion_v10_overlay_preview_live`

Construction aggregate:

- sampled frames:
  - `3072`
- eye ROI present:
  - `3071`
- eye ROI missing:
  - `1`
- raw-CSV geometry complete:
  - `16`
- state-only:
  - `2288`
- missing-only before completion:
  - `768`

Missing-only interpretation:

- all `768` unresolved rows were:
  - `raw_csv_file_missing`
- the missing bucket therefore reflected missing supervision source, not a broad model failure mode

No-CSV completion aggregate:

- attempted:
  - `768`
- completed:
  - `768`
- clean completed:
  - `753`
- fail-like completed:
  - `15`
- unresolved:
  - `0`

Final merged reading:

- total geometry-complete rows:
  - `784`
- geometry-complete from raw CSV:
  - `16`
- geometry-complete from no-CSV ROI-crop completion:
  - `768`
- remaining pupil-related missing:
  - `0`

Current dataset recommendation:

- keep:
  - `raw_csv geometry 16`
  - `no_csv clean completion 753`
  - `state_only 2288`
- conservative holdout:
  - `no_csv completion fail_like 15`

## 19. 2026-03-29 Session `101`-Only Repeat

Roots:

- preview batch:
  - `workspace_session_samples_12/roi_crop_prompted_session101_v10_same8samples`
- construction:
  - `workspace_session_samples_13/session101_dataset_construction_same8samples`
- completion:
  - `workspace_session_samples_14/session101_dataset_construction_same8samples_no_csv_completion_v10`
- overlay preview:
  - `workspace_session_samples_14/session101_dataset_construction_same8samples_no_csv_completion_v10_overlay_preview_live`

Preview aggregate:

- sampled frames:
  - `768`
- completed:
  - `768`
- eye_failed:
  - `0`
- pupil_failed:
  - `0`
- fail_like:
  - `15`

Construction aggregate:

- eye ROI present:
  - `768`
- pupil-related missing:
  - `768`
- CSV status:
  - `csv_file_missing=768`

Completion aggregate:

- attempted:
  - `768`
- completed:
  - `768`
- clean completed:
  - `753`
- fail-like completed:
  - `15`
- unresolved:
  - `0`

Interpretation:

- the focused `session_101` repeat exactly reproduces the no-CSV bucket found in the larger all-48 slice
- current recommendation:
  - do not drop the whole `session_101`
  - if a conservative training gate is needed, exclude or review only the `fail_like=15` subset

## 20. 2026-03-29 Current Dataset-Refinement Policy Memory

Fail-like handling:

- do not hard-delete fail-like rows from the sampled store or future H5-backed dataset roots
- event tensors are row-local, so neighboring event content is preserved even if a row is excluded from supervision
- the true continuity risk sits in:
  - `prev_annotation_ref`
  - `prev_state`
  - `valid_track`
- current recommended conservative policy:
  - keep the row
  - downweight or suppress direct supervision through `annotation_quality`
  - disable mask/track supervision through `mask_valid` and `valid_track`
  - do not use fail-like rows as trusted pair anchors

Current risk reading:

- fail-like exposure on the all-48 sampled slice:
  - `15 / 3072 = 0.49%` of all sampled rows
  - `15 / 768 = 1.95%` of no-CSV completion rows
  - `15 / 784 = 1.91%` of geometry-complete rows
- current interpretation:
  - unlikely to break training on its own
  - still worth conservative handling because the error mode is systematic and not random

H5 / annotation-storage reading:

- the heavy objects are:
  - frame arrays
  - event arrays
  - optional bitmap masks
- annotation metadata and fail-like flags are comparatively cheap
- practical recommendation:
  - keep frame/event arrays in the H5 store
  - keep annotation rows lightweight
  - archive bitmap masks primarily for accepted supervision rows
  - do not require heavy bitmap-mask storage for fail-like review-only rows in the primary training store

Mask-branch note:

- the active model still has a direct pixel-mask branch
- `search/mask_logits` is produced by `SearchMaskHead`
- final output shape:
  - `[B, 1, H, W]`
  - typically `[B, 1, 256, 256]`
- implication:
  - bitmap masks remain valuable for accepted training rows
  - they are not equally valuable for low-trust fail-like rows

## 21. 2026-03-30 Refined Production Dataset Plan Memory

Active plan doc:

- `docs/plan/20260330_182945_07_dataset_refinement_and_construction_plan.md`

Current recommended production dataset reading:

- canonical event coordinate:
  - `fixed_time_bin`
- in-bin event sampling:
  - `fixed_event_count` is allowed, but only as an input construction rule inside each time bin
- frame sync defaults:
  - `causal_prev_frame` for training / tracking
  - keep `nearest_frame_idx` for offline analysis
- fail-like handling:
  - keep the row in storage
  - do not use it as a trusted track anchor
- interpolation:
  - apply only to derived event features
  - do not synthesize raw event tuples and store them as if observed
- storage split:
  - H5 for heavy frame/event/time-grid/derived-feature arrays
  - JSON / JSONL for annotation geometry, provenance, quality flags, and pair metadata

Current implementation gap:

- the policy is now documented cleanly
- production builders / manifests still need explicit support for:
  - canonical time-bin export
  - in-bin fixed-count sampling metadata
  - fail-like clean-anchor pairing
  - derived-feature interpolation provenance

## 22. 2026-03-31 Event-Voxel Exploration Memory

Active analysis roots:

- `workspace_event_voxel_analysis/all48_first8_per_session`
- `workspace_event_voxel_analysis/user01_session201_interval_events`
- `workspace_event_voxel_analysis/user01_session201_event_guided_pupil_roi`
- `workspace_event_voxel_analysis/user01_session201_event_guided_pupil_detector_compare`
- `workspace_event_voxel_analysis/user01_session201_interval_events_3d`
- `workspace_event_voxel_analysis/user01_session201_interval_events_3d_with_rois`

Active scripts:

- `scripts/analyze_event_voxel_spatial_size.py`
- `scripts/render_session_interval_event_voxels.py`
- `scripts/render_event_guided_pupil_roi.py`
- `scripts/run_event_guided_pupil_detector_compare.py`
- `scripts/render_session_interval_event_3d.py`
- `scripts/render_session_interval_event_3d_with_rois.py`

Current event-voxel reading:

- all-48 first-8-per-session occupied-size statistics are broadly similar across session codes
  - `session_101` mean size:
    - `316.59 x 237.31`
  - `session_102` mean size:
    - `312.05 x 234.04`
  - `session_201` mean size:
    - `315.79 x 238.31`
  - `session_202` mean size:
    - `313.93 x 237.28`
- `user01/session_201` interval event counts are heavy-tailed
  - left mean:
    - `1947`
  - right mean:
    - `1712`
  - mean is much larger than the median, so burst intervals dominate the tail

Current event-guided ROI reading:

- event-guided ROI proposal worked well enough to produce stable exploratory ROI pages
- left interval policy usage:
  - `event_guided=4062`
  - `carry_prev_sparse=980`
  - `event_guided_high_burst=51`
- right interval policy usage:
  - `event_guided=4116`
  - `carry_prev_sparse=901`
  - `event_guided_high_burst=51`

Detector-comparison memory:

- on the representative `user01/session_201` subset, full-eye crop remains better than hard event-guided ROI crop
- left representative comparison:
  - full-eye:
    - `35 completed`
    - `0 fail_like`
    - `1 failed`
  - event-guided hard crop:
    - `34 completed`
    - `10 fail_like`
    - `2 failed`
- right representative comparison:
  - full-eye:
    - `36 completed`
    - `0 fail_like`
    - `0 failed`
  - event-guided hard crop:
    - `35 completed`
    - `9 fail_like`
    - `1 failed`

Current recommendation:

- keep event-voxel tooling as an exploratory / diagnostic layer
- do not promote the current event-guided hard crop into the main detection path
- if reused, event-guided ROI should act as:
  - a soft spatial prior
  - a prompt hint
  - a reranking cue
  rather than the sole detector crop

## 23. 2026-04-01 Consolidated Experiment Snapshot Memory

Active high-level summary doc:

- `docs/exps/20260401_040252_11_experiment_timeline_best_practice_todo.md`

Use this doc first when a quick restart is needed for:

- ROI-crop experiment chronology
- current recommended operating points
- production-facing remaining implementation tasks
