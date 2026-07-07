# Experiment Timeline, Current Best-Practice, And Remaining TODO

This document consolidates the major experiment lines completed so far and compresses them into three tables:

1. experiment timeline
2. current best-practice
3. remaining TODO

It is intended as the fastest high-level reference before resuming implementation or starting a new experiment cycle.

## 1. Experiment Timeline

| Date | Experiment | Scope | Key Result | Current Reading |
| --- | --- | --- | --- | --- |
| 2026-03-28 | Broad ROI-crop `v2` baseline | `48 users x 4 sessions x 2 eyes x 4 frames = 1536` | `completed=1506`, `eye_failed=0`, `pupil_failed=30`, `fail_like=28` | Established the original all-user `v2` reference baseline. |
| 2026-03-28 to 2026-03-29 | All-48 sampled `v2` baseline rerun | `48 users x 4 sessions x 2 eyes x 8 frames = 3072` | `completed=3019`, `eye_failed=1`, `pupil_failed=52`, `fail_like=60` | Clean reference on the larger sampled slice, but conservative misses remained. |
| 2026-03-28 to 2026-03-29 | `crop128` ungated follow-up | Same `3072` sampled slice | `completed=3048`, `pupil_failed=23`, `fail_like=463` | Recall improved, but pupil-iris overlap became too frequent. |
| 2026-03-28 to 2026-03-29 | `v6` hard-gated `crop128` | Same `3072` sampled slice | `completed=2585`, `pupil_failed=486`, `fail_like=0` | Hard gating removed overlap at the cost of too much recall. |
| 2026-03-28 to 2026-03-29 | `v8` `crop128` multibox | Same `3072` sampled slice | `completed=3051`, `pupil_failed=20`, `fail_like=223` | Stronger recall than the ungated crop, but still too geometry-noisy for default use. |
| 2026-03-28 to 2026-03-29 | `v9` native multibox | Same `3072` sampled slice | `completed=3046`, `pupil_failed=25`, `fail_like=65` | First low-risk native improvement over `v2`. |
| 2026-03-28 to 2026-03-29 | `v10` native primary + `crop128` rescue | Same `3072` sampled slice | `completed=3068`, `pupil_failed=3`, `fail_like=65` | Best current operating point for preview-time pupil completion. |
| 2026-03-28 to 2026-03-29 | `v11` source-resolution segmentation follow-up | Same `3072` sampled slice | `completed=3048`, `pupil_failed=23`, `fail_like=239` | Source-resolution segmentation alone did not solve the geometry issue. |
| 2026-03-29 | `v12` rescue soft-gated follow-up | Same `3072` sampled slice | `completed=3049`, `pupil_failed=22`, `fail_like=46` | Cleaner than `v10`, but too conservative to replace it. |
| 2026-03-29 | All-48 sampled dataset construction | Same fixed `3072` sampled slice | `eye_roi_present=3071`, `pupil_geometry_complete=16`, `pupil_state_only=2288`, `pupil_related_missing=768` | Confirmed that the main gap was missing raw CSV geometry, not eye ROI failure. |
| 2026-03-29 | All-48 no-CSV completion using `v10` | Missing-only bucket `768` | `attempted=768`, `completed=768`, `clean=753`, `fail_like=15`, `unresolved=0` | Closed the no-CSV bucket completely on the sampled slice. |
| 2026-03-29 | Focused `session_101` repeat | `48 users x session_101 x 2 eyes x 8 frames = 768` | `completed=768`, `clean=753`, `fail_like=15` | Reproduced the no-CSV completion behavior exactly; `session_101` should not be dropped wholesale. |
| 2026-03-29 to 2026-03-30 | Dataset-refinement / storage policy clarification | Production-facing plan work | `fail_like` kept in storage, `fixed_time_bin` chosen as canonical, interpolation restricted to derived features | The policy surface is documented cleanly, but not yet fully implemented in builders/manifests. |
| 2026-03-31 | Event-voxel exploration | All-48 first-8-per-session + `user01/session_201` deep dive | Similar per-session occupied sizes, heavy-tail interval event counts, stable event-guided ROI proposals | Event voxels are useful as analysis and soft-prior tools. |
| 2026-03-31 | Event-guided ROI detector comparison | `user01/session_201`, representative intervals | Full-eye crop beat hard event-guided crop on both eyes | Current event-guided ROI should remain a soft prior, not the detector crop itself. |

## 2. Current Best-Practice

| Area | Current Best-Practice | Why | Main Reference |
| --- | --- | --- | --- |
| Pupil completion on sampled frames | Use `v10` as the default ROI-crop operating point | Best current balance of completion and acceptable geometry on the `3072` sampled slice | `docs/exps/20260328_111235_09_all48_v2_handoff.md` |
| Clean comparison baseline | Keep `v2` as the clean reference baseline | Lower recall than `v10`, but simpler and still useful as a conservative geometry reference | `docs/exps/20260328_111235_09_all48_v2_handoff.md` |
| Rescue interpretation | Treat `v12` as a conservative comparison point, not the default | Rescue-only gating reduces fail-like cases but sacrifices too much recall | `docs/exps/20260328_111235_09_all48_v2_handoff.md` |
| Eye ROI source | Use Grounded-SAM eye bbox as the primary eye-region contract | Eye ROI coverage was effectively complete on the sampled construction slice | `docs/exps/20260329_014033_10_all48_sampled_dataset_construction_experiment.md` |
| Eye ROI mask storage | Prefer bbox-derived rasterized eye masks unless a stronger mask source is validated | The stable contract is bbox-first, not contour-perfect eye masks | `docs/exps/20260329_014033_10_all48_sampled_dataset_construction_experiment.md` |
| No-CSV completion policy | Keep clean `v10` completions; separate or review `fail_like=15` | The missing bucket was fully recoverable, but the residual fail-like subset is still low-trust | `docs/exps/20260329_014033_10_all48_sampled_dataset_construction_experiment.md` |
| `fail_like` handling | Keep the row, reduce trust, and do not use it as a trusted track anchor | Event continuity is row-local; the real risk is anchor contamination in tracking supervision | `docs/plan/20260330_182945_07_dataset_refinement_and_construction_plan.md` |
| Canonical event policy | Use `fixed_time_bin` as the canonical event time coordinate | Keeps the dataset time axis stable and leaves `fixed_event_count` as an input-construction rule | `docs/plan/20260330_182945_07_dataset_refinement_and_construction_plan.md` |
| In-bin event input policy | Use `fixed_event_count` only as a sampling policy inside each canonical bin | This preserves a single canonical time grid while keeping model input size controlled | `docs/plan/20260330_182945_07_dataset_refinement_and_construction_plan.md` |
| Frame sync for training | Default to `causal_prev_frame` | Avoids leakage while preserving a training-facing temporal contract | `docs/plan/20260330_182945_07_dataset_refinement_and_construction_plan.md` |
| Interpolation policy | Interpolate only derived event features, never raw event tuples | Raw-event fabrication would blur provenance and make debugging harder | `docs/plan/20260330_182945_07_dataset_refinement_and_construction_plan.md` |
| Storage contract | Store heavy frame/event arrays in H5 and lightweight annotation/provenance in JSON/JSONL | This keeps the canonical store compact and makes low-trust labels cheap to preserve | `docs/plan/20260330_182945_07_dataset_refinement_and_construction_plan.md` |
| Event-guided ROI usage | Use event-guided ROI only as a soft prior / prompt hint / reranking cue | Hard event-guided crop currently underperforms the full-eye crop | `docs/chat/20260328_231011_04_current_project_memory.md` |

## 3. Remaining TODO

| Priority | TODO | Status | Why It Still Matters | Likely First Surface |
| --- | --- | --- | --- | --- |
| P1 | Wire the refined dataset policy into the production `target_fps` path | Open | The production builders still do not explicitly export the new quality/provenance contract | `src/hbtxr/preprocess/target_fps_build.py` |
| P1 | Add production support for `annotation_quality`, `mask_valid`, and `valid_track`-driven manifest behavior | Open | The policy is documented, but the training-facing manifest still needs the actual gating logic | `src/hbtxr/preprocess/build_manifests.py` |
| P1 | Materialize fail-like clean-anchor pairing in production | Open | `fail_like` is conceptually resolved, but the production path still needs an explicit clean-anchor rule | `src/hbtxr/preprocess/build_manifests.py` |
| P2 | Export canonical time-bin metadata in the production dataset builder | Open | The new event policy depends on a stable canonical time grid | `src/hbtxr/preprocess/target_fps_build.py` |
| P2 | Record in-bin fixed-count sampling metadata and valid masks in production outputs | Open | The chosen event-input policy is not fully represented in the production export yet | `src/hbtxr/preprocess/target_fps_build.py` |
| P2 | Preserve interpolation provenance for derived event features | Open | Interpolated features need explicit provenance if they are introduced later | `src/hbtxr/preprocess/target_fps_build.py` |
| P2 | Turn the validated ROI-crop pupil completion logic into a reusable producer/store-builder | Open | `v10` is validated experimentally, but still lives mainly as a preview/exploration path | `scripts/run_roi_crop_prompted_preview_batch.py` and a future preprocess module |
| P3 | Integrate blink / close-eye policy into the production target-FPS label path | Open | The original ROI-crop handoff still identified blink-aware production integration as unfinished | `docs/exps/20260328_111235_09_all48_v2_handoff.md` and `src/hbtxr/preprocess/target_fps_build.py` |
| P3 | Decide the final training policy for the `fail_like=15` no-CSV subset | Open | The current recommendation is clear, but the exact training-store rule still needs implementation | Future annotation export + manifest filtering path |
| P3 | Explore event-guided ROI as a soft prior in reranking rather than as a hard crop | Open | Event voxels are useful, but current hard-crop replacement is not good enough | `scripts/run_event_guided_pupil_detector_compare.py` and future reranking logic |
| P4 | Add Eye Region aspect-ratio prior to eye-box selection if needed | Open | Current Grounded-SAM eye selection uses area/border heuristics, but not explicit aspect-ratio priors | `src/hbtxr/preprocess/groundedsam_eye_region_bbox.py` |

## 4. Fast Reading

- best current pupil operating point:
  - `v10`
- clean reference baseline:
  - `v2`
- dataset construction status on the fixed sampled slice:
  - no missing pupil-geometry rows remain after `v10` completion
- main remaining production gap:
  - policy is documented, but the `target_fps` and manifest builders still need to adopt it explicitly
- current event-voxel role:
  - analysis and soft prior only
