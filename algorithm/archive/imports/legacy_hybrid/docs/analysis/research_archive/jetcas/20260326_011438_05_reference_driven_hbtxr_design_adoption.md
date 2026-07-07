# Reference-Driven HBTXR_v3 Design Adoption

Last updated: 2026-03-25 KST

Role: `XR Tracking Architect`, `Dataset/Manifest Designer`, `Training Loss Engineer`

## Summary

- This document translates the comparative study in `docs/others/jetcas/20260326_011438_04_reference_ex_gaze_facet_swift_eye_analysis.md` into concrete HBTXR_v3 design decisions.
- The intended default for `HBTXR_v3_0` is:
  - `FACET` for dataset contract, resize policy, and ellipse-aware supervision
  - `EX-Gaze` for search/track separation, previous-state usage, and runtime gating philosophy
  - `Swift-Eye` only for selective occlusion/quality gating, not for its full offline interpolation stack
- The current `v3` ABI remains valid and should not be replaced by a new public interface.

## 1. Design Intent

The comparison across `EX-Gaze`, `FACET`, and `Swift-Eye` leads to one core conclusion:

- `HBTXR_v3_0` should stay a hybrid online tracker
- it should not collapse into a pure event detector (`FACET`)
- it should not depend on offline frame interpolation and synthetic occlusion pipelines (`Swift-Eye`)
- it should not fully switch its input ABI to sparse event patches (`EX-Gaze`)

Instead, `v3` should keep its current `frame + event + prev_state` contract and selectively absorb the most useful ideas from each reference.

## 2. Adopt / Defer / Reject Matrix

| Area | Adopt Now | Defer | Reject As Default | Why |
| --- | --- | --- | --- | --- |
| Dataset annotation | Dense eye ROI, pupil mask, pupil ellipse, quality flags | Full Grounded-SAM production integration | Annotation as a runtime dependency | FACET-style supervision and the current v3 canonical already align well |
| Canonical sample contract | `frame`, `event`, `prev_state`, `mask_target`, geometry targets, quality flags | Additional paired template/search contract | Replacing v3 ABI with Swift-style paired-only ABI | Current v3 ABI is already the correct integration point |
| Manifest | Explicit `resize_policy`, `event_window`, `prev_annotation_ref`, quality metadata | Extra runtime-state history rows | Hidden loader-only policy | FACET and EX-Gaze both show the value of explicit preprocessing provenance |
| Dataloader | `fixed_count` default, `time_bin` optional, `facet_square_direct` resize | EX-Gaze-style patch extraction as an ablation path | Making sparse patchification the only path | v3 should preserve a shared 256x256 multimodal tensor path |
| Model backbone | Keep shared partial `DeiT-Tiny` style multimodal backbone | Add optional local patch/event micro-encoder later | Replacing v3 with RoITransformer/Swin | Swin+FPN is strong but too far from the current hardware-friendly design |
| Scheduler/runtime | Keep host-side `Search | Track` FSM and gate by similarity/quality/event density | Add more Swift-like occlusion heuristics later | Offline interpolation-first runtime | EX-Gaze gives the better online runtime philosophy for HBTXR |
| Training pipeline | Keep 2-stage `Search Pretrain -> Hybrid Finetune` | Add stage-3 temporal refinement later if needed | Single monolithic training only | Current v3 structure is already aligned with EX-Gaze-style separation |
| Loss | FACET trig angle encoding, ellipse-aware geometry loss, EX-Gaze-style consistency, ROI restriction | Contrastive modality loss, template correlation loss | Pure event-only loss stack | The hybrid objective needs detection accuracy and drift control together |
| Occlusion handling | Quality/closed-eye masking and confidence gating | Dedicated occlusion estimator branch | Mandatory Timelens/LAMA dependency | Swift-Eye provides the idea, but not the default dependency set |
| Hardware path | Keep CPU-side scheduler and accelerator-friendly shared backbone | Add patch-track accelerator later | Full offline pipeline inside hardware path | v3 is targeting deployable hybrid tracking, not offline analysis only |

## 3. Concrete HBTXR Decisions By Project_Structure

### 3.1 Annotation

- Keep `Grounded-SAM assisted dense annotation` as a preprocessing source, not as part of the training runtime.
- Required annotation outputs stay:
  - eye region bounding box
  - pupil mask
  - pupil ellipse / region
  - annotation quality and validity flags

Decision source:

- `FACET` showed that strong dense supervision improves ellipse prediction quality.
- `EX-Gaze` showed that tracking needs stable previous-state supervision and continuous sequence annotations.

### 3.2 Dataset / Canonical / Manifest

- Keep the current `v3` canonical session package and manifest-centered design.
- Default event builder:
  - `policy = fixed_count`
  - `event_count_target = 5000`
  - `accumulation = fast_causal_linear`
  - `fast_causal_limit = 25`
- Optional ablation builder:
  - `policy = time_bin`
  - `time_bin_us` controlled in config/manifest
- Keep `facet_square_direct` as the default resize policy.
- Keep `prev_annotation_ref` and quality metadata mandatory in the manifest.

Decision source:

- `FACET` provides the cleanest dense sample contract and default fixed-count setting.
- `EX-Gaze` justifies explicit previous-state linkage and event-window provenance.
- `FACET Table 3` favors `fixed count 5000 + fast causal event volume`, so `v3` now adopts that pair as the default builder path.
- `paper-code gap`: public FACET code exposes `causal_linear` / `causal_linear_ori` plus post-clip saturation, while `HBTXR_v3_0` now implements the paper-side fast-causal limit directly as `fast_causal_linear(limit=25)`.

### 3.3 Model

- Keep the current hybrid ABI:
  - `frame [B,1,256,256]`
  - `event [B,2,256,256]`
  - `prev_state [B,6]`
- Keep the shared partial transformer backbone.
- Keep decoupled heads:
  - eye region search head
  - pupil search head
  - pupil residual track head
  - optional mask / auxiliary quality heads
- Do not replace the model with:
  - EX-Gaze sparse patch-only transformer
  - FACET pure event one-shot detector
  - Swift-Eye Swin/RoITransformer stack

Decision source:

- `EX-Gaze` contributes the search-track separation, not the public input ABI.
- `FACET` contributes supervision structure, not the whole system design.
- `Swift-Eye` contributes ideas for occlusion handling, not the default backbone.

### 3.4 Scheduler / Runtime

- Keep host-side `Search | Track` scheduler as the default runtime shape.
- Search mode should be responsible for:
  - relocalization
  - anchor refresh
  - re-entry after drift or closed-eye recovery
- Track mode should be responsible for:
  - event-driven residual updates
  - short-horizon state propagation
  - confidence/quality monitoring
- Required gating signals:
  - `search_conf`
  - `track_conf`
  - `track_quality`
  - `similarity`
  - `event_density`
  - `closed_eye_flag`

Decision source:

- `EX-Gaze` gives the most relevant online runtime pattern.
- `Swift-Eye` supports adding quality-based mode switching, but its offline interpolation path should remain optional research work.

## 4. Recommended Loss Stack For v3

### 4.1 Stage 1: Search / Relocalization

Use FACET-like geometry supervision on the search branch:

- `L_eye_bbox`
- `L_search_xy`
- `L_search_ab`
- `L_search_trig`
- `L_search_gwd`
- `L_mask`
- `L_search_conf`
- `L_constraint_center`

Recommended interpretation:

- `xy` is center regression
- `ab` is axis-length regression
- `trig` means angle is encoded as `(sin(2theta), cos(2theta))`
- `gwd` or equivalent ellipse-aware geometry loss stabilizes the full shape
- `constraint_center` keeps the prediction inside the ROI/search prior

### 4.2 Stage 2: Hybrid Tracking

Stage 2 should keep Stage 1 search supervision alive, then add track-specific residual losses:

- `L_track_xy`
- `L_track_logab`
- `L_track_uv`
- `L_track_gwd`
- `L_track_conf`
- `L_track_quality`
- `L_consistency`
- `L_constraint_center`

Recommended interpretation:

- use residual prediction relative to `prev_state`
- keep shape updates in stable parameterizations
- keep a geometry-aware track loss, not only center displacement
- add consistency between search-anchor and event-track output

### 4.3 Masking / Gating Rules

- If `closed_eye_flag == 1`:
  - suppress geometry-heavy losses
  - keep confidence/quality supervision if labels exist
- If `mask_valid == 0`:
  - disable mask and shape-dependent losses
- If `valid_track == 0`:
  - disable track geometry and consistency terms
- Use `annotation_quality` as a per-sample weight multiplier

This is the most important usable lesson from `Swift-Eye`: quality-aware gating is valuable, but it does not require adopting Swift-Eye's full offline pipeline.

## 5. What v3 Should Not Copy Literally

### 5.1 Do Not Copy EX-Gaze Sparse Patch Input As The Default

Reason:

- it would break the current v3 public ABI
- it would move too much policy from manifest/dataloader into a highly specialized preprocessor
- it is valuable as an ablation or accelerator-oriented variant, not as the default research surface

### 5.2 Do Not Copy FACET As A Pure Event-Only End-to-End Detector

Reason:

- HBTXR_v3 is intentionally hybrid
- frame search and relocalization are part of the deployment story
- the project requirements explicitly include `2-stage Track and Search Scheduler`

### 5.3 Do Not Copy Swift-Eye Offline Interpolation Dependency As Core v3

Reason:

- it introduces heavy external training/data synthesis dependencies
- it shifts the system from online hybrid tracking toward offline analysis
- it complicates hardware and runtime validation

## 6. Immediate Documentation-Level Actions

The reference study implies the following default documentation stance for `v3`:

- `docs/others/jetcas/20260326_011438_04_reference_ex_gaze_facet_swift_eye_analysis.md`
  - comparative evidence source
- this document
  - HBTXR adoption decision source
- `docs/prj/20260330_170419_01_v3_code_architecture_analysis.md`
  - code-level analysis of the active implementation
- `PLAN.md`
  - should treat this adoption mapping as the active reference-driven design direction

## 7. Recommended Next Implementation Priority

1. Normalize the active loss documentation around `trig + ellipse geometry + consistency + ROI restriction`.
2. Expose runtime thresholds explicitly in config rather than leaving them only as internal defaults.
3. Add a clean `track_quality` target path so Swift-Eye-style quality gating becomes measurable.
4. Keep EX-Gaze-like sparse patch tracking as a later optional experimental branch, not the main ABI.

## 8. Checklist

- [x] Compared `EX-Gaze`, `FACET`, and `Swift-Eye`
- [x] Mapped their ideas to HBTXR_v3 areas
- [x] Decided what to adopt now
- [x] Decided what to defer
- [x] Decided what not to use as the default path
- [x] Recorded the result as a design document
