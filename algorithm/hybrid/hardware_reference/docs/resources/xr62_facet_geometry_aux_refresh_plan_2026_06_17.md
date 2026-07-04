# XR-62 FACET Geometry Auxiliary Refresh Plan

## Purpose

XR-60 routed candidate heads and XR-61 auxiliary candidate heads did not promote center, P10, or P5. XR-61A came close to the center/P5 gate but still missed, while P10 remained below the XR-39 P10 gate. XR-62 pivots back to the FACET/EllSeg/RITnet geometry signal: keep the current heatmap-state main path, disable candidate routing, and add `track_state_aux` center/axis/angle supervision as a training-only geometry regularizer.

## Evidence Basis

- FACET analysis: `anlaysis/xr-eye-tracking/papers/FACET/analysis.md`
  - direct event-to-ellipse prediction
  - five-parameter ellipse state and angle-aware geometry loss
  - HGTXR primitive mapping: geometry/ellipse/mask head as auxiliary head/loss
- FACET codebase analysis: `anlaysis/xr-eye-tracking/codebases/FACET/analysis.md`
  - `Loss.py` includes focal/regression/trigonometric ellipse losses
  - configs expose fixed-count event accumulation and ellipse dataset fields
- Current HGTXR evidence:
  - active gates: center `<16.4701875601496`, P10 `>35.02295998845781`, P5 `>12.133503770828247`
  - XR-61A best test: `16.475529539585114 / 34.356718465260094 / 12.127126216888428`
  - XR-61B best P10-side test: `16.48967229127884 / 34.58843615395682 / 11.559524168287005`

## Ablation Matrix

Parent skill: `paper-idea-generator`; IDEA-Gen phase/domain: `P1 / Research Workflow`.

| Lane | Init | Teacher | Change | LR | Best metric | Expected use |
|---|---|---|---|---:|---|---|
| XR-62A | XR-56B best-P5 | XR-39 P10 soup | Heatmap-state + track-state aux geometry + P10/P5 soft guards | `3e-7` | P10 | recover P10 without losing center/P5 |
| XR-62B | XR-58A best-P10 | XR-39 P10 soup | Same geometry aux with lighter center guard | `5e-7` | P10 | test whether P10-near teacher branch benefits from geometry regularization |

## Control Variables

- Dataset/split: manifest1 train/val/test unchanged.
- Event builder: fixed-count `255000` with adaptive support range `160000..384000`, reference `4000003us`, scale power `0.5`.
- Main evaluated coordinate path: `track_center_heatmap_as_track_state=true`.
- Candidate head: disabled.
- Optimizer: AdamW.
- Full-test evaluator is the only promotion authority.

## Trainable Scope

```json
[
  "track_center_heatmap_head.*",
  "track_state_aux_head.*",
  "event_adapter.*",
  "patch_frontend.event_embed.proj.*"
]
```

## Acceptance Criteria

Promote only if full-test evaluation beats at least one active gate:

- center `<16.4701875601496`
- P10 `>35.02295998845781`
- P5 `>12.133503770828247`

If only one metric improves while another collapses, record as mixed-gate diagnostic, not active leader.

## Commands

```bash
DRY_RUN=1 bash scripts/external/run_xr62_facet_geometry_aux_refresh.sh a cuda:0
DRY_RUN=1 bash scripts/external/run_xr62_facet_geometry_aux_refresh.sh b cuda:1
```

Actual:

```bash
bash scripts/external/run_xr62_facet_geometry_aux_refresh.sh a cuda:0
bash scripts/external/run_xr62_facet_geometry_aux_refresh.sh b cuda:1
```

## Risks

- Geometry auxiliary can pull the latent state toward ellipse axes/angle without improving center hit-rate.
- Low LR may not move enough from XR-56B/XR-58A; high LR previously damaged center/P5 in XR-60.
- If XR-62 fails, the next P0 should move to data/provenance action: pseudo-label or dense-trajectory generation rather than more scalar P10 pressure.
