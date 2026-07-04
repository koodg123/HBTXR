# XR-48 P10-Boundary Heatmap Calibration Plan

## Prompt Pack

- Objective: improve the active HGTXR-SW accuracy gates by changing the training mechanism after XR-47 failed.
- Source artifacts: XR-43 center leader, XR-39 P10 leader, XR-47 negative result, heatmap-state config, P10 boundary loss implementation.
- IDEA-Gen phase: P1.
- Domain: Research Workflow.
- Parent skill: paper-idea-generator.
- Target model/dataset: HGTXR mode1 stage2 raw event-count tracker on manifest1 full train/val/test.
- Constraints: no subject-39 training slice; full train/val only; promote only on strict center/P10/P5 gate improvement.
- Assumptions: P10 boundary loss can supply metric-aligned pressure around 10 px without changing architecture.
- Required artifact: executable XR-48 runner plus validation record.
- Verification evidence: `bash -n`, dry-run for both lanes, required checkpoint checks, no train/val subject-39 leakage check, train/eval logs.
- Risks: center drift from XR39 P10 teacher; P5 not directly targeted.

## Task Card

```yaml
task_card:
  task_id: T-XR48
  sub_agent: "gpt-5.5"
  role: "expert"
  objective: "Recommend a mechanism-change experiment after XR-47 same-head calibration failed."
  file_ownership: []
  assigned_skill: ["cv-dl-expert", "ablation-study-designer", "caveman"]
  inputs:
    - "docs/Validation.md"
    - "src/hbtxr/loss/bundles/track.py"
    - "src/hbtxr/models/tracker/track_branch.py"
  outputs:
    - "P10-boundary heatmap-state calibration recommendation"
  validation:
    - "Existing code path supports P10 boundary loss"
    - "Current gates are explicitly listed"
    - "Leakage guard avoids subject-39 train/val slicing"
  dependencies: []
```

## Rationale

XR-47 used XR-43 init/self-teacher, low LR, heatmap-state head-only training, center L2, and weak state distillation. It did not promote any active gate. The next experiment should not be another LR/teacher-only sweep on the same loss surface.

XR-48 adds `loss.track_p10_boundary_weight` so gradients emphasize samples close to the 10 px decision boundary used by `metric_track_p10_pct`. This is a mechanism change while preserving the proven heatmap-state evaluated path.

## Active Gates

- Center: `<16.491429926667895`
- P10: `>35.02295998845781`
- P5: `>11.868197652271816`

## Ablation Matrix

| lane | init | teacher | LR | epochs | best metric | center L2 | P10 boundary | state distill | intent |
|---|---|---|---:|---:|---|---:|---:|---:|---|
| XR-48A | XR-43 center | XR-39 P10 | `2e-6` | 8 | P10 | `0.0030` | `0.010` | `0.00025` | direct metric-aligned P10 recovery |
| XR-48B | XR-43 center | XR-39 P10 | `2e-6` | 8 | P10 | `0.0040` | `0.006` | `0.00040` | lower boundary pressure with stronger center guard |

## Control Variables

- Data: `data/_internal/manifests/manifest1/{train,val,test}_manifest.jsonl`
- Event builder: fixed count `255000`, adaptive min/max `160000/384000`, reference `4000003 us`, scale power `0.5`
- Architecture: heatmap-state head enabled, grid `32`, heatmap as evaluated `track/state`
- Trainable scope: `track_center_heatmap_head.*`
- Optimizer: AdamW
- Eval checkpoints: `best_track_p10.pt`, `best_track_p5.pt`, `best_metric_track_center_px.pt` when present

## Execution

```bash
bash scripts/external/run_xr48_p10_boundary_heatmap_calibration.sh a cuda:0
bash scripts/external/run_xr48_p10_boundary_heatmap_calibration.sh b cuda:1
```

## Expected Evidence

- Static:
  - `bash -n scripts/external/run_xr48_p10_boundary_heatmap_calibration.sh`
  - `DRY_RUN=1 bash scripts/external/run_xr48_p10_boundary_heatmap_calibration.sh a cuda:0`
  - `DRY_RUN=1 bash scripts/external/run_xr48_p10_boundary_heatmap_calibration.sh b cuda:1`
- Leakage:
  - train/val manifests contain no subject `39` if subject IDs are present.
- Runtime:
  - train/eval exit `0`
  - final eval metrics compared against active gates.

## Decision Rule

Promote only if at least one strict gate improves:

- center `<16.491429926667895`
- P10 `>35.02295998845781`
- P5 `>11.868197652271816`

If neither lane promotes, close XR-48 as a negative result for P10-boundary heatmap calibration and move to a larger mechanism change: teacher retraining, explicit P5 objective, or non-head-only adapter update.

## Results

XR-48 ran to train/eval exit `0` on both lanes. Both lanes early-stopped at epoch `7/8`.

| lane/checkpoint | center | P10 | P5 |
|---|---:|---:|---:|
| XR-48A best-P10 | 16.49682899543217 | 34.431973586763654 | 11.298894902638027 |
| XR-48A best-P5 | 16.495972565242223 | 34.19685453006199 | 11.303146593911308 |
| XR-48B best-P10 | 16.49668344429561 | 34.431973586763654 | 11.417942530768258 |
| XR-48B best-P5 | 16.50094030073711 | 34.45663345200675 | 11.52423505101885 |

## Judgment

XR-48 did not promote any active gate.

The P10-boundary loss did not recover P10; both lanes stayed near `34.43-34.46`, below the active P10 gate `35.02295998845781`. Stronger center guard in XR-48B improved P5 relative to XR-48A but remained below the active P5 gate `11.868197652271816`, and center drifted above the active center gate.

Decision: close P10-boundary heatmap head-only calibration. Next branch should use a larger mechanism change: explicit P5-boundary objective, teacher retraining, or non-head-only adapter update.
