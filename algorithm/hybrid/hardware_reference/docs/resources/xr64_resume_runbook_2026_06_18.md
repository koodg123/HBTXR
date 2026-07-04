# XR-64 Resume Runbook

Date: 2026-06-18

## Status

Execution is paused by user directive. This runbook is for future resumption only.

Do not run commands in this document until the user explicitly resumes experiments.

## Goal

Convert XR-63 no-train teacher-target oracle headroom into leakage-safe trainable supervision:

- generate train/val teacher eval rows for `xr62a`, `xr39`, `xr56b`, `xr58a`.
- build train/val-only pseudo-target override JSON files.
- train XR-64A/B, optionally XR-64C.
- evaluate on test with `data.track_target_override_path=null`.

## Prompt Pack

Parent skill:

- `paper-idea-generator`
- IDEA-Gen domain: `Research Workflow`
- worker skill: `ablation-study-designer`

Objective:

- isolate whether sample-wise teacher-target supervision improves HGTXR center/P10/P5 gates beyond XR-62A/XR-39/XR-56B.

Known facts:

- active gates: center `<16.468481131962367`, P10 `>35.02295998845781`, P5 `>12.133503770828247`.
- XR-63 oracle upper bound: `16.04023192701366/36.48596938775512/13.41751700680271`.
- XR-63 oracle is not a valid trained model because it uses test-aligned teacher selection.
- XR-64 must use train/val generated targets only.

Assumptions:

- current checkpoints referenced by scripts still exist.
- `manifest1` train/val/test manifests are unchanged.
- CUDA is available when experiments resume.

Signoff evidence:

- all generated JSON files exist and validate.
- test eval uses no target override.
- promotion is measured only by full-test center/P10/P5 against current gates.

## Preconditions

Run these checks before resuming, but only after user approves experiment execution:

```bash
bash -n scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh scripts/external/run_xr64_teacher_target_construction.sh
test -f data/_internal/manifests/manifest1/train_manifest.jsonl
test -f data/_internal/manifests/manifest1/val_manifest.jsonl
test -f data/_internal/manifests/manifest1/test_manifest.jsonl
test -f runs/interpolated_checkpoints/xr39_mixedleader_soup_c25p45f30.pt
test -f runs/XR-62/xr62_xr56b_facetaux_p10_preserve_c255000_lr3e_7_g32_hm0_004_off0_0015_c0_0025_p10soft0_004_p5soft0_0020_auxc0_0006_auxa0_0125_auxt0_005_s0_00035_m160k_M384k_r4000kus_p0p5_20260618_003308/train/best_track_p10.pt
test -f runs/XR-56/xr56_xr52seed_xr39teacher_directp10p5soft_c255000_lr5e_7_g32_hm0_004_off0_0015_c0_0030_p10soft0_008_p5soft0_0020_s0_00035_m160k_M384k_r4000kus_p0p5_20260617_072340/train/best_track_p5.pt
test -f runs/XR-58/xr58_xr39self_p10teacher_anchored_c255000_lr1e_6_g32_hm0_004_off0_0015_c0_0005_p10soft0_014_p5soft0_0_disttrue_s0_00010_m160k_M384k_r4000kus_p0p5_20260617_082242/train/best_track_p10.pt
```

Optional CUDA readiness check:

```bash
.venv/bin/python scripts/external/check_raw_event_count_training_readiness.py --require-cuda
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
```

Current pause-state checker:

```bash
.venv/bin/python scripts/external/check_xr64_resume_artifacts.py --allow-missing-generated
.venv/bin/python scripts/external/check_xr64_resume_artifacts.py --allow-missing-generated --format summary
```

This should pass source preconditions and report missing generated XR-64 artifacts as warnings.

Reproducibility snapshot before execution:

```bash
git rev-parse HEAD
.venv/bin/python - <<'PY'
import torch, sys
print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
PY
df -h .
```

## Safe Resume Mode

Use split/teacher units first. This avoids the previous all-in-one failure mode where the helper stopped during `train/xr62a` and produced no rows.

### 1. Generate Train Eval Rows

Run one teacher at a time:

```bash
XR64_ACTIONS=eval XR64_SPLITS=train XR64_TEACHERS=xr62a XR64_EVAL_DEVICE=cuda:0 \
  bash scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh

XR64_ACTIONS=eval XR64_SPLITS=train XR64_TEACHERS=xr39 XR64_EVAL_DEVICE=cuda:0 \
  bash scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh

XR64_ACTIONS=eval XR64_SPLITS=train XR64_TEACHERS=xr56b XR64_EVAL_DEVICE=cuda:0 \
  bash scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh

XR64_ACTIONS=eval XR64_SPLITS=train XR64_TEACHERS=xr58a XR64_EVAL_DEVICE=cuda:0 \
  bash scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh
```

Expected files:

```text
data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/train/xr62a/eval_rows.json
data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/train/xr39/eval_rows.json
data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/train/xr56b/eval_rows.json
data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/train/xr58a/eval_rows.json
```

### 2. Generate Val Eval Rows

```bash
XR64_ACTIONS=eval XR64_SPLITS=val XR64_TEACHERS=xr62a XR64_EVAL_DEVICE=cuda:0 \
  bash scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh

XR64_ACTIONS=eval XR64_SPLITS=val XR64_TEACHERS=xr39 XR64_EVAL_DEVICE=cuda:0 \
  bash scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh

XR64_ACTIONS=eval XR64_SPLITS=val XR64_TEACHERS=xr56b XR64_EVAL_DEVICE=cuda:0 \
  bash scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh

XR64_ACTIONS=eval XR64_SPLITS=val XR64_TEACHERS=xr58a XR64_EVAL_DEVICE=cuda:0 \
  bash scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh
```

Expected files:

```text
data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/val/xr62a/eval_rows.json
data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/val/xr39/eval_rows.json
data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/val/xr56b/eval_rows.json
data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/val/xr58a/eval_rows.json
```

### 3. Build Override JSON Files

Build train/val overrides only after all eval rows exist:

```bash
XR64_ACTIONS=build XR64_SPLITS=train \
  bash scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh

XR64_ACTIONS=build XR64_SPLITS=val \
  bash scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh
```

Expected train files:

```text
data/_internal/manifests/manifest1/xr64_teacher_targets/xr64a_conservative_train_overrides.json
data/_internal/manifests/manifest1/xr64_teacher_targets/xr64b_threshold_train_overrides.json
data/_internal/manifests/manifest1/xr64_teacher_targets/xr64c_minerror_train_overrides.json
```

Expected val files:

```text
data/_internal/manifests/manifest1/xr64_teacher_targets/xr64a_conservative_val_overrides.json
data/_internal/manifests/manifest1/xr64_teacher_targets/xr64b_threshold_val_overrides.json
data/_internal/manifests/manifest1/xr64_teacher_targets/xr64c_minerror_val_overrides.json
```

## Validation Checks

Fail-fast file check after eval generation:

```bash
for p in \
  data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/train/xr62a/eval_rows.json \
  data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/train/xr39/eval_rows.json \
  data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/train/xr56b/eval_rows.json \
  data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/train/xr58a/eval_rows.json \
  data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/val/xr62a/eval_rows.json \
  data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/val/xr39/eval_rows.json \
  data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/val/xr56b/eval_rows.json \
  data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/val/xr58a/eval_rows.json
do
  test -s "$p" || { echo "missing eval rows: $p" >&2; exit 1; }
done
```

Fail-fast file check after override generation:

```bash
for p in \
  data/_internal/manifests/manifest1/xr64_teacher_targets/xr64a_conservative_train_overrides.json \
  data/_internal/manifests/manifest1/xr64_teacher_targets/xr64a_conservative_val_overrides.json \
  data/_internal/manifests/manifest1/xr64_teacher_targets/xr64b_threshold_train_overrides.json \
  data/_internal/manifests/manifest1/xr64_teacher_targets/xr64b_threshold_val_overrides.json \
  data/_internal/manifests/manifest1/xr64_teacher_targets/xr64c_minerror_train_overrides.json \
  data/_internal/manifests/manifest1/xr64_teacher_targets/xr64c_minerror_val_overrides.json
do
  test -s "$p" || { echo "missing override: $p" >&2; exit 1; }
done
```

Use this read-only JSON count check after generation:

```bash
.venv/bin/python -c '
import json, pathlib
paths = [
  "data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/train/xr62a/eval_rows.json",
  "data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/train/xr39/eval_rows.json",
  "data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/train/xr56b/eval_rows.json",
  "data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/train/xr58a/eval_rows.json",
  "data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/val/xr62a/eval_rows.json",
  "data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/val/xr39/eval_rows.json",
  "data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/val/xr56b/eval_rows.json",
  "data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/val/xr58a/eval_rows.json",
  "data/_internal/manifests/manifest1/xr64_teacher_targets/xr64a_conservative_train_overrides.json",
  "data/_internal/manifests/manifest1/xr64_teacher_targets/xr64a_conservative_val_overrides.json",
  "data/_internal/manifests/manifest1/xr64_teacher_targets/xr64b_threshold_train_overrides.json",
  "data/_internal/manifests/manifest1/xr64_teacher_targets/xr64b_threshold_val_overrides.json",
  "data/_internal/manifests/manifest1/xr64_teacher_targets/xr64c_minerror_train_overrides.json",
  "data/_internal/manifests/manifest1/xr64_teacher_targets/xr64c_minerror_val_overrides.json",
]
for p in paths:
    path = pathlib.Path(p)
    data = json.load(path.open())
    n = len(data.get("overrides", data)) if isinstance(data, dict) else len(data)
    print(f"{p}\\t{n}")
    assert n > 0
'
```

After all artifacts are generated, run strict checker:

```bash
.venv/bin/python scripts/external/check_xr64_resume_artifacts.py
.venv/bin/python scripts/external/check_xr64_resume_artifacts.py --format summary
```

Strict checker must return `0` before launching XR-64A/B.

Required summary state before launch:

```text
resume_status: ready_to_train
ready_to_train: true
can_run_lane: true
leakage_risk: none
missing_eval_rows: 0
missing_overrides: 0
```

Minimum acceptance:

- all row counts are nonzero.
- train override files are used only for training.
- val override files are retained for analysis/diagnostics. Current `run_xr64_teacher_target_construction.sh` uses the train override path for training and does not consume the val override path directly.
- no test override file exists.
- generated override summaries should have `sample_count > 0`.
- generated override `selection_counts` should sum to `sample_count`.

## Launch Order After Artifacts Exist

Primary lanes:

```bash
bash scripts/external/run_xr64_teacher_target_construction.sh a cuda:0
bash scripts/external/run_xr64_teacher_target_construction.sh b cuda:1
```

Diagnostic lane only after A/B:

```bash
bash scripts/external/run_xr64_teacher_target_construction.sh c cuda:0
```

After each lane starts, record:

```bash
tail -n 80 runs/_logs/xr64_teacher_target_<lane>_gpu*.log
```

Required log keys:

- `XR64_START`
- `XR64_TRAIN_EXIT`
- `XR64_RUN`
- `XR64_EVAL_EXIT`

Runner caveat:

- `run_xr64_teacher_target_construction.sh` finds `RUN_ROOT` with `ls -td runs/${EXPERIMENT}_*`.
- If repeated same-tag runs exist, verify the selected `XR64_RUN` path by timestamp and checkpoint presence before using it for conclusions.

## Lane Matrix

| Lane | Rule | Init | Teacher | LR | Target override weight | Purpose |
|---|---|---|---|---:|---:|---|
| XR-64A | conservative | XR-62A center | XR-39 P10 | `3e-7` | `0.0006` | center-safe oracle transfer |
| XR-64B | threshold-priority | XR-56B P5 | XR-39 P10 | `5e-7` | `0.0008` | P10/P5 recovery |
| XR-64C | min-error | XR-62A center | XR-39 P10 | `3e-7` | `0.0010` | overfit-risk diagnostic |

## Leakage Guards

Required:

- never create test target overrides.
- never train with `data.allow_test_target_override=true`.
- final test eval must use `data.track_target_override_path=null`.
- treat XR-63 as oracle evidence only, not promotion evidence.

Current runner behavior:

- prep helper refuses `test` split.
- training runner requires train override file.
- final eval overrides target path to `null`.

## Promotion Gates

Promote if full-test result beats any active gate:

| Metric | Promotion gate |
|---|---:|
| center | `<16.468481131962367` |
| P10 | `>35.02295998845781` |
| P5 | `>12.133503770828247` |

Tradeoff policy:

- center improvement with severe P10/P5 collapse is diagnostic, not clean promotion.
- P10 improvement with center/P5 collapse needs explicit tradeoff note.
- P5 improvement without P10 movement is useful but does not solve strict P10 gap.

## Stop Conditions

Stop or deprioritize lane if:

- eval rows fail validation.
- override JSON row count is zero.
- training cannot find target override file.
- validation center worsens by `>0.3 px` without P10/P5 improvement.
- generated logs show target override loaded from test split.

## Reporting Template

After each lane:

```text
Lane:
Run root:
Init:
Teacher:
Override:
Best checkpoint:
Test center/P10/P5:
Gate promoted:
Tradeoff:
Decision:
```

## Current State

At document creation:

- no train/val XR-64 eval rows exist.
- no XR-64 override JSON files exist.
- no XR-64A/B/C train run exists.
- experiments remain paused.
