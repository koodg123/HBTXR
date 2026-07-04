# XR-64 Teacher-Target Construction Implementation

## Summary

XR-64 now has the code path required for leakage-safe teacher-target training.

Implemented:

- `scripts/external/build_xr64_teacher_target_overrides.py`
- `scripts/external/run_xr64_teacher_target_construction.sh`
- `data.track_target_override_path`
- `data.allow_test_target_override`
- additive dataset fields:
  - `track_target_override_state`
  - `track_target_override_weight`
  - `meta.track_target_override_source`
- additive Stage2 losses:
  - `loss_track_target_override_center_l2`
  - `loss_track_target_override_axis_log`
  - `loss_track_target_override_angle_cos`
  - `loss_track_state_aux_target_override_center_l2`

`cur_state` remains the ground-truth metric target. Override targets are used only through explicit auxiliary loss weights.

## Leakage Guard

- Dataset construction rejects `data.track_target_override_path` on test manifests by default.
- Test split override is only allowed with `data.allow_test_target_override=true`, intended for diagnostics only.
- XR-64 runner explicitly clears target overrides during final test eval:
  - `--override data.track_target_override_path=null`
  - `--override data.allow_test_target_override=false`

## Validation

Passed:

```bash
python3 -m py_compile scripts/external/build_xr64_teacher_target_overrides.py scripts/external/analyze_p10_teacher_target_oracle.py src/hbtxr/data/dataset.py src/hbtxr/config/runtime_config.py src/hbtxr/loss/bundles/track.py src/hbtxr/loss/stage2.py
bash -n scripts/external/run_xr64_teacher_target_construction.sh
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_track_target_override.py tests/test_track_center_l2_loss.py
DRY_RUN=1 bash scripts/external/run_xr64_teacher_target_construction.sh a cuda:0
DRY_RUN=1 bash scripts/external/run_xr64_teacher_target_construction.sh b cuda:1
```

Builder smoke:

```bash
.venv/bin/python scripts/external/build_xr64_teacher_target_overrides.py \
  --reference-config runs/XR-62/eval_fixed255k_xr62_a_xr56b_facetaux_p10_preserve_adamw_lr3e_7_g32_hm0_004_off0_0015_c0_0025_p10soft0_004_p5soft0_0020_auxc0_0006_auxa0_0125_auxt0_005_best_track_p10_test_gpu0_w0_20260618_004307/hypers/resolved_config.json \
  --manifest data/_internal/manifests/manifest1/test_manifest.jsonl \
  --split test \
  --allow-test-diagnostic-output \
  --limit 64 \
  --baseline xr62a \
  --rule conservative \
  --output /tmp/xr64_override_smoke.json \
  --eval xr62a=runs/XR-62/eval_fixed255k_xr62_a_xr56b_facetaux_p10_preserve_adamw_lr3e_7_g32_hm0_004_off0_0015_c0_0025_p10soft0_004_p5soft0_0020_auxc0_0006_auxa0_0125_auxt0_005_best_track_p10_test_gpu0_w0_20260618_004307/eval/test/eval_rows.json \
  --eval xr39=runs/eval_fixed255k_xr39_mixedleader_soup_c25p45f30_gpu1_w0_20260617_010154/eval/test/eval_rows.json \
  --eval xr56b=runs/eval_fixed255k_xr56_b_xr52seed_xr39teacher_directp10p5soft_adamw_lr5e_7_g32_hm0_004_off0_0015_c0_0030_p10soft0_008_p5soft0_0020_state0_00035_best_track_p5_test_gpu1_w0_20260617_073652/eval/test/eval_rows.json \
  --eval xr58a=runs/eval_fixed255k_xr58_a_xr39self_p10teacher_anchored_adamw_lr1e_6_g32_hm0_004_off0_0015_c0_0005_p10soft0_014_p5soft0_0_disttrue_state0_00010_best_track_p10_test_gpu0_w0_20260617_083420/eval/test/eval_rows.json
```

Smoke output:

```json
{
  "split": "test",
  "rule": "conservative",
  "baseline": "xr62a",
  "sample_count": 64,
  "selection_counts": {
    "xr62a": 34,
    "xr39": 14,
    "xr56b": 3,
    "xr58a": 13
  }
}
```

## Remaining Work

Before launching XR-64 training:

1. Generate train eval rows for XR-62A, XR-39, XR-56B, and XR-58A.
2. Build train override files:
   - `data/_internal/manifests/manifest1/xr64_teacher_targets/xr64a_conservative_train_overrides.json`
   - `data/_internal/manifests/manifest1/xr64_teacher_targets/xr64b_threshold_train_overrides.json`
   - optional `xr64c_minerror_train_overrides.json`
3. Launch:
   - `bash scripts/external/run_xr64_teacher_target_construction.sh a cuda:0`
   - `bash scripts/external/run_xr64_teacher_target_construction.sh b cuda:1`

Do not build train overrides from test split.
