# XR-66 No-Train Failure Bucket Diagnostic - 2026-06-22

## Purpose

Run the first XR-66 diagnostic without training. The goal is to identify where XR-64C/XR-65 still fail after XR-65A/B did not promote P10/P5.

## Inputs

Baseline candidate:

- XR-64C best P10 eval rows:
  `runs/eval_fixed255k_xr64_c_xr62a_minerror_teacher_target_adamw_lr3e_7_to0_0010_best_track_p10_test_gpu0_w0_20260622_053934/eval/test/eval_rows.json`

XR-65 candidate:

- XR-65A best P5 eval rows:
  `runs/eval_fixed255k_xr65_a_xr64c_p10init_xr39teacher_threshold_recovery_adamw_lr2e_7_best_track_p5_test_gpu0_w0_20260622_061131/eval/test/eval_rows.json`

Generated diagnostics:

- `runs/diagnostics/xr66_failure_buckets_xr64c_bestp10_20260622.json`
- `runs/diagnostics/xr66_failure_buckets_xr65a_bestp5_20260622.json`

## Overall Comparison

| Candidate | Weighted center | Weighted P10 | Weighted P5 | Unweighted center | Unweighted P10 | Unweighted P5 |
|---|---:|---:|---:|---:|---:|---:|
| XR-64C best P10 | `16.401954641998156` | `34.491568727644356` | `12.161471640265713` | `16.4705002456678` | `33.64611260053619` | `11.483467381590707` |
| XR-65A best P5 | `16.403599453701027` | `34.4404701073071` | `12.212570260602964` | `16.471590326414614` | `33.64611260053619` | `11.57283288650581` |

Interpretation:

- XR-65A improves weighted P5 by `+0.051098620337251`.
- XR-65A worsens weighted center by `+0.001644811702871`.
- XR-65A worsens weighted P10 by `-0.051098620337258`.
- The changes are tiny and do not alter the failure structure.

## Main Failure Buckets

Similarity:

- Low similarity remains the largest broad failure bucket.
- XR-65A `similarity <=0.1`: weighted center `18.23511903032969`, weighted P10 `34.204275534441805`, weighted P5 `11.63895486935867`.
- XR-64C `similarity <=0.1`: weighted center `18.25965418000685`, weighted P10 `34.204275534441805`, weighted P5 `11.63895486935867`.
- Conclusion: XR-65 did not materially improve the low-similarity bucket.

Subject:

- Worst subject buckets remain `subject_id:42`, `39`, and `45`.
- XR-65A `subject_id:42`: weighted center `21.522116838454885`, weighted P10 `22.22222222222222`, weighted P5 `8.771929824561404`.
- XR-65A `subject_id:39`: weighted center `19.767562116648016`, weighted P10 `21.25984251968504`, weighted P5 `7.086614173228346`.
- XR-65A `subject_id:45`: weighted center `19.621061138880133`, weighted P10 `22.875816993464053`, weighted P5 `8.49673202614379`.

Session:

- Worst session buckets remain concentrated in `session_201` for users `45`, `42`, and `39`.
- XR-65A worst weighted sessions:
  - `user45/right/session_201`: center `24.610347196404984`, P10 `6.896551724137931`, P5 `3.4482758620689653`.
  - `user42/left/session_201`: center `24.2453010809484`, P10 `17.073170731707318`, P5 `7.317073170731708`.
  - `user42/right/session_201`: center `23.481124376130833`, P10 `17.94871794871795`, P5 `5.128205128205129`.

Eye side:

- Left eye remains worse than right eye.
- XR-65A left: weighted center `16.59736650837612`, weighted P10 `33.75262054507338`, weighted P5 `11.740041928721174`.
- XR-65A right: weighted center `16.219298586143672`, weighted P10 `35.094715852442675`, weighted P5 `12.662013958125623`.

## Decision

Do not train a generic XR-66 local fallback yet. The diagnostic points to concentrated failure by subject/session/left-eye and low-similarity buckets, not a global confidence cutoff that is obviously separable from the existing outputs.

Next best action:

1. Build a targeted failure-bucket manifest for the worst sessions/subjects and low-similarity rows.
2. Run a no-train side-by-side comparison across XR-64C, XR-65A, and previous P10 leader XR-39 on those buckets.
3. If the bucket comparison shows a consistent teacher advantage, then train a local or bucket-weighted fallback with a strict full-test fallback-off gate.

## Targeted Manifest Status

Generated targeted manifests:

| Bucket | Path | Splits | Rows |
|---|---|---|---:|
| Low similarity `similarity <= 0.1` | `data/_internal/manifests/manifest1/xr66_failure_buckets/low_similarity_le0p1` | train / val / test | `1624 / 254 / 493` |
| `user45/right/session_201` | `data/_internal/manifests/manifest1/xr66_failure_buckets/user45_right_session201_testonly` | test only | `31` |
| `user42/left/session_201` | `data/_internal/manifests/manifest1/xr66_failure_buckets/user42_left_session201_testonly` | test only | `46` |
| `user42/right/session_201` | `data/_internal/manifests/manifest1/xr66_failure_buckets/user42_right_session201_testonly` | test only | `46` |

Important caveat:

- The worst session buckets are test-only in the current split. They are valid for targeted evaluation, but not valid as train/val fine-tuning manifests.
- The only currently trainable targeted XR-66 bucket is the low-similarity subset.

Next execution rule:

1. Compare XR-39, XR-64C, and XR-65A on the low-similarity subset and the three test-only worst-session subsets.
2. Do not start XR-66 training until that comparison shows which teacher/checkpoint actually wins inside the failure buckets.
3. If training is justified, start with low-similarity bucket weighting or low-similarity curriculum, then require full-test promotion against XR-64C/XR-65 gates.

## Commands

```bash
.venv/bin/python scripts/external/summarize_eval_failure_buckets.py \
  --config configs/external/mode1_stage2_raw_event_count_lr1e-4_weakdistill_trackonly_heatmapstate_supportadaptive_headonly_fullwidth.yaml \
  --mode mode1 \
  --stage stage2 \
  --split test \
  --manifest data/_internal/manifests/manifest1/test_manifest.jsonl \
  --eval-rows runs/eval_fixed255k_xr64_c_xr62a_minerror_teacher_target_adamw_lr3e_7_to0_0010_best_track_p10_test_gpu0_w0_20260622_053934/eval/test/eval_rows.json \
  --state-key track_state \
  --override data.canonical_root=/home/kjm26/project/dataset/EV_Eye/canonical \
  --override data.mode1.event_builder.policy=fixed_count \
  --override data.mode1.event_builder.event_count_target=255000 \
  --override data.mode1.event_builder.adaptive_count.enabled=true \
  --override data.mode1.event_builder.adaptive_count.reference_us=4000003 \
  --override data.mode1.event_builder.adaptive_count.min_event_count=160000 \
  --override data.mode1.event_builder.adaptive_count.max_event_count=384000 \
  --override data.mode1.event_builder.adaptive_count.scale_power=0.5 \
  --output runs/diagnostics/xr66_failure_buckets_xr64c_bestp10_20260622.json

.venv/bin/python scripts/external/summarize_eval_failure_buckets.py \
  --config configs/external/mode1_stage2_raw_event_count_lr1e-4_weakdistill_trackonly_heatmapstate_supportadaptive_headonly_fullwidth.yaml \
  --mode mode1 \
  --stage stage2 \
  --split test \
  --manifest data/_internal/manifests/manifest1/test_manifest.jsonl \
  --eval-rows runs/eval_fixed255k_xr65_a_xr64c_p10init_xr39teacher_threshold_recovery_adamw_lr2e_7_best_track_p5_test_gpu0_w0_20260622_061131/eval/test/eval_rows.json \
  --state-key track_state \
  --override data.canonical_root=/home/kjm26/project/dataset/EV_Eye/canonical \
  --override data.mode1.event_builder.policy=fixed_count \
  --override data.mode1.event_builder.event_count_target=255000 \
  --override data.mode1.event_builder.adaptive_count.enabled=true \
  --override data.mode1.event_builder.adaptive_count.reference_us=4000003 \
  --override data.mode1.event_builder.adaptive_count.min_event_count=160000 \
  --override data.mode1.event_builder.adaptive_count.max_event_count=384000 \
  --override data.mode1.event_builder.adaptive_count.scale_power=0.5 \
  --output runs/diagnostics/xr66_failure_buckets_xr65a_bestp5_20260622.json
```
