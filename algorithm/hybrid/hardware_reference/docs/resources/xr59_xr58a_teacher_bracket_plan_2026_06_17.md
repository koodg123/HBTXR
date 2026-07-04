# XR-59 XR58A Teacher Bracket Plan

Date: 2026-06-17

## Prompt Pack

- Objective: close the remaining P10 gap from XR-58A best-P10 `34.90306201662336` to the active P10 gate `35.02295998845781`.
- Source artifacts: XR-58 closeout, `runs/xr58_xr39self_p10teacher_anchored_c255000_lr1e_6_g32_hm0_004_off0_0015_c0_0005_p10soft0_014_p5soft0_0_disttrue_s0_00010_m160k_M384k_r4000kus_p0p5_20260617_082242/train/best_track_p10.pt`, XR-39 P10 teacher.
- IDEA-Gen phase: P1.
- Domain: Research Workflow.
- Parent skill: `paper-idea-generator`.
- Target model/dataset: HGTXR mode1 stage2, fixed255k support-adaptive event-count, manifest1 train/val/test.
- Constraints: use full-test P10 as adoption metric; preserve active gates as comparison only, not as required for teacher candidate generation.
- Assumptions: XR-58A's anchored branch is closer to a usable P10 teacher than XR-58B; continuation from XR-58A is lower-risk than restarting from XR-39.
- Required artifact: two-lane runner, validation evidence, full-test closeout.
- Verification evidence: `bash -n`, lane A/B dry-run, train/eval exit `0`, full-test metric JSON.
- Risks: continuation may overfit validation P10, center/P5 may regress, and XR-39 may remain a hard P10 ceiling under current data/protocol.

## Ablation Matrix

| Lane | GPU | Init | Teacher | LR | P10 soft | Temp | Center L2 | State distill | Intent |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| A | 0 | XR-58A best-P10 | XR-39 P10 | `7.5e-7` | `0.016` | `0.9` | `0.0004` | `0.00008` | modest LR continuation |
| B | 1 | XR-58A best-P10 | XR-39 P10 | `5e-7` | `0.022` | `0.75` | `0.0003` | `0.00008` | stronger P10-soft pressure |

Control variables:

- Fixed255k support-adaptive window: `160k/255k/384k`, reference `4000003us`, scale power `0.5`.
- Heatmap representation: grid `32`, heatmap weight `0.004`, offset weight `0.0015`.
- Trainable scope: heatmap head, track head, event path, backbone stages 4/5, backbone norms.
- Optimizer: AdamW.
- Best/scheduler metric: `metric_track_p10_pct`.

## Runner

- `scripts/external/run_xr59_xr58a_teacher_bracket.sh`
- Post-train checkpoint eval helper: `scripts/external/eval_xr59_completed_checkpoints.sh`

Default commands:

```bash
bash scripts/external/run_xr59_xr58a_teacher_bracket.sh a cuda:0
bash scripts/external/run_xr59_xr58a_teacher_bracket.sh b cuda:1
```

## Validation Checklist

- [x] `bash -n scripts/external/run_xr59_xr58a_teacher_bracket.sh`
- [x] `DRY_RUN=1 bash scripts/external/run_xr59_xr58a_teacher_bracket.sh a cuda:0`
- [x] `DRY_RUN=1 bash scripts/external/run_xr59_xr58a_teacher_bracket.sh b cuda:1`
- [x] `bash -n scripts/external/eval_xr59_completed_checkpoints.sh`
- [x] actual lane A/B train completed
- [x] full-test checkpoint eval completed
- [x] full-test gate decision

## Decision Rule

Promote a teacher candidate only if full-test P10 exceeds `35.02295998845781`. If XR-59 misses, close scalar P10-soft teacher continuation and pivot to different P10 representation or data/protocol diagnostics.

## Closeout

XR-59 completed with no promotion. Both train lanes early-stopped at epoch `7/10`.

| Lane | Checkpoint | Epoch | Center px | P10 % | P5 % | Gate decision |
|---|---|---:|---:|---:|---:|---|
| A | `best_track_p10` | 1 | 16.520220368249074 | 34.300170864377705 | 11.376701021194458 | no promotion |
| A | `best_track_p5` | 1 | 16.520220368249074 | 34.300170864377705 | 11.376701021194458 | no promotion |
| B | `best_track_p10` | 1 | 16.51065547806876 | 34.43409944261823 | 11.287415306908743 | no promotion |
| B | `best_track_p5` | 1 | 16.51065547806876 | 34.43409944261823 | 11.287415306908743 | no promotion |

Evaluation summaries:

- `runs/eval_fixed255k_xr59_a_xr58a_continue_lrbracket_adamw_lr7_5e_7_g32_hm0_004_off0_0015_c0_0004_p10soft0_016_disttrue_state0_00008_best_track_p10_test_gpu0_w0_20260617_165302/eval/test/eval_summary.json`
- `runs/eval_fixed255k_xr59_a_xr58a_continue_lrbracket_adamw_lr7_5e_7_g32_hm0_004_off0_0015_c0_0004_p10soft0_016_disttrue_state0_00008_best_track_p5_test_gpu0_w0_20260617_165608/eval/test/eval_summary.json`
- `runs/eval_fixed255k_xr59_b_xr58a_continue_p10pressure_adamw_lr5e_7_g32_hm0_004_off0_0015_c0_0003_p10soft0_022_disttrue_state0_00008_best_track_p10_test_gpu1_w0_20260617_165301/eval/test/eval_summary.json`
- `runs/eval_fixed255k_xr59_b_xr58a_continue_p10pressure_adamw_lr5e_7_g32_hm0_004_off0_0015_c0_0003_p10soft0_022_disttrue_state0_00008_best_track_p5_test_gpu1_w0_20260617_165607/eval/test/eval_summary.json`

Decision: XR-59 regressed from XR-58A's P10 `34.90306201662336` to best XR-59 P10 `34.43409944261823`. Close the scalar P10-soft teacher-continuation family. Next P0 should change representation or protocol, for example a dedicated P10 calibration/classification head, subject/session failure-bucket data diagnostics, or a teacher trained with a different target construction.
