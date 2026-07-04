# Stage1 Promoted Baseline Downstream Plan - 2026-06-22

## Purpose

Record the Stage1 frame-based Search checkpoint that should seed downstream HGTXR software experiments, and define the next safe execution step.

## Active Baseline

- State: `promoted_followup`
- Run: `runs/NON_XR/raw/stage1_frame_search_centerpolish_selfdistill_lr7p5e6_xy2_ema999_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949`
- Checkpoint: `runs/NON_XR/raw/stage1_frame_search_centerpolish_selfdistill_lr7p5e6_xy2_ema999_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949/train/best_search_p10.pt`
- Source manifest: `docs/resources/current_stage1_frame_search_baseline_manifest.json`
- Gate: minimum `50` epochs satisfied; final manifest refresh observed the promoted lane at `80/80`.

## Promotion Evidence

Previous lane-B baseline:

- Run: `runs/NON_XR/raw/stage1_frame_search_sonly_adamw_lr5e4_xy1p5_fullwidth_20260621_204500`
- Best Search P10: `28.22439415050003`
- Best Search P5: `9.833782825829848`
- Best Search center px: `17.28884926831947`

Promoted self-distill center-polish lane:

- Best Search P10: `28.27717937613433`
- Best Search P5: `10.153863744915656`
- Best Search center px: `17.257583906065744`
- Delta: P10 `+0.052785225634302435 pp`, P5 `+0.3200809190858074 pp`, center `+0.03126536225372689 px` improvement.

No-distill center-polish lane is not the baseline despite higher P10 because its center error regresses:

- Best Search P10: `28.303010526693093`
- Best Search center px: `17.3741187104639`
- Center delta: `-0.08526944214442977 px` improvement, meaning worse than lane B.

## Current Runtime State

The Stage1 center-polish jobs completed `80/80`. Final refresh at `2026-06-22 03:18:35 KST`:

- GPU0: no-distill lane completed `80/80`.
- GPU1: self-distill lane completed `80/80`.
- Active Stage1 train processes: none.
- GPU state after completion: GPU0 `18 MiB` used / `15824 MiB` free; GPU1 `18 MiB` used / `15824 MiB` free.

Downstream planning should use the promoted checkpoint immediately. New GPU-heavy work is no longer blocked by Stage1 occupancy, but XR-64 training remains blocked until generated artifacts exist and the strict checker reports readiness.

## Stage1 Baseline Improvement Continuation - 2026-06-22

User direction changed the immediate priority back to Stage1: improve frame-based Search accuracy sufficiently and use the result as the baseline before further Stage2/XR downstream promotion work.

Current active baseline remains the promoted Lane J checkpoint above. Because the prior capacity/distillation branch collapsed badly, the next Stage1 work avoids larger-capacity expansion and uses conservative low-LR polishing from the promoted checkpoint.

New 2-GPU Stage1 polish run launched at `2026-06-22 10:39:14 KST`:

- Seed checkpoint:
  `runs/NON_XR/raw/stage1_frame_search_centerpolish_selfdistill_lr7p5e6_xy2_ema999_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949/train/best_search_p10.pt`
- Shared settings:
  - `stage=stage1`
  - `active_head=search`
  - enabled heads: Search only
  - `epochs=120`
  - `batch_size=8`
  - `num_workers=4`
  - optimizer: `AdamW`
  - scheduler: cosine
  - `warmup_epochs=2`
  - `min_lr=2.5e-7`
  - `weight_decay=5e-5`
  - pruning disabled
  - structural width ratio `1.0`
  - AMP disabled
  - `HBTXR_DISABLE_CUDNN=1`

Lane C: P10-polish no-distill branch.

- Run:
  `runs/NON_XR/raw/stage1_frame_search_promoted_p10polish_nodistill_lr3e6_xy2p25_20260622_103916`
- PID at launch: `2526793`
- GPU: `cuda:0`
- LR: `3e-6`
- `loss.search_xy_weight=2.25`
- `training.best_metric_name=metric_search_p10_pct`
- Distillation: disabled
- Log:
  `runs/NON_XR/shared/_logs/stage1_frame_search_warmstart_lane_c_stage1_promoted_polish_20260622_103914.log`

Lane D: center/P5-preserving weak self-distill branch.

- Run:
  `runs/NON_XR/raw/stage1_frame_search_promoted_selfdistill_lr2e6_xy2_ema9995_20260622_103916`
- PID at launch: `2526794`
- GPU: `cuda:1`
- LR: `2e-6`
- `loss.search_xy_weight=2.0`
- `training.best_metric_name=metric_search_p10_pct`
- Distillation: enabled
- Teacher init checkpoint: same as seed checkpoint
- EMA decay: `0.9995`
- Feature/state weights: `0.003`
- Prediction/mask weights: `0.0015`
- Log:
  `runs/NON_XR/shared/_logs/stage1_frame_search_selfdistill_lane_d_stage1_promoted_polish_20260622_103914.log`

Launch verification:

- Contract check passed for `data/_internal/manifests/manifest1`:
  train `5929`, val `844`, test `2238`.
- Torch CUDA smoke passed with `torch_cuda_device_count=2`.
- `nvidia-smi` before launch showed both RTX 5080 GPUs essentially idle.
- Post-launch logs show both lanes loaded the seed checkpoint with `loaded_count=142`, `partial=0`, `skipped=0`.
- Both lanes entered epoch `1/120`.

Promotion gate for this continuation:

1. Do not judge before at least `50` epochs.
2. Prefer a candidate only if it improves Search P10 over `28.27717937613433`.
3. Reject if the center error regresses materially beyond the current baseline center `17.257583906065744 px` unless P10 gain is large enough to justify a separate diagnostic branch.
4. Keep the current promoted Lane J checkpoint as the baseline until one of these new lanes beats it under the same reporter gate.

Watcher added at `2026-06-22 10:44:30 KST`:

- Host watcher PID: `2556516`
- Watcher log:
  `runs/NON_XR/shared/_logs/stage1_promoted_polish_min50_watch_20260622_host_104430.log`
- Watcher report JSON:
  `runs/NON_XR/shared/_logs/stage1_promoted_polish_min50_watch_20260622_host_104430_report.json`
- Watcher report summary:
  `runs/NON_XR/shared/_logs/stage1_promoted_polish_min50_watch_20260622_host_104430_report.txt`
- Poll interval: `300` seconds
- Stop condition: both follow-up runs reach at least `50` epochs or a promotion status appears.

First watcher report:

- State: `wait_min_epochs`
- Both follow-ups reached epoch `7/120`.
- Lane D current best: P10 `27.788634912023003`, P5 `9.533917607001538`, center `17.328227780899912`.
- Lane C current best: P10 `27.423630372533257`, P5 `9.382300358898235`, center `17.351748246066975`.
- Neither lane is currently promotable; continue training.

## XR-64 Readiness

Current XR-64 artifact check after completing XR-64 prep:

- `resume_status=ready_to_train`
- `ready_to_train=true`
- `can_run_lane=true`
- Missing train/val teacher eval rows: `0`
- Missing override JSON files: `0`
- Leakage risk: `none`
- Completed train eval rows: `xr62a/xr39/xr56b/xr58a`, each with `5929` rows.
- Completed val eval rows: `xr62a/xr39/xr56b/xr58a`, each with `844` rows.
- Completed override files: train/val `xr64a_conservative`, `xr64b_threshold`, and `xr64c_minerror`.
- XR-64A/B were relaunched with host GPU access after confirming the sandbox did not expose `/dev/nvidia*`.
- XR-64A/B train/eval completed and the six-candidate post-run matrix is complete.
- Promotion helper reports `decision_status=promoted`, `winning_lane=XR-64B`, `winning_checkpoint_kind=best_track_p5`, `center_promoted=true`, `p10_promoted=false`, and `p5_promoted=false`.
- XR-64C min-error diagnostic also completed after fixing Stage2 center-checkpoint saving.
- Best observed center result is XR-64C `best_track_p10`: center `16.464661524977004`, P10 `34.3554429258619`, P5 `12.044218056542533`, P1 `1.0153061594281878`.

Next prepared direction:

```bash
# Dry-run the next P10/P5 recovery lane from the XR-64C center-improved result.
DRY_RUN=1 bash scripts/external/run_xr65_p10p5_recovery_from_xr64c.sh a cuda:0
```

The next lane should preserve center near `16.46 px` while recovering P10 above `35.02295998845781` and P5 above `12.133503770828247`.

## Experiment Priority

1. Use the promoted Stage1 Lane J checkpoint as the frame-search baseline for downstream work.
2. Use XR-64C evidence as the new center-improved basis.
3. Launch XR-65 P10/P5 recovery from XR-64C only if it preserves the new center level.
4. If the recovery lane does not close the gap, move to paper-backed follow-ups:
   - XR-65 temporal-lite residual or self-distill from 3ET/AIS/ConvLSTM/SSM references.
   - XR-66 confidence-gated local update from EX-Gaze/Swift-Eye/E-BTS references.
   - FACET/EV-Eye local geometry or teacher-target variants only when diagnostics show the failure mode.

## XR-65 Recovery Runner

Prepared script:

```bash
bash scripts/external/run_xr65_p10p5_recovery_from_xr64c.sh <a|b> [device]
```

Default lanes:

- Lane A: XR-64C `best_track_p10.pt` init, XR-39 P10 teacher, threshold override pressure, LR `2e-7`.
- Lane B: XR-64C `best_track_p10.pt` init, XR-56B P5 teacher, conservative override pressure, LR `2e-7`.

Safety contract:

- Train-time target override is train-only.
- Test eval forces `data.track_target_override_path=null`.
- Test eval forces `data.allow_test_target_override=false`.
- Test eval forces target-override loss weights back to `0.0`.
- Promotable runs require all three checkpoint kinds: `best_track_p10`, `best_track_p5`, and `best_metric_track_center_px`.
- Treat center drift above `16.764661524977003` as rejection unless explicitly labeled diagnostic.

## Verification Commands

```bash
.venv/bin/python scripts/external/report_stage1_frame_search_followup.py \
  --baseline runs/NON_XR/raw/stage1_frame_search_sonly_adamw_lr5e4_xy1p5_fullwidth_20260621_204500 \
  --min-epochs 50 \
  --target-epochs 80 \
  --followup runs/NON_XR/raw/stage1_frame_search_centerpolish_nodistill_lr1e5_xy2_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949 \
  --followup runs/NON_XR/raw/stage1_frame_search_centerpolish_selfdistill_lr7p5e6_xy2_ema999_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949 \
  --format summary

.venv/bin/python scripts/external/write_stage1_frame_search_baseline_manifest.py --format summary
.venv/bin/python scripts/external/check_xr64_resume_artifacts.py --allow-missing-generated --format summary
.venv/bin/python scripts/external/emit_xr64_next_prep_command.py --format commands
bash -n scripts/external/run_xr65_p10p5_recovery_from_xr64c.sh
DRY_RUN=1 bash scripts/external/run_xr65_p10p5_recovery_from_xr64c.sh a cuda:0
.venv/bin/python -m pytest -q tests/test_xr65_runner_contract.py tests/test_trainer_checkpoint_specs.py
```
