# Stage1 Frame-Search Mid-Run Results

Date: 2026-06-21

## Goal

Raise Stage1 frame-based Search accuracy first, then use the improved Stage1 checkpoint as the baseline for downstream Stage2/XR-64 work.

## Agent Plan

Task cards:

```yaml
task_card:
  task_id: T-STAGE1-STATUS
  sub_agent: "codex-native"
  role: "evaluator"
  objective: "Check active Stage1 frame-search jobs, summarize metrics, and preserve evidence."
  file_ownership:
    - "docs/resources/stage1_frame_search_midrun_results_2026_06_21.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/log.md"
  assigned_skill:
    - "caveman:caveman-lite"
  inputs:
    - "runs/NON_XR/raw/stage1_frame_search_sonly_adamw_lr1e3_fullwidth_20260621_204500"
    - "runs/NON_XR/raw/stage1_frame_search_sonly_adamw_lr5e4_xy1p5_fullwidth_20260621_204500"
  outputs:
    - "mid-run metric summary"
    - "baseline promotion recommendation"
  validation:
    - "process status"
    - "nvidia-smi"
    - "summarize_training_history.py"
  dependencies: []
```

```yaml
task_card:
  task_id: T-STAGE1-NEXT
  sub_agent: "gpt5.3-codex-spark"
  role: "research"
  objective: "Propose follow-up Stage1 Search refinement experiments."
  file_ownership: []
  assigned_skill:
    - "caveman:caveman-lite"
  inputs:
    - "Stage1 Search baseline results"
    - "existing pretrained/distillation code paths"
  outputs:
    - "ranked next experiment plan"
  validation:
    - "repo code path citations"
  dependencies: []
```

Real sub-agent spawn for `T-STAGE1-NEXT` was attempted, but the runtime returned `agent thread limit reached`. The main agent completed the analysis and artifact update locally. No sub-agent output is claimed.

## Current Runs

Launch tag: `20260621_204500`

Both jobs are detached with `setsid`, PPID `1`, and `HBTXR_DISABLE_CUDNN=1`.

| Lane | GPU | Run | Main knobs |
|---|---:|---|---|
| A | 0 | `runs/NON_XR/raw/stage1_frame_search_sonly_adamw_lr1e3_fullwidth_20260621_204500` | Search-only, AdamW LR `1e-3`, `loss.search_xy_weight=1.0`, no distill, no pruning, full width |
| B | 1 | `runs/NON_XR/raw/stage1_frame_search_sonly_adamw_lr5e4_xy1p5_fullwidth_20260621_204500` | Search-only, AdamW LR `5e-4`, `loss.search_xy_weight=1.5`, no distill, no pruning, full width |

Last checked process state:

| PID | Lane | State | Elapsed | Status |
|---:|---|---|---|---|
| `1332213` | A | `Ssl` | `02:13:54` | active |
| `1332214` | B | `Rsl` | `02:13:54` | active |

Last checked GPU state:

| GPU | Memory used | Memory free | Utilization |
|---:|---:|---:|---:|
| 0 | `720 MiB` | `15123 MiB` | `15%` |
| 1 | `720 MiB` | `15123 MiB` | `34%` |

The user-required minimum of 50 epochs has been satisfied. The latest report-script check in this turn observed both lanes at 231 summarized epochs while both jobs were still active.

## Baseline Comparison

Old Stage1 reference:

`runs/NON_XR/raw/raw_mode1_stage1_best_adamw255k_200ep_2gpu_20260611_222452`

| Run | Epochs summarized | Best Search P10 | Best Search P5 | Best Search P1 | Best center px |
|---|---:|---:|---:|---:|---:|
| Old Stage1 reference | 200 | `23.3491` @165 | `6.3904` @189 | not used as gate | `18.3195` @154 |
| Lane A completed | 300 | `25.3246` | `8.0526` | latest reporter not used as gate | `17.0595` improvement vs old reference |
| Lane B completed | 300 | `28.2244` @175 | `9.8338` @172 | `1.0647` @189 | `17.2888` @176 |

Lane B is the current baseline candidate.

Lane B improvement over old Stage1 reference:

| Metric | Old | Lane B best | Delta |
|---|---:|---:|---:|
| Search P10 | `23.3491` | `28.2244` | `+4.8753 pp` |
| Search P5 | `6.3904` | `9.8338` | `+3.4434 pp` |
| Search center px | `18.3195` | `17.2888` | `-1.0307 px` |

Important caveat: lane B final promotion should use the saved `best_search_p10.pt`, not `last.pt`. The best Search P10 was reached earlier than epoch 300, and the reporter ranks checkpoints by best validation metric.

## Checkpoint Candidates

Current checkpoint candidates:

- Lane A P10: `runs/NON_XR/raw/stage1_frame_search_sonly_adamw_lr1e3_fullwidth_20260621_204500/train/best_search_p10.pt`
- Lane A P5: `runs/NON_XR/raw/stage1_frame_search_sonly_adamw_lr1e3_fullwidth_20260621_204500/train/best_search_p5.pt`
- Lane B P10: `runs/NON_XR/raw/stage1_frame_search_sonly_adamw_lr5e4_xy1p5_fullwidth_20260621_204500/train/best_search_p10.pt`
- Lane B P5: `runs/NON_XR/raw/stage1_frame_search_sonly_adamw_lr5e4_xy1p5_fullwidth_20260621_204500/train/best_search_p5.pt`

Promotion candidate:

`runs/NON_XR/raw/stage1_frame_search_sonly_adamw_lr5e4_xy1p5_fullwidth_20260621_204500/train/best_search_p10.pt`

## 2026-06-22 Promoted Center-Polish Baseline

The later center-polish follow-up gate has superseded lane B as the active Stage1 frame-search baseline.

Active promoted baseline:

- Run: `runs/NON_XR/raw/stage1_frame_search_centerpolish_selfdistill_lr7p5e6_xy2_ema999_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949`
- Checkpoint: `runs/NON_XR/raw/stage1_frame_search_centerpolish_selfdistill_lr7p5e6_xy2_ema999_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949/train/best_search_p10.pt`
- Gate: at least `50` epochs; final manifest refresh observed the promoted lane at `80/80`.
- Best Search P10: `28.27717937613433`
- Best Search P5: `10.153863744915656`
- Best Search center px: `17.257583906065744`
- Delta versus lane B: P10 `+0.052785225634302435 pp`, P5 `+0.3200809190858074 pp`, center `+0.03126536225372689 px` improvement.

Rejected higher-P10 candidate:

- Run: `runs/NON_XR/raw/stage1_frame_search_centerpolish_nodistill_lr1e5_xy2_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949`
- Best Search P10: `28.303010526693093`
- Best Search center px: `17.3741187104639`
- Decision: not promoted because it regresses center by `0.08526944214442977 px` versus lane B, even though P10 is higher.

Current decision: use the self-distill center-polish `best_search_p10.pt` as the Stage1 frame-search baseline for downstream work. Keep the 80-epoch jobs running to preserve late improvements, but do not wait for 80 epochs before unblocking downstream planning.

Latest final refresh at `2026-06-22 03:18:35 KST`:

- Lane I no-distill: completed `80/80`, best P10 `28.303010526693093`, best P5 `9.884321932522756`, best center `17.3741187104639`; still not promotable because center regresses.
- Lane J self-distill: completed `80/80`, best P10 `28.27717937613433`, best P5 `10.153863744915656`, best center `17.257583906065744`; remains active baseline.
- GPU state: both RTX 5080 GPUs are free after Stage1 completion.

## Follow-Up Experiments

The A/B lanes have now finished 300 epochs. S1C/S1D follow-up refinement is active.

Prepared runner:

`scripts/external/run_stage1_frame_search_warmstart_refine.sh`

Prepared status/promotion reporter:

`scripts/external/report_stage1_frame_search_baseline.py`

Prepared completion watcher:

`scripts/external/run_stage1_frame_search_refine_when_ready.sh`

Original watcher:

- PID: `2087936`
- Log: `runs/NON_XR/shared/_logs/stage1_frame_search_refine_when_ready_stage1_refine_after_300ep_20260621_v2.log`
- PID file: `runs/NON_XR/shared/_logs/stage1_frame_search_refine_when_ready_stage1_refine_after_300ep_20260621_v2.pid`
- Behavior: wait for PIDs `1332213 1332214`, require reporter promotion readiness, require complete 300-epoch candidates, require at least `12000 MiB` free on both GPUs, then launch `scripts/external/run_stage1_frame_search_warmstart_refine.sh`.

Resume status after stalled detached jobs:

- The original A/B PIDs `1332213` and `1332214` and watcher PID `2087936` were no longer alive before 300 epochs.
- No Python traceback, CUDA OOM, or explicit kill text was found in the A/B/watcher logs.
- The reporter still returned `complete_300=False`, so S1C/S1D were correctly blocked by the watcher policy.
- `scripts/external/run_stage1_frame_search_baseline_matrix.sh` now supports `LANE_A_RESUME` and `LANE_B_RESUME`.
- A/B were relaunched outside the Codex sandbox with `RUN_TAG=20260621_2258_resume`.
- Lane A resumed from `last.pt` as PID `2167860`, start epoch `239/300`, log `runs/NON_XR/shared/_logs/stage1_frame_search_lane_a_20260621_2258_resume.log`.
- Lane B resumed from `last.pt` as PID `2167861`, start epoch `238/300`, log `runs/NON_XR/shared/_logs/stage1_frame_search_lane_b_20260621_2258_resume.log`.
- The watcher was relaunched outside the sandbox as PID `2169848`, log `runs/NON_XR/shared/_logs/stage1_frame_search_refine_when_ready_stage1_refine_after_300ep_20260621_resume2258.log`, waiting on PIDs `2167860 2167861`.
- A supervisor was added as an additional guard: `scripts/external/run_stage1_frame_search_supervisor_until_complete.sh`.
- The warm-start runner now has a run-tag scoped lock so watcher and supervisor cannot launch duplicate S1C/S1D jobs for the same `RUN_TAG`.
- The supervisor was launched outside the sandbox as PID `2197421`, log `runs/NON_XR/shared/_logs/stage1_frame_search_supervisor_resume2258.log`, with `MAX_RELAUNCHES=5`.

Latest reporter summary:

```bash
.venv/bin/python scripts/external/report_stage1_frame_search_baseline.py --format summary
```

Result: lane B is ranked first and `promote_checkpoint=true`. The reporter guard keeps `launch_warmstart_refine_when_gpu_free=false` while A/B are incomplete, and flips it to true only after both 300-epoch candidates are complete.

Current completed-report result:

- Lane A: `epochs=300`, `complete_300=True`, best Search P10 `25.324573696784253`.
- Lane B: `epochs=300`, `complete_300=True`, best Search P10 `28.22439415050003`.
- Promotion checkpoint: `runs/NON_XR/raw/stage1_frame_search_sonly_adamw_lr5e4_xy1p5_fullwidth_20260621_204500/train/best_search_p10.pt`.
- `wait_for_300_epoch_completion=False`.
- `launch_warmstart_refine_when_gpu_free=True`.

Active follow-up runs:

- S1C warm-start no-distill, GPU0: `runs/NON_XR/raw/stage1_frame_search_warmstart_lr1e4_xy1p5_stage1_refine_after_300ep_20260621_resume2258_20260621_234220`.
- S1C log: `runs/NON_XR/shared/_logs/stage1_frame_search_warmstart_lane_c_stage1_refine_after_300ep_20260621_resume2258.log`.
- S1D weak self-distill, GPU1: `runs/NON_XR/raw/stage1_frame_search_selfdistill_lr7p5e5_xy1p5_stage1_refine_after_300ep_20260621_resume2258_manual_20260621_234506`.
- S1D log: `runs/NON_XR/shared/_logs/stage1_frame_search_selfdistill_lane_d_stage1_refine_after_300ep_20260621_resume2258_manual.log`.
- The watcher/supervisor launch path has been patched to force `DETACH=1` for future warm-start calls.

Follow-up reporter:

`scripts/external/report_stage1_frame_search_followup.py`

Current reporter decision:

- S1C: completed epoch `67/120` by reporter; best P10 `27.94025216012631`, best P5 `10.049416236157688`, best center `17.233418217245138`.
- S1D: completed epoch `61/120` by reporter; best P10 `27.629156058689333`, best P5 `9.926999344016021`, best center `17.34278841738431`.
- Both have crossed the `min_epochs=50` decision gate.
- Neither has beaten lane B baseline P10 `28.22439415050003`.
- Current status: `keep_baseline_unless_later_improves`; keep lane B as baseline and keep training S1C/S1D.

Min-50 watcher:

- Script: `scripts/external/run_stage1_frame_search_followup_until_min_epoch.sh`
- PID: `2485273`
- Log: `runs/NON_XR/shared/_logs/stage1_frame_search_followup_min50_20260621_235755.log`
- Report summary: `runs/NON_XR/shared/_logs/stage1_frame_search_followup_min50_20260621_235755_report.txt`
- Report JSON: `runs/NON_XR/shared/_logs/stage1_frame_search_followup_min50_20260621_235755_report.json`
- Poll interval: `300s`

Post-followup fallback queue:

- Script: `scripts/external/run_stage1_frame_search_post_followup_queue.sh`
- PID: `2638408`
- Log: `runs/NON_XR/shared/_logs/stage1_frame_search_post_followup_queue_v2_20260622_0027.log`
- Report summary: `runs/NON_XR/shared/_logs/stage1_frame_search_post_followup_queue_v2_20260622_0027_report.txt`
- Report JSON: `runs/NON_XR/shared/_logs/stage1_frame_search_post_followup_queue_v2_20260622_0027_report.json`
- Purpose: if S1C/S1D do not produce a promotable checkpoint, wait for the active follow-up PIDs to exit and for both GPUs to have at least `12000 MiB` free, then launch lower-LR S1E/S1F fallback lanes.
- Safety: it does not stop, resume, or modify the active S1C/S1D jobs.
- Update: the queue now also launches a fallback min-50 watcher for S1E/S1F after those lanes start.

Ranked next experiments:

1. **S1C warm-start no-distill fine-tune**
   - Base checkpoint: lane B `best_search_p10.pt`
   - Stage1 load path: `model.pretrained.enabled=true`, `model.pretrained.path=<base_ckpt>`, `model.pretrained.load_stage=stage1`
   - Search-only, full width, no pruning, no distillation
   - AdamW LR `1e-4`, `loss.search_xy_weight=1.5`, 120 epochs
   - Purpose: recover/stabilize the epoch-175 P10/P5 peak with lower LR.

2. **S1D warm-start weak self-distillation fine-tune**
   - Base checkpoint: lane B `best_search_p10.pt`
   - Teacher checkpoint: same lane B `best_search_p10.pt`
   - Search-only, full width, weak feature/state/prediction/mask distillation
   - AdamW LR `7.5e-5`, distillation weights `0.02/0.02/0.01/0.01`, EMA `0.996`, 120 epochs
   - Purpose: reduce late-epoch drift while preserving the best P10/P5 structure.

3. **S1E bounded auxiliary recovery**
   - Lower priority.
   - Re-enable only the minimum safe auxiliary target if S1C/S1D fail to improve P5/P1.
   - Do not reintroduce broad Stage1 multi-head training until there is evidence that Search-only has saturated.

## Code Path Evidence

- Stage1 pretrained loading is supported by `src/hbtxr/training/trainer.py::_maybe_apply_pretrained`.
- Stage1 teacher initialization can use `distillation.teacher_init_checkpoint` in `src/hbtxr/training/trainer.py::_initialize_model_state`.
- Stage1 default distillation targets include `search/pooled`, `search/state`, `search/eye`, `search/pupil`, and `search/mask_logits` in `src/hbtxr/loss/distillation.py`.
- Checkpoint payloads with `state_dict`, `model`, `student`, or `teacher` keys are supported by `src/hbtxr/models/preload/pretrained_loader.py`.

## Verification Commands

Commands used for the current status:

```bash
ps -p 1332213,1332214 -o pid,ppid,sid,stat,etime,cmd
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader
.venv/bin/python scripts/external/summarize_training_history.py \
  runs/NON_XR/raw/stage1_frame_search_sonly_adamw_lr1e3_fullwidth_20260621_204500 \
  runs/NON_XR/raw/stage1_frame_search_sonly_adamw_lr5e4_xy1p5_fullwidth_20260621_204500 \
  --key metric_search_p10_pct \
  --key metric_search_p5_pct \
  --key metric_search_p1_pct \
  --key metric_search_center_px \
  --key loss_total \
  --key loss_search_xy
```

Validation commands:

```bash
bash -n scripts/external/run_stage1_frame_search_baseline_matrix.sh
bash -n scripts/external/run_stage1_frame_search_warmstart_refine.sh
python3 -m py_compile scripts/external/report_stage1_frame_search_baseline.py
.venv/bin/python scripts/external/report_stage1_frame_search_baseline.py --format summary
python3 -m py_compile scripts/external/report_stage1_frame_search_followup.py
.venv/bin/python scripts/external/report_stage1_frame_search_followup.py --format summary
bash -n scripts/external/run_stage1_frame_search_followup_until_min_epoch.sh
bash -n scripts/external/run_stage1_frame_search_refine_when_ready.sh
bash -n scripts/external/run_stage1_frame_search_supervisor_until_complete.sh
```

## 2026-06-22 Live Refresh And Capacity Branch

Latest follow-up reporter command:

```bash
.venv/bin/python scripts/external/report_stage1_frame_search_followup.py --min-epochs 50 --target-epochs 120 --format json
```

Current status:

- Lane B remains the active Stage1 frame-search baseline candidate.
- Baseline checkpoint: `runs/NON_XR/raw/stage1_frame_search_sonly_adamw_lr5e4_xy1p5_fullwidth_20260621_204500/train/best_search_p10.pt`.
- Baseline best metrics: Search P10 `28.22439415050003`, Search P5 `9.833782825829848`, Search P1 `1.0646900500891343`, center `17.28884926831947`.
- S1C warm-start no-distill: epoch `89/120`, best P10 `27.94025216012631`, best P5 `10.049416236157688`, best center `17.233418217245138`; not promotable because primary P10 is `-0.284141990373719 pp` behind lane B.
- S1D weak self-distill: epoch `82/120`, best P10 `27.715633986131202`, best P5 `10.086478287318968`, best center `17.34278841738431`; not promotable because primary P10 is `-0.5087601643688267 pp` behind lane B and center is worse.
- Reporter recommendation: `keep_baseline_unless_later_improves`, `promote_checkpoint=None`, `continue_training=True`.

Active automation:

- S1C train PID `2402483` remains alive on GPU0.
- S1D train PID `2420379` remains alive on GPU1.
- Post-followup queue PID `2638408` remains alive and waits for S1C/S1D to finish before launching S1E/S1F.
- S1E planned lane: lower-LR warm-start no-distill, LR `5e-5`, run name `stage1_frame_search_warmstart_lr5e5_xy1p5_stage1_refine_after_min50_low_lr_20260622_000233`.
- S1F planned lane: lower-LR EMA self-distill, LR `3e-5`, EMA `0.999`, run name `stage1_frame_search_selfdistill_lr3e5_xy1p5_ema999_stage1_refine_after_min50_low_lr_20260622_000233`.
- Capacity-after-followups queue PID `2708973` is active:
  - Script: `scripts/external/run_stage1_frame_search_capacity_after_followups.sh`.
  - Log: `runs/NON_XR/shared/_logs/stage1_capacity_after_followups_queue_20260622_0041.log`.
  - Behavior: wait for S1E/S1F histories, require target completion and no promotion, check GPU memory, then launch S1G/S1H with `CAPACITY_RUN_TAG=stage1_capacity_after_fallback_20260622_0041`.

Capacity branch was launched in parallel because S1C/S1D crossed the 50-epoch gate but remained behind lane B, and both GPUs still had enough free memory:

- Script: `scripts/external/run_stage1_frame_search_capacity_distill.sh`.
- Run tag: `stage1_capacity_parallel_20260622_0050`.
- S1G: large Search-only student, embed dim `256`, depth `8`, heads `4`, partial warm-start from lane B, LR `2e-4`, 200 epochs, PID `2723776`.
- S1H: large Search-only student, baseline-size teacher override `192/6/3`, teacher checkpoint lane B `best_search_p10.pt`, LR `1e-4`, EMA `0.999`, distillation weights feature/state/prediction/mask `0.02/0.02/0.01/0.01`, 200 epochs, PID `2723777`.
- S1G run directory: `runs/NON_XR/raw/stage1_frame_search_large256_d8_partialwarm_lr2e4_stage1_capacity_parallel_20260622_0050_20260622_004413`.
- S1H run directory: `runs/NON_XR/raw/stage1_frame_search_large256_d8_baseteacher_distill_lr1e4_stage1_capacity_parallel_20260622_0050_20260622_004413`.
- S1G/S1H min-50 watcher PID `2730067`, log `runs/NON_XR/shared/_logs/stage1_capacity_parallel_min50_20260622_0050.log`.
- First capacity watcher poll reached epoch `1/200` for both lanes. Early metrics are not promotion evidence yet; the watcher remains in `wait_min_epochs`.
- Model role support was added so `model.teacher.*` overrides can build a smaller teacher while the student uses the larger base model.
- The older capacity-after-followups queue PID `2708973` was stopped after S1G/S1H started to avoid duplicate capacity launches. The S1E/S1F post-followup queue PID `2638408` remains active.

Decision:

- Do not promote S1C/S1D at the current snapshot.
- Keep lane B as the downstream baseline until a later S1C/S1D/S1E/S1F/S1G/S1H checkpoint beats lane B on primary Search P10 without center regression.

## 2026-06-22 Corrected Live State

The earlier S1G/S1H capacity branch did not survive to the 50-epoch gate.

- S1G large no-distill stopped during epoch `7/200`.
- S1H large baseline-teacher distill stopped during epoch `6/200`.
- Both logs ended with `DataLoader worker ... killed by signal: Killed`.
- Interpretation: the likely failure mode is host memory pressure from running four Stage1 jobs concurrently with high `num_workers`; this is not promotion evidence and not a model-quality conclusion.
- The S1G/S1H min-50 watcher PID `2730067` was terminated because the watched train processes had already exited and the watcher would otherwise wait indefinitely.

The current active branch is S1E/S1F lower-LR fallback:

- S1E process PID `2751991`, GPU0, run `runs/NON_XR/raw/stage1_frame_search_warmstart_lr5e5_xy1p5_stage1_refine_after_min50_low_lr_20260622_0057_20260622_004951`.
- S1F process PID `2751992`, GPU1, run `runs/NON_XR/raw/stage1_frame_search_selfdistill_lr3e5_xy1p5_ema999_stage1_refine_after_min50_low_lr_20260622_0057_20260622_004951`.
- Corrected min-50 watcher PID `2763859`, log `runs/NON_XR/shared/_logs/stage1_frame_search_fallback_min50_stage1_refine_after_min50_low_lr_20260622_0057_corrected_20260622_0059.log`.
- The initial fallback watcher had watched suffix-free paths; `scripts/external/run_stage1_frame_search_post_followup_queue.sh` now resolves timestamp-suffixed run directories before launching a watcher.

Latest fallback reporter snapshot:

- Baseline lane B: P10 `28.22439415050003`, P5 `9.833782825829848`, center `17.28884926831947`.
- S1E: epoch `7/120`, best P10 `27.67969512939453`, best P5 `9.851752245201254`, best center `17.437835072571378`; not min-ready.
- S1F: epoch `6/120`, best P10 `27.932390464926666`, best P5 `9.528302156700278`, best center `17.307134470849668`; not min-ready.
- Decision remains `wait_min_epochs`; Lane B remains the active Stage1 baseline.

Next action:

- Let S1E/S1F reach at least `50` epochs, then compare against Lane B.
- Retry the capacity branch only after S1E/S1F finish or clearly fail, and use a safer launch profile such as fewer data loader workers and lower concurrent job count.
- `scripts/external/run_stage1_frame_search_capacity_distill.sh` now defaults to `NUM_WORKERS=2` for safer future capacity retries.

## 2026-06-22 S1E/S1F Gate Refresh

Latest direct reporter command:

```bash
.venv/bin/python scripts/external/report_stage1_frame_search_followup.py \
  --min-epochs 50 \
  --target-epochs 120 \
  --followup runs/NON_XR/raw/stage1_frame_search_warmstart_lr5e5_xy1p5_stage1_refine_after_min50_low_lr_20260622_0057_20260622_004951 \
  --followup runs/NON_XR/raw/stage1_frame_search_selfdistill_lr3e5_xy1p5_ema999_stage1_refine_after_min50_low_lr_20260622_0057_20260622_004951 \
  --format summary
```

Current reporter snapshot:

- S1E: epoch `9/120`, best P10 `27.67969512939453`, best P5 `9.851752245201254`, best center `17.437835072571378`; not min-ready.
- S1F: epoch `9/120`, best P10 `27.932390464926666`, best P5 `9.528302156700278`, best center `17.307134470849668`; not min-ready.
- Lane B remains ahead on primary P10 by `0.5446990211054974 pp` over S1E and `0.29200368557336276 pp` over S1F.
- Process check confirmed S1E PID `2751991`, S1F PID `2751992`, and corrected watcher PID `2763859` are still alive.
- Decision remains `wait_min_epochs`; no baseline promotion until at least one fallback lane reaches `50` epochs and beats Lane B on primary Search P10 without center regression.

## 2026-06-22 Capacity Queue Refresh

S1E/S1F are still below the 50-epoch decision gate, but the next branch automation is now staged safely.

- Latest reporter snapshot from the capacity queue:
  - S1E epoch `14/120`, best P10 `27.83692773782982`, best P5 `9.851752245201254`, best center `17.437835072571378`.
  - S1F epoch `13/120`, best P10 `27.932390464926666`, best P5 `9.743935603015828`, best center `17.307134470849668`.
  - Status remains `wait_min_epochs`; Lane B still leads on primary Search P10.
- Added path auto-resolution to `scripts/external/run_stage1_frame_search_capacity_after_followups.sh`.
  - It now finds the latest S1E/S1F timestamped run directories when `FOLLOWUP_RUNS` is not explicitly provided.
  - This prevents stale suffix-free or older run paths from blocking the next branch.
- Launched corrected capacity-after queue:
  - PID `2800043`.
  - Log `runs/NON_XR/shared/_logs/stage1_capacity_after_s1ef_corrected_20260622_0120.log`.
  - Capacity run tag `stage1_capacity_after_s1ef_safe_workers2_20260622_0120`.
  - Behavior: wait for S1E/S1F completion, skip capacity retry if a fallback checkpoint is promotable, otherwise check GPU memory and launch the safer `NUM_WORKERS=2` capacity branch.

## 2026-06-22 Center-Preserve Polish Queue

S1E is now close to Lane B on P10, but its center error is still worse. Therefore the next staged branch was changed from immediate large-capacity retry to center-preserve polish.

- Warm-start runner update:
  - `scripts/external/run_stage1_frame_search_warmstart_refine.sh` now accepts `BEST_METRIC_NAME`.
  - This allows center-driven checkpoints such as `best_metric_search_center_px.pt` while still saving `best_search_p10.pt` and `best_search_p5.pt`.
- Added runner:
  - `scripts/external/run_stage1_frame_search_center_preserve_polish.sh`.
  - Default metric: `metric_search_center_px`.
  - Default epochs: `80`.
  - Default workers: `4`.
  - Lane I: no-distill center polish, LR `1e-5`, `loss.search_xy_weight=2.0`.
  - Lane J: weak self-distill center polish, LR `7.5e-6`, EMA `0.999`, distill weights feature/state/prediction/mask `0.005/0.005/0.0025/0.0025`.
- Queue change:
  - Previous capacity queue PID `2800043` was stopped.
  - New center-polish queue PID `2819081`.
  - Log `runs/NON_XR/shared/_logs/stage1_centerpolish_after_s1ef_queue_20260622_0128.log`.
  - Runner `scripts/external/run_stage1_frame_search_center_preserve_polish.sh`.
  - Tag `stage1_centerpolish_after_s1ef_20260622_0128`.
- First center-polish queue poll:
  - S1E epoch `20/120`, best P10 `28.218778646217203`, best P5 `9.851752245201254`, best center `17.437835072571378`.
  - S1F epoch `19/120`, best P10 `27.932390464926666`, best P5 `9.743935603015828`, best center `17.307134470849668`.
  - Status remains `wait_min_epochs`.

Decision:

- Keep Lane B as baseline until a candidate crosses `min_epochs=50` and beats Lane B on P10 without center regression.
- If S1E/S1F complete without promotion, run center-polish before retrying the larger capacity branch.

## 2026-06-22 S1E/S1F Pre-Gate Refresh

Latest direct reporter snapshot:

- S1E epoch `23/120`, best P10 `28.218778646217203`, best P5 `9.974169173330631`, best center `17.437835072571378`.
- S1F epoch `22/120`, best P10 `27.932390464926666`, best P5 `9.743935603015828`, best center `17.307134470849668`.
- S1E is within `0.005615504282825867 pp` of Lane B on P10 and is ahead on P5 by `0.14038634750078316 pp`, but still regresses center by `0.14898580425190744 px`.
- S1F has better center than S1E but remains behind Lane B on both P10 and center.
- Status remains `wait_min_epochs`; no candidate can be promoted before epoch `50`.

Runtime state:

- S1E PID `2751991` alive.
- S1F PID `2751992` alive.
- Min-50 watcher PID `2763859` alive.
- Center-polish queue PID `2819081` alive and waiting.

## 2026-06-22 S1E/S1F Second Pre-Gate Refresh

Latest reporter snapshot:

- S1E epoch `26/120`, best P10 `28.218778646217203`, best P5 `9.974169173330631`, best center `17.437835072571378`.
- S1F epoch `25/120`, best P10 `27.932390464926666`, best P5 `9.743935603015828`, best center `17.307134470849668`.
- S1E remains a P10 near-tie with Lane B but is still `0.005615504282825867 pp` behind on P10 and `0.14898580425190744 px` worse on center.
- S1F remains behind on primary P10 and center.
- Status remains `wait_min_epochs`; promotion remains blocked by the explicit `50` epoch gate.

Runtime state:

- S1E PID `2751991` alive.
- S1F PID `2751992` alive.
- Min-50 watcher PID `2763859` alive.
- Center-polish queue PID `2819081` alive.

## 2026-06-22 S1E/S1F Third Pre-Gate Refresh

Latest direct reporter snapshot:

- S1E epoch `31/120`, best P10 `28.218778646217203`, best P5 `9.974169173330631`, best center `17.437835072571378`.
- S1F epoch `30/120`, best P10 `28.078392352697986`, best P5 `9.743935603015828`, best center `17.307134470849668`.
- S1E remains `0.005615504282825867 pp` behind Lane B on P10, `0.14038634750078316 pp` ahead on P5, and `0.14898580425190744 px` worse on center.
- S1F improved P10 relative to the previous snapshot but remains `0.14600179780204314 pp` behind Lane B on P10 and `0.01828520253019761 px` worse on center.
- Status remains `wait_min_epochs`; no promotion before epoch `50`.

Queue state:

- Center-polish queue PID `2819081` is alive.
- Latest queue poll saw S1E `27/120`, S1F `26/120`, and remained in `wait_min_epochs`.

## 2026-06-22 S1E/S1F Fourth Pre-Gate Refresh

Latest direct reporter snapshot:

- S1E epoch `36/120`, best P10 `28.218778646217203`, best P5 `9.974169173330631`, best center `17.437835072571378`.
- S1F epoch `35/120`, best P10 `28.15363937953733`, best P5 `9.743935603015828`, best center `17.307134470849668`.
- S1E remains `0.005615504282825867 pp` behind Lane B on P10, `0.14038634750078316 pp` ahead on P5, and `0.14898580425190744 px` worse on center.
- S1F remains `0.07075477096269722 pp` behind Lane B on P10, `0.08984722281402036 pp` behind on P5, and `0.01828520253019761 px` worse on center.
- Status remains `wait_min_epochs`; no promotion before epoch `50`.

Runtime state:

- S1E PID `2751991` alive on GPU0.
- S1F PID `2751992` alive on GPU1.
- Corrected min-50 watcher PID `2763859` alive.
- Center-polish queue PID `2819081` alive and waiting.
- GPU memory remained healthy at roughly `15 GiB` free per GPU during the check.

## 2026-06-22 S1E/S1F Min-50 Gate Result

Latest direct reporter snapshot after both fallback lanes crossed the minimum gate:

- S1E epoch `55/120`, best P10 `28.218778646217203`, best P5 `9.974169173330631`, best center `17.437835072571378`.
- S1F epoch `53/120`, best P10 `28.15363937953733`, best P5 `9.743935603015828`, best center `17.307134470849668`.
- Lane B baseline remains P10 `28.22439415050003`, P5 `9.833782825829848`, center `17.28884926831947`.
- S1E is still `0.005615504282825867 pp` behind Lane B on primary P10 and regresses center by `0.14898580425190744 px`, despite improving P5 by `0.14038634750078316 pp`.
- S1F is `0.07075477096269722 pp` behind Lane B on P10, `0.08984722281402036 pp` behind on P5, and regresses center by `0.01828520253019761 px`.
- Reporter status: `keep_baseline_unless_later_improves`.
- `promote_checkpoint=None`; active baseline remains `runs/NON_XR/raw/stage1_frame_search_sonly_adamw_lr5e4_xy1p5_fullwidth_20260621_204500/train/best_search_p10.pt`.

Decision:

- Do not promote S1E or S1F at the min-50 gate.
- Let S1E/S1F continue toward 120 epochs because the reporter still returns `continue_training=True`.
- Keep center-preserve polish staged as the next branch if S1E/S1F finish without promotion.

## 2026-06-22 S1E/S1F Post-Gate Continuation

Latest continuation snapshot:

- S1E epoch `67/120`, best P10 `28.218778646217203`, best P5 `9.974169173330631`, best center `17.437835072571378`.
- S1F epoch `64/120`, best P10 `28.15363937953733`, best P5 `9.743935603015828`, best center `17.307134470849668`.
- Reporter status remains `keep_baseline_unless_later_improves`.
- `promote_checkpoint=None`.
- S1E PID `2751991`, S1F PID `2751992`, and center-polish queue PID `2819081` are alive.

Decision:

- Continue S1E/S1F toward the `120` epoch target.
- Do not launch extra concurrent Stage1 jobs while these two jobs are alive.
- Keep Lane B as the active baseline until a completed or later-best checkpoint beats Lane B under the reporter gate.

## 2026-06-22 Post-Centerpolish Capacity Queue Staging

Latest continuation snapshot:

- S1E epoch `69/120`, best P10 `28.218778646217203`, best P5 `9.974169173330631`, best center `17.437835072571378`.
- S1F epoch `67/120`, best P10 `28.15363937953733`, best P5 `9.743935603015828`, best center `17.307134470849668`.
- Reporter status remains `keep_baseline_unless_later_improves`.
- `promote_checkpoint=None`; Lane B remains active baseline.

New automation:

- Added `scripts/external/run_stage1_frame_search_after_centerpolish_capacity_queue.sh`.
- Launched host queue PID `3000547`.
- Log: `runs/NON_XR/shared/_logs/stage1_after_centerpolish_capacity_queue_20260622_0141.log`.
- Behavior:
  - Wait for center-polish lane I/J run directories from tag `stage1_centerpolish_after_s1ef_20260622_0128`.
  - After those runs exist, start `scripts/external/run_stage1_frame_search_capacity_after_followups.sh` against the center-polish runs.
  - If center-polish completes without promotion, launch the safer capacity branch with `NUM_WORKERS=2` via `scripts/external/run_stage1_frame_search_capacity_distill.sh`.

Decision:

- Keep current S1E/S1F jobs running.
- Keep center-polish queued after S1E/S1F completion.
- Keep post-centerpolish capacity queued after center-polish non-promotion, so the accuracy-improvement path does not stop after one failed branch.

## 2026-06-22 S1E/S1F Continuation Plateau Check

Latest reporter snapshot:

- S1E epoch `77/120`, best P10 `28.218778646217203`, best P5 `9.974169173330631`, best center `17.437835072571378`.
- S1F epoch `74/120`, best P10 `28.15363937953733`, best P5 `9.743935603015828`, best center `17.307134470849668`.
- Reporter status remains `keep_baseline_unless_later_improves`; `promote_checkpoint=None`.

Direct `history.json` check:

- S1E best P10 occurred at epoch `15`, best P5 at epoch `23`, and best center at epoch `3`.
- S1E latest epoch `77` validation metrics were P10 `26.15116848135894`, P5 `9.258760398288942`, P1 `0.1347708971995228`, center `17.70168143398357`.
- S1F best P10 occurred at epoch `35`, best P5 at epoch `13`, and best center at epoch `1`.
- S1F latest epoch `74` validation metrics were P10 `26.965409296863484`, P5 `9.07232734392274`, P1 `0.5615454079969874`, center `17.785983782894206`.

Runtime state:

- S1E PID `2751991` alive.
- S1F PID `2751992` alive.
- Center-polish queue PID `2819081` alive.
- Post-centerpolish capacity queue PID `3000547` alive.
- No center-polish or capacity follow-up run directories have been created yet.

Decision:

- Do not promote S1E/S1F.
- Do not stop them before the configured target, but treat the current branch as plateauing unless a later epoch updates the best checkpoint.
- Keep queued center-polish and post-centerpolish capacity paths as the next accuracy-improvement branches.

## 2026-06-22 S1E/S1F Continuation Refresh

Latest reporter snapshot:

- S1E epoch `80/120`, best P10 `28.218778646217203`, best P5 `9.974169173330631`, best center `17.437835072571378`.
- S1F epoch `77/120`, best P10 `28.15363937953733`, best P5 `9.743935603015828`, best center `17.307134470849668`.
- Reporter status remains `keep_baseline_unless_later_improves`; `promote_checkpoint=None`.

Latest direct history rows:

- S1E epoch `80`: P10 `26.160153370983195`, P5 `9.232929283717894`, P1 `0.3706199538032964`, center `17.88450043606308`.
- S1F epoch `77`: P10 `26.10624487894886`, P5 `8.707322840420705`, P1 `0.6794699362988742`, center `17.84767130185973`.

Runtime state:

- S1E PID `2751991` alive.
- S1F PID `2751992` alive.
- Center-polish queue PID `2819081` alive.
- Post-centerpolish capacity queue PID `3000547` alive.
- No center-polish or downstream capacity directories exist yet.

Decision:

- Lane B remains the active baseline.
- Continue to target completion; queued follow-up branches remain staged.

## 2026-06-22 S1E/S1F Later Continuation Refresh

Latest reporter snapshot:

- S1E epoch `83/120`, best P10 `28.218778646217203`, best P5 `9.974169173330631`, best center `17.437835072571378`.
- S1F epoch `79/120`, best P10 `28.15363937953733`, best P5 `9.743935603015828`, best center `17.307134470849668`.
- Reporter status remains `keep_baseline_unless_later_improves`; `promote_checkpoint=None`.

Latest direct history rows:

- S1E epoch `83`: P10 `26.089398564032788`, P5 `8.738769369305304`, P1 `0.2751572447003059`, center `17.880303625790578`.
- S1F epoch `80`: P10 `26.918239575512004`, P5 `8.611860095330005`, P1 `0.4099281418998286`, center `17.691625046280194`.

Runtime state:

- S1E PID `2751991` alive.
- S1F PID `2751992` alive.
- Center-polish queue PID `2819081` alive.
- Post-centerpolish capacity queue PID `3000547` alive.
- No center-polish or downstream capacity directories exist yet.

Decision:

- Lane B remains the active baseline.
- Continue S1E/S1F to the configured target unless they fail externally.

## 2026-06-22 S1E/S1F Extended Continuation Refresh

Latest reporter snapshot:

- S1E epoch `93/120`, best P10 `28.218778646217203`, best P5 `9.974169173330631`, best center `17.437835072571378`.
- S1F epoch `90/120`, best P10 `28.15363937953733`, best P5 `9.743935603015828`, best center `17.307134470849668`.
- Reporter status remains `keep_baseline_unless_later_improves`; `promote_checkpoint=None`.

Latest direct history rows:

- S1E epoch `93`: P10 `26.446541462304456`, P5 `7.851527645902814`, P1 `0.5446990390993515`, center `17.97235798385908`.
- S1F epoch `90`: P10 `26.60938953903486`, P5 `8.505166485624493`, P1 `0.6626235674012382`, center `17.68777278234374`.

Runtime state:

- S1E PID `2751991` alive.
- S1F PID `2751992` alive.
- Center-polish queue PID `2819081` alive.
- Post-centerpolish capacity queue PID `3000547` alive.
- No center-polish or downstream capacity directories exist yet.

Decision:

- Lane B remains the active baseline.
- Continue monitoring to 120 epochs; queued follow-up branches remain active.

## 2026-06-22 S1E/S1F Near-Completion Refresh

Latest reporter snapshot:

- S1E epoch `96/120`, best P10 `28.218778646217203`, best P5 `9.974169173330631`, best center `17.437835072571378`.
- S1F epoch `92/120`, best P10 `28.15363937953733`, best P5 `9.743935603015828`, best center `17.307134470849668`.
- Reporter status remains `keep_baseline_unless_later_improves`; `promote_checkpoint=None`.

Latest direct history rows:

- S1E epoch `96`: P10 `26.581312377497834`, P5 `8.716307586094118`, P1 `0.25269542550140955`, center `18.04789892232643`.
- S1F epoch `92`: P10 `26.19047675942475`, P5 `8.409703776521503`, P1 `1.0163971523068986`, center `17.752043238225973`.

Runtime state:

- S1E PID `2751991` alive.
- S1F PID `2751992` alive.
- Center-polish queue PID `2819081` alive.
- Post-centerpolish capacity queue PID `3000547` alive.
- No center-polish or downstream capacity directories exist yet.

Decision:

- Lane B remains active baseline.
- Continue S1E/S1F to target completion and let queued center-polish handle the next branch if no late promotion occurs.

## 2026-06-22 Queue Safety Fix and Requeue

Issue found:

- `scripts/external/run_stage1_frame_search_capacity_after_followups.sh` previously entered `complete_no_promotion` when all follow-ups were min-ready and `any_followup_complete` was true.
- That could launch center-polish after only S1E reached 120 while S1F was still running, creating an avoidable four-job concurrency risk.

Fix:

- Updated `decision_state()` to require all watched follow-up runs to have `complete_target_epochs=true` before returning `complete_no_promotion`.
- Dry-run validation with current S1E/S1F state returned `min_ready_no_promotion`, confirming the queue no longer launches early.

Runtime change:

- Stopped old center-polish queue PID `2819081`.
- Launched new all-complete center-polish queue PID `3079899`.
- New queue log: `runs/NON_XR/shared/_logs/stage1_centerpolish_after_s1ef_queue_allcomplete_20260622_0158.log`.

Latest queue reporter snapshot:

- S1E epoch `100/120`, best P10 `28.218778646217203`, best P5 `9.974169173330631`, best center `17.437835072571378`.
- S1F epoch `96/120`, best P10 `28.15363937953733`, best P5 `9.743935603015828`, best center `17.307134470849668`.
- Status remains `keep_baseline_unless_later_improves`; queue decision `min_ready_no_promotion`.

Decision:

- Lane B remains active baseline.
- Continue S1E/S1F to target completion.
- Center-polish will now start only after both S1E and S1F are complete and no promotion is available.

## 2026-06-22 Active-Run Refresh After Restart Request

Latest reporter snapshot:

- S1E epoch `107/120`, best P10 `28.218778646217203`, best P5 `9.974169173330631`, best center `17.437835072571378`.
- S1F epoch `102/120`, best P10 `28.15363937953733`, best P5 `9.743935603015828`, best center `17.307134470849668`.
- Reporter status remains `keep_baseline_unless_later_improves`; `promote_checkpoint=None`; `continue_training=True`.

Baseline:

- Lane B run: `runs/NON_XR/raw/stage1_frame_search_sonly_adamw_lr5e4_xy1p5_fullwidth_20260621_204500`.
- Lane B best P10 `28.22439415050003`, best P5 `9.833782825829848`, best center `17.28884926831947`.

Decision:

- Do not promote S1E/S1F yet.
- S1E is close on P10 but still below Lane B and center-regressed.
- S1F does not beat Lane B.
- Center-polish and downstream capacity are correctly waiting because S1E/S1F have not both completed 120 epochs.
- GPT-5.3-Codex-Spark sub-agent spawn failed due usage limit; main-agent validation was used.

## 2026-06-22 GPT-5.5 Audit and Later Active-Run Refresh

Audit:

- GPT-5.5 sub-agent `Curie the 2nd` completed a read-only audit.
- It inspected histories, reporter JSON, queue logs, and `docs/track`.
- It did not edit files or launch train/eval jobs.

Latest reporter snapshot:

- S1E epoch `113/120`, best P10 `28.218778646217203`, best P5 `9.974169173330631`, best center `17.437835072571378`.
- S1F epoch `108/120`, best P10 `28.15363937953733`, best P5 `9.743935603015828`, best center `17.307134470849668`.
- Reporter status remains `keep_baseline_unless_later_improves`; `promote_checkpoint=None`; `continue_training=True`.

Latest rows:

- S1E epoch `113`: P10 `26.48584965040099`, P5 `7.980683074807221`, center `17.999366805238544`.
- S1F epoch `108`: P10 `26.176999649911558`, P5 `8.645552833125276`, center `17.734962881736035`.

Decision:

- Lane B remains active baseline.
- Continue S1E/S1F to 120 epochs.
- Center-polish should launch only after both complete without promotion.

## 2026-06-22 S1E/S1F Completion and Center-Polish Launch

S1E/S1F final gate:

- S1E reached `120/120`; best P10 `28.218778646217203`, best P5 `9.974169173330631`, best center `17.437835072571378`.
- S1F reached `120/120`; best P10 `28.15363937953733`, best P5 `9.743935603015828`, best center `17.307134470849668`.
- Neither run is promotable against Lane B.

Center-polish launch:

- The all-complete queue reached `complete_no_promotion`.
- It launched `scripts/external/run_stage1_frame_search_center_preserve_polish.sh`.
- Lane I no-distill run: `runs/NON_XR/raw/stage1_frame_search_centerpolish_nodistill_lr1e5_xy2_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949`.
- Lane J weak self-distill run: `runs/NON_XR/raw/stage1_frame_search_centerpolish_selfdistill_lr7p5e6_xy2_ema999_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949`.

Early center-polish result:

- Lane I epoch `3/80`: P10 `28.303010526693093`, P5 `9.884321932522756`, center `17.3741187104639`.
- Lane J epoch `3/80`: P10 `28.27717937613433`, P5 `10.153863744915656`, center `17.257583906065744`.

Interpretation:

- Lane I improves P10/P5 over Lane B but still has worse center.
- Lane J improves P10, P5, and center over Lane B at epoch 3.
- No baseline promotion is allowed until the 50-epoch minimum gate is reached.

# 2026-06-22 Center-Polish Active Refresh

## Current baseline

- Run: `runs/NON_XR/raw/stage1_frame_search_sonly_adamw_lr5e4_xy1p5_fullwidth_20260621_204500`.
- Epochs: `300`.
- Best P10: `28.22439415050003`.
- Best P5: `9.833782825829848`.
- Best center: `17.28884926831947`.

## Active center-polish lanes

- Lane I no-distill:
  - Run: `runs/NON_XR/raw/stage1_frame_search_centerpolish_nodistill_lr1e5_xy2_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949`.
  - Epochs: `29/80`.
  - Best P10: `28.303010526693093`, delta vs Lane B `+0.0786163761930645`.
  - Best P5: `9.884321932522756`, delta vs Lane B `+0.05053910669290751`.
  - Best center: `17.3741187104639`, center improvement delta vs Lane B `-0.08526944214442977`.
- Lane J weak self-distill:
  - Run: `runs/NON_XR/raw/stage1_frame_search_centerpolish_selfdistill_lr7p5e6_xy2_ema999_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949`.
  - Epochs: `28/80`.
  - Best P10: `28.27717937613433`, delta vs Lane B `+0.052785225634302435`.
  - Best P5: `10.153863744915656`, delta vs Lane B `+0.3200809190858074`.
  - Best center: `17.257583906065744`, center improvement delta vs Lane B `+0.03126536225372689`.

## Decision

- Status: `wait_min_epochs`.
- Promotion checkpoint: none.
- Continue training: yes.
- Lane J is the leading baseline candidate, but no baseline switch is valid before epoch `50/80`.
- Baseline manifest: `docs/resources/current_stage1_frame_search_baseline_manifest.json`.
- Active watcher: `runs/NON_XR/shared/_logs/stage1_after_centerpolish_capacity_queue_manifest_hook_20260622_0236b.log`, host PID `3300765`.

## 2026-06-22 50-Epoch Gate Result

- Status: `promote_followup`.
- Active Stage1 frame-search baseline checkpoint:
  - `runs/NON_XR/raw/stage1_frame_search_centerpolish_selfdistill_lr7p5e6_xy2_ema999_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949/train/best_search_p10.pt`
- Promoted run:
  - `runs/NON_XR/raw/stage1_frame_search_centerpolish_selfdistill_lr7p5e6_xy2_ema999_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949`
- Gate evidence:
  - Lane J reached at least `51/80`, above the required minimum `50` epochs.
  - Best P10 `28.27717937613433`, delta vs Lane B `+0.052785225634302435 pp`.
  - Best P5 `10.153863744915656`, delta vs Lane B `+0.3200809190858074 pp`.
  - Best center `17.257583906065744`, center improvement vs Lane B `+0.03126536225372689 px`.
- Lane I no-distill reached at least `53/80` and has higher P10 `28.303010526693093`, but is not promotable because center regresses to `17.3741187104639`.
- Manifest state:
  - `docs/resources/current_stage1_frame_search_baseline_manifest.json` is `promoted_followup`.

Decision:

- Use Lane J as the current Stage1 frame-based Search baseline for downstream Stage2/XR-64 planning.
- Keep Lane I/J running toward `80/80` to collect final stability evidence before freezing the full experiment report.
