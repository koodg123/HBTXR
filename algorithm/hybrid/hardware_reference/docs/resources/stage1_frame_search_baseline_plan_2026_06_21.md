# Stage1 Frame-Search Baseline Plan - 2026-06-21

## Goal

Build a stronger Stage1 frame-based Search baseline before continuing Stage2/XR-64 work.

Current promoted Stage1 reference:

- Run: `runs/NON_XR/raw/raw_mode1_stage1_best_adamw255k_200ep_2gpu_20260611_222452`
- Best validation Search P10: `23.34905708960767` at epoch `165`
- Best validation Search center: `18.319516586807538 px` at epoch `154`
- Final epoch 200 validation Search: center `18.71338979253229 px`, P10 `20.153863942848062`

Current promoted Stage1 frame-search baseline after the 2026-06-22 center-polish gate:

- Run: `runs/NON_XR/raw/stage1_frame_search_centerpolish_selfdistill_lr7p5e6_xy2_ema999_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949`
- Checkpoint: `runs/NON_XR/raw/stage1_frame_search_centerpolish_selfdistill_lr7p5e6_xy2_ema999_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949/train/best_search_p10.pt`
- Best validation Search P10: `28.27717937613433`
- Best validation Search P5: `10.153863744915656`
- Best validation Search center: `17.257583906065744 px`
- Promotion basis versus the lane-B 300-epoch baseline: P10 `+0.052785225634302435 pp`, P5 `+0.3200809190858074 pp`, center `+0.03126536225372689 px` improvement.
- Manifest: `docs/resources/current_stage1_frame_search_baseline_manifest.json`

The original 200-epoch run above is retained as historical provenance only. New downstream Stage2/XR-64 work should use the 2026-06-22 self-distill center-polish checkpoint unless a later completed follow-up supersedes it.

New baseline promotion gate:

- Primary: validation `metric_search_p10_pct > 23.34905708960767`
- Secondary: validation `metric_search_center_px < 18.319516586807538`
- Guardrail: no NaN, CUDA execution confirmed, `best_search_p10.pt` written, history contains finite val metrics.

## Rationale

The previous 200/200 run proved that the fresh Stage1 checkpoint can train, but the downstream Stage2 run did not promote. Before further Stage2 construction, Stage1 should be isolated as a frame Search baseline and improved directly.

The current Stage1 preset still trains eye/mask/Search together unless overridden. For this baseline, the experiment uses:

- `model.heads.active=search`
- eye/event/track/mask/aux heads disabled
- full-width model
- no distillation
- no pruning
- Search P10 as checkpoint metric

This removes multi-task interference and makes the Stage1 checkpoint a clearer Search prior for later Stage2 initialization.

## Runner

Script:

```bash
bash scripts/external/run_stage1_frame_search_baseline_matrix.sh
```

Default behavior:

- Launches two detached lanes.
- GPU0 lane A: `lr=1e-3`, default Search loss weights.
- GPU1 lane B: `lr=5e-4`, higher `loss.search_xy_weight=1.5`.
- Both lanes use 300 epochs, batch size 8, 8 workers, AdamW, cosine schedule, warmup 5, min LR `1e-6`.
- Runs are created under `runs/NON_XR/raw/`.
- Logs and PID files are written under `runs/NON_XR/shared/_logs/`.

Dry-run command:

```bash
DRY_RUN=1 bash scripts/external/run_stage1_frame_search_baseline_matrix.sh
```

Monitor:

```bash
tail -f runs/NON_XR/shared/_logs/stage1_frame_search_lane_a_<RUN_TAG>.log
tail -f runs/NON_XR/shared/_logs/stage1_frame_search_lane_b_<RUN_TAG>.log
```

## Task Cards

```yaml
task_card:
  task_id: STAGE1-FS-001
  sub_agent: "codex-native"
  role: "analyst"
  objective: "Audit current Stage1 frame Search baseline and define promotion gate."
  file_ownership:
    - "docs/resources/stage1_frame_search_baseline_plan_2026_06_21.md"
  assigned_skill:
    - "agent-hierarchy-runtime-manager"
    - "caveman:caveman"
  inputs:
    - "runs/NON_XR/raw/*stage1*/train/history.json"
    - "configs/external/mode1_stage1.yaml"
    - "src/hbtxr/loss/stage1.py"
  outputs:
    - "baseline gate and experiment plan"
  validation:
    - "history summary confirms current best Stage1 Search metrics"
  dependencies: []
```

```yaml
task_card:
  task_id: STAGE1-FS-002
  sub_agent: "codex-native"
  role: "implementer"
  objective: "Create runnable two-GPU Stage1 frame Search baseline matrix."
  file_ownership:
    - "scripts/external/run_stage1_frame_search_baseline_matrix.sh"
  assigned_skill:
    - "bash-scripting"
  inputs:
    - "scripts/external/train_hbtxr.py"
    - "data/_internal/manifests/manifest1"
  outputs:
    - "detached two-lane training runner"
  validation:
    - "bash -n passes"
    - "readiness check passes before launch"
    - "logs and pid files emitted"
  dependencies:
    - "STAGE1-FS-001"
```

## Follow-Up Decision

After both lanes complete:

1. Summarize histories with `scripts/external/summarize_training_history.py`.
2. Promote the lane/checkpoint only if it beats the current Stage1 gate.
3. Use the promoted `best_search_p10.pt` as the new Stage2 initialization baseline.
4. If neither lane promotes, keep the previous Stage1 run and try a narrower LR/loss bracket:
   - `lr=7.5e-4`, `search_xy_weight=1.25`
   - `lr=3e-4`, `search_xy_weight=1.5`, `search_ab_weight=0.75`
