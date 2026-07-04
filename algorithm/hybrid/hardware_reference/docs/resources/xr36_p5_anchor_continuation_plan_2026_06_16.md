# XR-36 P5-Anchor Continuation Plan - 2026-06-16

## Purpose

XR-34B is the center/P10 leader, but P5 remains below the XR-29 gate. XR-35 no-train interpolation did not restore P5. XR-36 is the next trainable branch: continue from the strongest center/P10 or balanced interpolation anchor while using the XR-29 checkpoint as the P5 anchor reference.

## Runner

- Script: `scripts/external/run_xr36_p5_anchor_continuation.sh`
- Base runner: `scripts/external/run_xr27_trackheatmap_p10teacher_probe.sh`
- Contract: fixed255k support-adaptive event count, `160000/255000/384000`, `reference_us=4000003`, `scale_power=0.5`
- Head: heatmap-state grid `32`
- Loss ratio: original heatmap/offset/center `0.005/0.001/0.001`
- Distillation state similarity: disabled, consistent with the promoted XR-27 to XR-35 branch
- Checkpointing metric: `metric_track_center_px`; eval still covers best-P10, best-P5, and best-center

## Lanes

| Lane | Device | Init | Teacher/reference | LR | Purpose |
|---|---|---|---|---:|---|
| XR-36A | `cuda:0` | XR-34B best-P10 | XR-29 best-center | `5e-5` | gently recover P5 from the center/P10 leader |
| XR-36B | `cuda:1` | XR-35 alpha `0.09375` | XR-29 best-center | `8.75e-5` | start from the best XR-35 P5 tradeoff and test stronger correction |

## Commands

```bash
tmux new-session -d -s hgtxr_xr36a_p5anchor_gpu0_20260616 \
  'cd /home/kjm26/project/PRJXR/XR-VIT/HGTXR/software && bash scripts/external/run_xr36_p5_anchor_continuation.sh a > runs/_logs/xr36a_p5anchor_gpu0_20260616.log 2>&1'
```

```bash
tmux new-session -d -s hgtxr_xr36b_p5anchor_gpu1_20260616 \
  'cd /home/kjm26/project/PRJXR/XR-VIT/HGTXR/software && bash scripts/external/run_xr36_p5_anchor_continuation.sh b > runs/_logs/xr36b_p5anchor_gpu1_20260616.log 2>&1'
```

## Gate

- Center promotion: `<16.53321223940168`
- P10 promotion: `>33.77168447630746`
- P5/unified promotion: `>11.50467722075326`

## Decision

- Promote a single checkpoint only if it restores P5 without losing the center/P10 gains beyond the stated active gate policy.
- If XR-36 does not restore P5, prefer a true P5-aware loss/metric change over more interpolation or LR-only replay.

## Results

Both lanes completed with train exit `0`, early-stopped at epoch `7/10`, and produced best-P10, best-P5, and best-center full-test summaries.

| Lane | Checkpoint | Epoch | Center px | P10 | P5 | Decision |
|---|---|---:|---:|---:|---:|---|
| XR-36A | best-P10 | 7 | 16.59279990025929 | 34.39710958344596 | 11.738095617294311 | P5 leader; strong unified/P10-P5 checkpoint |
| XR-36A | best-P5 | 2 | 16.576952314376832 | 34.74064704350063 | 11.415816688537598 | P10 leader |
| XR-36A | best-center | 3 | 16.599328325475966 | 33.54209257534572 | 10.998724787575858 | no active gate |
| XR-36B | best-P10 | 7 | 16.53305721793856 | 34.19387831687927 | 11.502551344462804 | center leader and P10 promotion |
| XR-36B | best-P5 / best-center | 3 | 16.844227249281747 | 32.883078956604 | 11.231718029294695 | no active gate |

Evidence:

- Logs: `runs/_logs/xr36a_p5anchor_gpu0_20260616.log`, `runs/_logs/xr36b_p5anchor_gpu1_20260616.log`
- Summaries: `runs/eval_fixed255k_xr27_trackheatmap_xr36*test*/eval/test/eval_summary.json`

Decision:

- Active center gate is now `<16.53305721793856` from XR-36B best-P10.
- Active P10 gate is now `>34.74064704350063` from XR-36A best-P5.
- Active P5 gate is now `>11.738095617294311` from XR-36A best-P10.
- XR-36A best-P10 is the best practical P10/P5 checkpoint, but it does not own the strict center gate.
- Next P0: XR-37 no-train interpolation between XR-36B best-P10 and XR-36A best-P10 to test whether center can be preserved while keeping the XR-36A P5/P10 gain.
