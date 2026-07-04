# XR-38 Center-Preserve P10/P5 Recovery Plan - 2026-06-17

## Prompt Pack

- Objective: preserve XR-37 alpha `0.50` center leader while recovering XR-36A P10/P5.
- Parent skill: `paper-idea-generator`, IDEA-Gen domain `Research Workflow`, using ablation-study-designer controls.
- Task family: event-based pupil tracking with heatmap-state coordinate head.
- Fixed controls: fixed255k support-adaptive event count, min/base/max `160000/255000/384000`, `reference_us=4000003`, `scale_power=0.5`, heatmap grid `32`, head-only trainable `track_center_heatmap_head.*`, AdamW, loss ratio `0.005/0.001/0.001`.
- Current gates before XR-38: center `<16.507612899371555`, P10 `>34.74064704350063`, P5 `>11.738095617294311`.

## Evidence

- XR-37 alpha `0.50`: `16.507612899371555 / 34.33205857958112 / 11.539116007941109`.
- XR-36A best-P5 P10 leader: `16.576952314376832 / 34.74064704350063 / 11.415816688537598`.
- XR-36A best-P10 P5/practical leader: `16.59279990025929 / 34.39710958344596 / 11.738095617294311`.

## Ablation Matrix

| Lane | Init | Reference checkpoint | LR | Selection metric | Goal |
|---|---|---|---:|---|---|
| XR-38A | XR-37 alpha `0.50` | XR-36A best-P10 | `2.5e-5` | `metric_track_center_px` | keep center and recover P5/P10 softly |
| XR-38B | XR-37 alpha `0.50` | XR-36A best-P5 | `1.25e-5` | `metric_track_p10_pct` | recover strict P10 with lower LR drift |

## Command

```bash
bash scripts/external/run_xr38_center_preserve_p10p5_recovery.sh a
bash scripts/external/run_xr38_center_preserve_p10p5_recovery.sh b
```

## Promotion Rule

- Promote center only if test center `<16.507612899371555`.
- Promote P10 only if test P10 `>34.74064704350063`.
- Promote P5 only if test P5 `>11.738095617294311`.
- If no strict gate improves, keep XR-37 alpha `0.50` as center leader and XR-36A checkpoints as P10/P5 leaders.

## Closeout

| Lane / checkpoint | Center | P10 | P5 | Decision |
|---|---:|---:|---:|---|
| XR-38A best-center | 16.556500560896737 | 33.96471167291914 | 11.519983339309693 | no promotion |
| XR-38A best-P10 | 16.57439456837518 | 33.92984774453299 | 11.455357497079032 | no promotion |
| XR-38A best-P5 | 16.513289058208464 | 34.1815484387534 | 11.56462620326451 | no promotion |
| XR-38B best-P10 | 16.510086681161606 | 34.707483761651176 | 11.223214626312256 | near P10 miss |
| XR-38B best-P5 | 16.513694180761064 | 33.93452457700457 | 11.843962955474854 | P5 promoted |

- Decision: XR-38B best-P5 promotes the strict P5 gate only.
- Active gates after XR-38: center `<16.507612899371555`, P10 `>34.74064704350063`, P5 `>11.843962955474854`.
- Next bounded option: XR-39 no-train soup/interpolation around XR-37 alpha `0.50`, XR-38B best-P5, and XR-36A best-P5 before launching another training branch.

## Risks

- Teacher/reference checkpoint has limited direct force because current heatmap config keeps state distillation weight at `0.0`; movement mainly comes from init, LR, and supervised heatmap loss.
- XR-38B P10 selection can drift center; low LR bounds risk but does not eliminate it.
- If both lanes fail, next bounded option should be no-train soup around XR-37 alpha `0.50`, XR-36A best-P5, and XR-36A best-P10 before more training.
