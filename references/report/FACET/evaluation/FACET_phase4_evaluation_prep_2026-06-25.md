# FACET Phase 4 Evaluation Prep Log

Date: 2026-06-25

## Scope

이 문서는 `FACET_reproduction_plan_2026-06-25.md`의 Phase 4, 즉 full expanded DeanDataset으로 EPNet을 학습한 뒤 FACET 논문 metric과 비교하기 위한 평가/비교 경로 준비 기록이다.

현재 full `DeanDataset_full_unet`과 full EPNet checkpoint는 아직 없다. 따라서 이 문서의 실행 결과는 논문 재현 결과가 아니라, 평가 파이프라인이 동작하는지 확인한 smoke 결과이다.

## Implemented Changes

### EPNet checkpoint evaluation script

추가 파일:

- `EvEye/utils/scripts/evaluate_epnet_checkpoint.py`

역할:

- EPNet config와 checkpoint를 입력으로 받는다.
- val dataloader를 순회해 validation metrics를 평균낸다.
- model parameter count를 계산한다.
- THOP 기반 FLOPs를 계산한다.
- 현재 runtime에서 inference latency를 측정한다.
- local report에서 확인된 FACET Table II reference와 비교한 JSON/Markdown artifact를 저장한다.

입력:

```text
--config
--checkpoint
--output-json
--output-md
--max-batches
--device
--latency-warmup
--latency-iterations
```

출력:

- JSON metric artifact
- Markdown comparison table

## Paper Reference Values

현재 local analysis 문서에서 확인된 Table II reference:

```text
P1:                 99.59%
mean pixel error:   0.2030
params:             3.92M
FLOPs:              3.44G
latency:            0.5302 ms
```

주의:

- P10/P5/P3/IoU/AP의 논문 reference 값은 현재 저장된 report text에서 확인되지 않았다.
- latency reference는 논문 기준이며, 현재 Python/CPU runtime latency와 직접 비교하면 안 된다.

## Smoke Evaluation

Command:

```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/matplotlib-facet \
/home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python \
EvEye/utils/scripts/evaluate_epnet_checkpoint.py \
  --config DavisEyeEllipse_EPNet_local_train_smoke.yaml \
  --checkpoint runs/logs/EPNet_local_train_smoke/version_0/checkpoints/epoch=00-val_mean_distance=38.6962.ckpt \
  --output-json /home/kjm26/project/PRJXR/HBTXR/references/report/FACET/FACET_phase4_epnet_eval_smoke_2026-06-25.json \
  --output-md /home/kjm26/project/PRJXR/HBTXR/references/report/FACET/FACET_phase4_epnet_eval_smoke_2026-06-25.md \
  --max-batches 2 \
  --device cpu \
  --latency-warmup 2 \
  --latency-iterations 5
```

Generated artifacts:

```text
references/report/FACET/FACET_phase4_epnet_eval_smoke_2026-06-25.json
references/report/FACET/FACET_phase4_epnet_eval_smoke_2026-06-25.md
```

Result summary:

```text
evaluated_batches:    2
val_mean_distance:    38.69616508483887
params_m:             3.89828
flops_g:              3.422324568
latency_ms_cpu:       20.294669223949313
```

Metric table summary:

| Metric | Smoke Current | Paper Table II reference | Interpretation |
|---|---:|---:|---|
| P1 | 0.0 | 0.9959 | smoke checkpoint is not trained enough |
| mean pixel error | 38.6962 | 0.2030 | smoke checkpoint is not a reproduction result |
| params M | 3.89828 | 3.92 | close; architecture parameter count is aligned |
| FLOPs G | 3.42232 | 3.44 | close; THOP estimate is aligned with paper scale |
| latency ms | 20.2947 | 0.5302 | CPU Python latency, not TensorRT latency |

## Interpretation

The evaluation path is now available and validated. The smoke checkpoint is intentionally weak because it was trained for only two batches. Therefore:

- metric accuracy values are not meaningful reproduction numbers.
- parameter count and FLOPs are useful architecture sanity checks.
- latency must be re-measured on the target runtime. The paper latency appears to be optimized runtime latency, while this smoke run measured CPU Python inference.

## Full Reproduction Evaluation Template

After Phase 3 creates `DeanDataset_full_unet` and Phase 4 trains the final EPNet checkpoint:

```bash
PYTHONPATH=. FACET_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib-facet \
/home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python \
EvEye/utils/scripts/evaluate_epnet_checkpoint.py \
  --config DavisEyeEllipse_EPNet_full_unet.yaml \
  --checkpoint /path/to/full_epnet_checkpoint.ckpt \
  --output-json /home/kjm26/project/PRJXR/HBTXR/references/report/FACET/FACET_reproduction_results_<date>.json \
  --output-md /home/kjm26/project/PRJXR/HBTXR/references/report/FACET/FACET_table2_comparison_<date>.md \
  --device cuda:0 \
  --latency-warmup 50 \
  --latency-iterations 200
```

Expected full report fields:

```text
P10
P5
P3
P1
mean pixel error
IoU
AP
parameter count
FLOPs
inference latency
dataset root
checkpoint path
config path
runtime device
number of evaluated batches
```

## Current Phase 4 Status

Completed:

- evaluation script for EPNet checkpoints
- parameter count measurement
- FLOPs measurement
- runtime latency measurement
- JSON artifact output
- Markdown comparison artifact output
- smoke validation on existing EPNet checkpoint

Not completed:

- full `DeanDataset_full_unet` generation
- full EPNet training
- full validation split evaluation
- TensorRT or deployment-equivalent latency measurement
- final Table II comparison with reproduction checkpoint

Current blocker:

- GPU driver access is unavailable in the current session.
- full U-Net checkpoint and full expanded dataset are prerequisites for the final EPNet reproduction result.

