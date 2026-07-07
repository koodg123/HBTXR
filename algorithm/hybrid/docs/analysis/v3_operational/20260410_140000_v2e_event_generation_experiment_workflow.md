# v2e Event Generation 실험 가이드

이 문서는 `v2e`를 `mode2 target-FPS` 경로에서 synthetic event-generation backend로 비교 실험하는 가장 짧은 운영 절차를 정리합니다.

핵심 원칙:

- `v2e`는 frame interpolation backend가 아니다.
- 먼저 target frame을 만든 뒤, 그 frame grid에 맞춰 synthetic events를 생성한다.
- 가장 안전한 비교는 같은 target-FPS / 같은 interpolation backend / 같은 session filter에서 `event_generation_backend=none` 과 `event_generation_backend=v2e`를 나란히 만드는 것이다.

## 1. 실험 전에 확인할 것

- 외부 repo:
  - `Third/EI`
- Python env:
  - `.venv`
- 경로 설정:
  - `configs/paths/ev_eye_groundedsam_paths.json`

이 프로젝트는 `v2e` import 시 optional viewer/output dependency가 없어도 in-memory event generation이 가능하도록 adapter를 둡니다. 그래도 `.venv`에 `screeninfo`, `easygui`, `engineering-notation` 정도는 설치돼 있는 편이 안전합니다. 기본 root는 `Third/EI`이고, `packages/v2e`는 fallback only입니다.

## 2. 가장 권장하는 첫 실험

목표:

- 같은 session에서
  - baseline: `none`
  - candidate: `v2e`
- 두 target session store를 만들고
- frame별 event count, 총 event 수, polarity 비율을 비교한다.

가장 짧은 명령:

```bash
sh exps/scripts/run_v2e_event_experiment.sh \
  --target-fps 50 \
  --user-id 1 \
  --eye left \
  --session-code 101 \
  --max-sessions 1 \
  --interpolation-backend linear_blend \
  --v2e-device cpu \
  --overwrite
```

event 폭주가 보이면 아래 파라미터를 같이 조절한다.

```bash
  --v2e-pos-thres 0.5 \
  --v2e-neg-thres 0.5 \
  --v2e-sigma-thres 0.02 \
  --v2e-cutoff-hz 30 \
  --v2e-leak-rate-hz 0.0 \
  --v2e-shot-noise-rate-hz 0.0
```

산출물 기본 위치:

- `workspace/v2e_experiments/default/none_target_data`
- `workspace/v2e_experiments/default/v2e_target_data`
- `workspace/v2e_experiments/default/reports/summary.json`

## 3. TimeLens / TimeLens-XL과 같이 쓰는 경우

`v2e`는 frame interpolation 다음 단계이므로 아래처럼 조합한다.

TimeLens:

```bash
sh exps/scripts/run_v2e_event_experiment.sh \
  --target-fps 2000 \
  --user-id 1 \
  --eye left \
  --session-code 101 \
  --max-sessions 1 \
  --interpolation-backend timelens \
  --timelens-checkpoint /path/to/attention.bin \
  --timelens-device cuda:0 \
  --v2e-device cuda:0 \
  --overwrite
```

TimeLens-XL:

```bash
sh exps/scripts/run_v2e_event_experiment.sh \
  --target-fps 2000 \
  --user-id 1 \
  --eye left \
  --session-code 101 \
  --max-sessions 1 \
  --interpolation-backend timelens_xl \
  --timelens-xl-checkpoint /path/to/TimeLens_1.pt \
  --timelens-xl-device cuda:0 \
  --v2e-device cuda:0 \
  --overwrite
```

## 4. compare-only 모드

이미 `none_target_data`와 `v2e_target_data`를 만들어 둔 경우 비교만 따로 돌릴 수 있다.

```bash
PYTHONPATH=src .venv/bin/python exps/scripts/compare_v2e_event_generation.py \
  --baseline-root ./workspace/v2e_experiments/default/none_target_data \
  --candidate-root ./workspace/v2e_experiments/default/v2e_target_data \
  --target-fps 50 \
  --output-dir ./workspace/v2e_experiments/default/reports
```

## 4.1 threshold sweep

`v2e`가 under-generate 또는 over-generate 하는지 빠르게 보려면 sweep를 먼저 돌리는 편이 좋다.

```bash
sh exps/scripts/run_v2e_event_sweep.sh \
  --experiment-root ./workspace/v2e_experiments/sweep50 \
  --target-fps 50 \
  --user-id 1 \
  --eye left \
  --session-code 101 \
  --max-sessions 1 \
  --interpolation-backend linear_blend \
  --v2e-device cpu \
  --v2e-pos-thres-values 0.50,0.45,0.40 \
  --v2e-sigma-thres 0.02 \
  --v2e-cutoff-hz 30 \
  --v2e-leak-rate-hz 0.0 \
  --v2e-shot-noise-rate-hz 0.0 \
  --overwrite
```

주요 산출물:

- `reports/sweep_summary.json`
- `reports/<candidate_label>/summary.json`
- `reports/<candidate_label>/*_event_count_plot.png`

## 5. 해석할 때 우선 보는 값

`summary.json`에서 먼저 볼 것:

- `baseline_total_events`
- `candidate_total_events`
- `event_ratio_candidate_vs_baseline`
- `baseline_pos_fraction`
- `candidate_pos_fraction`
- per-session `event_count_curve_mae`

세션별 `*_event_count_plot.png`는 target frame index별 event count curve를 보여준다.

해석 기준:

- 총 event 수만 크게 늘고 curve shape가 완전히 깨지면 `v2e`가 과도하게 noisy할 수 있다.
- total event ratio가 적당하고, curve shape가 baseline과 비슷하면서 더 motion-aware한 구간에서만 차이가 커지면 좋은 후보일 수 있다.
- polarity 비율이 한쪽으로 심하게 쏠리면 threshold/noise 파라미터 calibration이 필요할 가능성이 크다.

## 5.1 v2e calibration에 바로 쓰는 옵션

- `--v2e-pos-thres`
- `--v2e-neg-thres`
- `--v2e-sigma-thres`
- `--v2e-cutoff-hz`
- `--v2e-leak-rate-hz`
- `--v2e-shot-noise-rate-hz`
- `--v2e-refractory-period-s`
- `--v2e-seed`

## 6. 현재 제한

- 현재 기본 실험은 `v2e` 파라미터 sweep까지 자동화하지 않는다.
- 즉 첫 단계는 `backend 연결 + none/v2e 비교`에 집중한다.
- 다음 단계 후보:
  - `pos_thres`, `neg_thres`, `sigma_thres` calibration
  - session별 event ratio sweep
  - downstream `mode2 stage2` 성능 비교

## 7. 추천 실험 순서

1. `linear_blend + none vs v2e` 작은 session 1개
2. `timelens_xl + none vs v2e` 같은 session 1개
3. event count / polarity ratio / per-frame curve 비교
4. 그 다음에만 더 큰 target-FPS나 여러 session으로 확대
