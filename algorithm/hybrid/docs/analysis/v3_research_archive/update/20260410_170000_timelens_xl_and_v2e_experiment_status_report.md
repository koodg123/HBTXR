# TimeLens-XL 및 v2e 실험 현황 보고서

이 문서는 `HBTXR_v3_0`에서 최근 진행한 `TimeLens-XL` fine-tuning 실험과 `v2e` synthetic event-generation 실험의 현재 상태를 한 번에 정리한다.

범위:

- `TimeLens-XL` fine-tuning export / launcher / reconnect
- `linear_blend` 대비 `TimeLens-XL` interpolation 비교
- `v2e` event-generation backend 연결
- `v2e` calibration 실험과 현재까지의 해석

## 1. 현재 결론 요약

### 1.1 TimeLens-XL

- HBTXR에서 `TimeLens-XL` fine-tuning dataset export와 native launcher가 구현되어 있다.
- small smoke run과 real-mini run이 모두 성공했고, native checkpoint 생성도 확인했다.
- 생성된 checkpoint를 HBTXR mode2 interpolation runtime이 다시 읽도록 reconnect 경로도 열렸다.
- 다만 현재 real-mini 결과는 `linear_blend`를 크게 압도하는 상태는 아니다.
- sampled pair 기준으로는 `TimeLens-XL`이 `linear_blend`에 비해 소폭의 motion-aware 보정을 주는 정도로 보인다.

### 1.2 v2e

- HBTXR에서 `v2e`를 synthetic event-generation backend로 비교 실험할 수 있는 표면이 구현되어 있다.
- `target-FPS` build 경로에 `v2e` threshold / noise 파라미터를 넘길 수 있다.
- optional dependency가 일부 없어도 in-memory event generation이 가능하도록 adapter가 보강되었다.
- 가장 큰 안정성 문제였던 absolute timestamp 입력은 상대 timestamp 정규화로 완화했다.
- 현재 calibration 결과를 보면 `threshold=0.50`은 보수적이고, `0.45` 이하로 낮추면 baseline 대비 event 수가 빠르게 증가한다.

## 2. 반영된 구현 범위

### 2.1 TimeLens-XL 관련 구현

핵심 구현:

- `src/hbtxr/preprocess/timelens_xl_finetune.py`
- `exps/scripts/export_timelens_xl_finetune_dataset.py`
- `exps/scripts/finetune_timelens_xl.py`
- `exps/scripts/run_export_timelens_xl_finetune_dataset.sh`
- `exps/scripts/run_finetune_timelens_xl.sh`
- `src/hbtxr/preprocess/interpolation.py`

운영 문서:

- `docs/prj/20260410_110000_timelens_xl_finetuning_workflow.md`

### 2.2 v2e 관련 구현

핵심 구현:

- `src/hbtxr/preprocess/event_generation.py`
- `src/hbtxr/preprocess/target_fps_build.py`
- `src/hbtxr/preprocess/v2e_experiment.py`
- `src/hbtxr/utils/external_packages.py`
- `exps/scripts/build_target_fps_dataset.py`
- `exps/scripts/compare_v2e_event_generation.py`
- `exps/scripts/run_v2e_event_experiment.py`
- `exps/scripts/run_v2e_event_experiment.sh`
- `exps/scripts/run_v2e_event_sweep.py`
- `exps/scripts/run_v2e_event_sweep.sh`

운영 문서:

- `docs/prj/20260410_140000_v2e_event_generation_experiment_workflow.md`

테스트:

- `tests/test_v2e_experiment_v3.py`
- `tests/test_interpolation_v3.py`
- `tests/test_external_packages_v3.py`
- `tests/test_surface_split_v3.py`

## 3. TimeLens-XL 현재 상태

### 3.1 smoke run

smoke run에서는 native trainer가 끝까지 실행되고 checkpoint가 생성되는 것까지 확인했다.

예시 산출물:

- `workspace/timelens_xl_finetune_runs_smoke/TimeLens_native_smoke/weights/TimeLens_0.pt`
- `workspace/timelens_xl_finetune_runs_smoke/TimeLens_native_smoke/weights/TimeLens_1.pt`

이 단계의 의미는 “품질 검증”보다 “학습 경로 / 체크포인트 생성 / reconnect 경로” 확인에 가깝다.

### 3.2 real-mini run

real EV-Eye mini-session 기반 run도 수행했다.

대표 로그:

- `workspace/timelens_xl_finetune_runs_realmini/TimeLens_realmini/training_record.txt`

마지막 epoch 기준 기록:

- `Charbonier = 0.0072`
- `train PSNR = 30.2466`
- `val l1_loss = 0.0025`
- `val PSNR = 40.4634`
- `val SSIM = 0.9960`

이 run은 strict supervised middle-GT fine-tuning이라기보다, current export surface 기반의 real-mini 적응 실험에 가깝다.

### 3.3 reconnect 상태

generated checkpoint를 HBTXR interpolation runtime이 다시 읽도록 연결했다.

예시 산출물:

- `workspace/timelens_xl_finetune_runs_realmini/direct_reconnect_pair_000001.png`

single-pair reconnect는 성공했다. 다만 full-session materialization을 대규모로 재검증하는 작업은 아직 남아 있다.

### 3.4 linear_blend 대비 비교 결과

비교 산출물:

- `workspace/timelens_xl_finetune_runs_realmini/linear_blend_vs_timelens_xl_multi/summary.json`
- `workspace/timelens_xl_finetune_runs_realmini/comparison_gifs_from_validation/raw_vs_raw_plus_interpolated_from_validation.gif`
- `workspace/timelens_xl_finetune_runs_realmini/comparison_gifs_raw_runtime/raw_vs_raw_plus_interpolated_direct_runtime.gif`

sampled pair 6개 기준 pupil crop 차이:

- 가장 차이가 큰 구간:
  - `000410 -> 000411`
  - `crop_mae = 0.5197`
  - `endpoint_delta_mean = 8.1469`
- 가장 차이가 작은 구간:
  - `000210 -> 000211`
  - `crop_mae = 0.2823`

현재 해석:

- `TimeLens-XL` 결과는 `linear_blend`와 매우 비슷한 편이다.
- 움직임이 큰 pair에서만 차이가 조금 커진다.
- 즉 지금 단계에서는 `linear_blend`를 명확히 압도했다기보다, “안정적으로 붙었고 소폭 보정이 보이는 상태”로 보는 것이 맞다.

## 4. v2e 현재 상태

### 4.1 backend 연결 상태

`v2e`는 현재 HBTXR에서 frame interpolation backend가 아니라 synthetic event-generation backend로 붙어 있다.

비교 실험 흐름:

1. 같은 target frame grid를 생성
2. baseline `none` 또는 candidate `v2e`로 event packet 생성
3. 총 event 수, polarity 비율, frame-wise curve를 비교

### 4.2 안정화 작업

초기에는 absolute epoch timestamp가 `v2e` 내부 lowpass 계산으로 그대로 들어가면서 비정상적으로 큰 delta-time 경고가 발생했다.

현재는 `src/hbtxr/preprocess/event_generation.py`에서:

- frame timestamp를 첫 frame 기준 상대시간으로 정규화해서 `v2e`에 전달
- output event timestamp만 다시 원래 시간축으로 복원

이 변경 후 기존의 매우 큰 `delta_time/tau` 경고는 사라졌고, 현재는 `0.02s` 간격 기준의 정상적인 lowpass 경고만 남는다.

### 4.3 full real-mini 기준 비교

baseline:

- source: `none`
- temporary experiment root: `/tmp/v2e_realmini_exp_cal`
- summary: `/tmp/v2e_realmini_exp_cal/reports/summary.json`

`threshold=0.50`, `sigma=0.02`, `cutoff=30`, `leak=0`, `shot_noise=0` 기준:

- `baseline_total_events = 122395`
- `candidate_total_events = 61548`
- `event_ratio_candidate_vs_baseline = 0.5029`
- `baseline_pos_fraction = 0.5366`
- `candidate_pos_fraction = 0.4907`
- `event_count_curve_mae = 355.16`

현재 해석:

- 이 설정은 event 폭주 방지에는 성공했다.
- 그러나 baseline 대비 event 수를 절반 수준으로 생성해 under-generate 쪽이다.

### 4.4 fast calibration 결과

긴 full-session CPU run 대신, baseline target frame 일부를 재사용해 threshold 방향성을 빠르게 본 결과를 남긴다.

임시 summary:

- `/tmp/v2e_realmini_fast_sweep8/summary.json`

8-frame calibration 결과:

- `0.50`
  - total events: `65 / 77`
  - ratio: `0.8442`
  - curve MAE: `7.75`
- `0.45`
  - total events: `93 / 77`
  - ratio: `1.2078`
  - curve MAE: `9.00`
- `0.40`
  - total events: `129 / 77`
  - ratio: `1.6753`
  - curve MAE: `11.75`
- `0.35`
  - total events: `209 / 77`
  - ratio: `2.7143`
  - curve MAE: `19.75`

현재 해석:

- `0.50`은 조금 부족하다.
- `0.45`는 이미 baseline보다 약간 많다.
- `0.40` 이하로 내려가면 event가 빠르게 과생성된다.
- 현재 조건에서는 `0.47~0.48` 근처를 다음 후보로 보는 것이 자연스럽다.

## 5. 현재까지의 검증 상태

TimeLens-XL와 v2e 관련 변경에 대해 아래 검증이 통과했다.

- `python3 -m py_compile` 관련 파일 통과
- `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_v2e_experiment_v3.py tests/test_interpolation_v3.py`
  - `14 passed`
- 이전 단계에서:
  - `tests/test_external_packages_v3.py`
  - `tests/test_surface_split_v3.py`
  - 관련 help smoke
  도 통과한 상태였다.

## 6. 남은 작업

### 6.1 TimeLens-XL

- pretrained checkpoint 기반의 더 긴 real fine-tuning 재실행
- full-session reconnect materialization 재검증
- downstream `mode2` 학습 성능 비교

### 6.2 v2e

- `0.47`, `0.48`, `0.49` 근처 추가 sweep
- full `191` target frame 기준으로 최적 threshold 재검증
- 필요 시 polarity imbalance를 기준으로 `pos_thres` / `neg_thres` 비대칭 조정
- baseline `none` 대비 downstream 학습 효과 비교

## 7. 현재 판단

- `TimeLens-XL`
  - 학습 경로와 reconnect는 성공적으로 열렸다.
  - 현재 품질은 안정적이지만, `linear_blend` 대비 개선 폭은 아직 크지 않다.
- `v2e`
  - backend 연결과 calibration 경로는 열렸다.
  - 가장 중요한 안정성 이슈였던 absolute timestamp 문제는 완화되었다.
  - threshold는 이제 `0.50`과 `0.45` 사이를 더 촘촘히 보는 단계로 들어갈 수 있다.
