# HBTXR_v3_0 실제 EV-Eye Mode2 Optimizer-Pool CPU 파일럿

최종 갱신: 2026-03-29 KST

Role: `데이터 파이프라인 엔지니어`, `학습 벤치마크 엔지니어`, `CUDA 환경 감사자`

## 요약

이 문서는 2026-03-29에 실행한 첫 bounded `real EV-Eye` `mode2` optimizer-pool CPU 파일럿을 정리한다.

실행 범위:

- 실제 EV-Eye frame/event 소스
- 제한된 부분집합
  - `user01/left/session_102`
  - `user01/right/session_102`
- `100 FPS` target-FPS build
- CPU 전용
- `mode2 stage1` optimizer-pool sweep
- `mode2 stage2` optimizer-pool sweep

해석 시 주의할 점:

- 실제 데이터 기반이지만 full EV-Eye benchmark는 아니다.
- 두 세션만 사용한 bounded subset이다.
- CUDA benchmark가 아니다.
- stage2는 1 epoch 파일럿에서 후보 전반에 `NaN` train loss가 발생했기 때문에, 최종 순위가 아니라 stability screen으로 해석해야 한다.

## 1. 실행 기준

### 실제 데이터 소스

- raw EV-Eye root
  - `E:\WSL\Shared\dataset\Eye\EV_Eye\raw_data\Data_davis`
- canonical EV-Eye root
  - `E:\WSL\Shared\dataset\Eye\EV_Eye\canonical`

이 canonical root는 다음 산출물을 재사용하는 기반으로 사용됐다.

- `events.npz`
- eye-region bbox 전용 annotation row
- `frame_index.jsonl`

### 임시 실제 데이터 subset 계약

현재 target-FPS builder는 `events.npz`를 요구하지만, raw EV-Eye 세션은 `events.txt`를 제공한다.

이번 bounded pilot에서는 아래와 같이 임시 bridge를 사용했다.

- raw 세션 레이아웃 유지
  - `frames/`
  - `timestamps.txt`
  - `user_1.csv`
- canonical 세션 레이아웃 재사용
  - `events/events.npz`
- eye ROI annotation은 다음 소스에서 bbox-only JSONL row로 합성
  - `labels/eye_region.json`
  - `labels/frame_index.jsonl`

이 bridge는 임시 pilot workspace 안에서만 사용했으며, benchmark 종료 후 삭제했다.

### Target-FPS build 결과

설정:

- `target_fps=100`
- `frame_storage_mode=materialized_target_frames`
- `session_store_format=npz`
- `interpolation_backend=linear_blend`

결과:

- planned sessions: `2`
- planned target frames: `21,426`
- materialized sessions: `2`
- rebinned events materialized: `8,146,197`
- pending annotation rows: `0`

### 실제 pilot manifest subset

전체 `manifest2`는 다음 크기였다.

- total rows: `21,426`

CPU 비교 비용을 제한하기 위해 실제 파일럿은 아래 subset만 사용했다.

- train subset: `64 rows`
- val subset: `64 rows`

## 2. 실행 루트

Bootstrap run:

- [hbtxr_mode2_stage1_real_ev_eye_bootstrap_cpu_pilot_20260329_011211](../../runs/hbtxr_mode2_stage1_real_ev_eye_bootstrap_cpu_pilot_20260329_011211)

Stage1 pool run:

- [hbtxr_mode2_stage1_real_ev_eye_optimizer_pool_cpu_pilot_20260329_011237](../../runs/hbtxr_mode2_stage1_real_ev_eye_optimizer_pool_cpu_pilot_20260329_011237)

Stage2 pool run:

- [hbtxr_mode2_stage2_real_ev_eye_optimizer_pool_cpu_pilot_20260329_011502](../../runs/hbtxr_mode2_stage2_real_ev_eye_optimizer_pool_cpu_pilot_20260329_011502)

후보 optimizer:

- `adamw`
- `lion`
- `prodigy`
- `adopt`
- `adam_mini`
- `adema_mix`
- `ivon`
- `sophia_g`
- `mars`
- `soap`
- `musgd`

공통 제약:

- `epochs=1`
- `num_workers=0`
- `device=cpu`
- stage1 `batch_size=8`
- stage2 `batch_size=1`

## 3. Stage1 결과

주요 관찰:

- `metric_search_p10_pct`는 모든 optimizer에서 `0.0`으로 동일했다.
- 따라서 이번 pilot에서는 `val.loss_total`이 가장 유용한 보조 판별 지표였다.

`val.loss_total` 기준 Stage1 순위:

| 순위 | Optimizer | `val.loss_total` | `elapsed_sec` |
|---|---|---:|---:|
| 1 | `adam_mini` | 423.8818 | 13.3796 |
| 2 | `sophia_g` | 424.4201 | 11.1533 |
| 3 | `lion` | 424.5163 | 11.3774 |
| 4 | `mars` | 424.6501 | 11.0968 |
| 5 | `adema_mix` | 424.6777 | 11.1557 |
| 6 | `adamw` | 424.7567 | 12.0783 |
| 7 | `adopt` | 425.9746 | 11.5530 |
| 8 | `soap` | 427.8786 | 11.4684 |
| 9 | `musgd` | 429.0080 | 11.3589 |
| 10 | `prodigy` | 429.2177 | 11.3707 |
| 11 | `ivon` | 429.2179 | 11.0588 |

해석:

- bounded CPU pilot 기준으로는 `adam_mini`가 가장 낮은 validation loss를 보였다.
- 그러나 주 metric이 동률이었으므로, 이를 최종 Stage1 권고안으로 받아들이면 안 된다.

## 4. Stage2 결과

주요 관찰:

- `metric_track_p10_pct`는 `10/11` optimizer에서 `100.0`이었다.
- `soap`는 `0.0`으로 collapse했다.
- 모든 후보에서 1 epoch train 동안 `train_loss=NaN`이 발생했다.
- validation은 `10/11` 후보에서 finite였지만, 이번 실행은 사실상 stability screen에 가깝다.

finite `val.loss_total` 기준 Stage2 순위:

| 순위 | Optimizer | `val.loss_total` | `val.metric_track_center_px` | `elapsed_sec` |
|---|---|---:|---:|---:|
| 1 | `adam_mini` | 438.4396 | 0.1300 | 14.9707 |
| 2 | `mars` | 455.1349 | 0.1646 | 19.9202 |
| 3 | `sophia_g` | 455.9976 | 0.1181 | 18.0756 |
| 4 | `adamw` | 457.6546 | 0.1629 | 18.7888 |
| 5 | `adema_mix` | 457.6701 | 0.1628 | 14.9656 |
| 6 | `adopt` | 460.7784 | 0.1993 | 15.3022 |
| 7 | `musgd` | 471.9880 | 0.2243 | 19.8703 |
| 8 | `prodigy` | 483.4749 | 0.1112 | 16.8003 |
| 9 | `ivon` | 483.4985 | 0.1112 | 18.5794 |
| 10 | `lion` | 485.8306 | 0.2798 | 24.5263 |
| 11 | `soap` | `NaN` | `NaN` | 19.7047 |

해석:

- `adam_mini`가 가장 낮은 finite validation loss를 기록했다.
- `prodigy`, `ivon`은 finite run 중 가장 낮은 `metric_track_center_px`를 보였지만, `val.loss_total`은 상대적으로 높았다.
- `soap`는 이 bounded pilot에서 실패했다.
- 전체 후보에서 `train_loss=NaN`이 발생했으므로, 이번 Stage2 결과는 신뢰 가능한 최종 순위가 아니다.

## 5. CUDA benchmark 상태

2026-03-29 기준으로 프로젝트 환경의 CUDA 실행 가능 여부를 점검했다.

확인된 사실:

- 프로젝트 interpreter
  - `E:\WSL\Shared\ETRI_SYNC\HBTXR\paper_works\CODE\HBTXR_v3_0\.venv\Scripts\python.exe`
- 해당 환경 내부
  - `torch.cuda.is_available() == False`
  - `torch.cuda.device_count() == 0`
  - `torch.version.cuda == None`
- 시스템 GPU는 `nvidia-smi`로 확인됨
  - `NVIDIA GeForce RTX 4070 Ti`
  - `12282 MiB`
  - driver `591.86`

결론:

- 현재 프로젝트 `.venv`에서는 실제 CUDA throughput / memory benchmark를 실행할 수 없었다.
- 병목은 하드웨어 부재가 아니라 환경 불일치다.
- 이 저장소에 CUDA-enabled PyTorch 환경이 연결될 때까지 CUDA benchmark는 보류다.

## 6. 운영 결론

이번 패스에서 확인된 것:

- real EV-Eye bounded `mode2` 데이터를 target-FPS build 경로에 bridge할 수 있다.
- stage1 optimizer-pool 실행이 real-data subset에서 완료된다.
- stage2 optimizer-pool 실행이 real-data subset에서 완료된다.
- 이전 synthetic pilot 임시 디렉터리는 정리되었다.
  - `.tmp_mode2_pool_pilot/`
- 실제 데이터용 임시 pilot workspace도 결과 캡처 후 정리되었다.

이번 패스로는 아직 증명되지 않은 것:

- full-dataset EV-Eye optimizer ranking
- 실제 데이터에서의 안정적 Stage2 수렴
- CUDA throughput ranking
- CUDA peak-memory ranking

## 7. 권장 다음 단계

권장 순서:

1. 이 저장소에 CUDA-enabled PyTorch 환경을 연결한다.
2. 같은 bounded real EV-Eye subset을 CUDA로 재실행하고 다음을 수집한다.
   - wall-clock
   - step time
   - peak GPU memory
3. 현재 두 세션 subset에서 더 큰 real EV-Eye target-FPS subset으로 확장한다.
4. 그 다음에만 optimizer별 권고안을 pilot 수준 이상으로 승격한다.
