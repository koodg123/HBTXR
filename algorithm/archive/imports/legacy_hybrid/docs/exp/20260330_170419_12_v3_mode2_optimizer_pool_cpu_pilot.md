# HBTXR_v3_0 Mode2 Optimizer Pool CPU 파일럿

최종 갱신: 2026-03-29 KST

Role: `최적화 시스템 아키텍트`, `학습 벤치마크 엔지니어`, `합성 데이터셋 검증 리뷰어`

## 요약

이 문서는 `mode2` optimizer-pool preset과 smoke-path 정리가 끝난 뒤 처음 실행한 실제 CPU 파일럿 결과를 정리한다.

이번 파일럿의 범위는 다음과 같다.

- synthetic target-FPS canonical workspace
- CPU 전용
- `mode2 stage1` optimizer-pool sweep
- `mode2 stage2` optimizer-pool sweep

해석 시 주의할 점:

- 실제 EV-Eye 벤치마크가 아니다.
- CUDA 벤치마크가 아니다.
- 목적은 경량 실행 파일럿이다.
  - 현재 구현된 optimizer가 모두 `mode2`에서 완주하는지
  - stage1/stage2 pool report가 end-to-end로 생성되는지
  - 대규모 실행 전에 상대 경향을 관찰할 수 있는지

## 1. 실행 기준

워크스페이스 기준:

- 아래 테스트와 같은 mini session 패턴으로 만든 synthetic target-FPS workspace를 사용했다.
  - [test_target_fps_canonical_v3.py](../../tests/test_target_fps_canonical_v3.py)

실행 루트:

- stage1 bootstrap
  - `runs/hbtxr_mode2_stage1_bootstrap_cpu_pilot_20260329_004722`
- stage1 pool
  - `runs/hbtxr_mode2_stage1_optimizer_pool_cpu_pilot_20260329_004734`
- stage2 pool
  - `runs/hbtxr_mode2_stage2_optimizer_pool_cpu_pilot_20260329_004801`

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
- `batch_size=1`
- `num_workers=0`
- `device=cpu`
- 매우 작은 synthetic `mode2` manifest

## 2. Stage1 결과

주요 관찰:

- `metric_search_p10_pct`는 모든 optimizer에서 `0.0`으로 동일했다.
- 주 지표가 동률이었기 때문에 이번 실행에서 가장 유용한 보조 구분자는 `val.loss_total`이었다.

`val.loss_total` 기준 Stage1 순위:

| 순위 | Optimizer | `val.loss_total` | `val.metric_search_center_px` | `elapsed_sec` |
|---|---|---:|---:|---:|
| 1 | `adopt` | 419.3124 | 207.9951 | 1.6544 |
| 2 | `adam_mini` | 420.0957 | 206.4349 | 1.4424 |
| 3 | `adema_mix` | 439.2731 | 218.1780 | 1.5076 |
| 4 | `adamw` | 439.3921 | 218.2427 | 2.4226 |
| 5 | `mars` | 441.3722 | 219.2937 | 1.7392 |
| 6 | `sophia_g` | 441.4869 | 219.2450 | 1.5055 |
| 7 | `lion` | 441.5187 | 219.2862 | 1.5428 |
| 8 | `soap` | 448.1663 | 222.6282 | 1.8336 |
| 9 | `musgd` | 450.2628 | 223.8744 | 1.8515 |
| 10 | `prodigy` | 452.9863 | 225.3701 | 1.6715 |
| 11 | `ivon` | 452.9875 | 225.3703 | 1.4593 |

해석:

- 이 작은 synthetic Stage1 파일럿에서는 `adopt`, `adam_mini`가 가장 낮은 validation loss를 보였다.
- 모든 optimizer가 정상 종료했다.
- 다만 search metric 자체는 이 축소된 셋업에서 충분히 구분력이 없었다.

## 3. Stage2 결과

주요 관찰:

- `metric_track_p10_pct`는 모든 optimizer에서 `98.7654`로 동일했다.
- 따라서 실제로 구분에 도움이 된 값은 다음 두 개였다.
  - 낮을수록 좋은 `val.metric_track_center_px`
  - `val.loss_total`

`val.metric_track_center_px` 기준 Stage2 순위:

| 순위 | Optimizer | `val.metric_track_center_px` | `val.loss_total` | `elapsed_sec` |
|---|---|---:|---:|---:|
| 1 | `soap` | 0.1392 | 499.6333 | 3.9351 |
| 2 | `adamw` | 0.1784 | 477.5470 | 4.7599 |
| 3 | `musgd` | 0.2082 | 493.9846 | 4.2504 |
| 4 | `adema_mix` | 0.2278 | 477.3322 | 3.2830 |
| 5 | `adopt` | 0.2450 | 468.6893 | 3.6644 |
| 6 | `adam_mini` | 0.2453 | 441.4904 | 3.9997 |
| 7 | `mars` | 0.2696 | 480.6760 | 3.3960 |
| 8 | `sophia_g` | 0.2843 | 480.6802 | 3.6190 |
| 9 | `lion` | 0.3273 | 482.2992 | 3.8701 |
| 10 | `prodigy` | 0.4180 | 510.7205 | 3.7934 |
| 11 | `ivon` | 0.4180 | 510.7223 | 3.1007 |

`val.loss_total` 기준 보조 해석:

- `adam_mini`가 가장 낮은 validation loss를 기록했다: `441.4904`
- `adopt`가 그다음이었다: `468.6893`
- `adema_mix`와 `adamw`가 그 뒤를 이었다.

해석:

- 이 작은 synthetic Stage2 파일럿에서는 `soap`가 가장 낮은 tracking center error를 보였다.
- 반면 validation loss 기준으로는 `adam_mini`가 가장 좋았다.
- 주 tracking percentage metric이 포화됐기 때문에, 이번 결과는 최종 순위가 아니라 상대 sanity check로 해석해야 한다.

## 4. 운영 결론

이번 파일럿으로 확인된 것:

- 현재 구현된 모든 optimizer가 `mode2 stage1`에서 완주한다.
- 현재 구현된 모든 optimizer가 `mode2 stage2`에서 완주한다.
- 실제 pool runner가 다음 산출물을 정상 생성한다.
  - `optimizer_pool_report.json`
  - `optimizer_pool_summary.csv`
  - `optimizer_pool_rankings.json`
- stage2 pool도 일반적인 stage1 bootstrap checkpoint와 함께 정상 동작한다.

이번 파일럿으로 아직 증명되지 않은 것:

- real EV-Eye 기준 순위
- CUDA 기준 throughput 순위
- multi-epoch 기준 수렴 순위

## 5. 권장 다음 단계

권장 실행 순서:

1. 더 큰 synthetic `mode2` budget으로 같은 후보군을 재실행한다.
2. real EV-Eye `mode2` subset으로 재실행한다.
3. CUDA wall-clock과 peak-memory 결과를 수집한다.
4. 그 다음에만 optimizer별 권고안을 README 수준 문서로 승격한다.
