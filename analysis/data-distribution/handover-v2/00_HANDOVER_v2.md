# HANDOVER v2 — HBTXR (HW) Latency · Invocation · Raw 4-State (v1 이후 종합)

- 프로젝트: **HBTXR** (Runtime-Aware Hybrid Event–Frame Eye Tracking, IEEE JETCAS-2026-0074, ZCU104)
- 데이터셋: **EV-Eye** (48 subj × 2 eyes × 4 sessions = 384 sessions, ~1.49M frames@25Hz, **~2.76B raw events**, Train 1–36 / Val{37,40,45,48} / Test 37–48)
- 작성일: 2026-06-30
- 범위: **handover-v1 이후**의 모든 대화·산출물 종합 (HW Latency 시트 진화, raw 4-state, 스케줄러 개념 정립, 리플라이 검증, 디렉토리 정리)
- 위치: `E:\DATASET\codes\data-distribution\handover-v2\` (01_excel/02_docs/03_tables/04_code)

> v1은 4-state·velocity·invocation·trade-off·리플라이 1차 검토까지 다룸. v2는 그 이후 — 특히 **(HW) Latency 시트의 모델 정립과 11세대 반복, raw 이벤트 전수 4-state, 스케줄러 개념 Q&A**를 종합.

---

## 1. Raw 4-State 분포 (전수 집계 완료)

EV-Eye 전체 48명을 raw 단위로 4-state(Fixation/Saccade/Smooth/Blink) 집계:
- **Frame 표** (`4State_Frame_rawcount_all48`): raw **frame count**(total, blink 포함). Train Sacc 1.17%/Blink 2.27%, Test Sacc 0.88%/Blink 1.95%.
- **Event 표** (`4State_Event_rawcount_all48`): 각 **raw event를 timestamp의 motion 상태로 직접 집계** (2.76B 이벤트 전수, 96+288 세션 청크 처리). Train Sacc 1.58%/Blink 1.11%, Test Sacc 1.37%/Blink 0.97%.
- 방법: event 절대 timestamp → `rel = abs − event_startime` → matcell 모션 타임라인(Saccade=5ms I-VT speed>493, Blink=frame-anchor gap, Fix/Smooth=세션타입)에 매핑.
- **핵심**: Frame과 Event의 motion 비율이 다름(Event Saccade%↑) — saccade는 이벤트 밀도가 높기 때문. "timestamp 직접 집계"의 결과(proportional 아님).

## 2. (HW) Latency 시트 — 모델 진화 (11세대)

같은 표를 여러 invocation 모델로 반복. 각 세대의 모델과 의의:

| 버전 | I_S / I_T / R_S | 의의 |
|---|---|---|
| _filled | 시트 R_S(7~27%) 유지, 빈칸 추정 | 1차 채움 |
| _rewritten | I_S=matcell frame-upd, I_T=8×N_F, R_S~10.6% | 큐시뮬 p95/p99 |
| _corrected | I_S=N_F(valid), R_S=**11.11% 상수**(1:8) | N_F=valid 정의 |
| _matcell | I_S=matcell frame-upd, I_T=matcell event-upd±4%, **R_S 8.45~13.07** | per-subject 변동 |
| _modelA | N_EV=8×total, I_T=8×valid(N_EV≠I_T), R_S 11.15~11.62(blink구동) | I_T<N_EV 분리 |
| _raw | #Frames=total raw, N_EV=8×N_F, R_S=11.11% 상수 | raw 통일 |
| _eventdriven | voxel=K events(K=220.5), N_EV=N_E/K, **R_S 7.5~13.36** | event-driven |
| _E2E_IoUskip | IoU τ=0.5 게이트 search(rare), **R_S~0.44%**, **E2E 통일·IP II** | quality-skip |
| _E2E_matcellRS | E2E 레이아웃 + matcell R_S 8.45~13.07 | 합성 |
| _E2E_final | IoU-skip base + R_S 8.45~13.07(event-driven track) | |
| **_E2E_final_B** (최신) | + **track-skip(blink+IoU<τ voxel skip)**, Skipped Voxels>0, **R_S 8.66~13.52** | 대칭 skip |

## 3. 핵심 개념 정립 (대화에서 확정)

- **Invocation ≠ Sample count**: I_S=#Frames, I_T=#Voxels는 "1 sample=1 invocation, skip 없음" 가정의 결과. 실제는 quality-skip으로 Invocation < Sample.
- **R_S = I_S/(I_S+I_T)** (전체 추론 중 search 비율). cadence로 결정 — 5ms 고정이면 frame:voxel=1:8 → **11.11% 상수**; event-driven이면 **8.45~13.07% 변동**(이벤트 밀도); IoU-게이트면 **~0.44%**(search 드묾).
- **Frame/Event 개수 정의 차이**: Frame은 total(31237) vs valid(=total−blink). Event는 raw(44.7M)/5ms-voxel(248k)/matcell행(294k)/event-update(265k) — **4계층**이라 표마다 다른 양.
- **II**: **IP II**(HLS 합성 kernel II, mode별 상수, 데이터 무관) ≠ **System II**(DMA start–end+큐 포함, 변동). DMA 포함하면 IP II 아님.
- **p95/p99 vs Worst**: 이전엔 worst=service / p95·p99=큐포함 e2e라 순서 깨짐. **mode-gated(5ms 슬롯당 1추론, service<5ms→큐누적 없음)** → e2e≈service, p95/p99=service분포(worst 상한) → **best≤avg≤p95≤p99≤worst** 성립.
- **Skip (대칭, quality 기반)**: Search-skip(track 양호→재anchor 불필요), Track-skip(blink/IoU<τ로 lock 상실→다음 frame까지 event voxel skip, search 재anchor). 둘 다 quality 연동.

## 4. 리플라이 테이블 검증 (Q/HW/Benchmark) 주요 발견

- 🔴 **논문 LUT/FF 10× 오타 의심**: (HW)Resources 합계 145K LUT/227K FF(ZCU104 230K/461K의 63/49%)인데 본문 14.5K/22.7K — DSP/BRAM/URAM(814/184/12)은 동일. 시트가 옳고 본문 오타로 판단.
- 🟠 **전력 1.34W = PL 가속기 동적 단독**; full-chip = 4.65W(static 0.673+dyn-PS 2.45+dyn-PL 1.525). 범위 명시 필요.
- 🟠 **(Q)Bit-sweep 베이스라인 1.279px** ↔ 논문 0.18px (7× 불일치), Pixel Error가 Drop Ratio 역산 수식.
- 4-State/Speed 시트의 채워진 값은 내 계산과 **정확 일치**(Event=EventStream, Frame=I-VT@40ms, Speed=velocity 비율).
- 시트 R_S(7~27%)는 matcell 실측(9~13%)·5ms모델(≤12.5%)과 불일치 → 권장값으로 재작성.

## 5. 디렉토리 정리 (2026-06-30)

- root의 superseded 13개 → `_archive/` 이동: DataDistribution v1/v2/v3, 테스트전용 4State(2), (HW) 이전 8세대(filled·rewritten·corrected·matcell·modelA·raw·eventdriven·E2E_matcellRS).
- root에는 최신 13개 xlsx + 인덱스(README.md) 유지. cache/tables/src/figures/reports 보존.

## 6. 미결 / TODO / 주의

1. 🔴 **(HW)_E2E_final_B의 8× 이슈**: Skipped Voxels=8×(blink+IoU<τ)가 5ms 고정 가정인데 I_T는 event-driven matcell이라 혼재. → **event-driven 실측 기반 재계산 필요**(blink/loss 구간의 실제 event-update를 matcell에서 직접 카운트).
2. **(HW) 최종본 미확정**: R_S 정의(11.11% 상수 / event-driven 변동 / IoU-skip rare) 중 택일 필요.
3. **Latency·II·skip은 Test만 + 시뮬/추정**: Train/Val latency 측정 없음. p95/p99=service분포 추정, IP II=min best-case 추정 → **보드 invocation 로그/HLS 합성값으로 대체** 권장.
4. **논문 본문 수정**: LUT/FF 10× 오타, 전력 범위(1.34W=PL/full 4.65W) 명시.
5. **τ 의존**: track/search skip은 IoU/confidence τ=0.5 기준. τ 바꾸면 R_S·skip 변동(τ0.3→0.24%, 0.5→0.45%, 0.7→0.85% search 등).

## 7. provenance / 파라미터
원본 `E:\DATASET\eveye\{raw_data\Data_davis, processed_data\Frame_event_pupil_track_result(matcell), Data_davis_predict, Pixel_error_evaluation, Benchmark_implementation_code}`. v_sacc=493px/s, r=35px(GT 타원 a≈37,b≈32), voxel 5ms/frame 40ms, τ=0.5, K(event-driven)=220.5. 핵심 스크립트(04_code): event_motion_raw.py(raw event 집계), velocity4state·event_4state·frame_center_4state, velocity_hist, invocation_designspace·invocation_final·refresh_sweep.

산출물 전체 목록은 `00_MANIFEST_v2.md` 참조.
