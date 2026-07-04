# MANIFEST v2 — handover-v2 산출물 인덱스

위치: `E:\DATASET\codes\data-distribution\handover-v2\` (01_excel 13 · 02_docs 1 · 03_tables 39 · 04_code 44)

## 01_excel/ (현재 산출물 13)

| 파일 | 내용 | 상태 |
|---|---|---|
| 4State_Frame_rawcount_all48.xlsx | Frame 4-state, raw frame count, 48명 | 최신 |
| 4State_Event_rawcount_all48.xlsx | Event 4-state, raw event timestamp-motion, 48명(2.76B) | 최신 |
| HW_Invocation_raw_all48.xlsx | invocation(N_F·N_E·N_EV·I_S·I_T·R_S) 48명 | 최신 |
| JETCAS_REPLY_TABLES (HW)_E2E_final_B.xlsx | **(HW) Latency 최신** E2E+IP II+R_S 8.66~13.52+track-skip(B) | 최신(8× 이슈) |
| JETCAS_REPLY_TABLES (HW)_E2E_final.xlsx | E2E + R_S 8.45~13.07 (track-skip 전) | 참고 |
| JETCAS_REPLY_TABLES (HW)_E2E_IoUskip.xlsx | E2E + IoU-skip(τ0.5, R_S~0.44%) | 참고 |
| DataAug_4state_excl.xlsx | Frame 4-state(I-VT@40ms) before/after-aug | |
| EventStream_4state_excl.xlsx | Event 4-state(5ms grid) | |
| FrameCenter_4state_excl.xlsx | Frame center literal 속도 4-state | |
| Velocity_distribution.xlsx | 속도 25-bin 히스토그램 | |
| Invocation_designspace.xlsx | Search/Track 설계공간 스윕 | |
| Invocation_final.xlsx | 확정 스케줄러 invocation | |
| Refresh_sweep.xlsx | watchdog 주기 trade-off | |

## 02_docs/
- HBTXR_Invocation_Tradeoff.docx — invocation·refresh 트레이드오프(표 5개)

## 03_tables/ (CSV 39)
4-state(aug/event/framecenter/all48), velocity(frame/event raw/interp), invocation(designspace/final/refresh_sweep), 리플라이 채움(latency_rewrite/fill 등), event_motion_counts·testset_sampling.

## 04_code/ (Python 44)
event_motion_raw.py(raw event 모션 집계, 핵심), velocity4state·event_4state·frame_center_4state, velocity_hist, invocation_designspace·invocation_final·refresh_sweep, common 등.

## 비교: v1 → v2 신규
- raw 4-state Frame/Event(all 48, 2.76B 전수 집계) ★신규
- HW Invocation/Latency 11세대(_filled~_E2E_final_B) ★신규
- 스케줄러 개념 정립(invocation/R_S/II/p95-p99/skip 정의) ★신규
- 리플라이 검증(LUT/FF 10× 오타, 전력 범위, Bit-sweep) ★신규

## 원본/아카이브 (복사 안 함)
- EV-Eye: `E:\DATASET\eveye\...`
- superseded 13개: `..\_archive\`
- 이전 패키지: `..\handover-v1\`
- 업로드 보존: JETCAS_REPLY_TABLES (Q|HW|Benchmark).xlsx, jetcas_draft_final_hbtxr.zip

## 최우선 미결 (자세히 00_HANDOVER_v2.md §6)
1. (HW)_E2E_final_B의 8×(5ms) ↔ event-driven 혼재 → event-driven 실측 재계산.
2. (HW) 최종 R_S 정의 확정.
3. 보드 invocation 로그/HLS 합성값으로 II·p95/p99·skip 대체.
