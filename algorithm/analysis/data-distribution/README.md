# data-distribution — 디렉토리 인덱스 (정리: 2026-06-30)

HBTXR(JETCAS) 리벗용 EV-Eye 데이터 분석 산출물. 예전/최신이 섞여 있어 superseded 버전을 `_archive/`로 분리했습니다.

## 최신 산출물 (root)

| 파일 | 내용 |
|---|---|
| `4State_Frame_rawcount_all48.xlsx` | Frame 4-state, **raw frame count**, 전체 48명(Train/Val/Test) |
| `4State_Event_rawcount_all48.xlsx` | Event 4-state, **raw event를 timestamp motion으로 직접 집계**, 48명 (2.76B 이벤트 전수) |
| `HW_Invocation_raw_all48.xlsx` | (HW) invocation: N_Frames·N_Events·N_EventVoxels·I_S·I_T·R_S, 48명 |
| `JETCAS_REPLY_TABLES (HW)_E2E_final_B.xlsx` | **(HW) Latency 최신본** — E2E 통일·IP II·R_S 변동(8.66~13.52%)·track-skip(B) |
| `JETCAS_REPLY_TABLES (HW)_E2E_final.xlsx` | (HW) E2E + R_S 8.45~13.07 (track-skip 전) |
| `JETCAS_REPLY_TABLES (HW)_E2E_IoUskip.xlsx` | (HW) E2E + IoU-skip(τ0.5, R_S~0.44%) 기준본 |
| `DataAug_4state_excl.xlsx` | Frame 4-state (I-VT@40ms) before/after-aug |
| `EventStream_4state_excl.xlsx` | Event 스트림 4-state (5ms grid) |
| `FrameCenter_4state_excl.xlsx` | Frame GT-center literal 속도 4-state |
| `Velocity_distribution.xlsx` | 속도 25-bin 히스토그램 (Frame/Event × RAW/Interp) |
| `Invocation_designspace.xlsx` | Search/Track 설계공간 스윕 |
| `Invocation_final.xlsx` | 확정 스케줄러(5ms+40ms) invocation |
| `Refresh_sweep.xlsx` | watchdog 주기 trade-off |

## 폴더

- `handover-v2/` — **최신 핸드오버 패키지** (HANDOVER_v2 + MANIFEST + 산출물 복사)
- `handover-v1/` — 이전 핸드오버 패키지
- `cache/` — 분석 캐시(스크립트 의존, 유지). event_motion_counts.csv 등
- `tables/` — 분석 원자료 CSV
- `src/` — 분석 Python 스크립트
- `figures/` — 그림
- `reports/` — 트레이드오프 DOCX 등
- `_archive/` — superseded(예전) 버전 13개: DataDistribution v1/v2/v3, 테스트전용 4State, (HW) 이전 11세대 중 8개

## 비고
- `~$...` = Excel 임시 락(파일 열림). 무시.
- (HW) 최신본 `_E2E_final_B`에는 미해결 이슈: Skipped Voxels의 8×(5ms 고정)가 event-driven I_T와 혼재 → event-driven 실측 기반 재계산 대기.
