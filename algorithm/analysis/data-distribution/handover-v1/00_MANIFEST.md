# MANIFEST — workspace_etri_laptop

전체 패키지(바이너리 포함): `E:\DATASET\codes\data-distribution\workspace_etri_laptop\`
Google Drive(텍스트 핸드오버): `Z:\내 드라이브\temp_workspace\workspace_etri_laptop\`

## 01_excel/ (Excel 산출물 — E: 폴더에 위치)

| 파일 | 내용 |
|---|---|
| `DataAug_4state_excl.xlsx` | Frame 4-state(I-VT@40ms). before / after-aug(×2/×8/×1) / summary, per-subject 48 |
| `EventStream_4state_excl.xlsx` | Event 스트림 4-state(5ms grid). before/after/summary. Total=matcell 행 |
| `FrameCenter_4state_excl.xlsx` | Frame GT-center literal 속도(col5==1 anchors, 40ms) 4-state. raw/med3/after |
| `Velocity_distribution.xlsx` | 속도 25-bin 히스토그램(0–10@1px, 10–50@5px, …≥1200). Frame/Event × RAW/Interp + Count·Ratio |
| `Invocation_designspace.xlsx` | Search/Track 설계공간: cadence×idle 12조합(N_Track), trigger 4종(N_Search), 예시 full-config |
| `Invocation_final.xlsx` | 확정(5ms Track+40ms Search): per-subject I_S/I_T/비율 + IoU needed/deg(τ=0.3/0.5/0.7) |
| `Refresh_sweep.xlsx` | watchdog W{40..∞}×τ{0.3/0.5/0.7} Search% 트레이드오프 + 권장(W200,τ0.5) per-subject |
| `JETCAS_REPLY_TABLES (HW)_rewritten.xlsx` | **권장 재작성본**: matcell I_S + 5ms voxel I_T, R_S~10.6%, II·p95/p99(큐 시뮬). 초록=실측, 노랑=추정 |
| `JETCAS_REPLY_TABLES (HW)_filled.xlsx` | (이전) 시트 R_S 기반 채움본 — 재작성본으로 대체됨, 참고용 |

## 02_docs/

| 파일 | 내용 |
|---|---|
| `HBTXR_Invocation_Tradeoff.docx` | 스케줄러 설정·invocation 모델·12.5% 기준·IoU 정당성·W 스윕·권장(W200)·가정/한계 (표 5개) |

## 03_tables/ (분석 원자료 CSV)

- 4-state: `tbl_4state_aug_before/after.csv`(Frame I-VT40), `tbl_4state_event_before/after.csv`(Event), `tbl_4state_framecenter_before.csv`, `tbl_4state_before/after_excl.csv`, split_*.csv
- 속도: `tbl_vel_frame_raw/interp.csv`, `tbl_vel_event_raw/interp.csv`
- invocation: `tbl_invoc_track_grid.csv`, `tbl_invoc_search_grid.csv`, `tbl_invocation_final.csv`, `tbl_refresh_sweep.csv`
- 리플라이 채움: `tbl_latency_rewrite_final.csv`(재작성), `tbl_latency_fill_sheetRS.csv`, `tbl_latency_fill_noforced.csv`
- 샘플링: `tbl_testset_sampling.csv`(test 37–48 이벤트 전수 카운트·voxel)

## 04_code/ (Python 스크립트)

핵심: `velocity4state.py`, `event_4state.py`, `frame_center_4state.py`, `velocity_hist.py`, `invocation_designspace.py`, `invocation_final.py`, `refresh_sweep.py`, `common.py`. 그 외 4-state/오차/그림 빌더 포함(총 43개).

## 원본/외부 참조 (복사 안 함, 위치만)

- EV-Eye raw/processed: `E:\DATASET\eveye\` (Data_davis, processed_data, Benchmark_implementation_code)
- 리플라이 원본 업로드본: `JETCAS_REPLY_TABLES (Q|HW|Benchmark).xlsx` (업로드 보존)
- HBTXR 논문 초안: `jetcas_draft_final_hbtxr.zip` (업로드 보존)

## 바이너리를 Google Drive로 옮기려면

1) E:\…\workspace_etri_laptop\01_excel, 02_docs 의 파일을 Drive 폴더로 드래그, 또는
2) Drive 폴더에 대한 셸 접근을 부여 → 자동 복사.
