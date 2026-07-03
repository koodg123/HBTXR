# MANIFEST — motion-label-packages

이어작업용 self-contained 패키지. 시작점: `00_HANDOVER.md`.

## outputs/ (최종 산출물)
| 파일 | 행수 | 내용 |
|---|---|---|
| `RowLabels_test37_48_motion_NoBlink.csv` | 366,190 | **error table용** — Fix/Sacc/Smooth, Blink 제외. cols: row_id, sample_idx, subject, eye, session_code, frame_idx, timestamp(abs μs), rel_ts, speed_pxps, motion_state |
| `RowLabels_test37_48_motion_ALLframes.csv` | 373,463 | 전체 프레임 4-state(Blink 포함) + ev_density |
| `RowLabels_test37_48_README.md` | — | 컬럼·방법·join 가이드 |

## code/ (재현 코드)
| 파일 | 역할 |
|---|---|
| `build_rowlabels_test.py` | **메인 빌더** (quota-calibrated row-level 라벨 생성). 경로 하드코딩 — 새 환경서 상단 경로 수정 |
| `velocity_dist.py` | 속도 산출 lineage (matcell 궤적, median win9, fsacc40 계열) |
| `ivt_states.py` | I-VT 3-state 분류 기준(>493=Saccade, 세션타입) |
| `motion_labeler.py` | 대안 라벨러(detected-center 속도, 참고용) |
| `build_test_tables.py` | test 37-48 4-state 집계표 빌더(참고) |
| `common.py`, `pupil_detect.py` | 헬퍼(세션맵, GT 로딩 등) |

## data/ (의존 데이터)
| 파일 | 내용 |
|---|---|
| `vel4state_sessions.csv` | **quota 소스** — per-session total/blink/valid/fsacc40/fsacc5 |
| `4State_Frame_rawcount_all48.xlsx` | **타깃 표** — 라벨 집계가 이것과 일치해야 함 |
| `test37_48_manifest_reconcile.csv` | per-subject frames/valid/skip_no_ellipse/skip_no_events (366,171↔366,190의 18~19 차이 설명) |

## 외부 의존 (패키지에 미포함, 경로 참조)
- EV-Eye 원본: `E:\DATASET\eveye\raw_data\Data_davis\user{37-48}\{left,right}\session_*\frames\timestamps.txt`
- matcell: `E:\DATASET\eveye\processed_data\Frame_event_pupil_track_result\{left,right}\update_20_point_user*_session_*.mat`
- manifest: `E:\DATASET\eveye\DeanDataset_full_unet\{manifest.json, progress_state.json}`
- HBTXR inference 결과(사용자 보유): join 대상.

## join 키
**(subject, eye, session_code, frame_idx)** — `sample_idx`/`row_id`는 내부 순서, 사용 금지.
