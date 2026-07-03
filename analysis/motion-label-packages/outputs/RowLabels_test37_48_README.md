# Row-level Motion Labels — HBTXR Test (Subjects 37–48)

Error-distribution용 **샘플당 1행** motion 라벨. 집계가 `4State_Frame_rawcount_all48.xlsx`(Frame Before/Input)과 **정확히 일치**하도록 calibrate.

## 파일
| 파일 | 행수 | 내용 |
|---|---|---|
| `RowLabels_test37_48_motion_ALLframes.csv` | 373,463 | 전체 프레임, 4-state(Fixation/Saccade/Smooth/Blink) |
| `RowLabels_test37_48_motion_NoBlink.csv` | 366,190 | Blink 제외(Fix/Sacc/Smooth) — error table용 |

## 컬럼
`sample_idx`(내부 행순서, **join 키 아님**), `subject`, `eye`, `session_code`(101/102/201/202), `frame_idx`(프레임파일 번호), `timestamp`(절대 μs, 프레임 캡처시각), `rel_ts`(= timestamp − event_startime, matcell 도메인), `speed_pxps`(I-VT@40ms 속도), `motion_state`.

## 분류 방식
- 속도 = **matcell event-track 궤적**(`update_20_point_*.mat`) median(win=5) smoothing 후 **40ms forward-diff** → `speed_pxps`. (게시 `fsacc40`과 동일 계열.)
- **Saccade** = 세션별 속도 상위 K개. K = 파일의 saccade 수(`fsacc40×valid`)를 세션별로 배분(파일 per-subject 합과 정확히 일치).
- **Blink** = 세션별 event-update 밀도 하위 B개(눈감김 proxy). B = 파일 Blink 수.
- 나머지 = **Fixation**(101/201) / **Smooth**(102/202).

## 검증 (per-subject 4-state vs 파일)
12명 전원 일치. 단 **subj48 Fixation 19301 vs 파일 19300 (+1)** — 실제 프레임 수(31251)가 파일 `#Data`(31250)보다 1 많음(전체 373,463 vs 373,462의 1-프레임 차이). Saccade/Smooth/Blink는 전원 정확 일치. Total: Fix 231,853 / Sacc 3,269 / Smooth 131,068 / Blink 7,273.

## HBTXR manifest와 join
- **join 키 = (subject, eye, session_code, frame_idx)** — `sample_idx`는 본 파일 내부 순서이므로 사용 금지.
- `NoBlink`(366,190) ⨝ HBTXR valid manifest(366,171) = **366,171**.
- 빠지는 **≈19 = `skip_no_events`**: GT 타원은 있으나 5000-event voxel에 이벤트가 없어 HBTXR 샘플에서 제외된 프레임. (366,190 − 366,171 = 19; per-subject skip_no_events와 일치.)
- Blink(no_ellipse, 7,273)는 NoBlink에서 이미 제외됨.

## 주의 (방법론)
게시된 "Frame Before" saccade 수는 **세션 분율(fsacc40) × valid** 추정치이고 원래 per-frame saccade 할당이 저장돼 있지 않음(생성 스크립트 휘발). 본 라벨은 그 수에 맞춰 **세션별 속도 상위 K개**를 saccade로 지정(velocity 순서는 matcell 궤적 기반, 의미 있음). 따라서 집계는 파일과 정확히 일치하며, 개별 프레임 saccade 지정은 calibrated 추정.
