# HANDOVER — Row-level Motion Labels for HBTXR Test Error-Distribution

- 프로젝트: **HBTXR** (Runtime-Aware Hybrid Event–Frame Eye Tracking, IEEE JETCAS-2026-0074, ZCU104) / 데이터셋: **EV-Eye**
- 목적: HBTXR **test(subjects 37–48)** 추론 결과를 **motion(Fixation/Saccade/Smooth)별 Error Distribution**으로 분석하기 위한 **샘플당 1행 motion 라벨** 생성
- 작성일: 2026-06-30
- 위치: `E:\DATASET\codes\motion-label-packages\`
- 원본 분석 디렉토리: `E:\DATASET\codes\data-distribution\` (cache/src 등 전체)

---

## 1. 결론 (먼저)

- `outputs/RowLabels_test37_48_motion_NoBlink.csv` (366,190행) 와 `..._ALLframes.csv` (373,463행) 가 최종 산출물.
- 집계가 `data/4State_Frame_rawcount_all48.xlsx`의 test 값과 **per-subject·per-state 정확 일치** (subj48 Fixation만 +1, 사유는 §4).
- HBTXR inference 결과와 **(subject, eye, session_code, frame_idx)** 로 inner-join → motion별 error 분석.

---

## 2. 핵심 Q&A (이번 세션에서 확정한 것)

### Q1. test 샘플 수 366,171 의 정체 / 18(=19) 차이는?
- **366,171 = subjects 37–48의 `valid` 합** (`DeanDataset_full_unet/progress_state.json`). HBTXR 실제 inference 샘플 수.
- Frame-Before(Blink 제외) 합 = **366,190** = `frames(373,463) − skip_no_ellipse(7,273)`.
- 차이 **366,190 − 366,171 = 19 = `skip_no_events`**: GT(UNet) 타원은 있으나 5000-event voxel에 이벤트가 없어 HBTXR 샘플에서 빠진 프레임. (`data/test37_48_manifest_reconcile.csv`에 per-subject 수치, per-subject skip_no_events와 정확히 일치.)
- Blink(`skip_no_ellipse`, 7,273)는 NoBlink 파일에서 이미 제외됨. → **NoBlink(366,190) ⨝ manifest(366,171) = 366,171**.

### Q2. "Frame Before/Input" saccade 수(subj37=275 등)는 어떻게 나온 값인가?
- **per-frame 라벨이 아니라 세션 단위 분율 추정**: `Saccade_session = fsacc40 × valid`. (`data/vel4state_sessions.csv`의 `fsacc40` 컬럼.)
- `fsacc40` = **matcell event-track 궤적**(`Frame_event_pupil_track_result/.../update_20_point_*.mat`, col[2]=t, [3]=x, [4]=y, [5]=frame/event flag) 을 **median(win=5) smoothing → 40ms forward-diff 속도 → >493 px/s** 비율.
  - 검증: 40ms grid + forward1 + median win3~9 조합이 per-session fsacc40을 오차 ~0.0004로 재현. (central-2step diff=80ms 윈도우는 속도 절반→ 과소.)
- 즉 **원본 per-frame saccade 할당은 저장돼 있지 않음**(생성 스크립트는 /tmp 휘발).

### Q3. 그래서 row-level 라벨을 어떻게 만들었나? (사용자 선택 = "파일대로 나오게")
- **Quota-calibrated**: 세션별 saccade 수 K, blink 수 B를 파일 값에 맞춰 고정.
  - **Saccade** = 세션별 **속도 상위 K개** 프레임. K는 파일의 per-subject Saccade를 세션별(`fsacc40×valid` 가중)로 배분.
  - 세션 그룹 분배는 파일의 Fixation/Smooth로 핀: `K_smooth = Σvalid(102,202) − Smooth_file`, `K_sacc = Saccade_file − K_smooth` → Fix/Smooth 자동 일치.
  - **Blink** = 세션별 **event-update 밀도 하위 B개** (눈감김 proxy). B = `vel4state.blink` (per-subject 합 = 파일 Blink).
  - 나머지 = **Fixation**(101/201) / **Smooth**(102/202).
- 속도 = matcell 궤적 median(win5)+40ms forward-diff (fsacc40과 동일 계열). frame↔matcell 정렬: `est = frame_abs_ts[0] − matcell_frameRow0_rel`, `rel = abs_ts − est` (검증: 정확 일치, 프레임 간격 40ms 균일).

### Q4. (직전 개념질문) Event는 왜 증강(보간)했나? 이미 촘촘한데?
- 맞음. **Event는 이미 dense(μs / 5ms voxel 200Hz)라 시간 보간 불필요.** Frame(25Hz/40ms)만 saccade under-sampling → 보간 정당.
- Event에 적용한 ×N은 "보간"이 아니라 **class-rebalancing oversampling**(드문 saccade 복제). loss 가중으로 대체 권장. → **Interpolated 표는 Frame만 의미, Event는 raw 유지가 맞음.**

---

## 3. 라벨링 방법 요약 (재현용)

```
입력: matcell event-track 궤적(update_20_point_*.mat), 프레임 timestamps.txt, vel4state_sessions.csv, 4State_Frame_rawcount_all48.xlsx
1) 궤적 정렬·median(win5) smoothing
2) per-frame 속도 = |traj(rel+40ms) − traj(rel)| / 40ms,  rel = abs_ts − est
3) per-frame event-density = [rel−20ms, rel+20ms] 내 event-update row 수
4) 세션별 quota: K(saccade)=파일 기반 배분, B(blink)=vel4state.blink
5) 라벨: 속도 top-K → Saccade; (나머지 중) density 하위 B → Blink; 그 외 → Fixation(101/201)/Smooth(102/202)
파라미터: v_sacc=493 px/s, WIN=5, DT=40ms
```

코드: `code/build_rowlabels_test.py` (단독 실행). 의존: numpy, pandas, scipy. **경로 하드코딩**이므로 새 환경에서 상단 `EV/FE/DD/CD` 경로 수정 필요.

---

## 4. 검증 결과

per-subject 4-state(mine vs 파일): **12명 전원 일치**. 단 **subj48 Fixation 19,301 vs 파일 19,300 (+1)** — 실제 프레임 수(31,251)가 파일 `#Data`(31,250)보다 1 많음(전체 373,463 vs 373,462). Saccade/Smooth/Blink는 전원 정확 일치.
- Total(mine): Fix 231,853 / Sacc **3,269** / Smooth **131,068** / Blink **7,273** / NonBlink **366,190**.
- 파일: Fix 231,852 / Sacc 3,269 / Smooth 131,068 / Blink 7,273.

---

## 5. 이어서 작업하는 법 (How to continue)

1. **HBTXR inference 결과** (per valid sample의 예측 pupil center/IoU + GT) 를 준비.
2. `outputs/RowLabels_test37_48_motion_NoBlink.csv` 를 **(subject, eye, session_code, frame_idx)** 키로 inner-join. (⚠ `sample_idx`/`row_id`는 파일 내부 순서이므로 join 키로 쓰지 말 것.)
3. 결과 366,171행 (skip_no_events 19개 자동 제외). `motion_state` 별로 error(center-distance, IoU, p-accuracy) 통계·분포(violin/box) 산출.
4. Error table: **Fixation / Saccade / Smooth** 3열만 채우고 Blink(Input)은 n/a.

### 다음 후보 작업 (미결)
- saccade 비율↑ 학습용 oversampling(Frame×N) 적용·실험 (Event는 raw 유지 권장).
- 라벨 robustness: 속도 윈도우(win3/5/9) 민감도, 또는 순수 threshold@493 버전과 비교(총합 ~93%, subject편차 최대 ~26%).
- HBTXR manifest의 실제 sample_idx와 본 라벨을 결합한 단일 평가 CSV로 통합.

---

## 6. Provenance / 파라미터
- 원본: `E:\DATASET\eveye\{raw_data\Data_davis, processed_data\Frame_event_pupil_track_result, DeanDataset_full_unet\{manifest.json,progress_state.json}}`.
- 세션맵: 101=session_1_0_1(saccade), 102=_1_0_2(smooth), 201=_2_0_1(saccade), 202=_2_0_2(smooth). pattern1=saccade, pattern2=smooth.
- v_sacc=493 px/s, voxel 5ms, frame 40ms(25Hz), events_per_sample=5000, split=session-order 0.8 (UNet train/val) — **test 37–48은 subject-independent eval protocol(별도)**.
