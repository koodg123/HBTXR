# data-distribution — HANDOVER (EV-Eye 4-state 분포 분석 & Supervision 정당성)

위치: `E:\DATASET\codes\data-distribution`  ·  문서화: 2026-06-29
프로젝트: HBTXR (IEEE JETCAS Major Revision, JETCAS-2026-0074) — EV-Eye 기반 hybrid event–frame eye tracking.
목적: EV-Eye 데이터의 **운동상태(Fixation/Saccade/Smooth/Blink) 분포**를 Subject별·스트림별(Frame/Event)·보간 전후로 정량화하여, 학습 단계의 **Frame/Event Interpolation supervision(특히 Saccade 증강)을 정당화**하는 리뷰어 응답 근거를 만든다.

---

## 1. 핵심 결론 (한 줄)
**Frame(25Hz)에서는 Saccade가 0.00%로 학습 불가하지만, 동일 데이터의 Event(μs)에서는 ~8.5%로 풍부**하다 → Frame의 0%는 25Hz 측정 아티팩트. 그래서 이벤트가 가리키는 saccade 구간에 **TimeLens-XL 200Hz 프레임 보간 + V2E 이벤트 보간**을 적용하면 Frame에서도 Saccade가 **~9.3%로 복원**되어 학습 가능해진다. 평가·GT는 보간이 아닌 원본 실측으로만 수행(순환평가 방지).

## 2. 데이터 / Split
- EV-Eye: 48명 × 양안 × 4 서브세션(384세션), ~1,490,548 프레임(25Hz), 이벤트 raw `events.txt`(세션당 ~346MB).
- 세션→운동: pattern 1(_0_1)=saccade(101/201), pattern 2(_0_2)=smooth(102/202).
- Split(EX-Gaze config 기준): **Train=1–36, Val={37,40,45,48}, Test=37–48** (Val⊂Test 중첩 — 표의 `InVal` 플래그로 구분).

## 3. 4-state 분류 기준
| 스트림 | 판정 물리량 | 임계 | 시간해상도 | Saccade 결과 |
|---|---|---|---|--:|
| Frame (보간 전) | 동공중심 속도(px/s) | speed>493 (saccade-세션 90pct) | 25Hz(40ms) | ~0.00% |
| Event (보간 전) | 이벤트 레이트(ev/s) | 4·m<rate≤12·m (m=세션중앙), >12·m=Blink | 20ms bin (μs) | ~8.5% |
| Fused (프레임 타임라인) | 프레임±20ms 이벤트수 cnt | cnt>4·m, Blink=프레임마스크 | 프레임 정합 | ~17.6%* |
| Frame (보간 후, emulated) | — | 비-blink를 이벤트비율로 재분배, Blink=프레임UNet | 200Hz(5ms) | ~9.3%(실측 [TBD]) |
- Blink: Frame=UNet 마스크 빈 프레임(`skip_no_ellipse`, 신뢰) / Event=극단 버스트(>12·m, 과대추정 경향).
- *Fused Saccade 17.6%는 ±20ms·4× 임계의 상단추정(post-saccade 정착 포함). 임계 민감.

## 4. 주요 결과 (share %)
**Frame (보간 전, progress_state 전 프레임):** Fixation ~63% / **Saccade 0.00%** / Smooth ~35% / Blink ~2%
**Event (보간 전, 20ms bin, 전48 3.08M bins):** Fixation ~53% / **Saccade ~8.5%** / Smooth ~29% / Blink ~9%
**After-interp (emulated, split):** **Saccade ~9.3%** (Train 9.33/Val 9.35/Test 9.33), Blink ~2%
(상세: `tables/split_4state_summary.csv` 및 per-subject CSV)

## 5. 데이터 출처 / 방법
- Frame 4-state: `DeanDataset_full_unet/progress_state.json`의 session_summaries(frames/valid/skip_no_ellipse, 전 프레임) → 마스크 재판독 없이 산출.
- Event 4-state: 원시 `raw_data/Data_davis/.../events.txt`를 20ms bin → 레이트 → 임계. (2코어 병렬·재개형)
- Fused: 프레임 마스크(blink) + 프레임±20ms 이벤트버스트(saccade). 프레임 ts와 이벤트 ts가 동일 DAVIS 클럭임을 확인.
- After-interp: 실 보간 데이터 부재 → **emulated**(가정: 200Hz 보간이 event 해상도로 saccade 복원).

## 6. 파일 인덱스
- `src/` (31 .py): 분류·집계 코드.
  - 4-state 핵심: `blink_detect.py`, `build_4state_blink.py`, `ivt_states.py`, `event_4state_all.py`, `event_4state_test.py`, `run_test_mask.py`, `fuse_4state_test.py`, `build_splits_4state.py`, `build_test_tables.py`, `pupil_detect.py`, `common.py`.
  - (참고) 정확도/오차 분석 코드도 포함: `eveye_metrics.py`, `build_sec12/3/4/5.py`, `build_persubject_tables.py`, `inventory.py`, `motion_labeler.py`, `iou_frame.py`, `frame_aligned_err.py`, `eveye_pe_aggregate.py`, `exgaze_eval.py`, `verify_mapping.py`, `error_harness.py`, `make_figure*.py` — 이 출력들은 별도 작업(정확도)이며 일부 출력은 재구성 필요(§8).
- `tables/` (11 CSV): split_4state_{frame_before,event_before,after_emulated,summary}, tbl_4state_test37_48, tbl_test37_48_{4state_detail,sample_counts}, tbl_event_4state_{all48,test37_48}, tbl_fused_4state_test37_48, tbl_4state_blink_per_subject. **모든 표에 비율(%) 포함.**
- `figures/` (9 PNG): fig_split_4state_{frame_before,event_before,after_emulated}, fig_4state_test37_48, fig_event_4state_{all48,test37_48}, fig_fused_4state_test37_48, fig_test37_48_sample_counts, fig_4state_blink_per_subject.
- `reports/` (4 MD): `CRITERIA_4state.md`(기준+비율), `Response_supervision_reviewer_reply_KR.md`(리뷰어 응답 최종형), `Response_supervision_justification_KR.md`/`_v2.md`(초안·서술형).
- `cache/` (6 CSV): frames_4state_{test37_48,blink,blink_test37_48}, event_4state_{all48,test37_48}, fused_4state_test37_48 (per-frame/세션 라벨).

## 7. 재현 (실행)
```bash
export MT_CODES=/mnt/e/DATASET/codes          # 출력 루트(코드/표/그림)
export EVEYE_ROOT=/mnt/e/DATASET/eveye        # 데이터 루트(raw_data, processed_data)
cd src
python3 event_4state_all.py        # Event 4-state 전48 (재개형, 반복실행)
python3 run_test_mask.py           # Frame mask 4-state (재개형)
python3 fuse_4state_test.py        # 융합(이벤트버스트+마스크blink)
python3 build_splits_4state.py     # Train/Val/Test 표·그림 + summary (progress_state + event cache)
python3 build_test_tables.py       # per-subject 상세·표본수 표
```
(주의: 마운트에 stale `__pycache__`가 있으면 `/tmp` 복사 후 실행 권장. 경로는 `common.py`가 `MT_CODES`/`EVEYE_ROOT`로 해석.)

## 8. 한계 / 주의 (정직성)
- **Saccade 임계 민감**: Frame 0% / Event(레이트) 8.5% / Fused 17.6% — 절대값은 임계·bin폭에 민감. 견고한 결론은 "Event ≫ Frame".
- **보간 후는 EMULATED**: 실 TimeLens-XL/V2E 산출이 디스크에 없음 → 9.3%는 추정. 실 GPU 실행 산출로 교체 필요(§9).
- **Event Blink 과대추정**(~9%) → 신뢰 Blink는 Frame-UNet(~2%).
- **평가 무결성**: 증강은 학습 전용, 평가·GT는 원본 실측(순환평가 방지). 보간 GT 금지.
- subject39/left/101: UNet 예측 마스크 누락으로 융합에서 제외(raw 프레임은 존재).
- Frame stride: Frame 4-state(progress_state)는 전 프레임. Frame mask-classify(run_test_mask)는 stride2.

## 9. TODO
- [ ] 실제 TimeLens-XL(200Hz)+V2E 보간을 WSL/GPU에서 생성 → 보간 후 분포·표·그림을 실측으로 교체(현재 [TBD]).
- [ ] Saccade 임계를 자극 타이밍(saccade 세션 11×11 그리드, 세션당 ~120회)에 보정해 생리학적 saccade율 확정.
- [ ] (옵션) 정확도/오차 분석(sec1–5, Reviewer_Response DOCX) 출력 재생성 — 코드는 src/에 있으나 캐시(velocity_at_gt/pe_allframes/iou) 재빌드 필요.
- [ ] supervision 응답을 영문화 / 리뷰어 응답 DOCX 통합.
