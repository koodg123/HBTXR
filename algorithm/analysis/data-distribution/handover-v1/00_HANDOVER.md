# HANDOVER — HBTXR JETCAS 리벗 데이터 분석 패키지

- 프로젝트: **HBTXR** (A Runtime-Aware Hybrid Event–Frame Eye Tracking System, IEEE JETCAS-2026-0074, ZCU104 FPGA)
- 데이터셋: **EV-Eye** (48 subjects × 2 eyes × 4 sessions = 384 sessions, ~1.49M frames @25Hz, 9,011 GT 타원 annotation, 분할: Train 1–36 / Val {37,40,45,48} / Test 37–48)
- 작성일: 2026-06-30
- 패키지 위치(전체·바이너리 포함): `E:\DATASET\codes\data-distribution\workspace_etri_laptop\`
- Google Drive 사본(텍스트 핸드오버): `Z:\내 드라이브\temp_workspace\workspace_etri_laptop\`

> ⚠️ 샌드박스 제약: 분석 환경은 Google Drive(Z:) 볼륨에 **바이너리(xlsx/docx)를 직접 기록할 수 없습니다.** 따라서 Excel/Word 산출물은 위 E: 경로에 있으며, Drive 폴더에는 본 HANDOVER와 MANIFEST(텍스트)만 자동 배치됩니다. Excel/Word는 E: 폴더에서 Drive로 드래그하거나, Drive 폴더에 대한 셸 접근을 부여해 주시면 복사해 드립니다.

---

## 1. 이번 세션 범위

EV-Eye 원본 데이터·EV-Eye 벤치마크 코드·HBTXR 논문 초안·JETCAS 리플라이 테이블을 바탕으로, 4-state 모션 분포 / 속도 분포 / Search·Track invocation / refresh 트레이드오프를 정량화하고, 리플라이 테이블의 채워진 값 검증·빈칸 채움·상호 정합 분석을 수행함.

## 2. 핵심 발견 & 결정 (요약)

1. **세션→모션 매핑**: pattern1(_0_1)=saccade 세션(코드 101/201), pattern2(_0_2)=smooth 세션(102/202). Fixation/Smooth는 속도가 아니라 **세션 타입**으로 구분(중첩 때문).
2. **Blink** = UNet/Grounded-SAM이 동공을 못 찾은 프레임(`skip_no_ellipse`), ~2%.
3. **4-state(배타적)**: Blink 우선 → Saccade(speed>493px/s, v_sacc=90th pct) → 나머지를 세션타입별 Fixation/Smooth.
4. **Saccade 검출은 시간해상도에 지배됨**: Frame-center(40ms)·이벤트트랙 40ms 재샘플 모두 ~1.2%, 이벤트 native 5ms는 ~2.1%. 원인은 라벨 종류가 아니라 **샘플링 간격**(40ms는 30–80ms saccade를 평균화). 위치/속도 모두 EV-Eye에 **annotation 없음** — 위치(타원중심)+timestamp만 제공, 속도는 유한차분 I-VT로 계산.
5. **증강 정책**: Fixation×2, Smooth×2, Saccade×8, Blink×1 (per-class 보간). 전체 ×8 균일 증강은 Fix/Smooth를 과증가시켜 폐기.
6. **EV-Eye invocation 구조(`frame_event_pupil_track.m`)**: Search=Frame update(프레임마다 UNet centroid), Track=Event update(동공 edge 밴드 이벤트 20개 누적 시 ICP, "update_20_point"). matcell col6=frame_or_event, col2=pixel_num(blink 보조). **이벤트 update 평균 ~200Hz ≈ 5ms** → 사용자가 고른 5ms voxel과 일치.
7. **확정 스케줄러 모델**: Track=5ms voxel(200Hz), Search=프레임(40ms) anchor refresh, busy=Queue(FIFO), non-preemptive. R_S ≈ **10~11%**(matcell 실측 9~13%와 정합).
8. **Refresh 트레이드오프**: 40ms 강제 refresh는 결정적 12.5% Search지만 대부분 redundant(τ0.5에서 needed≈0.3%). watchdog 주기 W↑ → Search 급감(W200≈5×↓, W∞≈0.15%).
9. **리플라이 테이블 검증 핵심 발견**:
   - 🔴 **논문 LUT/FF 10× 오타 의심**: (HW) Resources 합계 145K LUT/227K FF(ZCU104 230K/461K의 63/49%)인데 논문 본문 14.5K/22.7K → 10배 불일치(DSP/BRAM/URAM은 814/184/12 동일). 시트가 맞고 논문이 오타로 판단.
   - 🟠 **전력 1.34W는 PL 가속기 동적전력 단독**; full-chip = static 0.673 + dyn-PS 2.45 + dyn-PL 1.525 = **4.65W**. 범위 명시 필요.
   - 🟠 **(Q) Bit-sweep 베이스라인 1.279px**가 논문 보고(~0.18px)와 7× 불일치, Pixel Error가 Drop Ratio에서 역산되는 비표준 수식.
   - **시트 R_S(7~27%, 평균16%)는 matcell 실측(9~13%)·5ms-voxel 모델(≤12.5%)과 불일치** → 권장값(~10.6%)으로 재작성함.
10. **상호 정합(Benchmark↔HW) 결론**: 치명적 모순 없음. 단 정의 차이 명시 필요 — HW N_F=valid 프레임(=4-State #Data−Blink), 4-State(F) #Data=total. "이벤트 수"는 3정의(matcell 행 294k / event-update 265k / 5ms-voxel 8×N_F 248k). I_S<N_F 격차(~1,600/subject)는 **matcell 처리 윈도우(마지막 200프레임 드롭)** 때문이지 blink 아님.

## 3. 산출물 (자세한 내용은 00_MANIFEST.md)

- `01_excel/` — 4-state(Frame/Event/Frame-center), 속도분포, invocation(설계공간/확정/스윕), 리플라이(HW) 재작성·초안
- `02_docs/` — Invocation·Refresh 트레이드오프 문서(DOCX)
- `03_tables/` — 모든 분석 CSV(원자료)
- `04_code/` — 분석 Python 스크립트 전체

## 4. 데이터/방법 provenance

- 원본: `E:\DATASET\eveye\raw_data\Data_davis`(frames@25Hz, events.txt), `processed_data\Frame_event_pupil_track_result`(matcell), `Data_davis_predict`(UNet 마스크), `Pixel_error_evaluation`.
- 파라미터: v_sacc=493 px/s, 동공원 r=35px(GT 타원 a≈37,b≈32 평균), voxel=5ms, frame=40ms(25Hz), IoU/conf τ=0.5.
- 속도식: v=√(Δx²+Δy²)/Δt, 위치 median-9 평활, 그리드 재샘플(I-VT).

## 5. 미결 항목 / TODO / 주의

1. **(HW) 재작성 보정 권장**: I_S=matcell frame-update(아티팩트) → **I_S=N_F(valid)** 로 바꾸면 R_S=11.11%(구조상수), Benchmark와 완전 정합. 재작성본 노트의 "blink" 문구를 "matcell 200-frame tail drop"으로 수정 필요.
2. **시트 R_S(16%) 출처 확인**: 보드 실측 로그면 ground truth; 아니면 권장 ~10.6%/11.1% 채택.
3. **논문 LUT/FF 10× 오타** 본문 수정(14.5K→145K, 22.7K→227K) 및 **전력 범위(1.34W=PL-only, full 4.65W)** 명시.
4. **Benchmark Search-Track Invocation 시트**(Frame/Event 분리) 미작성 — 정의 정리 후 채움.
5. **II·p95/p99**는 시뮬레이션 추정치(보드 invocation 로그 확보 시 대체). Track p95/p99(~1.25/1.37ms)는 non-preempt search contention 반영.
6. (HW) Latency #Events는 정확 전수 카운트, #Event Voxels=8×N_F.

## 6. 재현 방법

`04_code/`의 스크립트는 환경변수 `EVEYE_ROOT`(raw/processed_data 루트)와 `MT_CODES`(출력 루트)를 사용. 핵심 진입점:
- `velocity4state.py`/`event_4state.py`/`frame_center_4state.py` → 4-state
- `velocity_hist.py` → 속도 25-bin 히스토그램
- `invocation_designspace.py`/`invocation_final.py`/`refresh_sweep.py` → invocation·트레이드오프
- 리플라이 재작성: matcell flag 카운트 + 5ms/40ms 균일 큐 시뮬(FIFO, non-preempt)
