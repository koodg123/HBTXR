# 14 · 동공/홍채 혼동 (pupil↔iris) — 발견·수정

2026-07-01. 사용자 지적으로 발견: GSAM2가 동공 대신 **홍채(iris)**를 분할하는 케이스. center 기반 평가로는 안 잡히는 맹점.

## 1. 발견 (증거)
- **크기 비교**(GSAM2 마스크 등가반경 / GT 동공 등가반경, anchor n=469): median **1.03**(정상=동공), 그러나 **4/469(0.9%)**가 ratio **1.9–2.7** = 홍채/과분할.
- **육안 확인**(`overlay/iris_check/`): ratio 2.72 케이스 — GT(초록)=동공, GSAM2(빨강)=**홍채 전체**. 저대비 subject.
- ⚠️ **맹점**: 동공·홍채는 **동심원** → center 오차(median 0.75px)로는 혼동을 **전혀 못 잡음**. 지금까지 평가가 놓치고 있던 문제.
- GT 동공 등가반경 ~15px, 홍채 ~45–60px(2.5–3배, 면적 6–9배). 정상 GSAM2 면적 ~800(p95), 홍채 케이스 3200–4355.

## 2. 원인
`sam2_center`가 SAM2 `multimask_output=True`의 3개 마스크 중 **SAM-score 최고**를 선택. 저대비 프레임에선 **홍채(공막 대비 경계 뚜렷) score > 동공(홍채 대비 경계 약함)** → 홍채 선택. 동공 마스크는 보통 후보에 **이미 있으나** 점수 기준으로 탈락.

## 3. 수정 (구조/크기 사전지식 활용)
- **(주) multimask 동공-선택 by 크기**: 후보를 **면적 ∈ [min, pupil_area_max]**로 제한(홍채=대면적 배제, 동공⊂홍채는 동심원이라 크기로 분리), 그 중 SAM-score 최고 선택. `pupil_area_max=3000`(정상 동공 ≤~2000·p95~900 vs 홍채 ≥3200 → 3000이 gap).
- **(백스톱) 크기 sanity**: 최종 마스크 반경 > 1.6×기대(~15px)면 홍채로 간주(로그/플래그).
- **(Plan B, 필요시)** multimask에 동공-크기 후보가 없으면 → 박스 내 **최암부 point**로 SAM2 재프롬프트(동공 분할 유도).
- **평가 보강**: center뿐 아니라 **반경/면적 비율·(가능하면)mask IoU**를 지표에 추가 → 혼동을 측정 가능하게.

## 4. 구현
`08_run_gsam2.py`:
- `sam2_center(predictor, box, pupil_max=None, min_area=120)`: pupil_max 주면 면적范위 후보 중 argmax(score), 없으면 전체 argmax(기존).
- `process_frame`: `pmax = a.pupil_area_max if a.pupil_select else None` → 3개 sam2_center 호출(primary/jitter/tta)에 전달.
- argparse: `--pupil-select`(플래그), `--pupil-area-max`(기본 3000).
- **하위호환**: `--pupil-select` 없으면 기존 동작(변경 없음).

## 5. 검증 계획 (현재 harness 실행 종료 후 GPU 여유 시)
1. `iris_fix_validate.py`로 **4개 홍채 케이스** 재검출 → ratio ~1.0로 회복되는지(동공 선택).
2. 정상 anchor 표본 → center 무회귀 + 반경비 median ~1.0 유지.
3. 통과 시 전 4,669 재실행에 `--pupil-select` 포함.

## 6. 롤아웃 (마커+홍채 both)
현재 실행(ROI+geom, 마커 수정)이 끝나면 → **iris 수정 추가한 단일 재실행**:
```
08 --roi 25,10,325,195 --min-box 18 --geom-select --pupil-select --pupil-area-max 3000 \
   --prompt "black pupil." --box-thr 0.20 --text-thr 0.15 --repeats 4 --tta --save-masks --overwrite
```
→ 14 재플래그 → 11/12 재빌드 → 10 재측정(+반경비 지표) → before/after.

## 7. 영향 범위
- **center 정확도(주 지표)엔 영향 없음**(동심원 → 이미 정확). 홍채 혼동은 **mask/반경/면적/IoU를 쓸 때** 문제.
- 0.9%(4/469)로 드물지만, 사용자 지적대로 **평가가 못 보던 실질 오류** → 수정+지표 보강 필요.

## 8. 검증·적용 결과 (2026-07-02)
**핵심: 홍채혼동은 ROI+geom harness가 이미 해결. `--pupil-select`는 불필요한 백스톱(회귀 없음).**
- `iris_fix_validate.py`(4 홍채케이스+30 정상, pupil-select OFF vs ON): 4개 홍채케이스가 **OFF에서도 ratio ~1.03**(옛 1.9–2.7), ON과 완전 동일. → 원인은 whole-image 박스 오검출이었고 **ROI crop[25,10,325,195]+geom-select가 박스를 동공으로 교정**하면 SAM2 multimask top-score가 이미 동공.
- **전 483앵커 잔여 홍채 0건**: 현 GSAM2(bur314la1, ROI+geom) radius_ratio median 1.02, p95 1.07, **max 1.17**, ratio>1.5 = **0/479**. → **전체 재실행 불필요**(pupil-select 무효과 확정, ~40분 GPU 절약).
- **다운스트림 재실행 완료**: `14`(mislabel **161→4**, 0.09% — 마커 오검출 소멸) → `11`(perframe 4669 재빌드) → `12`(오버레이) → `10`(center 재측정) + **신규 `10b_shape_metrics.py`**(반경비+IoU).
- **10b 결과**: GSAM2 radius_ratio median 1.016·**iris-suspect 0/479**, **mask IoU median 0.914**(U-Net 0.853보다 높음). U-Net radius_ratio 0.993·iris 0/483. → 홍채혼동 해소가 반경·IoU 지표로 정량 확인. 저장 `results/label_shape_gt_unet_gsam2.md`.
- 코드 유지: `08 --pupil-select`는 향후 ROI를 안 쓰는 경우 대비 백스톱으로 보존(기본 off).
