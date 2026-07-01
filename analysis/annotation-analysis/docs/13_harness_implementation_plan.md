# 13 · GSAM2 검출 harness 구현 계획 (구체)

2026-07-01. 목표: U-Net fallback 없이 GSAM2가 스스로 동공 검출. 설계 배경은 [12](12_gsam2_detection_harness.md). 우선 **A1(ROI 크롭)+C(기하 선택)**, 부족 시 D(auto-mask)·B(iris→pupil).

## 0. 성공 기준 (측정 가능)
anchor gold(483, 사람 GT) 재평가 기준:
- **mislabel(‖gsam2−GT‖>10px) 14 → ≤2** (U-Net fallback 없이). 잔여는 near-blink로 blink 정책 분류 허용.
- **정상 프레임 median ‖gsam2−GT‖ ≤ 0.85px** (현재 0.75px 대비 무회귀).
- valid rate ≥ 99%.

## 1. 파라미터 (데이터 기반, 튜닝 가능)
- **Eye-ROI(고정)**: `x∈[25,325], y∈[10,195]` (346×260). 근거: 동공 cx[85,265]·cy[61,177] 전부 포함, 마커(cy≥161·post는 하단 edge까지) 밴드 상당 제거. → `--roi 25,10,325,195`.
- **크기 필터**: min `w,h ≥ 18px` 또는 `area ≥ 250` (마커 w~15·area~266 배제, 정상 w~32·area~779 유지). max는 기존 `--max-box-frac 0.55`. ※ ROI가 1차 방어라 min은 느슨하게(작은 수축동공 보호).
- **GDINO**: ROI에서 false positive 감소 → `--box-thr 0.20 --text-thr 0.15`, prompt `"black pupil."`.
- **기하 점수**(argmax conf 대체): `score = det + 0.5·darkness_norm + 0.5·roundness − centrality_penalty`
  - darkness = 박스 내부 평균밝기 낮을수록↑(동공=최암부), roundness = min(w,h)/max(w,h)(~1), centrality = ROI 중앙 대비 거리 페널티(약하게). plausible area [150,2500] 밖은 제외.

## 2. Phase 0 — PoC (구현 전 검증, ~10분)
14개 mislabel anchor에 대해 **ROI 크롭 후 GDINO "black pupil." 재검출**만 돌려 top 박스가 동공(±10px GT)에 오는지 확인.
- 통과(≥10/14): Phase 1 진행. 미달: ROI/threshold 조정 or D(auto-mask)로 우회.
- 스크립트: 임시 `poc_roi.py`(08 함수 재사용, 크롭+GDINO+SAM2 1스테이지).

## 3. Phase 1 — 08에 A1+C 구현
`08_run_gsam2.py` 수정(기존 옵션 유지, 신규 추가):
- 신규 인자: `--roi x0,y0,x1,y1`, `--min-box <px>`, `--geom-select`, (Phase2용 `--automask-fallback`).
- `process_frame` 변경:
  1. `crop = image_source[y0:y1, x0:x1]`; offset `(x0,y0)`.
  2. GDINO on `crop` → boxes(crop 좌표).
  3. 필터: min-size(신규) + max-frac(기존).
  4. **선택**: `--geom-select`면 §1 기하 점수 최대, 아니면 기존 argmax.
  5. SAM2 on `crop` + box → mask(crop 크기).
  6. **좌표 복원**: center `+= (x0,y0)`; box `+= (x0,y0,x0,y0)`; mask는 346×260 캔버스의 ROI 위치에 paste. repeats/tta도 crop 내 수행 후 복원.
- 출력 스키마 불변(gsam2.json/perframe 그대로) → 하위 파이프라인 무변경.

## 4. Phase 2 — 복구(Phase1 미달 시, 비U-Net)
- `--automask-fallback`: ROI 내 GDINO 실패 시 `SAM2AutomaticMaskGenerator(crop)` → 후보 중 **어둡+둥금+area∈[150,2500]+ROI중앙** 최고점 선택(D). 
- 대안 B(iris→pupil): GDINO "iris"(ROI) → iris 박스 → 그 안에서 pupil(darkest disc/SAM2). near-blink에 강건.

## 5. Phase 3 — 롤아웃 & 재측정
1. 전 4,669 재실행: `08 --roi ... --geom-select --save-masks --overwrite` (fallback 불필요 — harness 자체가 검출).
2. `14_flag_gsam2_mislabel` 재실행 → mislabel ~0 기대(잔여=near-blink).
3. `11_build_perframe` 재빌드, `12_overlay` 재렌더(회복 확인).
4. `10_label_noise` 재측정 → **U-Net fallback 없는** 순수 GSAM2-vs-GT label noise 보고(현재 0.75px는 14건 fallback 포함값).
5. before/after 비교표(mislabel율·median·valid).

## 6. 실행 커맨드 (Phase 1 검증)
```bash
cd scripts
PYTHONPATH=<repo>/third/Grounded-SAM-2 ../.venv-gsam2/bin/python 08_run_gsam2.py \
  --out ../samples --sam2-ckpt ../weights/gsam2/sam2.1_hiera_large.pt \
  --sam2-cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --gdino-ckpt ../weights/gsam2/groundingdino_swint_ogc.pth \
  --gdino-cfg <repo>/.../GroundingDINO_SwinT_OGC.py \
  --prompt "black pupil." --box-thr 0.20 --text-thr 0.15 \
  --roi 25,10,325,195 --min-box 18 --geom-select \
  --repeats 4 --tta --save-masks --overwrite --anchors-only   # 먼저 anchor로 검증
python 13_gsam2_mislabel.py   # 재평가 (mislabel율)
```

## 7. 리스크 & 완화
| 리스크 | 완화 |
|---|---|
| 고정 ROI 밖 동공(카메라 정렬 변화) | ROI 여유 크게(검증: 전 동공 포함) / 안되면 A2 적응적(GDINO "eye"→크롭) |
| 크롭이 SAM2 문맥 바꿔 정상 회귀 | anchor gold 정상프레임 median 무회귀 확인(≤0.85px) |
| min-size가 작은 수축동공 제거 | min 느슨(18px), ROI를 1차 방어로 |
| near-blink 강제검출 부정확 | blink 정책 분류(강제 X), 별도 보고 |

## 8. 산출물
- 수정 `08_run_gsam2.py`(신규 옵션), (Phase0) `poc_roi.py`, 재측정 결과 `results/`, before/after 비교, 본 계획 갱신.
- **원본 데이터 미수정**(사용자 방침 유지).
```
```
