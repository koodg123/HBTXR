# HBTXR — 주석 정밀도 / 라벨 노이즈: 종합 보고서

*리뷰어 응답용 분석. 재훈련·신규 인간주석 없이 추론 결과와 기존 라벨만 사용. `src/run_all.py`로 생성;
스칼라는 `results/precision/precision_summary.json`(346×260), `.../frame64_scalars.json`(64×64).*

## 0. 실험 환경

| 항목 | 값 |
|---|---|
| Sample | GT-anchor 윈도우, **users 1-10**, 483 anchor(모션당 161), 양안 |
| ⚠ Subject split | HBTXR: **train 1-32 / val 33-36 / test 37-48** → samples 1-10은 **TRAIN** subject. Layer A/B/C(라벨 품질)는 split 무관; STEP 5(E_orig)만 영향 → **낙관치(optimistic)로 명시** |
| 주 프레임 | **346×260**(native). **64×64**(모델 프레임)도 병기 |
| 프레임 환산 | **비등방**: x×5.406, y×4.063 (346/64, 260/64). 단일 스칼라 환산 금지; radial 값은 좌표에서 재계산 |
| 보고된 "0.1812 px" | **64×64, U-Net dense 의사라벨 대비**로 확정(`Metric.py RESOLUTION=(64,64)`) |
| 소스 | human_ellipse, unet, gsam2(+repeats), y_pred, ellseg/ritnet/edge_guided/deepvog/yoloe |
| 순환성 방지 | **y_pred는 모든 정밀도/불확실성 추정에서 제외**; STEP 5에서만 사용 |

**STEP 0 (데이터+코드 검증).** 사람 **타원**(VIA CSV `cx,cy,rx,ry,theta`)이 원본 주석. **mask**
(`Data_davis_labelled_with_mask`, HDF5)는 이 타원을 `cv2.ellipse`로 **rasterize한 파생물**(축보정
IoU≈0.95, 상수 (−1,−1)px 생성 오프셋) — 따라서 표현 floor는 **rasterization floor**이며 human_mask는
독립 소스가 **아님**. EV-Eye **U-Net은 이 파생 mask로 학습됨** → 사람 라벨과 비독립.

---

## 1. 주석 정밀도 (Annotation Precision)

보고된 pixel error는 수작업 주석된 pupil-ellipse 중심에 대해 측정되므로, 이 라벨이 모든 정확도 수치의
기준(reference)이다. 동공 중심은 **연속 좌표**이므로 annotation precision을 **주석된 중심 좌표의 공간적
정밀도**로 정의하고, center-distance / ICC / Bland–Altman / three-cornered-hat로 정량화한다 — 범주형
일치도(Cohen/Fleiss κ)는 쓰지 않는다. threshold(TP/FP) 정밀도는 §3의 P_n으로 별도 보고한다. 라벨은
**부동소수점**으로 저장되므로 **uniform-quantization floor는 적용·주장하지 않는다.** 독립 재주석이 불가하므로
σ_human을 재훈련·신규주석 없이 세 가지 상보적 추정량으로 **구간(bracket)** 추정한다.

**(a) 반복성(repeatability) 하한.** 입력 섭동(box-jitter / TTA) 하 Grounded-SAM2의 run-to-run 산포
(`gsam2.json.repeats`, 프레임당 5회). *La.2-human(I-VT fixation 내 GT 프레임간 std)은 **생략** — 사람
라벨이 sparse keyframe이라 시간 인접쌍이 없음 → 개선 GSAM2를 반복성 proxy로 사용(결정).*

**(b) 재현성(reproducibility, 주축).** 사람 타원·EV-Eye U-Net·Grounded-SAM2를 세 개의 독립 중심
추정기로 보고, pairwise center-distance 분산으로부터 three-cornered-hat(3CH) 분해로 σ_human 분리(축별
x,y). HBTXR는 순환성 방지로 제외.

**(c) 표현(representation) floor.** 사람 타원중심 vs 사람 mask centroid — 단 mask는 **파생**(rasterize)
이므로 이는 **rasterization floor**(상수 ~1px 생성 오프셋)이지 사람 변동성이 아님.

### 표 1 — σ_human 구간 및 정밀도 추정량

| 추정량 | 346×260 px | 64×64 px | 비고 |
|---|---|---|---|
| **σ_human — 3CH {h, gsam2, unet}** | **0.552** | **0.119** | 비음수 ✓ (양 프레임) |
| σ_human — 3CH 독립 {h, gsam2, ellseg} | 0.577 | 0.134 | 확증; **64×64서 음수분산** → 해당 프레임서 이 triple 폐기, RMS를 상한으로 |
| **σ_human bracket** | **[0.552, 0.864]** | **[0.119, 0.186]** | [3CH 하한, human↔SAM2 RMS 상한] |
| 표현(rasterization) floor — La.3 | 1.414 | 0.302 | mask **파생**(IoU 0.873); 상수 오프셋이 지배, 정밀도 아님 |
| GSAM2 섭동 σ — La.1 | 0.367 | 0.069 | 반복성(box-jitter/TTA) |
| fixation F2F jitter — GSAM2 (La.2 proxy) | 0.214 | — | 방법 안정성(사람반복 proxy) |
| fixation F2F jitter — U-Net | 0.220 | — | 방법 안정성 |

**내부 정합성 점검(Lb.3).** 3CH로 복원한 SAM2 산포 σ_s = **0.262**(346) / **0.060**(64×64)는 직접 측정한
섭동 반복성 **0.367 / 0.069**과 근접(비율 0.72 / 0.87) → 분해를 독립적으로 뒷받침. **{h,gsam2,unet}
triple의 모든 3CH 분산은 비음수**(양 프레임) → 해당 triple엔 공통 계통편향 미검출; {h,gsam2,ellseg}
triple은 coarse한 64×64서 소폭 음수분산 → 64×64서 해당 분해 폐기하고 human↔SAM2 RMS를 상한으로 유지.

**결과.** σ_human ≈ **0.55 px(346×260) / 0.12 px(64×64)**, 구간 **[0.55, 0.86] / [0.12, 0.19]**.
→ *그림 `fig/fig_3ch.png`.*

---

## 2. 라벨 노이즈 및 불확실성 (Label Noise & Uncertainty)

annotation precision이 단일 방법의 내부 일관성이라면, **label uncertainty**는 참 중심에 대한 총 잔여
모호성으로 **정밀도 이상(≥)**이다. 방법 간 높은 일치가 공통 계통 오프셋을 배제하지 않는다(예: 눈꺼풀 폐색 시
모든 방법이 가시 경계에 중심을 두어 서로는 일치하나 참값과 함께 어긋남). 따라서 정밀도 구간은 불확실성의
**하한**이다. label uncertainty를 **방법 간 불일치(inter-method disagreement)**로 정량화하고
**human↔SAM2** 거리를 실효 상한으로 채택한다.

### 표 2 — pairwise 불일치 및 형상 일치도

| pair | radial RMS 346×260 | radial RMS 64×64 | Bland–Altman bias (x,y) px | mask IoU (vs GT 타원) |
|---|---|---|---|---|
| **human ↔ gsam2** | **0.864** | **0.186** | (+0.48, +0.38) | gsam2/GT = **0.914** |
| human ↔ unet | 1.752 | — | **(+1.02, +1.14)** 계통 | unet/GT = 0.853 |
| gsam2 ↔ unet | 1.167 | — | — | gsam2/unet = 0.910 |

- **U-Net은 사람 GT 대비 계통 오프셋 (+1.02, +1.14)px**(Bland–Altman): 불일치가 주로 *bias*이지 산포가
  아님(그래서 centered 분산을 쓰는 3CH가 여전히 깨끗한 σ_human 산출). U-Net은 사람-파생 mask로 학습되었으므로
  leak-free 비교가 필요하면 held-out 프레임으로 한정해야 함.
- **GSAM2 densification은 학습 supervision 전용이며 평가 기준이 아님.** 편향 미주입 입증: co-labeled
  프레임서 GSAM2-vs-human 일치도 = **center median 0.77 px(346×260), IoU 0.914** — audit 라벨이 사람
  타원을 tight하게 추종.

**실효 불확실성 band** = **[σ_human, human↔SAM2 RMS] = [0.55, 0.86] px(346×260) / [0.12, 0.19] px(64×64)**.

---

## 3. 보고 오차 대 라벨 불확실성 (Reported Error vs Label Uncertainty)

아래 모든 양은 **동일한 64×64 프레임**(보고 0.1812의 프레임)에서 비교한다. native→64×64 축소가 비등방
(×5.41 / ×4.06)이므로 오차·불확실성은 **단일 스칼라로 환산하거나 프레임 간 비교하지 않는다.**

### 표 3 — budget (동일 프레임)

| 양 | 64×64 px | 346×260 px |
|---|---|---|
| 표현(rasterization) floor | 0.30 | 1.41 |
| **σ_human (3CH 하한)** | **0.119** | 0.552 |
| **human↔SAM2 RMS (상한)** | **0.186** | 0.864 |
| **보고 0.1812 (vs U-Net dense)** | **0.181** | ≈0.85 (iso; 범위 0.74–0.98) |
| **정정 E_orig (vs 사람 GT, ckpt pe0.5401)** | **1.174** (median; mean 2.33, p95 10.9) | **5.698** (mean 11.4, p95 56) |

*→ 그림 `fig/fig_budget.png`.*

**해석 (측정값에 맞게 수정).**

1. **보고 0.1812 px는 라벨노이즈 band [0.119, 0.186](64×64) *안*에 위치** — 즉 주석-노이즈 floor에 있다.
   이는 **dense U-Net 의사라벨** 대비 floor 근처에서 측정된 값이라 모델 오차와 라벨 노이즈가 구별되지 않으며,
   바로 그래서 "너무 좋아" 보인다. (측정된 σ_human ≈ 0.12 기준으로 이 값은 floor *에 있는* 것이지 floor
   *아래*가 아니다 — 후자는 주장하지 않는다.)

2. **원본 사람 타원 라벨로 정정하면 오차는 1.17 px(64×64) / 5.70 px(346×260)로 상승** — σ_human band보다
   분명히 **위**다. 사람 GT(dense 자기라벨이 아님)로 평가하면 수치가 "floor에 있음"(아티팩트)에서 "floor
   위"(실측 가능한 실제 오차)로 이동한다. 라벨을 **누수(leak)**하는 모델은 이렇게 될 수 없다(floor에 고정되고
   무거운 꼬리가 없음; 여기선 p95≈11px/57px, 13%가 gross >20px). **→ 누수 반증.**

3. **이 ckpt는 label-noise-limited가 아니다.** band 안에 있을 모델과 달리, 가용 ckpt의 정정 오차
   (1.17px @64×64)는 σ_human band의 ~6–10배 — sub-pixel 라벨 모호성이 아니라 여유(headroom)가 있는 실제
   모델 오차. **두 가지 캐비앳:** (a) 가용 ckpt는 **`pe0.5401`**(자체 dense-라벨 val distance ≈0.54px
   @64×64)로 논문 0.1812를 만든 ckpt가 **아님**; (b) samples가 **TRAIN** subject → subject-independent
   값(test 37-48)은 이보다 **크다(≥)**.

### 표 4 — threshold 정밀도 P_n (abstain 분리)

| 소스 | 프레임 | P_n ≤10px | ≤5px | ≤1px | abstain |
|---|---|---|---|---|---|
| gsam2 vs human (방법 간) | 64×64 | 1.00 | 1.00 | 1.00 | 4/483 |
| gsam2 vs human (방법 간) | 346×260 | 0.99 | 0.99 | 0.73 | 4/483 |
| **pred vs human (HBTXR)** | 64×64 | 0.94 | 0.90 | **0.41** | 0 |
| pred vs human (HBTXR) | 346×260 | 0.74 | 0.44 | 0.02 | 0 |

모델의 threshold 정밀도는 **독립** 방법들이 서로 일치하는 수준(GSAM2↔human 1px @64×64서 ~100%)에 근접하며,
이는 독립 주석자들끼리 일치하는 허용오차까지 모델이 GT를 맞추고 있음과 정합한다. 64×64 P_n의 sub-pixel 규모는
**축소된 평가 해상도의 직접 결과**(native σ_human≈1.3px가 64×64서 ≈0.12–0.3px로 대응)이지 초해상 localization이
아니다.

### subject별 / 모션별 (346×260, E_orig; ⚠ TRAIN subject)

| 모션 | E_orig median | p95 |
|---|---|---|
| fixation | 5.78 | — |
| saccade | 4.40 | — |
| smooth_pursuit | 7.45 | — |

subject별 median E_orig 4.58(user10) – 6.92(user8, 최악); gross 실패율이 전 10 user 균일 → event-mode
모달리티 난이도이지 특정 subset 아티팩트 아님.

---

## 4. 범위 및 한계 (Scope & caveats)

- **3CH 독립성**은 가정이며 비음수로 사후검증: {h,gsam2,unet} triple은 양 프레임서 깨끗, {h,gsam2,ellseg}
  triple은 coarse한 64×64서 음수분산 → 해당 프레임서 분해 폐기하고 human↔SAM2 RMS를 상한으로 유지. 모든 방법이
  공유하는 계통편향은 포착 못 함(불확실성은 하한).
- **GSAM2 섭동 산포**는 해당 라벨러의 반복성을 특성화하지 사람 GT를 특성화하지 않음.
- **정확도는 원본 사람 라벨이 존재하는 프레임에서 평가**; 고주파 tracking 능력은 별도 throughput claim.
- **★ ckpt / subject 캐비앳.** 평가 ckpt(`pe0.5401`)는 0.1812 모델이 아니고, 평가 subject(1-10)는 HBTXR
  **학습** subject다. 둘 다 정직한 subject-independent 헤드라인을 위 수치보다 **상향**시킨다. 깨끗한 수치는
  **test subject 37-48**(사람 GT + U-Net 존재 확인됨)에서 audit을 재수집해 동일 파이프라인을 재실행해야 얻는다.

---
*그림: `fig/fig_3ch.png`, `fig/fig_budget.png`. 테이블: `tables/*.csv`. 스칼라:
`results/precision/{precision_summary.json, frame64_scalars.json}`. Gate 로그:
`results/precision/run_gate_log.txt`. 스키마: `results/e0_schema_report.md`.*
