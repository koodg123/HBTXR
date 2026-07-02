# 결과별 상세 해석 (Annotation Precision / Label Noise)

각 실험 결과에 대해 **측정 대상 → 값(346×260 / 64×64) → 해석 → 한계**를 정리한다. 수치 출처:
`scalars/precision_summary.json`(346×260), `scalars/frame64_scalars.json`(64×64).
프레임 환산은 **비등방**(x×5.406, y×4.063)이므로 radial 값은 좌표에서 재계산했다.

전제(고정): samples=**users 1-10 = HBTXR 학습 subject**(test=37-48) → La/Lb/Lc(라벨 품질)는 무관,
**STEP 5(E_orig)만 낙관치**. y_pred는 STEP 5에서만 사용. mask는 타원 **rasterize 파생**(→ 표현 floor는
rasterization floor). 보고 0.1812는 **64×64, U-Net dense 대비**.

---

## La.1 — GSAM2 반복성 (perturbation repeatability)
- **측정**: 같은 프레임에 box-jitter/TTA 5회 반복 시 GSAM2 중심들의 산포(radial σ).
- **값**: σ = **0.367 px(346×260) / 0.069 px(64×64)**, p95 0.420.
- **해석**: 자동 라벨러(GSAM2)의 **run-to-run 정밀도**. 같은 입력에 대해 sub-pixel(64×64서 0.07px)로 안정 →
  검출기가 결정론적에 가깝고 노이즈가 작다. 이는 **정밀도의 하한 후보**(방법 자체의 흔들림)이다.
- **한계**: 이는 **GSAM2의 반복성**이지 사람 GT의 정밀도가 아니다. 5회 반복이라 σ 추정 자체는 다소 얇다.

## La.2 — Fixation 프레임간(F2F) jitter
- **측정**: I-VT fixation 구간(참 운동≈0) 내 연속 프레임 중심 변위 — 소스별.
- **값(346×260)**: U-Net **0.220**, GSAM2 **0.214**, y_pred **2.294**.
- **해석**: fixation에선 실제 눈 운동이 ~0이므로 프레임간 변위 = **방법의 시간적 안정성**. U-Net·GSAM2 모두
  **~0.22px**로 매우 안정(정밀). 반면 **y_pred(모델)은 2.29px로 10배 큼** → 이는 GT 정밀도가 아니라
  **모델의 시간적 jitter**(event 재구성 노이즈)로 별도 해석해야 한다.
- **한계**: 사람 GT의 F2F는 라벨이 sparse해 측정 불가(SKIP) → 개선 GSAM2(0.214)를 사람-반복 proxy로 사용.

## La.3 — 표현 floor (rasterization floor)
- **측정**: 사람 타원중심 vs 사람 mask centroid 거리(매칭된 주석 프레임 30세션).
- **값**: dist = **1.414 px(346×260) / 0.302 px(64×64)**, 타원-rasterize vs mask IoU **0.873**.
- **해석**: mask가 **타원에서 rasterize된 파생물**임이 확정(코드 `cv2.ellipse` + 데이터 IoU≈0.95(축보정)).
  그래서 이 거리는 **사람 변동성이 아니라 rasterization(이산화) floor**이며, 전 프레임 **상수 (−1,−1)px 생성
  오프셋**이 지배한다. 즉 human_mask는 human_ellipse와 사실상 같아 **독립 소스가 아니다.**
- **한계**: 이 값을 "정밀도"로 보고하면 픽셀화 잡음을 정밀도로 오해하게 됨 → floor로만 사용.

## Lb.1 — 방법 간 pairwise 불일치 + Bland-Altman + IoU
- **측정**: (human,gsam2)/(human,unet)/(gsam2,unet) center-distance RMS, 축별 BA bias/LoA, mask IoU.
- **값**: radial RMS **human-gsam2 0.864/0.186**, human-unet 1.752, gsam2-unet 1.167 (346/64).
  BA bias: **human-unet (+1.02,+1.14)px**(계통), human-gsam2 (+0.48,+0.38). IoU: gsam2/GT **0.914**,
  unet/GT 0.853, gsam2/unet 0.910.
- **해석**:
  - **GSAM2가 사람 GT와 가장 가깝다(RMS 0.86, IoU 0.914)** → 독립 audit로서 신뢰. U-Net은 더 멀다(1.75).
  - **U-Net의 불일치는 대부분 계통 offset (+1.0,+1.1)px**(BA) → 랜덤 산포가 아니라 편향. 그래서 centered
    분산을 쓰는 3CH가 U-Net을 써도 깨끗한 σ_human을 뽑을 수 있다.
  - IoU 0.91(gsam2)은 형상까지 잘 맞음 → center뿐 아니라 마스크 수준 일치.
- **한계**: U-Net은 사람-파생 mask로 학습 → leak-free 비교엔 held-out 프레임 한정 필요.

## Lb.2 — Three-Cornered-Hat (σ_human 분해) ★주축
- **측정**: {human, gsam2, unet} 세 독립 추정기의 pairwise 분산으로 각 소스 고유 σ를 분해(축별). 비음수 검사.
- **값**: **σ_human = 0.552 px(346) / 0.119 px(64), 비음수 ✓**. 독립 triple{h,gsam2,ellseg} σ_human
  0.577/0.134(단 64×64서 음수분산→그 프레임 폐기). σ_gsam2 0.262/0.060, σ_unet 0.66/0.153.
  **bracket [0.552, 0.864] / [0.119, 0.186]**.
- **해석**: **사람 GT 주석의 고유 노이즈 ≈ 0.55px(346) / 0.12px(64×64)**. 이는 GT 자체가 갖는 불확실성 →
  "모델이 이보다 정확할 수는 없다"의 기준선. 두 개의 서로 다른 triple({unet},{ellseg})이 근접값(0.55,0.58)을
  주어 **교차 확증**. σ_gsam2가 셋 중 가장 작음 → GSAM2가 가장 정밀한 추정기.
- **한계**: 3CH는 세 소스의 **노이즈 독립**을 가정. 64×64 coarse 프레임서 {h,gsam2,ellseg}가 음수분산
  (공통편향 신호) → 그 분해는 폐기하고 human↔SAM2 RMS(상한)를 유지(스펙대로).

## Lb.3 — 내부 정합성 교차검증
- **측정**: 3CH가 분리한 σ_gsam2 vs La.1 직접측정 반복성 σ.
- **값**: 3CH σ_gsam2 **0.262/0.060** vs La.1 repeat **0.367/0.069** (ratio 0.72/0.87).
- **해석**: 두 경로(분산분해 vs 직접 반복측정)가 **같은 크기의 GSAM2 노이즈**를 준다 → 3CH 분해가
  독립적으로 **신뢰 가능**함을 뒷받침. (완전 일치가 아닌 0.7~0.9 비율은 3CH가 프레임간 성분도 일부 포함하기 때문.)
- **한계**: 완전 동일치는 아님 — 독립성/공통편향의 잔여 영향 가능.

## Lc.1 — Threshold 정밀도 P_n (abstain 분리)
- **측정**: 사람 GT ±{10,5,1}px 이내 프레임 비율. gsam2(방법 간)·pred(모델). abstain(무검출) 분리.
- **값**:
  - gsam2 vs human: (64×64) **1.00/1.00/1.00**, (346) 0.99/0.99/0.73, abstain 4/483.
  - pred vs human: (64×64) 0.944/0.896/**0.406**, (346) 0.74/0.44/0.02, abstain 0.
- **해석**: **GSAM2는 사람 GT와 1px(64×64) 이내로 ~100% 일치**(방법 간 일치 상한). 모델은 10px 94%/5px
  90%/1px 41%(64×64) → **독립 방법들이 서로 일치하는 허용오차 근처까지** GT를 맞춤. 64×64서 sub-pixel P_n은
  **축소 해상도의 결과**이지 초해상이 아님(native σ_human≈1.3px가 64×64서 0.12–0.3px로 대응).
- **한계**: 1px(346×260) 임계는 매우 엄격(≈0.18px@64×64). abstain은 P_n에서 실패로 계상(precision과 분리).

## STEP 5 — 정정 오차 & budget (y_pred 전용)
- **측정**: E_orig = ‖y_pred − 사람 타원중심‖. 분포·per-subject·per-motion. 0.1812 환산.
- **값**: median **5.698 px(346) / 1.174 px(64)**, mean 11.43/2.33, p95 56.5/10.9, **gross>20px 13%**.
  ≤2/5/10px = 8/44/74%. per-motion(346): fixation 5.78, saccade 4.40, smooth 7.45. worst user8(6.92)/
  best user10(4.58). **0.1812(64×64) → 0.85px(346, 범위 0.74–0.98).**
- **해석 (핵심)**:
  1. **보고 0.1812는 라벨노이즈 band [0.119, 0.186](64×64) *안*** = **floor에 위치**. dense 의사라벨 대비
     floor 근처 측정이라 모델·라벨노이즈 구별 불가 → "너무 좋아" 아티팩트. *(측정 σ_human=0.119 < 0.1812
     이므로 "floor 아래라 물리적 불가"는 성립 안 함 — floor "에" 있음으로 수정.)*
  2. **사람 GT로 정정하면 1.17/5.70px로 상승 = band 위** → dense→human 전환이 "floor"에서 "floor 위"로
     이동시킴 = **실측 가능한 실제 오차**. 누수 모델은 이럴 수 없음(floor 고정+꼬리 없음; 여기선 p95 11/57px,
     gross 13%) → **누수 반증.**
  3. **이 ckpt는 label-noise-limited 아님**: 정정오차(1.17@64)가 band의 6–10배 → sub-pixel 라벨모호성이
     아닌 실제 모델오차(여유 존재).
- **한계 ★**: (a) ckpt `pe0.5401`은 0.1812 생성 모델이 **아님**(자체 dense val≈0.54px@64); (b) samples가
  **train subject**라 낙관 → subject-independent(test 37-48) 값은 이보다 큼.

## 그림 해석
- **fig_3ch.png**: σ_human(0.55/0.58, 두 triple 일치) ≫ σ_gsam2(0.26) — GSAM2가 가장 정밀. 점선(La.1
  repeat 0.37)이 σ_gsam2 근처 → 3CH 신뢰. **해석: 사람 GT가 셋 중 가장 노이즈 큰 소스**(사람 주석이 자동
  방법보다 덜 정밀). 
- **fig_budget.png**: 보고 0.181(환산 0.85)·raster floor 1.41·σ_human lo 0.58·inter-method 0.86은 모두
  **~1px 근방(라벨노이즈 대역)**, 정정 E_orig 5.70만 홀로 높음. **해석: 라벨노이즈는 ~0.6–0.9px에 밀집,
  정정오차는 그 위 = 모델 실오차.**

---

## 종합 해석 (한 문단)
사람 GT 주석의 정밀도는 **σ_human ≈ 0.12px(64×64) / 0.55px(346×260)**이며, 서로 독립인 세 추정량(3CH,
독립 triple, inter-method RMS)이 [0.12, 0.19]/[0.55, 0.86]로 일관되게 구간화한다(내부 교차검증: 3CH
σ_gsam2 ≈ La.1 repeat). 보고된 **0.1812는 이 라벨노이즈 floor에 위치**하는데, 이는 **dense U-Net 의사라벨**
대비 값이라 모델오차와 라벨노이즈가 구별되지 않는 "self-labeling 아티팩트"다. **원본 사람 타원으로 정정하면
오차가 floor 위(1.17px@64 / 5.70px@346)로 올라가 측정이 유의미**해지며, 무거운 꼬리(gross 13%)와 함께
**누수가 아님을 반증**한다. 단, 이 값은 학습 subject·pe0.5401 ckpt 기준이므로 진짜 subject-independent
수치는 test 37-48 재수집으로 확정해야 한다.
