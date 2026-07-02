# Subject-independent E_orig (HBTXR event-img64 vs human GT)

2026-07-03. 미해결 항목 ③ 해소: 진짜 subject-independent `E_orig = ‖y_pred − y_orig‖`(anchor).
이전 5.70px는 **학습 subject(1–10)** 값이라 낙관적이었음 → 이제 held-out val(33–36)·test(37–48)로 재측정.

## 방법
- 모델: **event-mode HBTXR img64** ckpt `weights/hbtxr/hbtxr-imgsz64-epoch66-pe0.5401.ckpt` (subject-independent 학습, users 1–10 제외 → 33–48은 완전 미학습).
- `09_inject_pred.py --mode event --n-events 5000 --img-size 64` on `samples_{val,test}` (anchors-only), valid 100%.
- `E_orig = ‖(pred.cx,cy) − (humanGT.cx,cy)‖` (native 346×260 px), anchor n = val 69 / test 138.

## 결과
| split | n | median | mean | p90 | p95 | max | gross>10px | ≤2px | ≤5px |
|---|---|---|---|---|---|---|---|---|---|
| train (1–10)* | 483 | 5.70 | 11.43 | 24.59 | 56.46 | 101.7 | — | — | — |
| **val (33–36)** | 69 | **7.81** | 16.68 | 49.82 | 60.53 | 73.65 | 44.9% | 5.8% | 36.2% |
| **test (37–48)** | 138 | **6.11** | 12.71 | 34.11 | 63.20 | 90.31 | 24.6% | 8.0% | 34.1% |

\* train = 이전 측정(학습 subject, 참고용).

## 해석 (리벗 관련)
1. **진짜 subject-independent E_orig = 6.11px(test)·7.81px(val)** — 학습 subject 5.70px 대비 **test +7%·val +37%**. 즉 이전 5.70은 낙관적이었으나 **크게 빗나가진 않음**(held-out에서도 동급 수준).
2. **여전히 label-noise floor(≈0.75px)·σ_human(≈0.55px)의 8–10배** → **NOT label-noise-limited**. 보고된 0.1812px(64×64 dense U-Net 라벨)는 실제 pred-vs-사람GT 오차(6.11px)의 **1/34** → 정밀도 floor 산물이지 실제 정확도가 아님.
3. **event-mode gross-fail 24.6%(test)·44.9%(val)** (>10px) — 이벤트 표현의 일부 프레임 실패가 mean/p90/p95를 끌어올림(median은 robust). subject 편차(val>test)는 표본 작음(n69)·subject 난이도 차.
4. **결론 강화**: held-out에서도 모델 오차가 라벨 노이즈보다 훨씬 커, "0.1812px = 라벨 노이즈 한계에 도달한 정밀도"라는 리뷰어 우려는 **정확도(E_orig)와 무관** — 0.1812는 정밀도(재현성) floor일 뿐.

## 산출
- `samples_{val,test}/label/*/pred.json` (event img64 pred, anchor).
- 비교 baseline: `docs/09_hbtxr_pred_results.md`(train 5.70), 정밀도 `results/precision/`.
