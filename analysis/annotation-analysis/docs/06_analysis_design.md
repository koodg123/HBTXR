# 06 · 평가 설계 (10_eval 사양 + 리벗 구성)

## 라벨 소스 기호
```
y_orig  : 사람 GT (anchor, VIA 타원중심)          [진실, frozen]
y_unet  : U-Net 의사라벨 center (dense)           [audit / 0.1812의 라벨]
y_gsam2 : Grounded-SAM2 center (dense)            [독립 audit]
y_pred  : HBTXR 예측 center                       [모델]
```
모든 center 346×260 px. 거리 = 유클리드 px.

## A. Corrected primary accuracy (헤드라인 교체)
- `E_orig_i = ‖y_pred_i − y_orig_i‖` (anchor에서만, frame/event/hybrid 모드별)
- 보고: mean/median/p95/p99/max + **subject-level cluster-bootstrap 95% CI**
- 0.1812(=`‖y_pred − y_unet‖`, dense 라벨 기준)와 나란히 제시하여 **dense-vs-human reference gap** 노출.

## B. Annotation Precision (표1)
- 사람 불가 → 자동 proxy:
  1. **fixation F2F jitter**: 각 annotator(y_unet/y_gsam2/y_pred)의 fixation 윈도우 내 프레임간 center 변화(px). 참 운동≈0 → 변동=측정정밀도.
  2. **GSAM2 perturbation 산포**: `gsam2.json.repeats`의 std(같은 프레임, 박스지터/TTA).
- 표: metric × {mean, median, p95, p99} (+ "human N/A, automated proxy" 주석).

## C. Label Noise (표2)
- anchor에서 `‖y_orig − y_unet‖`, `‖y_orig − y_gsam2‖` (+ valid rate, rejection 사유).
- 의미: 원래 사람 GT 대비 audit 라벨의 편차 = label noise floor 후보.

## D. Label Uncertainty (표3)
- sample별: `U_i = median{‖y_orig−y_unet‖, ‖y_orig−y_gsam2‖, ‖y_unet−y_gsam2‖}` (anchor).
- `E_i = ‖y_pred − y_orig‖`.
- 보고: `E_i` vs `U_i` 분포, **`E_i≤U_i` 비율**, `E_i≤2U_i` 비율, **Spearman(E_i, U_i)**, per-subject/per-motion.
- 해석: 헤드라인 = **median E vs median U**. `E≲U`면 "label-noise-limited". Spearman>0이면 "어려운(불확실) 라벨에서 error 큼 → 모델이 나쁜 게 아니라 라벨 한계".

## E. 그림
- CDF(E_i), CDF(U_i), scatter(x=U_i, y=E_i), boxplot(motion·subject별 E_i/U_i).

## F. 10_eval 입출력
- 입력: `manifest_*.csv` + `label/{key}/{gt,unet_dense,gsam2,pred}.json`
- 로직: anchor 매칭으로 y_orig/y_unet/y_gsam2/y_pred 정렬 → 거리·통계 → 표·그림.
- 없는 소스는 스킵(예: gsam2/pred 아직이면 label-noise(orig vs unet)·dense-vs-human gap만 먼저 산출).
- 출력: `../results/` 표(csv/md) + `plots/` + `rebuttal_draft.md`.

## G. 리벗 문구 골격
```
We separate annotation precision, label noise, and label uncertainty.
Annotation precision (no human re-annotation available) is estimated as the
frame-to-frame pupil-center jitter during fixations and the spread of Grounded-SAM2
under prompt perturbation. Label noise is the disagreement between the original
EV-Eye human labels and U-Net / Grounded-SAM2 audit centers. Label uncertainty U_i
is the per-sample pairwise disagreement among these sources. Our previously reported
0.1812 px was measured against the U-Net dense pseudo-labels used in supervision
(a dense pseudo-label reference, not human GT); recomputing against the original human labels yields <X> px
(median <..>, p95 <..>), which is [comparable to / below] the label-noise floor of
<σ> px. E_i and U_i are positively correlated (Spearman <r>), indicating the residual
error is dominated by label uncertainty rather than model error.
```

## H. 미결 파라미터
- 0.1812 모드(frame/event/hybrid) → `--mode`
- subject-independent ckpt + 그 config의 img_size/patch_size(G)
- (정밀 dense-label 대비용) 학습머신 full_unet anchor 라벨 확보 여부
