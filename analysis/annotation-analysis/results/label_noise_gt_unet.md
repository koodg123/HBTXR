# Label Noise & Uncertainty — GT (human) vs U-Net

2026-07-01. anchor 483개(U-Net valid 483, 무효 0). 단위 px(346×260). 스크립트 `scripts/10_label_noise.py`. 플롯 `results/plots/`.

## Table 2 — Label Noise  `‖y_orig − y_unet‖`
| 구간 | median | mean | std | p95 | max |
|---|---|---|---|---|---|
| **ALL** | **1.575** | 1.645 | 0.602 | 2.44 | 6.84 |
| fixation | 1.715 | 1.772 | 0.688 | 2.64 | 6.30 |
| saccade | 1.525 | 1.543 | 0.456 | 2.23 | 3.96 |
| smooth_pursuit | 1.559 | 1.621 | 0.615 | 2.40 | 6.84 |

**분해 (핵심):**
- **계통 bias** Δ = (Δx **−1.02**, Δy **−1.14**), **|Δ| = 1.53px** — U-Net center가 사람 GT보다 일관되게 **좌상단**으로 치우침(`plots/gt_unet_offset_scatter.png`).
- **랜덤 dispersion** σx 0.52 · σy 0.68 (bias 제거 후) · RMS 1.75.
- → **1.57px 라벨노이즈의 대부분(1.53px)은 systematic offset**이고 랜덤 성분은 ~0.5–0.7px/축. **bias를 빼면 잔차는 독립 검출기 수준**.

**tail (깨끗함, U-Net 오검출 없음):** d>2px 19.3% · d>3px 1.4% · d>5px 0.4% · **d>10px 0%**. GSAM2와 달리 gross mislabel **0건**.

## Table 3 — Label Uncertainty
- **pairwise U_i = ‖y_orig − y_unet‖**: median **1.575px**, high-unc U_i>1px **92.8%** · >2px 19.3%.
- **3-source U_i = median(‖o−u‖,‖o−g‖,‖u−g‖)** (GSAM2 non-mislabel, n=469): median **1.066px** (더 robust).

## 맥락 / ⚠️ 중요 caveat
- **GT-vs-GSAM2(독립) median 0.73px  <  GT-vs-U-Net median 1.57px.** 즉 사람 라벨로 **학습된** U-Net이, 학습 안 된 **독립** GSAM2보다 사람 GT에서 **더 멀다**. 원인 = U-Net dense 마스크의 **~1.5px 계통 offset**(SAM2는 실제 동공 경계를 더 충실히 분할해 사람 타원중심과 잘 맞음).
- **비독립성**: U-Net은 사람 마스크로 학습됨 → GT-vs-U-Net은 원래 "U-Net이 학습 라벨을 얼마나 재현하나"라 **독립 노이즈 측정이 아님**(보통 낮게 나옴). 그런데 여기선 계통 bias 때문에 오히려 독립 GSAM2보다 큼 → **bias가 U-Net dense 라벨의 실제 성질**.
- **provenance**: y_unet = EV-Eye 공식 `Data_davis_predict`(학습머신 `DeanDataset_full_unet`의 proxy, docs/05 G1). 정확한 수치는 학습 라벨로 교체 필요.

## 시사점
- **U-Net dense 라벨은 사람 GT 대비 ~1.5px 계통 편향**이 있고, 이는 **보정 가능**(Δ 감산). 보정 후 U-Net-vs-사람 랜덤 노이즈는 ~0.5px/축로 GSAM2 독립 floor(0.73px)와 유사.
- reference로서 **독립성·정확도는 GSAM2 > U-Net**(단 GSAM2는 ~3% mislabel, U-Net은 0%). 상호 보완.

## 플롯
- `plots/gt_unet_offset_scatter.png` — offset 산포(계통 bias 입증)
- `plots/label_noise_cdf.png` — GT-vs-U-Net vs GT-vs-GSAM2 CDF
- `plots/gt_unet_hist.png` — ‖y_orig−y_unet‖ 분포
