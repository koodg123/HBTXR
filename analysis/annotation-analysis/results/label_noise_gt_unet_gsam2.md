# Label Noise vs human GT — U-Net & GSAM2 (detector-failure→U-Net fallback)

anchors n=483. GSAM2 detector-failure(mislabel/invalid)→U-Net fallback: 4건. 단위 px(346×260), std=sample(ddof=1).

## OVERALL
| source | mean | median | p95 | p99 | std | n |
|---|---|---|---|---|---|---|
| GT vs U-Net | 1.645 | 1.575 | 2.440 | 3.539 | 0.602 | 483 |
| GT vs GSAM2 (fallback) | 0.792 | 0.778 | 1.522 | 1.805 | 0.385 | 483 |

## fixation
| source | mean | median | p95 | p99 | std | n |
|---|---|---|---|---|---|---|
| GT vs U-Net | 1.772 | 1.715 | 2.644 | 4.624 | 0.690 | 161 |
| GT vs GSAM2 (fallback) | 0.919 | 0.861 | 1.681 | 1.955 | 0.439 | 161 |

## saccade
| source | mean | median | p95 | p99 | std | n |
|---|---|---|---|---|---|---|
| GT vs U-Net | 1.543 | 1.525 | 2.231 | 2.659 | 0.457 | 161 |
| GT vs GSAM2 (fallback) | 0.709 | 0.680 | 1.287 | 1.633 | 0.349 | 161 |

## smooth_pursuit
| source | mean | median | p95 | p99 | std | n |
|---|---|---|---|---|---|---|
| GT vs U-Net | 1.621 | 1.559 | 2.400 | 3.023 | 0.617 | 161 |
| GT vs GSAM2 (fallback) | 0.750 | 0.734 | 1.294 | 1.601 | 0.329 | 161 |

## notes
- GT-vs-U-Net bias Δ=(-1.017,-1.138) |Δ|=1.526px (U-Net 노이즈 대부분이 계통 offset).
- GT-vs-GSAM2 RAW(non-failure, n=479): median 0.773 p95 1.438 std 0.367.
- U-Net은 사람 라벨로 학습(비독립), GSAM2는 독립. y_unet=공식 Data_davis_predict proxy.
