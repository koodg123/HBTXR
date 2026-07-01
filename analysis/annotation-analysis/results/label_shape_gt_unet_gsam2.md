# Shape metrics vs human GT — radius_ratio & mask IoU (anchors)

radius_ratio = r_equiv(det)/r_equiv(GT), ~1.0=pupil, **>1.5 = iris/over-seg suspect** (center-blind; docs/14). IoU = det_mask ∩ GT-ellipse / ∪. GSAM2 mislabel-flagged excluded.

## OVERALL
| metric | mean | median | p95 | p99 | std | n |
|---|---|---|---|---|---|---|
| radius_ratio U-Net | 0.995 | 0.993 | 1.054 | 1.097 | 0.040 | 483 |
| radius_ratio GSAM2 | 1.019 | 1.016 | 1.067 | 1.126 | 0.028 | 479 |
| mask IoU U-Net | 0.839 | 0.853 | 0.908 | 0.923 | 0.061 | 483 |
| mask IoU GSAM2 | 0.908 | 0.914 | 0.955 | 0.960 | 0.036 | 479 |

- **iris-suspect(ratio>1.5) U-Net: 0/483** (max 1.30)

- **iris-suspect(ratio>1.5) GSAM2: 0/479** (max 1.17)

## fixation
| metric | mean | median | p95 | p99 | std | n |
|---|---|---|---|---|---|---|
| radius_ratio U-Net | 0.988 | 0.987 | 1.046 | 1.104 | 0.042 | 161 |
| radius_ratio GSAM2 | 1.016 | 1.011 | 1.064 | 1.167 | 0.032 | 159 |
| mask IoU U-Net | 0.822 | 0.835 | 0.905 | 0.919 | 0.071 | 161 |
| mask IoU GSAM2 | 0.894 | 0.900 | 0.950 | 0.959 | 0.041 | 159 |

## saccade
| metric | mean | median | p95 | p99 | std | n |
|---|---|---|---|---|---|---|
| radius_ratio U-Net | 0.991 | 0.991 | 1.043 | 1.069 | 0.033 | 161 |
| radius_ratio GSAM2 | 1.015 | 1.015 | 1.058 | 1.085 | 0.024 | 161 |
| mask IoU U-Net | 0.854 | 0.861 | 0.908 | 0.930 | 0.049 | 161 |
| mask IoU GSAM2 | 0.916 | 0.920 | 0.956 | 0.962 | 0.029 | 161 |

## smooth_pursuit
| metric | mean | median | p95 | p99 | std | n |
|---|---|---|---|---|---|---|
| radius_ratio U-Net | 1.005 | 1.001 | 1.069 | 1.101 | 0.042 | 161 |
| radius_ratio GSAM2 | 1.026 | 1.021 | 1.078 | 1.128 | 0.027 | 159 |
| mask IoU U-Net | 0.842 | 0.856 | 0.908 | 0.919 | 0.059 | 161 |
| mask IoU GSAM2 | 0.914 | 0.918 | 0.953 | 0.960 | 0.032 | 159 |

## notes
- ROI+geom harness(marker fix)로 GSAM2 홍채혼동 해소 확인용. GT 타원 rasterize로 IoU.
- U-Net mask=공식 predict .gif, GSAM2 mask=ROI+geom .png(0/255).
