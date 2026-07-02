# 10_eval — final accuracy / label-noise / label-uncertainty (anchors)

anchors n=483. All distances px @346x260 (Euclidean). GSAM2 mislabel-flagged excluded. CI = subject-level cluster bootstrap (10 users, B=5000).


## A. Corrected primary accuracy  E_orig = ||y_pred − y_orig|| (HBTXR vs human GT)

| metric | mean | median | p95 | p99 | max | std | n |
|---|---|---|---|---|---|---|---|
| E_orig (pred vs human GT) | 11.431 | 5.698 | 56.455 | 84.727 | 101.667 | 17.044 | 483 |
| E_unet (pred vs U-Net dense) | 10.853 | 4.870 | 55.610 | 84.282 | 103.142 | 17.247 | 483 |

- **median E_orig = 5.698 px** (95% CI [5.20, 6.12]); mean 11.431 (95% CI [10.18, 12.62]).
- dense reference median E_unet = 4.870 px (model tracks its U-Net training labels ~1.2x closer than human GT).
- The paper's reported **0.1812 px** is a dense-label (U-Net) figure; against sparse **human GT** the honest error is **5.70 px median**. Not a leakage claim — a dense-vs-human reference gap.
- distribution: 44% ≤5px, 74% ≤10px; **heavy tail** (mean 11.4, p95 56, p99 85) = **13% event-mode gross failures** (>20px, uniform across all 10 users → modality difficulty, not a subset/bug). pred mode=`event`. **Median is the robust headline.**

## Table 1 · Annotation precision (automated proxy — human re-annotation N/A)

| proxy | mean | median | p95 | p99 | max | std | n |
|---|---|---|---|---|---|---|---|
| fixation f2f jitter — U-Net | 0.654 | 0.220 | 3.509 | 5.501 | 12.565 | 1.144 | 1607 |
| fixation f2f jitter — GSAM2 | 0.653 | 0.214 | 3.537 | 5.582 | 6.301 | 1.139 | 1576 |
| GSAM2 box/TTA repeat spread | 0.371 | 0.367 | 0.420 | 0.514 | 0.559 | 0.035 | 479 |

- In fixation windows true motion ≈ 0, so detector frame-to-frame jitter is a precision proxy: U-Net median 0.22 px, GSAM2 median 0.21 px. GSAM2 box-jitter/TTA spread median 0.37 px. **Not human precision** (no repeats).

## B. Label noise vs human GT (anchor)

| pair | mean | median | p95 | p99 | max | std | n |
|---|---|---|---|---|---|---|---|
| ||GT − U-Net|| | 1.645 | 1.575 | 2.440 | 3.539 | 6.844 | 0.602 | 483 |
| ||GT − GSAM2|| | 0.782 | 0.773 | 1.438 | 1.702 | 1.948 | 0.367 | 479 |

(see also results/label_shape_gt_unet_gsam2.md for radius/IoU; GSAM2 IoU 0.914, 0 iris.)

## C. Label uncertainty  U_i  vs model error  E_i

U_i(design) = median{||GT-UNet||, ||GT-GSAM2||, ||UNet-GSAM2||}; U_i(multi) = inter-annotator spread over {UNet,GSAM2,EllSeg,RITnet,Edge-Guided}.

| quantity | mean | median | p95 | p99 | max | std | n |
|---|---|---|---|---|---|---|---|
| E_i = ||pred − GT|| | 11.431 | 5.698 | 56.455 | 84.727 | 101.667 | 17.044 | 483 |
| U_i (design, 3-source) | 1.142 | 1.041 | 1.701 | 3.539 | 6.099 | 0.521 | 483 |
| U_i (multi, 5-source) | 0.691 | 0.660 | 1.072 | 1.352 | 2.209 | 0.208 | 483 |

- **E_i ≤ U_i : 3.7%**   |   E_i ≤ 2·U_i : 9.1%
- **Spearman(E_i, U_i) = +0.088** (p=5.2e-02) — negligible coupling; model error is largely independent of label ambiguity.
- median E_i / median U_i = 5.47 → model error is ABOVE the label-uncertainty floor (**NOT label-noise-limited**).

## per-motion medians (px)

| motion | E_orig | ||GT-UNet|| | ||GT-GSAM2|| | U_design | n |
|---|---|---|---|---|---|
| fixation | 5.775 | 1.715 | 0.852 | 1.118 | 161 |
| saccade | 4.402 | 1.525 | 0.680 | 1.018 | 161 |
| smooth_pursuit | 7.454 | 1.559 | 0.727 | 1.020 | 161 |

## per-subject E_orig median (px)
| user | median | n |
|---|---|---|
| user1 | 4.830 | 52 |
| user2 | 6.896 | 47 |
| user3 | 5.809 | 46 |
| user4 | 6.127 | 54 |
| user5 | 4.887 | 45 |
| user6 | 5.439 | 47 |
| user7 | 5.855 | 50 |
| user8 | 6.923 | 48 |
| user9 | 5.481 | 50 |
| user10 | 4.583 | 44 |

## figures
- plots/eval_cdf.png  - plots/eval_scatter_UE.png  - plots/eval_box_motion.png

