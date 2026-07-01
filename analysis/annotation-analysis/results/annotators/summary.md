# Annotator accuracy vs human GT (quality.good anchors)

GT anchors: 483 total, 483 good. Center err in px @346x260. radius_ratio>1.5 = iris/over-seg suspect (center blind to it).

| source                 | valid% | mean | median | p95 | p99 | std | n | iris? |
|------------------------|-------|-------|-------|-------|-------|-------|-------|-------|
| U-Net (EV-Eye)         | 100.0 | 1.645 | 1.575 | 2.440 | 3.539 | 0.602 |  483 | 0/0 |
| GSAM2 (audit)          |  99.2 | 0.782 | 0.773 | 1.438 | 1.702 | 0.367 |  479 | 0/0 |
| ritnet                 | 100.0 | 1.152 | 0.594 | 1.933 | 5.206 | 6.065 |  483 | 0/483 |
| ellseg                 | 100.0 | 0.668 | 0.621 | 1.291 | 1.597 | 0.399 |  483 | 0/483 |
| edge_guided            |  99.0 | 0.774 | 0.718 | 1.388 | 1.715 | 0.535 |  478 | 0/478 |
| deepvog                |  87.4 | 12.354 | 0.907 | 118.187 | 213.930 | 42.562 |  422 | 11/422 |
| yoloe                  |  98.1 | 1.649 | 1.401 | 3.298 | 6.996 | 1.440 |  474 | 38/474 |

## Per-motion median / p95 (px)

| source                 | fixation med/p95 | saccade med/p95 | smooth_p med/p95 |
|------------------------|----------------|----------------|----------------|
| U-Net (EV-Eye)         | 1.71/2.64 (n161) | 1.53/2.23 (n161) | 1.56/2.40 (n161) |
| GSAM2 (audit)          | 0.85/1.65 (n159) | 0.68/1.29 (n161) | 0.73/1.24 (n159) |
| ritnet                 | 0.74/2.67 (n161) | 0.53/1.25 (n161) | 0.52/2.09 (n161) |
| ellseg                 | 0.65/1.40 (n161) | 0.65/1.23 (n161) | 0.60/1.17 (n161) |
| edge_guided            | 0.71/1.49 (n158) | 0.74/1.39 (n161) | 0.72/1.29 (n159) |
| deepvog                | 0.99/122.91 (n128) | 0.93/176.04 (n151) | 0.85/8.08 (n143) |
| yoloe                  | 1.51/3.24 (n156) | 1.28/4.33 (n160) | 1.34/2.31 (n158) |

Notes: mask->uniform ellipse-fit for ALL sources (fair center/radius). GSAM2 baseline excludes mislabel-flagged. Tools trained off-domain (EllSeg/RITnet/Edge-Guided/DeepVOG on other IR sets; YOLOE/SAM3 RGB) — cross-dataset audit, not tuned.
