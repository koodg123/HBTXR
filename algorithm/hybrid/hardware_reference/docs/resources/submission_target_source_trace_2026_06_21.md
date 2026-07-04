# HGTXR-SW Submission Target Source Trace

Date: 2026-06-21

This read-only trace links the second-goal paper target values to the source manuscript under:

`/home/kjm26/project/PRJXR/XR-VIT/PAPER_WORKS/10_submission_initial/main.tex`

It does not change the manuscript and does not run train/eval.

## Source Values

| Source | Line Hint | Values |
|---|---:|---|
| Canonical pupil-state target | 260-264 | `q_k=(x_k,y_k,a_k,b_k,theta_k)`, with `a_k >= b_k`, `theta_k in [0,pi)` |
| Track residual target | 274-277 | `Delta s_k* = q_k - g(alpha_k*)` |
| Table `tab:mode`, hybrid row | 650 | P10 `99.97`, P5 `99.72`, P1 `99.61`, pixel error `0.1812`, latency `0.43` |
| Table `tab:slim`, Option A | 717 | pixel error `0.1812`, power `1.34`, latency `0.43` |
| Table `tab:algo`, HBTXR column | 760-763 | P10 `99.97`, P5 `99.72`, P1 `99.61`, pixel error `0.1812` |
| Table `tab:accel`, deployment range | 801, 806 | error range `0.1812--0.357`, latency range `0.26--0.43` |

## Current Relation To Software Metrics

Direct comparison remains invalid because current software metrics are still recorded as post-transform track metrics, not proven paper-frame hybrid scheduler metrics.

Before direct comparison, require:

- coordinate-frame match,
- hybrid scheduler-mode evaluation,
- paper-frame full-test P1 evidence,
- split protocol match,
- target definition match,
- trained XR-64-or-later result dependency.

Machine-readable source: `docs/resources/submission_target_source_trace_2026_06_21.json`.
