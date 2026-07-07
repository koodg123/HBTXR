# HBTXR v8 revised

Applied updates requested from the v7 review, except for adding a separate hardware-parameter summary table.

## Main changes
- Added a real overview figure in place of the placeholder.
- Added a supervision/data-contract figure in Section III-B.
- Replaced the model-architecture placeholder with a cut-point-aware architecture diagram.
- Added a Search/Track scheduler FSM figure.
- Replaced hardware overview and hardware timeline placeholders with concrete diagrams.
- Removed the evaluation placeholder figure.
- Added explicit residual-decode equations for Track mode.
- Rewrote event accumulation as fast causal accumulation with a causal weighting equation.
- Added explicit scheduler-signal definitions for Search confidence, Track confidence, Track quality, similarity, density, and eye-state flag.
- Added the ellipse covariance definition used in the geometry loss.
- Made teacher/student slimming more explicit with structural shrink factors.
- Kept the Network Slimming description aligned with: reduced embed dim / MLP ratio / channel count / depth, followed by Feature KD and RKD.
- Renamed the event-side head to `Event-Validation Head` to avoid awkward Search/Track terminology overlap.
- Expanded the mode-wise table with `Backbone span` and `Active heads`.
- Reworked the algorithm comparison table to replace sparse Params/FLOPs fields with `Prev. state` and `Search/Track` columns.
- Added a `Task` row to the accelerator comparison table and clarified that HBTXR hardware values are projected ZCU104 targets.
- Expanded the ablation table with `w/o implicit model pruning`, `w/o Track early-exit`, `w/o Feature KD`, and `w/o RKD`.
- Removed AHCO-style meta wording from the body text and conclusion.
- Compressed the non-linear approximation subsection from five detailed operator bullets into three representative datapath mappings.

## Layout / formatting
- Recompiled and checked that no table or equation extends beyond the column boundary.
- Final PDF length remains 12 pages.
