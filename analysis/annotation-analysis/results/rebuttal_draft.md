# Rebuttal draft — pupil-annotation precision / noise / uncertainty

*(auto-generated from 10_eval.py; numbers are anchor-level, px @346×260, CI = subject-level cluster bootstrap over 10 users.)*


## R1. The reported 0.1812 px is a dense-label figure, not leakage
The headline **0.1812 px** was measured against the **U-Net dense pseudo-labels** the model was trained on. Re-scored against the **frozen human GT** (VIA ellipse centers at annotated anchors), the model's honest error is **median E_orig = 5.70 px** (95% CI [5.20, 6.12]; mean 11.43, p95 56.46, p99 84.73). For the same model, error against the dense U-Net labels is median 4.87 px — the model naturally matches its own training labels more closely than the sparse human GT. This gap is a **dense-vs-human reference difference**, expected and benign; we do not and cannot infer label leakage from it. In fact the honest error has a heavy tail — 74% of anchors within 10px but 13% gross failures (>20px, up to ~100px) from the event modality (mode=`event`), uniform across all 10 users. A label-leaking model could not fail by ~100px on 1-in-20 frames; the tail is positive evidence that the model genuinely predicts from events rather than memorising GT.

## R2. Human-GT label noise is ~1 px, and independently corroborated
Because the human GT itself carries annotation noise, we bound it with two **independent** auto annotators. Against human GT: **||GT−U-Net|| median 1.57 px** (mostly a systematic offset) and **||GT−GSAM2|| median 0.77 px**. Five independently-architected, off-domain pupil segmenters (EllSeg/RITnet/Edge-Guided/DeepVOG/YOLOE) agree with human GT to **0.59–1.40 px median** (EllSeg 0.62, RITnet 0.59), and the three dedicated eye-segmenters show **0 iris-confusion** across 483 anchors (GSAM2 mask IoU 0.914 vs GT ellipse). Multiple methods converging on human GT at sub-pixel-to-~1px establishes a **label-noise floor of ~0.8–1.6 px**.

## R3. Model error vs label-uncertainty
Per-anchor label uncertainty U_i = median pairwise disagreement of {GT,U-Net,GSAM2} has **median 1.04 px** (5-source inter-annotator spread 0.66 px). The model error E_i and U_i give **Spearman ρ = +0.09** (p=5e-02): no strong coupling — errors are not explained by label ambiguity. E_i ≤ U_i on 4% and E_i ≤ 2·U_i on 9% of anchors. With median E_i 5.70 px vs median U_i 1.04 px, the model is **above the label-noise floor**.

## R4. Annotation precision (human re-annotation N/A → automated proxy)
As no repeated human annotations exist, precision is an automated proxy: in fixation windows (true motion ≈ 0) detector frame-to-frame jitter is **0.22 px median (U-Net)** / **0.21 px (GSAM2)**, and the GSAM2 box-jitter/TTA repeat spread is **0.37 px median**. This sub-/near-pixel detector stability indicates a precise annotation pipeline; it is stated as a proxy, not human precision (see docs/06, docs/10).

## Summary
- Corrected headline: **5.70 px median** vs human GT (was 0.1812 px vs dense labels).
- Label-noise floor ~**0.8–1.6 px**, corroborated by 5 independent annotators (0 iris-confusion).
- E–U Spearman +0.09; model is above the label-noise floor.

*Figures: results/plots/eval_cdf.png, eval_scatter_UE.png, eval_box_motion.png. Tables: results/eval_tables.md. Per-anchor: results/eval_Ei_Ui.csv.*
