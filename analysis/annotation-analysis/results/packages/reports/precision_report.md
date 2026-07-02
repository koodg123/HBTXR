# Annotation Precision / Label Noise — synthesis (346×260 px)

Samples = users 1-10 (⚠ HBTXR **TRAIN** subjects; test=37-48). Layers La/Lb/Lc measure label quality (subject-split-agnostic); only STEP5 E_orig is affected by the train-subject caveat.

## Label-noise / precision budget
| quantity | px (346×260) | note |
|---|---|---|
| σ_human (3CH {h,gsam2,unet}) | **0.552** | shared_bias=False |
| σ_human (3CH indep {h,gsam2,ellseg}) | 0.577 | corroborates; shared_bias=False |
| σ_human bracket | [0.552, 0.864] | [3CH, Human↔GSAM2 RMS] |
| inter-method RMS (human–gsam2) | 0.864 | independent audit vs GT |
| GSAM2 perturbation σ (La.1) | 0.367 | box-jitter/TTA repeatability |
| fixation F2F jitter — GSAM2 (La.2) | 0.214 | method stability (human-repeat proxy) |
| rasterization floor (La.3) | 1.414 | mask DERIVED (IoU 0.873); ~1px constant gen-offset, not precision |
| **reported 0.1812 (64×64 vs dense) → 346×260** | **0.8492** | range [0.7361, 0.9796]; sits AT the σ_human floor |
| corrected E_orig (pred vs human GT) | **5.698** | median; ⚠ TRAIN subj, optimistic |

## Reading
- Human-GT annotation noise is **σ_human ≈ 0.552 px** (bracket [0.552, 0.864]), independently corroborated by the {h,gsam2,ellseg} triple (0.577) and by inter-method RMS (0.864).
- The reported **0.1812 px is a 64×64, dense-label figure ≈ 0.8492 px in 346×260** — i.e. **at the label-noise floor**. A metric measured against dense pseudo-labels near the annotation-noise floor cannot separate model error from label noise; that is why it looks 'too good'.
- Against **human GT**, the honest error is **5.698 px median** (mean 11.43, p95 56.5, 13% gross >20px) — well ABOVE the σ_human floor → genuine model error, NOT label leakage (a leaking model would sit at the floor with no heavy tail). ⚠ This E_orig is on TRAIN subjects (1-10); the subject-independent value (test 37-48) will be ≥ this.
- Bland-Altman shows U-Net carries a **systematic (+1.02, 1.14) px offset** from human GT (so U-Net noise is mostly bias, not scatter).

*Figures: fig/fig_3ch.png (σ decomposition), fig/fig_budget.png (budget). Tables: tables/*.csv. Scalars: results/precision/precision_summary.json.*
