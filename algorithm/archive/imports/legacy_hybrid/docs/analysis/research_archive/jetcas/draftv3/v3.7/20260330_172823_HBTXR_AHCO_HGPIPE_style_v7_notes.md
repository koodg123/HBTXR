# HBTXR v7 change notes

This revision applies the requested consistency update to the manuscript narrative.

## Main changes

1. **Half-backbone early-exit moved into implicit model pruning / network slimming**
   - The abstract, introduction, contribution list, algorithm description, and conclusion were revised so that the front-half Track early-exit is no longer described as the default full-model behavior.
   - Instead, the manuscript now states that the **deployment student produced by implicit model pruning** keeps the full Search path and introduces a **Track-side cut point** after the front half of the student backbone.
   - Section III-C and Section III-E were updated accordingly, including the cut-point notation `L_c = floor(L^S / 2)`.

2. **AUX Head removed from the paper narrative**
   - The back-end head description now uses only:
     - Eye Region Head
     - Pupil Search Head
     - Event Search Head
     - Pupil Track Head
     - Search Mask Head
   - AUX-head wording was removed from the algorithm section, scheduler description, training/setup prose, and hardware head-module prose.

3. **Hardware-side explanation kept intact in structure and tone**
   - The hardware section still uses the same multi-stage pipeline / streaming / cross-layer coupling / intra-layer dataflow narrative.
   - Only terminology directly tied to the deleted AUX head was cleaned up for consistency.

4. **Table/layout QA**
   - The updated PDF was recompiled and visually checked.
   - The comparison tables on the later pages remain within the page/column layout in the rendered output.

## Files

- `HBTXR_AHCO_HGPIPE_style_v7.tex`
- `HBTXR_AHCO_HGPIPE_style_v7.pdf`

