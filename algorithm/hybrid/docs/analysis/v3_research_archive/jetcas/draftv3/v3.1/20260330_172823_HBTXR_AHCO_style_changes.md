# HBTXR → AHCO-YOLO style revision notes

This revision adapts the JETCAS draft toward the narrative and organizational style of the AHCO-YOLO paper while keeping the HBTXR technical content and reported numbers intact.

## Main style changes applied

1. **Abstract reframing**
   - Rewritten in a more unified “co-optimization framework” tone.
   - Emphasizes system-level synergy among supervision, model design, runtime scheduling, and hardware mapping.
   - Keeps the original HBTXR metrics unchanged.

2. **Introduction rewrite**
   - Reorganized to more explicitly contrast prior directions and their limitations.
   - Added stronger “To address these issues, we propose…” framing.
   - Inserted an integrated-pipeline description mirroring AHCO-YOLO’s holistic narrative.

3. **Contribution style update**
   - Converted the contribution list to AHCO-like contribution bullets with short emphasized labels.
   - Sharpened wording around geometry-stable supervision, Search/Track transformer, runtime scheduler, and deployment architecture.

4. **Related work / motivation update**
   - Preserved the original subsection ladder but shifted the closing subsection to a stronger motivation-oriented positioning.
   - Increased emphasis on why isolated optimization is insufficient for XR eye tracking.

5. **Method and accelerator tone adjustment**
   - Tightened section-openers so they read more like a co-design paper rather than a pure model paper.
   - Strengthened the link between algorithmic decomposition and hardware consequences.

6. **Experimental narrative update**
   - Rewrote the main-results discussion and prior-work comparison in a more source-native, careful benchmarking style.
   - Reinforced deployment-oriented interpretation rather than overclaiming cross-paper normalization.

7. **Conclusion rewrite**
   - Recast the conclusion around the AHCO-like theme that practical performance comes from joint design across the stack.
   - Ends with a clearer system-level takeaway and open directions.

## Deliverables

- `HBTXR_AHCO_style_revision.pdf`: IEEE-style revised manuscript
- `HBTXR_AHCO_style_revision.tex`: editable LaTeX source
