# HBTXR AHCO-style restructured draft (v2)

This version rewrites the manuscript into the requested AHCO-YOLO-style section hierarchy.

## Main structural changes
- Reorganized the paper into:
  1. Abstract
  2. Introduction
  3. Related Works
  4. Proposed HBTXR Algorithm
  5. Proposed HBTXR Hardware
  6. Experimental Results
  7. Conclusion
- Split the algorithm section into framework design, supervision pipeline, modality-aware transformer, scheduler, and training/loss pipeline.
- Split the hardware section into co-design principles, quantization/non-linear approximation, accelerator architecture, per-module descriptions, and CPU-FPGA partition.
- Split experiments into setup, proposed evaluation, algorithm comparison, accelerator comparison, and ablation.

## Important note
The original `JETCAS_Draft_V3.pdf` provides verified latency/accuracy numbers, but it does **not** provide finalized FPGA resource or power measurements for HBTXR. Therefore:
- latency / accuracy rows are filled with verified draft values
- resource / power entries in the hardware implementation table are intentionally left as `TBD`

## Files
- `HBTXR_AHCO_restructured_v2.tex`: editable LaTeX source
- `HBTXR_AHCO_restructured_v2.pdf`: compiled PDF
