# HGTXR Hardware ADR

## ADR-2026-06-15-01: Keep C3b As Third-Goal Gate While Adding VREF Experiments

- Status: accepted
- Context: ViT accelerator references suggest useful quantization, memory, pipeline, sparse attention, and MoE ideas.
- Decision: C3b remains the immediate board-smoke gate. VREF experiments are added as successors/ablations.
- Consequences:
  - P2-ViT and ME-ViT can proceed before board access as SW/audit tasks.
  - Attention replacement and MoE routing require paper-scope review.
  - LUT-heavy methods remain negative controls under current resource policy.

## ADR-2026-06-15-02: Memory Placement Policy

- Status: accepted
- Decision: large frame/global/QKV/hidden/deep FIFO buffers prefer URAM; small tables, short control buffers, and tiny FIFOs prefer LUTRAM or BRAM.
- Rationale: user explicitly requested higher URAM/DSP use with lower LUT pressure, while avoiding wasteful URAM use on small memories.

## ADR-2026-06-16-01: C3b Protection Before VREF Promotion

- Status: accepted
- Context: C3b is the current ZCU104 board-smoke candidate and final signoff still has external blockers.
- Decision: VREF work may proceed as software/static/audit work, but no VREF successor may replace or mutate the C3b baseline unless it passes the C3b protection checklist.
- Required successor gates: no C3b overwrite, latency `<= 37508072`, WNS `>= 4.415 ns`, DSP `<= 604`, LUT `<= 126506`, URAM `<= 64`, valid bit/hwh topology, physical smoke evidence, and XR-VITs reference policy.
- Consequence: C3b remains the reference baseline while VREF improvements are evaluated as bounded successors.
