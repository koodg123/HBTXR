#!/usr/bin/env python3
from pathlib import Path


UPDATES = [
    (
        "docs/Validation-2026-06-05-Status.md",
        "## E2E AXI-Stream Q4W/Q8A Shell Validation - 2026-06-09",
        """

## E2E AXI-Stream Q4W/Q8A Shell Validation - 2026-06-09

| Check | Status | Evidence |
|---|---|---|
| AXI-Stream DMA-facing top | Passed smoke/csynth | hgtxr_e2e_axis_top exposes AXIS input/output, m_axi weights, m_axi runtime_state, and AXI-Lite control. |
| Conv patch embedding | Implemented shell | hgtxr_conv_patch_embedding consumes 256x256 streamed frame and fills token buffer. |
| Global Buffer | Implemented shell | HgtxrGlobalBuffer holds token buffer and pooled vector. |
| Two ATTN units | Implemented shell | hgtxr_e2e_attn_unit<0> and <1> are instantiated by the controller. |
| Two MLP units | Implemented shell | hgtxr_e2e_mlp_unit<0> and <1> are instantiated by the controller. |
| ViT block order | Corrected | Controller now runs ATTN0 -> MLP0 -> ATTN1 -> MLP1 -> ... for six blocks. |
| C++ smoke | Passed | g++ E2E AXIS testbench produced runtime_state=1 count=6 last=1. |
| HLS csim | Blocked by environment | Same Ubuntu/WSL Vitis csim header issue: features-time64.h cannot include bits/wordsize.h. |
| HLS csynth | Passed | solution_e2e_q4w8a: 3.650 ns / 273.97 MHz, 1,610,485..1,610,491 cycles, 8.052 ms, 52 BRAM, 18 DSP, 3798 FF, 6411 LUT. |
| Weight AXI live-read | Passed | hgtxr_e2e_axis_top_csynth.rpt includes gmem_e2e_weights_m_axi_U and AR/R ports. |
| Numerical equivalence | Incomplete | This is an E2E interface/scheduler shell, not yet full HG-PIPE dense QKV/WO/W1/W2/LN/GELU/Softmax numerical equivalence. |
""",
    ),
    (
        "docs/track/PROGRESS-2026-06-06.md",
        "## E2E AXI-Stream ViT Shell Progress - 2026-06-09",
        """

## E2E AXI-Stream ViT Shell Progress - 2026-06-09

- [x] Add DMA-facing AXI-Stream top hgtxr_e2e_axis_top.
- [x] Add conv-style patch embedding shell.
- [x] Add explicit global token buffer.
- [x] Add two ATTN units and two MLP units.
- [x] Correct controller order to ATTN0 -> MLP0 -> ATTN1 -> MLP1 for ViT equivalence.
- [x] Add MLP-based head shell.
- [x] Add ZCU104 Q4W/Q8A E2E define and Vitis HLS csim/csynth scripts.
- [x] Run g++ E2E AXIS smoke.
- [x] Run E2E AXIS csynth and record report-backed resource/latency.
- [ ] Replace shell pooled-attention/synthetic MLP with full HG-PIPE dense QKV/WO/W1/W2 and table math.
- [ ] Resolve Vitis HLS csim environment and run HLS vector equivalence.

Current E2E shell csynth: 273.97 MHz, 1,610,485..1,610,491 cycles / 8.052 ms at 5 ns target, 52 BRAM, 18 DSP, 3798 FF, 6411 LUT, 0 URAM.
""",
    ),
    (
        "docs/track/HANDOFF-2026-06-06.md",
        "## Latest Continuation: E2E AXI-Stream Q4W/Q8A ViT Shell",
        """

## Latest Continuation: E2E AXI-Stream Q4W/Q8A ViT Shell

Added a separate DMA-facing HLS top named hgtxr_e2e_axis_top. It preserves the older hgtxr_top flow and creates a new E2E shell with AXI-Stream frame input/output, m_axi weight/runtime ports, conv patch embedding, global token buffer, two ATTN units, two MLP units, controller sequencing, and an MLP-style state head.

Important correction: for ViT/HG-PIPE equivalence, the controller order is ATTN -> MLP, not MLP -> ATTN. The implemented shell now runs ATTN0 -> MLP0 -> ATTN1 -> MLP1 -> ... across six blocks.

Validation completed:

- g++ E2E AXI-Stream smoke: runtime_state=1 count=6 last=1.
- Static validation passed with the new files registered.
- Vitis HLS csim is blocked by the existing Ubuntu/WSL Vitis header issue.
- Vitis HLS csynth passed for solution_e2e_q4w8a.

E2E shell csynth result: target 5.00 ns, estimated 3.650 ns / 273.97 MHz, latency 1,610,485..1,610,491 cycles / 8.052 ms, resources 52 BRAM_18K, 18 DSP, 3798 FF, 6411 LUT, 0 URAM. The report includes gmem_e2e_weights_m_axi_U and AR/R ports, so the weight AXI read path is live.

This is not yet final HG-PIPE numerical equivalence. Next work should replace pooled-context attention and synthetic MLP seed math with packed Q4 dense QKV/WO/W1/W2, bias, 3-head split/scale, LayerNorm rsqrt LUT, GELU LUT, Softmax exp/recip LUT, and quantization tables.
""",
    ),
    (
        "docs/track/log.md",
        "## 2026-06-09 - E2E AXI-Stream Q4W/Q8A shell",
        """

## 2026-06-09 - E2E AXI-Stream Q4W/Q8A shell

- Added hgtxr_e2e_axis_top with AXI-Stream input/output, m_axi weights/runtime_state, conv patch embedding, global token buffer, two ATTN units, two MLP units, ATTN-first controller order, and MLP-style head.
- Added zcu104_e2e_q4w8a_defines.h and E2E csim/csynth Tcl scripts.
- g++ E2E AXIS smoke passed: runtime_state=1 count=6 last=1.
- Vitis HLS csim remains blocked by the Ubuntu/WSL header issue.
- solution_e2e_q4w8a csynth passed: 273.97 MHz, 1,610,485..1,610,491 cycles, 8.052 ms, 52 BRAM, 18 DSP, 3798 FF, 6411 LUT, 0 URAM.
""",
    ),
    (
        "docs/track/CHANGELOG.md",
        "## 0.1.20 - 2026-06-09",
        """

## 0.1.20 - 2026-06-09

- Added a separate E2E AXI-Stream Q4W/Q8A ViT shell top for DMA-facing execution.
- Added conv patch embedding, global buffer, two ATTN units, two MLP units, ATTN-first controller schedule, and MLP head shell.
- Added ZCU104 E2E define and Vitis HLS csim/csynth scripts.
- Recorded E2E shell csynth resource and latency evidence in the HLS summary CSV.
""",
    ),
]


def main() -> None:
    for rel, marker, entry in UPDATES:
        path = Path(rel)
        text = path.read_text()
        if marker not in text:
            path.write_text(text.rstrip() + entry)


if __name__ == "__main__":
    main()
