# Goal Audit - 2026-06-10 Third Goal

Scope: current evidence-based audit for active `[3차목표]` items (0)..(11).

Status legend:

- Proved: current files and command evidence directly support the requirement.
- Partial: meaningful implementation exists, but final objective still needs stronger proof.
- Blocked-by-choice: implementation can continue, but user asked for direction choice before major branch work.
- Missing path: referenced source path is absent or differs in this Ubuntu workspace.

## Summary

Current best hardware points:

- AXIS PAR8 baseline: `hgtxr_e2e_axis_top`, `active196_b6_ff768`, Q4W/Q8A, `dsp_mixed_stream`, CSim `[32, -13, 26, -6, 14, -11]`, CSynth `3.744 ns`, `71,316,968 cycles`, `306 BRAM_18K`, `332 DSP`, `43,740 FF`, `81,144 LUT`, `64 URAM`.
- A2 memory-mapped wrapper: `hgtxr_e2e_m_axi_top(frame, weights, out_state, runtime_state)`, full CSim `[32, -13, 26, -6, 14, -11]`, safe-LUTRAM CSynth `3.744 ns`, `81,514,836 cycles`, `326 BRAM_18K`, `334 DSP`, `46,502 FF`, `84,220 LUT`, `64 URAM`.
- C PAR16 candidate: AXIS `HGTXR_E2E_PAR=16`, reduced CSim `[18, -3, 3, -2, 1, -4]`, full CSynth `3.953 ns`, `37,508,072 cycles`, `338 BRAM_18K`, `604 DSP`, `59,507 FF`, `127,916 LUT`, `64 URAM`.

Current decision gate:

- Option A board implementation is the recommended next proof, but not a direct old-flow command.
- A2 HLS wrapper is complete; remaining A2 work is package/BD/PYNQ for the memory-mapped wrapper.
- A1 keeps the AXIS top and builds AXIS/DMA Vivado/PYNQ after or alongside A2 board work.
- A3 replaces old flow in place with higher regression risk.
- B tunes HLS timing first.
- C sweeps higher parallelism.
- E aligns final trained weights/LUT/calibration.

## Requirement Checklist

| Item | Requirement | Status | Evidence | Gap / next action |
|---|---|---:|---|---|
| 0 | Analyze plans, progress, HANDOVER, docs, then continue work | Proved | `docs/Master-Plan.md`, `docs/Sub-Plan.md`, `docs/track/HANDOVER-2026-06-10-E2E.md`, `docs/track/PROGRESS.md`, `docs/track/NEXT-DECISION-2026-06-10-E2E.md` updated | Continue updating the same artifacts after selected implementation path |
| 1 | Ubuntu Linux paths, `/tools/Xilinx`, changed cwd | Proved | HLS command docs use `/tools/Xilinx/Vitis_HLS/2023.2`; cwd is `hardware`; `docs/Spec.md` records spec-kit CLI state in Ubuntu | None for path migration; board commands still need selected flow |
| 2 | Proper experiment plan/spec first, use spec-kit if possible, maximize Spark | Partial | `docs/Spec.md`, `docs/Master-Plan.md`, `docs/Sub-Plan.md`; repeat check `command -v spec-kit || command -v specify || echo manual-spec`; Spark attempts recorded | Spec maintained manually because spec-kit CLI unavailable; Spark quota exhausted until 2026-06-15 23:18, GPT5.5 fallback used |
| 3 | Prioritize ZCU104 cyclic hardware accelerator baseline over full paper | Proved | ZCU104 configs, E2E resource policies, `dsp_mixed_stream` fit, board decision docs | Need board implementation/timing proof via A1/A2/A3 |
| 4 | Run paper ViT model E2E | Partial | `active196_b6_ff768` covers `blocks=6`, `active_tokens=196`, `ff_dim=768`, full CSim/CSynth pass | Board execution and final trained-paper equivalence not proven |
| 5 | Q4 weights, Q8 activations, arbitrary weights allowed if SW/HW equal | Proved for synthetic baseline | `configs/zcu104_e2e_q4w8a_defines.h` sets `HGTXR_BIT_WIDTH=8`, `HGTXR_WEIGHT_BIT_WIDTH=4`; generated specs/goldens and CSim prove SW/HW equality | Final trained weights still open under Option E |
| 6 | Parameterize tiling, parallelism, bus width, bit width, buffer size, FIFO depth | Proved for current baseline | `HGTXR_TILING_FACTOR`, `HGTXR_PARALLELISM_FACTOR`, `HGTXR_E2E_PAR=16|32`, `HGTXR_BUS_WIDTH=256`, `HGTXR_BIT_WIDTH=8`, `HGTXR_BUFFER_SIZE=256`, `HGTXR_FIFO_DEPTH=128` | PAR32 remains a risk-managed C stress test |
| 7 | Analyze HG-PIPE code and reflect needed pieces | Proved for LUT/math families | `docs/resources/hgpipe_reference_analysis_2026_06_08.md`, `refs/hgpipe_lut_math_contract.json`, validation artifact | Remaining HG-PIPE matmul/root/network/patch families not fully promoted |
| 8 | Implement LayerNorm, GeLU, Softmax, Quantization from HG-PIPE paper/code | Partial | Isolated HG-PIPE GeLUQ, SoftmaxQ, LayerNormQ, attention Q/K/V/A quant helpers and validation over HG-PIPE refs; named reduced gates exist | Default full E2E path promotion still gated by SW mirror/golden regeneration |
| 9 | Refer to `PAPER_PRJXR/05_RESOURCES/DeiT-Tiny C-Syn Results.png` | Missing path / partial substitute | Requested path not found; same filename found at `/home/kjm26/project/PRJXR/XR-VIT/HGPIPE/DeiT-Tiny C-Syn Results.png`; resource metrics CSV exists | Confirm whether HGPIPE image is intended replacement; image-specific OCR/manual values not current proof |
| 10 | Refer to `/home/kjm26/project/PRJXR/XR-VIT` experiments and code | Partial | Current docs/resources include parsed `impl_repos` metrics and HGTXR/HGPIPE references | Keep using current evidence; some requested sibling paths differ from present workspace |
| 11 | Use HLS code in `/home/kjm26/project/PRJXR/XR-VITs` for ZCU104 fit/low latency | Missing path | `find /home/kjm26/project/PRJXR/XR-VITs` reports no such directory | User must provide corrected path, or continue with available `XR-VIT/HGTXR`, `XR-VIT/HGPIPE`, and parsed `impl_repos` evidence |

## Completion Judgment

Goal is not complete.

Proved:

- Ubuntu `/tools/Xilinx` migration is reflected in scripts/docs.
- ZCU104 Q4W/Q8A E2E HLS baseline exists and passes full-scale CSim/CSynth.
- DSP/URAM/LUT balance improved: PAR16 raises DSP to `604`, cuts latency to `37,508,072` cycles, and keeps URAM useful at `64/96`; LUT rises to `127,916`, so board routing/timing risk remains.
- HG-PIPE LayerNorm/GeLU/Softmax/Quantization reference contracts exist and validate isolated/ref-gated math.
- Parameter knobs are present and active in current config.

Not yet proved:

- Board implementation timing.
- Board/PYNQ runtime execution.
- E2E packaged IP for `hgtxr_e2e_axis_top`.
- Final trained-paper weight/LUT/calibration equivalence.
- Default-path promotion of all HG-PIPE integer math in full active196/b6 mode.
- Correctness against `/home/kjm26/project/PRJXR/XR-VITs`, because that path is absent.

## Recommended Next Choices

1. A2: fastest board-smoke path.
   - HLS wrapper and CSim/CSynth are complete.
   - Next: build wrapper package/BD/PYNQ flow.
   - Expected: quickest `.bit/.hwh` path, with AXI-MM driver close to old PYNQ style.

2. A1: clean architecture path.
   - Package existing AXIS top.
   - Build AXI DMA BD and PYNQ stream driver.
   - Expected: preserves validated public E2E interface, but DMA packing and BD work are larger.

3. B: timing-risk reduction before board build.
   - Tune HLS slack around `-0.09 ns`.
   - Expected: lower chance of Vivado implementation timing failure.

4. C: more parallelism.
   - PAR16 is complete.
   - Next optional stress: PAR32.
   - Expected: more DSP, lower latency if memory/timing hold; high LUT/timing risk.

5. E: paper fidelity.
   - Replace synthetic weights/LUTs with final trained artifacts.
   - Expected: stronger paper reproduction, but slower path to board baseline.
