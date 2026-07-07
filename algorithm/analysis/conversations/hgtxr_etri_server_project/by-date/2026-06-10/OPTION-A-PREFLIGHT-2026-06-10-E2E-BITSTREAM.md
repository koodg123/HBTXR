# Option A Preflight - E2E Bitstream/Implementation

Date: 2026-06-10

Scope: prepare, but do not run, board implementation/timing for the selected E2E point.

Selected HLS point:

- Top: `hgtxr_e2e_axis_top`
- Scale: `active196_b6_ff768`
- Resource policy: `dsp_mixed_stream`
- QKV cache: `HGTXR_E2E_QKV_WEIGHT_CACHE=1`
- CSim: passed, output `[32, -13, 26, -6, 14, -11]`
- CSynth: passed, `3.744 ns`, `71,316,968 cycles`, `306 BRAM_18K`, `332 DSP`, `43,740 FF`, `81,144 LUT`, `64 URAM`

## Preflight Finding

The existing bitstream/PYNQ flow is not yet wired for `hgtxr_e2e_axis_top`.

Existing flow assumptions:

- `vivado/scripts/package_ip.tcl` sources `create_hls_project.tcl`.
- `create_hls_project.tcl` sets top `hgtxr_top`, not `hgtxr_e2e_axis_top`.
- `vivado/scripts/build_bitstream.tcl` default IP repo is `hardware/generated/hgtxr_hls/solution/impl/ip`.
- `build_bitstream.tcl` instantiates `xilinx.com:hls:hgtxr_top:1.0`.
- `build_bitstream.tcl` connects old memory-mapped ports:
  - `m_axi_gmem_frame`
  - `m_axi_gmem_event_pos`
  - `m_axi_gmem_event_neg`
  - `m_axi_gmem_prev_state`
  - `m_axi_gmem_out_state`
  - `m_axi_gmem_runtime_state`
- Current E2E top has different interface:
  - AXIS input: `axis_in`
  - AXIS output: `axis_out`
  - m_axi weight port: `gmem_e2e_weights`
  - m_axi runtime port: `gmem_e2e_runtime`
  - AXI-Lite control: `weights`, `num_pixels`, `runtime_state`, `return`
- Existing PYNQ driver `pynq/hgtxr/hgtxr_overlay.py` is also old memory-mapped-buffer oriented and writes `frame`, `event_pos`, `event_neg`, `prev_state`, `out_state`, and `runtime_state`.
- Current visible `hardware/generated/hgtxr_e2e_hls` tree contains the CSim log, but no E2E `component.xml` or visible `csynth.rpt`; the E2E IP package must be regenerated before any board-flow proof.

Therefore, running current `build_bitstream.tcl` directly is expected to fail or build the wrong IP path. Option A needs a small implementation-flow decision first.

## Sub-Agent Audit Capture

- Spark A1 explorer was attempted first, but GPT5.3-Codex-Spark quota was exhausted until 2026-06-15 23:18.
- GPT5.5 fallback A1 explorer completed read-only and made no file edits.
- GPT5.5 A2 evaluator completed read-only and made no file edits.

## Sub-Option A1 - Build E2E AXIS BD With DMA

Work plan:

- Add or clone an E2E-specific HLS package Tcl that exports `hgtxr_e2e_axis_top`.
- Add or clone an E2E-specific Vivado bitstream Tcl.
- Instantiate `hgtxr_e2e_axis_top`.
- Add AXI DMA or equivalent stream infrastructure for `axis_in` and `axis_out`.
- Connect `gmem_e2e_weights` and `gmem_e2e_runtime` to PS HP/HPC through SmartConnect.
- Connect AXI-Lite control to PS HPM.
- Generate bit/hwh and copy to an E2E overlay package path.

BD/IP blocks likely needed:

- Zynq UltraScale+ MPSoC.
- `hgtxr_e2e_axis_top_0` HLS IP.
- AXI DMA in simple mode with MM2S and S2MM enabled, scatter-gather disabled.
- AXI SmartConnect for AXI-Lite control to HLS `s_axi_control` and DMA `S_AXI_LITE`.
- AXI SmartConnect for DDR access from DMA MM2S/S2MM plus HLS `gmem_e2e_weights` and `gmem_e2e_runtime`.
- Optional AXIS Data FIFO if backpressure/debug visibility is needed.

Expected result:

- Board design matches the current public E2E AXIS top.
- Closest path to validating the existing CSim/CSynth interface.

Cost/runtime:

- Medium implementation effort plus long Vivado implementation runtime.

Risk:

- DMA register/runtime setup required.
- PYNQ driver update required for stream send/receive plus weight buffer pointer.
- More integration surface than memory-mapped wrapper.
- DMA length width must support `65536 * 32 = 2,097,152` input bytes because the current 256-bit AXIS reader consumes one 256-bit beat per pixel and uses the low 8-bit lane.
- `num_pixels` is currently ignored by the E2E top, so short DMA transfers can hang.
- PYNQ packing must send one 256-bit beat per pixel and read six 256-bit output beats, using the low 16-bit lane for each state output.

Validation evidence:

- HLS IP `component.xml` for `hgtxr_e2e_axis_top`.
- Vivado `validate_bd_design` passes.
- Routed timing WNS/TNS.
- Utilization report.
- `.bit` and `.hwh` copied to package.
- PYNQ stream test returns `[32, -13, 26, -6, 14, -11]`.

Choose when:

- User wants to preserve the E2E AXIS contract already validated by CSim.

## Sub-Option A2 - Add E2E Memory-Mapped Wrapper

Work plan:

- Add a new wrapper top around E2E internal functions, not an in-place replacement and not a nested call to `hgtxr_e2e_axis_top`.
- Recommended top signature:
  - `hgtxr_e2e_m_axi_top(frame, weights, out_state, runtime_state)`
  - `frame`: `m_axi`, bundle `gmem_frame`, depth `65536`
  - `weights`: `m_axi`, bundle `gmem_e2e_weights`, depth `HGTXR_E2E_WEIGHT_DEPTH`
  - `out_state`: `m_axi`, bundle `gmem_out_state`, depth `6`
  - `runtime_state`: `m_axi`, bundle `gmem_runtime_state`, depth `1`
  - all pointer arguments plus return on `s_axilite bundle=control`
- Wrapper exposes memory-mapped frame/input, output state, weights, and runtime state similar to existing `hgtxr_top` flow.
- Reuse more of existing `build_bitstream.tcl` and PYNQ driver patterns through cloned E2E MM scripts/classes, not by overwriting old flow.
- Keep the original AXIS top for HLS regression.

Expected result:

- Faster path to reuse current Vivado/PYNQ infrastructure.
- Avoids DMA stream integration for the first board proof.

Cost/runtime:

- Medium code/HLS work, then CSynth and Vivado implementation.

Risk:

- New wrapper needs SW/HW CSim equivalence.
- Resource and timing may differ from current AXIS CSynth point, but expected DSP/URAM should stay effectively unchanged; BRAM/LUT/FF may rise slightly from AXI-MM adapters/control.
- Could drift from the already-validated public E2E AXIS top if not carefully mirrored.
- Existing HLS timing risk around `-0.09 ns` remains; A2 is not a timing fix.

Validation evidence:

- New wrapper CSim matching `[32, -13, 26, -6, 14, -11]`.
- Wrapper CSynth resource/timing under ZCU104 budget.
- Existing or lightly modified Vivado BD passes.
- PYNQ memory-mapped test passes.

Choose when:

- User wants shortest path to board smoke using existing infrastructure.

## Sub-Option A3 - Patch Existing Build Script Directly

Work plan:

- Modify `build_bitstream.tcl` in place to instantiate the E2E IP and new ports.
- Modify existing PYNQ driver in place.

Expected result:

- Single flow name, fewer scripts.

Cost/runtime:

- Medium.

Risk:

- High regression risk for existing `hgtxr_top` overlay flow.
- Harder to compare old and E2E board paths.

Validation evidence:

- Existing static validation still passes.
- Old overlay assumptions either intentionally retired or separately preserved.
- E2E bit/hwh generated and tested.

Choose when:

- User explicitly wants to replace the old overlay flow rather than keep both.

## Recommended Sub-Choice

Default recommendation: A1 if preserving the validated E2E AXIS contract is more important; A2 if fastest board smoke using existing infrastructure is more important.

Main agent recommendation:

- Ask user to choose A1 or A2 before editing implementation-flow scripts.
- Do not run Vivado implementation until E2E packaging/BD path is selected and static-validated.

## Likely Commands After Sub-Choice

Current HLS verification command already proven:

```bash
HGTXR_E2E_SCALE=active196_b6_ff768 \
HGTXR_E2E_RESOURCE_POLICY=dsp_mixed_stream \
LD_LIBRARY_PATH=/tmp:/usr/lib/x86_64-linux-gnu:/usr/lib/gcc/x86_64-linux-gnu/13:/lib/x86_64-linux-gnu \
/tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f vivado/scripts/run_e2e_q4w8a_csim.tcl
```

Current HLS CSynth command already proven:

```bash
HGTXR_E2E_SCALE=active196_b6_ff768 \
HGTXR_E2E_RESOURCE_POLICY=dsp_mixed_stream \
LD_LIBRARY_PATH=/tmp:/usr/lib/x86_64-linux-gnu:/usr/lib/gcc/x86_64-linux-gnu/13:/lib/x86_64-linux-gnu \
/tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f vivado/scripts/run_e2e_q4w8a_csynth.tcl
```

Expected future package command after adding E2E package Tcl:

```bash
HGTXR_E2E_SCALE=active196_b6_ff768 \
HGTXR_E2E_RESOURCE_POLICY=dsp_mixed_stream \
/tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f vivado/scripts/package_e2e_q4w8a_ip.tcl
```

Expected future implementation command after adding E2E Vivado Tcl:

```bash
/tools/Xilinx/Vivado/2023.2/bin/vivado -mode batch \
  -source vivado/scripts/build_e2e_bitstream.tcl \
  -tclargs -jobs 4
```

These future commands should not be run until the chosen E2E board path exists.
