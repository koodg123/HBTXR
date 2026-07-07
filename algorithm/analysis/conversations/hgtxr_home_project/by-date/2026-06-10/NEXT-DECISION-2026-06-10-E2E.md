# Next Decision - After A2 E2E m_axi Bitstream

Date: 2026-06-10 KST

Scope: active `[3차목표]` HGTXR ZCU104 E2E Q4W/Q8A hardware baseline.

## Current State

- User selection in force: Path 1 = A2 then A1, Path 2 = C in parallel where feasible, E pending.
- A2 HLS wrapper: complete.
- A2 HLS CSim: passed with `[32, -13, 26, -6, 14, -11]`, `runtime_state=2`.
- A2 HLS CSynth after A1 safe LUTRAM: `3.744 ns`, `81,514,836 cycles`, `326 BRAM_18K`, `334 DSP`, `46,502 FF`, `84,220 LUT`, `64 URAM`.
- A2 Vivado bitstream: complete.
- A2 routed timing: `WNS=1.926 ns`, `TNS=0.000 ns`, `WHS=0.010 ns`, `THS=0.000 ns`.
- A2 placed resources: `149 Block RAM Tile`, `148 RAMB36`, `2 RAMB18`, `64 URAM`, `246 DSP`.
- A2 routed power estimate: `4.060 W`.
- A2 PYNQ package: `hardware/pynq/hgtxr/hgtxr_e2e_m_axi.bit` and `hardware/pynq/hgtxr/hgtxr_e2e_m_axi.hwh`.
- Board-ready preflight: `ok=36`, `warn=6`, `fail=0`.

## User Decision Rule

When multiple credible directions exist, do not choose silently. Present options with:

- Work plan
- Expected result
- Cost/runtime
- Resource/timing risk
- Validation evidence required
- Recommendation and reason

Then wait for user selection before large implementation, long board build, broad default-path promotion, or next major HLS sweep.

## Option R - Run A2 PYNQ Runtime Smoke

Work plan:

- Use packaged `hgtxr_e2e_m_axi.bit/.hwh`.
- Load `HgtxrE2EMaxiOverlay` on ZCU104/PYNQ.
- Allocate frame, optional packed Q4 weight buffer, output state, and runtime state.
- Run deterministic smoke and compare against CSim-level expectation.
- Capture board log, output vector, runtime state, timeout/no-timeout, and overlay load result.

Expected result:

- Proves board-side control, address registers, DDR mapping, and memory-mapped E2E execution.
- Converts A2 from bitstream-ready to board-runtime-ready.

Cost/runtime:

- Medium if board access is ready.
- High if PYNQ environment, device nodes, permissions, or overlay loading fail.

Risks:

- Needs physical ZCU104/PYNQ access.
- Current default `weights_u32=None` uses zero weights and should produce live-weight bit `0` / `runtime_state=1`; CSim full expected `[32, -13, 26, -6, 14, -11]` used initialized test weights, so runtime test must either pass matching packed weights or use a zero-weight board golden.
- HWH register naming must match PYNQ register map; parser now supports Vivado attribute-style HWH but board proof still pending.

Validation required:

- Overlay loads with `hgtxr_e2e_m_axi.bit/.hwh`.
- IP found by name or fallback discovery.
- AP_DONE observed before timeout.
- Output vector and `runtime_state` recorded.
- No kernel crash, DMA issue, or MMIO register error.

Choose when:

- Fastest closure of selected A2 path is priority.

Recommendation:

- Strong next step if board access exists. It verifies the artifact already built.

## Option A1 - Build E2E AXIS + AXI DMA Board Flow

Work plan:

- Preserve `hgtxr_e2e_axis_top` as public stream interface.
- Add separate AXIS/DMA HLS package Tcl and Vivado BD Tcl.
- Instantiate AXI DMA simple mode for `axis_in` and `axis_out`.
- Connect AXI-Lite control and DDR paths.
- Add PYNQ stream helper/test.

Expected result:

- Clean architecture matching the existing AXI-Stream top.
- Gives direct board proof for the long-term stream interface, separate from A2 m_axi smoke.

Cost/runtime:

- Medium implementation.
- High Vivado runtime.

Risks:

- Full input transfer is one 256-bit beat per pixel: `65536 * 32 = 2,097,152` bytes.
- `num_pixels` currently ignored, so short DMA transfer can hang.
- DMA/PYNQ integration surface is larger than A2 memory-mapped runtime.

Validation required:

- E2E AXIS IP `component.xml`.
- `validate_bd_design` pass.
- Routed timing/utilization.
- `.bit/.hwh` package.
- PYNQ stream output and `last` semantics match CSim.

Choose when:

- Interface purity and long-term DMA architecture matter more than immediate A2 board smoke.

Recommendation:

- Do after Option R unless user explicitly wants stream architecture next.

## Option C2 - Continue Parallelism Track

Work plan:

- Treat PAR16 result as current best HLS latency/DSP point.
- Decide whether to run PAR32 stress CSynth or move PAR16 into implementation.
- Keep legal lane factors only: `16` or `32`; avoid `12/24` because packed 64-lane fast path is not divisible.
- Record resource/timing/latency table beside A2/PAR8.

Expected result:

- PAR16 already moved DSP `332 -> 604` and latency `71,316,968 -> 37,508,072` cycles.
- PAR32 may raise DSP further and lower latency if memory/banking/timing hold.

Cost/runtime:

- Medium-to-high for CSynth.
- High if implementing PAR16/PAR32 bitstream.

Risks:

- PAR16 LUT already `127,916`; PAR32 may push LUT/timing sharply.
- Higher HLS parallelism may not route as cleanly as A2 PAR8 m_axi.
- Board implementation resource can differ from HLS estimates.

Validation required:

- Reduced CSim for selected PAR.
- Full active196/b6 CSynth.
- Report table: clock, latency, BRAM/DSP/FF/LUT/URAM, II bottlenecks.
- If promoted to board: Vivado routed timing and bit/hwh.

Choose when:

- User prioritizes DSP utilization and low latency over immediate board runtime proof.

Recommendation:

- PAR16 is useful; PAR32 is stress-only until PAR16 board/timing risk is understood.

## Option B2 - DSP Pipeline Warning Cleanup

Work plan:

- Analyze Vivado DRC `DPIP-2` and `DPOP-4` warnings from the A2 bitstream.
- Try small HLS-source changes or directives to enable DSP input/MREG/PREG pipelining.
- Focus on hot multiply sites: patch embedding, LayerNorm, QKV, output projection.
- Re-run CSim and CSynth before any bitstream rerun.

Expected result:

- Potential power/Fmax improvement.
- May reduce future timing risk for PAR16/PAR32.
- Current routed DRC report count: `DPIP-2=210`, `DPOP-4=178`.

Cost/runtime:

- Medium. Likely multiple CSynth iterations.

Risks:

- Extra pipeline latency can perturb loop II/control latency.
- Source-level added registers can increase FF or change expected cycle counts.
- Current A2 route already meets timing; this is optimization, not blocker.

Validation required:

- CSim output unchanged.
- CSynth reports resource and latency delta.
- Vivado DRC warning count reduced if bitstream rerun.
- Routed timing still passes.

Choose when:

- User wants clean hardware quality/power/Fmax before more architecture work.

Recommendation:

- Defer until after A2 PYNQ runtime unless board is unavailable.

## Option E - Paper Weight/LUT/Calibration Alignment

Status: pending by user choice.

Work plan:

- Inventory final trained checkpoint, Q4 packed weights, calibration, and HG-PIPE nonlinear LUT provenance.
- Replace synthetic arbitrary weights only when final artifacts are available.
- Regenerate SW/HW golden vectors.

Expected result:

- Moves from synthetic functional baseline to paper-fidelity baseline.

Cost/runtime:

- High and artifact-dependent.

Risks:

- Required final artifacts may still be missing.
- Scale/tolerance contract may need redesign.

Validation required:

- Artifact provenance.
- SW/HW golden regeneration.
- Native comparator, Vitis CSim, CSynth.

Choose when:

- User unpauses E and paper fidelity becomes priority.

Recommendation:

- Keep pending for now.

## Recommendation Snapshot

1. If ZCU104/PYNQ board is available: choose **Option R**.
2. If board is unavailable but architecture work should continue: choose **Option A1**.
3. If latency/DSP is priority: choose **Option C2**, preferably PAR16 implementation before PAR32 stress.
4. If cleanup/quality is priority: choose **Option B2**.
5. Keep **Option E** pending.
