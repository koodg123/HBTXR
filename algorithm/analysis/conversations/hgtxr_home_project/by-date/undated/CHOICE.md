# CHOICE - Decision Process And Current Options

Date: 2026-06-10 KST

Scope: active `[3차목표]` HGTXR ZCU104 E2E Q4W/Q8A cyclic hardware accelerator work.

## Decision Rule

When multiple credible directions exist, the main agent must not silently choose a branch. Before any large implementation, long HLS/Vivado run, board-flow change, or default datapath promotion, present the choices and wait for user selection.

Each choice report must include:

- Work plan.
- Expected result.
- Cost/runtime.
- Resource, timing, or integration risk.
- Validation evidence required.
- Recommendation and reason.

Allowed branch-neutral work before user selection:

- Read-only analysis.
- Documentation and handover updates.
- Static checks.
- Small regression tests for existing tooling.
- Preflight checks that do not commit to A1/A2/A3/B/C/E.

Disallowed before user selection:

- Long Vivado implementation runs.
- New board-flow script execution.
- Replacing existing overlay/PYNQ flow.
- Promoting opt-in HG-PIPE integer math into default E2E path.
- Sweeping higher parallelism as the next main direction.

## Current Selected HLS Point

- Top: `hgtxr_e2e_axis_top`
- Scale: `active196_b6_ff768`
- Resource policy: `dsp_mixed_stream`
- Parallelism: `PAR=8`
- Weight/activation: Q4W/Q8A
- QKV cache: `HGTXR_E2E_QKV_WEIGHT_CACHE=1`
- CSim: passed with `[32, -13, 26, -6, 14, -11]`, `failures=0`
- CSynth: `3.744 ns`, `71,316,968 cycles`, `306 BRAM_18K`, `332 DSP`, `43,740 FF`, `81,144 LUT`, `64 URAM`
- Known risk: HLS timing slack near `-0.09 ns`; board implementation/PYNQ runtime not proven.

## Current Execution Selection

- Path 1: run A2 first, then A1.
- Path 2: run C in parallel when tool/runtime contention allows.
- E: pending.

Current execution evidence:

- A2 memory-mapped wrapper added and validated: full CSim output `[32, -13, 26, -6, 14, -11]`, runtime_state `2`; full CSynth `3.744 ns`, `81,514,836 cycles`, `334 BRAM_18K`, `334 DSP`, `46,438 FF`, `84,196 LUT`, `64 URAM`.
- A1 small-memory LUTRAM refinement added for `gb.pooled`; QKV weight-cache URAM default remains off because the all-Q/K/V cache URAM experiment reached `112/96 URAM`.
- C PAR=16 full AXIS CSynth passed: `3.953 ns`, `37,508,072 cycles`, `338 BRAM_18K`, `604 DSP`, `59,507 FF`, `127,916 LUT`, `64 URAM`; reduced PAR=16 CSim passed with `[18, -3, 3, -2, 1, -4]`.
- A2 package/board-flow scaffold now completed: `hgtxr_e2e_m_axi_top` IP export, Vivado block design, implementation, and bitstream generation passed.
- A2 board artifact evidence: routed WNS `1.926 ns`, TNS `0.000 ns`, WHS `0.010 ns`, THS `0.000 ns`; placed utilization `148 RAMB36`, `2 RAMB18`, `64 URAM`, `246 DSP`; `hgtxr_e2e_m_axi.bit/.hwh` packaged under `hardware/pynq/hgtxr/`.

## Current Gate Status

Neutral preflight:

- `ok=36`
- `warn=6`
- `fail=0`

Board-ready preflight:

- `ok=36`
- `warn=6`
- `fail=0`

Board-ready remaining warnings are non-blocking:

- E2E CSim log is not visible under generated trees, although functional CSim was run and recorded separately.
- Legacy `pynq/hgtxr/hgtxr.hwh` still contains old `hgtxr_top_0` and is ignored for E2E m_axi proof.
- `spec-kit`/`specify`, requested `PAPER_PRJXR` image path, and requested `XR-VITs` sibling are absent in this workspace.

## Option A1 - E2E AXIS + AXI DMA Board Flow

Work plan:

- Preserve `hgtxr_e2e_axis_top` as the public E2E interface.
- Add E2E-specific HLS package Tcl.
- Add E2E-specific Vivado BD/bitstream Tcl.
- Instantiate AXI DMA simple mode for `axis_in` and `axis_out`.
- Connect weight/runtime AXI-MM ports and AXI-Lite control.
- Add PYNQ stream driver/test.

Expected result:

- Cleanest architecture match to the currently validated AXIS top.
- Produces a board path that directly validates the public E2E stream interface.

Cost/runtime:

- Medium implementation work.
- High Vivado runtime.

Risks:

- DMA packing must send one 256-bit beat per pixel.
- Full input transfer length is about `2,097,152` bytes.
- `num_pixels` is currently ignored, so short DMA transfers can hang.
- PYNQ stream integration surface is larger than memory-mapped smoke.

Validation required:

- E2E IP `component.xml`.
- `validate_bd_design` pass.
- Routed timing and utilization reports.
- `.bit`/`.hwh` package.
- PYNQ stream output `[32, -13, 26, -6, 14, -11]`.

Choose when:

- Interface purity and long-term architecture are more important than fastest smoke test.

## Option A2 - E2E Memory-Mapped Wrapper

Work plan:

- Add `hgtxr_e2e_m_axi_top(frame, weights, out_state, runtime_state)`.
- Reuse E2E internals, not nested AXIS top calls.
- Keep original AXIS top for regression.
- Clone/adapt existing memory-mapped build/PYNQ patterns.
- Run fresh wrapper CSim and CSynth.

Expected result:

- Fastest path to ZCU104 board smoke.
- Reuses more existing old-flow infrastructure while avoiding in-place replacement.

Cost/runtime:

- Medium HLS/code work.
- Medium-to-high Vivado runtime after wrapper passes.

Risks:

- Wrapper resource/timing may differ from AXIS CSynth.
- Existing `-0.09 ns` timing risk remains.
- Must prevent drift from validated AXIS behavior.

Validation required:

- Wrapper CSim output `[32, -13, 26, -6, 14, -11]`.
- Wrapper CSynth under ZCU104 budget.
- E2E MM Vivado package/BD pass.
- PYNQ memory-mapped test pass.

Choose when:

- Fastest credible board smoke is the priority.

## Option A3 - Replace Existing Board Flow In Place

Work plan:

- Modify current `build_bitstream.tcl` and PYNQ driver directly for E2E.

Expected result:

- Single board-flow path.

Cost/runtime:

- Medium.

Risks:

- Highest regression risk.
- Old `hgtxr_top` overlay assumptions become harder to preserve or compare.

Validation required:

- Static validation.
- E2E bit/hwh package.
- Clear decision that old overlay flow is retired or separately preserved.

Choose when:

- User explicitly wants to replace old overlay flow.

## Option B - HLS Timing Pre-Tune

Work plan:

- Inspect current HLS timing offenders.
- Try small source/pragmas only.
- Re-run CSim and CSynth, not Vivado implementation.

Expected result:

- Better HLS timing margin before expensive implementation.

Cost/runtime:

- Medium, likely multiple CSynth iterations.

Risks:

- HLS timing may not correlate perfectly with implementation timing.
- Banking/pipeline changes can increase BRAM/LUT.

Validation required:

- CSim pass.
- CSynth timing/resource comparison.
- QKV remains `II=1`.
- ZCU104 resource fit preserved.

Choose when:

- Reducing timing risk before board implementation is priority.

## Option C - Higher Parallelism Sweep

Work plan:

- Sweep legal packed-lane candidates `PAR=16` first, then `PAR=32` only as a stress point.
- Keep QKV cache, selective URAM, and small LUTRAM.
- Compare full active196/b6 CSynth resource and latency.

Expected result:

- Higher DSP use.
- Lower latency if memory/cache ports and timing hold.

Cost/runtime:

- Medium-to-high.

Risks:

- Timing can degrade.
- BRAM/LUT can rise.
- Diminishing returns if non-parallel sections dominate.
- Avoid `PAR=12/24` because 64 packed weight lanes are not divisible by those factors and the aligned fast path can break.

Validation required:

- CSim for selected points.
- CSynth table with latency, II, BRAM/DSP/FF/LUT/URAM, timing.

Choose when:

- DSP utilization and lower HLS latency are priority over immediate board proof.

Current PAR=16 result:

- Full AXIS CSynth passed with `HGTXR_E2E_PAR=16`.
- DSP increased `332 -> 604`.
- Latency improved `71,316,968 -> 37,508,072` cycles.
- LUT increased `81,144 -> 127,916`.
- URAM stayed `64/96`.
- Resource judgment: fit, useful latency/DSP tradeoff, but LUT headroom is much tighter (`55%`).

## Option E - Paper Weight/LUT/Calibration Alignment

Work plan:

- Replace deterministic arbitrary Q4 weights with final paper-trained weights when available.
- Import/generate final calibration and nonlinear LUTs.
- Regenerate SW golden vectors and HLS headers.
- Re-run reduced, staged, full CSim/CSynth gates.

Expected result:

- Moves baseline toward paper reproduction rather than synthetic functional proof.

Cost/runtime:

- High.

Risks:

- Required final artifacts may be missing.
- Numerical tolerance and scale contract may need redefinition.
- Resource/timing can shift.

Validation required:

- Artifact provenance.
- SW/HW golden regeneration.
- Native comparator, CSim, CSynth.
- Per-layer numerical comparison.

Choose when:

- Paper-fidelity is priority over board smoke.

## Recommendation Snapshot

- A2 implementation artifact proof: complete.
- Fastest remaining board proof: choose A2 PYNQ runtime smoke.
- Cleanest long-term interface: choose A1 AXIS/DMA board flow.
- Push DSP use/latency further: choose C/PAR16 implementation or PAR32 stress.
- Reduce DSP DRC warning/power/Fmax risk: choose B2 DSP pipeline cleanup.
- Move toward paper reproduction: choose E when unpaused.

Current main-agent recommendation: run A2 PYNQ runtime smoke next if ZCU104 board is available. If board is unavailable, proceed with A1 read/write implementation planning or C PAR16/PAR32 HLS exploration only after user choice.

## Next Decision After A2 Bitstream

Current next choices:

- R: A2 PYNQ runtime smoke using `hgtxr_e2e_m_axi.bit/.hwh`.
- A1: E2E AXIS + AXI DMA board flow.
- C2: continue higher parallelism path, either PAR16 board implementation or PAR32 HLS stress.
- B2: DSP pipeline warning cleanup for `DPIP-2`/`DPOP-4`.
- E: still pending.

See `docs/track/NEXT-DECISION-2026-06-10-E2E.md` for work plan, expected result, cost, risks, validation evidence, and recommendation for each option.

Sub-agent follow-up:

- GPT5.3-Codex-Spark A1 sidecar failed due usage limit until 2026-06-15 23:18.
- GPT5.5 A1 sidecar recommended A2 PYNQ runtime smoke before long A1 AXIS/DMA implementation.
- GPT5.5 C/DSP sidecar recommended A2 PYNQ runtime smoke first, then DSP `DPIP-2`/`DPOP-4` cleanup, then PAR32 stress only after user choice.

## Current User Selection

Status: selected on 2026-06-10 KST.

Selected execution paths:

- Path 1: run A2 first, then A1.
- Path 2: run C in parallel with Path 1 where tool/runtime contention allows.
- E: pending for now.

Execution policy:

- A2 is the first critical-path board-smoke implementation.
- A1 starts after A2 establishes the memory-mapped E2E wrapper baseline, unless A1 read-only preparation is useful and does not touch A2-owned files.
- C can proceed in parallel as an HLS sweep/report track, but long CSynth jobs should be scheduled so they do not starve A2 validation.
- E remains documentation/artifact-inventory only; no default paper-weight/LUT/calibration promotion until the user unpauses it.

Primary file ownership for next work:

- A2 owns new memory-mapped wrapper files, cloned E2E MM package/build scripts, and cloned PYNQ MM smoke support.
- A1 owns new AXIS/DMA package/build scripts and PYNQ stream support after A2 baseline.
- C owns sweep configs, generated sweep reports, and resource comparison documents.
- E owns only pending notes and artifact inventory while paused.

## A1 Progress After A2 Baseline

Status: scaffold implemented on 2026-06-10 KST.

- Added A1 package/build scripts without overwriting A2:
  - `hardware/vivado/scripts/package_e2e_axis_ip.tcl`
  - `hardware/vivado/scripts/build_e2e_axis_dma_bitstream.tcl`
- A1 generated HLS/IP path is `hardware/generated/hgtxr_e2e_axis_hls`.
- A1 board output names are `hgtxr_e2e_axis_dma.bit/.hwh`.
- Added PYNQ AXIS/DMA helper and smoke CLI:
  - `hardware/pynq/hgtxr/e2e_axis_dma_overlay.py`
  - `hardware/pynq/hgtxr/run_e2e_axis_dma_smoke.py`
- Off-board fake-DMA validation passes through the shared PYNQ unittest suite.

Remaining A1 choice:

- A1-build: run A1 HLS IP export and Vivado AXIS/DMA bitstream build.
- A1-board: run physical PYNQ DMA smoke after A1 artifacts exist.
- Hold A1 and continue C if higher DSP/latency exploration is preferred.

## A1 Build Choice Result

Status: A1-build completed on 2026-06-10 KST.

Decision taken:

- Proceeded with A1 HLS IP export and AXIS/DMA Vivado bitstream build after A2 board artifact baseline.
- Kept A2 m_axi artifacts intact and wrote A1 artifacts under separate names: `hgtxr_e2e_axis_dma.bit/.hwh`.
- E remains pending.

Result:

- A1 IP export succeeded: `hardware/generated/hgtxr_e2e_axis_hls/solution_e2e_q4w8a/impl/ip/component.xml`.
- A1 HLS core resources: `298 BRAM_18K`, `332 DSP`, `43,804 FF`, `81,168 LUT`, `64 URAM`.
- A1 Vivado build initially found a real DMA width issue: DMA `MM2S` was `32b`, HLS `axis_in` was `256b`.
- Fixed the A1 Vivado script to force 256-bit DMA stream/AXI widths.
- A1 bitstream build then passed: routed `WNS=4.120 ns`, `TNS=0`, `WHS=0.009 ns`, `THS=0`.
- A1 PYNQ artifacts now exist under `hardware/pynq/hgtxr/`.

Remaining choices:

- A1-board: run physical ZCU104 AXIS/DMA smoke.
- C-PAR16-board: package/board-test the already validated PAR16 resource point.
- C-PAR32-HLS: run PAR32 stress to push DSP/latency harder, with expected LUT/timing pressure.
- E: still pending.

## C Path Choice After A1 Build

Status: choices prepared on 2026-06-10 KST; no long PAR32 run started yet.

User-selected context:

- Path 1 A2 -> A1 is now implemented through board artifacts.
- Path 2 C should continue.
- E remains pending.

Implemented safety preparation:

- Added isolated E2E HLS project naming support to `hardware/vivado/scripts/run_e2e_q4w8a_csim.tcl` and `hardware/vivado/scripts/run_e2e_q4w8a_csynth.tcl`.
- `HGTXR_E2E_RUN_TAG=par32` now writes to `hardware/generated/hgtxr_e2e_hls_par32`.
- `HGTXR_E2E_PROJECT_NAME=<name>` can force an exact generated project directory.
- A1 and A2 package wrappers now default to separate project names, while allowing `HGTXR_E2E_PROJECT_NAME` override for isolated C package runs.
- AXIS/DMA Vivado build now accepts `-artifact_name` so PAR16 board attempts can produce separate PYNQ/overlay artifacts without overwriting `hgtxr_e2e_axis_dma.bit/.hwh`.

Available C choices:

| Choice | Work | Expected result | Risk |
|---|---|---|---|
| C1: PAR16 board/package | Reuse validated `HGTXR_E2E_PAR=16` HLS point, package IP, then try Vivado implementation or board smoke path. | Current best measured DSP/latency HLS point: `604 DSP`, `37,508,072 cycles`, `64 URAM`. If routed, strongest board candidate for higher DSP utilization. | LUT already high at `127,916`; Vivado route/timing may fail or require floorplanning/cleanup. |
| C2: PAR32 HLS stress | Run isolated CSynth with `HGTXR_E2E_PAR=32 HGTXR_E2E_RUN_TAG=par32`. | Higher DSP pressure and possibly lower latency than PAR16 if memory banking and scheduling hold. Produces report without overwriting PAR16/baseline evidence. | Expected LUT/timing pressure is high; HLS may fail, overuse LUT/BRAM, or produce a non-board-fit point. |
| C3: DSP/LUT cleanup first | Inspect and reduce known DSP pipeline warnings and LUT-heavy control/array pressure before increasing PAR. | Lower routing risk for C1/C2; may preserve DSP while reducing LUT or improving timing margin. | Slower path to a headline DSP increase; gains are less predictable than a direct PAR32 sweep. |

Main-agent recommendation:

- If the priority is board-feasible high-DSP proof: choose C1 first.
- If the priority is maximum DSP/latency exploration: choose C2 now, using the isolated `par32` output directory.
- If the priority is reducing implementation risk before long runs: choose C3 before C2.

Next command candidates:

- C1 HLS/IP package: `HGTXR_E2E_PAR=16 HGTXR_E2E_RESOURCE_POLICY=dsp_mixed_stream HGTXR_E2E_PROJECT_NAME=hgtxr_e2e_axis_par16_hls /tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f hardware/vivado/scripts/package_e2e_axis_ip.tcl`
- C1 isolated Vivado build: `/tools/Xilinx/Vivado/2023.2/bin/vivado -mode batch -source hardware/vivado/scripts/build_e2e_axis_dma_bitstream.tcl -tclargs -project_name hgtxr_e2e_axis_dma_par16_overlay -bd_name hgtxr_e2e_axis_dma_par16_system -hls_ip_repo hardware/generated/hgtxr_e2e_axis_par16_hls/solution_e2e_q4w8a/impl/ip -artifact_name hgtxr_e2e_axis_dma_par16`
- C2 stress: `HGTXR_E2E_PAR=32 HGTXR_E2E_RESOURCE_POLICY=dsp_mixed_stream HGTXR_E2E_RUN_TAG=par32 /tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f hardware/vivado/scripts/run_e2e_q4w8a_csynth.tcl`

## C1 PAR16 Board Choice Result

Status: C1 PAR16 package/build completed on 2026-06-10 KST.

Decision taken:

- Proceeded with C1 first because it was the strongest already-measured higher-DSP point and had lower risk than an immediate PAR32 stress run.
- Kept A1 `hgtxr_e2e_axis_dma.bit/.hwh` intact by using isolated project, block-design, overlay, and artifact names.
- E remains pending.

Result:

- C1 HLS/IP export succeeded at `hardware/generated/hgtxr_e2e_axis_par16_hls/solution_e2e_q4w8a/impl/ip/component.xml`.
- C1 HLS core estimate: `3.953 ns`, `37,508,072 cycles`, `338 BRAM_18K`, `604 DSP`, `59,507 FF`, `127,916 LUT`, `64 URAM`.
- C1 isolated Vivado build routed successfully: `WNS=4.723 ns`, `TNS=0`, `WHS=0.010 ns`, `THS=0`.
- C1 routed power estimate: `3.475 W`.
- C1 PYNQ artifacts now exist as `hardware/pynq/hgtxr/hgtxr_e2e_axis_dma_par16.bit/.hwh`.
- C1 artifact SHA256: bit `d34cc3cb108c508153a3f69531b80be81567343f38ceb52e517e578556e42e11`, hwh `6d32cafffc6b9c9b33793bed34f114c199450e1167895ca4d448c56012c471a8`.

Interpretation:

- DSP utilization materially improved versus A1/A2: `604 DSP` versus about `332-334 DSP`.
- URAM remains at `64`, preserving the intended large-buffer mapping.
- LUT is high at `127,916`, so C1 is a board-routed candidate but not yet a comfortable resource point.
- Vivado wrapper utilization is not the authoritative DSP/URAM source for this OOC/IP flow; use the HLS C1 report for HGTXR core accounting.

Remaining choices:

- C1-board: run physical ZCU104 AXIS/DMA PAR16 smoke.
- C2: run isolated PAR32 HLS stress if maximum DSP/latency pressure is worth high LUT/timing risk.
- C3: do DSP/LUT cleanup first if the priority is reducing C1 LUT pressure before PAR32.
- E: still pending.

## C3 DSP/LUT Cleanup Choice Result

Status: C3 cleanup implemented and full PAR16/MEM8 CSynth completed on 2026-06-10 KST.

Decision taken:

- Proceeded with C3 after C1 because C1 already proved a routed high-DSP candidate, but its HLS core LUT pressure was high at `127,916`.
- Deferred C2/PAR32 because pushing parallelism again before lowering LUT pressure would likely increase timing/routing risk.
- Kept E pending per user decision.

Implemented cleanup:

- Added `HGTXR_E2E_MEM_BANK_PAR` so compute parallelism and memory-bank partition factor can be controlled separately.
- Default remains unchanged: if `HGTXR_E2E_MEM_BANK_PAR` is not set, it follows `HGTXR_E2E_DENSE_PAR`.
- Added an aligned packed-weight vector fast path under `HGTXR_E2E_WEIGHT_VEC_ALIGNED_FASTPATH` to remove the boundary `word1`/mux path when lane alignment is guaranteed.
- Extended CSim/CSynth Tcl flows to accept and validate `HGTXR_E2E_MEM_BANK_PAR=1|2|4|8|16|32`.

Measured C3 result:

| Point | Latency cycles | BRAM_18K | DSP | FF | LUT | URAM | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| C1 PAR16 baseline | `37,508,072` | `338` | `604` | `59,507` | `127,916` | `64` | Fast high-DSP routed candidate, high LUT pressure |
| C3 PAR16 MEM8 | `48,598,922` | `292` | `604` | `56,966` | `113,124` | `64` | Same DSP/URAM with lower LUT/BRAM/FF, slower due memory-port pressure |

Result:

- LUT decreased by `14,792` versus C1, about `11.6%`.
- BRAM decreased by `46`; FF decreased by `2,541`.
- DSP stayed at `604`; URAM stayed at `64`.
- Timing estimate improved from C1 `-0.30 ns` slack to C3 `-0.05 ns` slack in HLS estimate.
- Latency increased by `11,090,850` cycles, about `29.6%`, because the MLP W2 loop now reports `Final II=2` under the reduced memory-bank factor.

Interpretation:

- C3 MEM8 is a resource-pressure reduction candidate, not the low-latency default.
- It is useful if route/LUT pressure is the priority.
- C1 remains the better low-latency high-DSP board candidate until physical smoke or further cleanup proves otherwise.
- A follow-up C3b run with PAR16/MEM16 plus aligned fast path only can isolate the fast-path benefit without the MEM8 latency penalty.

Remaining choices:

- C1-board: run physical ZCU104 AXIS/DMA PAR16 smoke.
- C3b: run PAR16/MEM16 fast-path-only CSynth to isolate LUT impact without memory-port reduction.
- C2: run isolated PAR32 stress after accepting high LUT/timing risk.
- E: still pending.

## C3b MEM16 Fastpath-Only Choice Result

Status: C3b PAR16/MEM16 fastpath-only CSynth completed on 2026-06-10 KST.

Decision taken:

- Ran C3b after C3/MEM8 to isolate whether the aligned packed-weight fastpath or the reduced memory-bank factor caused the C3 LUT/latency changes.
- Kept compute parallelism at `PAR16`.
- Restored memory partitioning to `MEM_BANK_PAR=16`, matching C1.
- Kept E pending per user decision.

Measured result:

| Point | Latency cycles | BRAM_18K | DSP | FF | LUT | URAM | Key readout |
|---|---:|---:|---:|---:|---:|---:|---|
| C1 PAR16 baseline | `37,508,072` | `338` | `604` | `59,507` | `127,916` | `64` | Fast high-DSP routed candidate |
| C3 PAR16/MEM8 | `48,598,922` | `292` | `604` | `56,966` | `113,124` | `64` | Lower LUT, slower due MLP W2 `II=2` |
| C3b PAR16/MEM16 | `37,508,072` | `332` | `604` | `59,505` | `126,506` | `64` | Latency returns to C1; small LUT reduction only |

Interpretation:

- C3b removes the C3/MEM8 MLP W2 latency penalty: W2 loop returns to `II=1`.
- Fastpath-only LUT benefit is small: `1,410` LUT less than C1, about `1.1%`.
- Most C3/MEM8 LUT reduction came from reducing memory banking, not from the aligned fastpath.
- C3b is the safer low-latency high-DSP C candidate if we want C1-like speed with a small LUT improvement.
- C3/MEM8 is still useful as a resource-pressure candidate when LUT pressure matters more than latency.

Remaining choices:

- C1/C3b board path: package or rebuild board artifact from the preferred PAR16/MEM16 candidate if low latency is priority.
- C3c: try intermediate banking only if a supported factor is added; current Tcl supports `1|2|4|8|16|32`, so `12` would require new support and divisibility review.
- C2: run PAR32 stress if maximum DSP pressure is worth high LUT/timing risk.
- E: still pending.

## C3b Board Candidate Choice Result

Status: C3b PAR16/MEM16 AXIS/DMA board candidate completed on 2026-06-10 KST.

Decision taken:

- Chose the C1/C3b board path from the remaining C choices because C3b is the low-latency high-DSP candidate.
- Kept C3/MEM8 as a resource-pressure candidate only.
- Kept C2 PAR32 stress and E pending.
- Used isolated names to avoid overwriting A1/C1 artifacts:
  `hgtxr_e2e_axis_par16_c3b_mem16_hls`,
  `hgtxr_e2e_axis_dma_c3b_mem16_overlay`,
  `hgtxr_e2e_axis_dma_c3b_mem16_system`,
  and `hgtxr_e2e_axis_dma_c3b_mem16`.

Measured result:

| Point | HLS cycles | HLS BRAM_18K | HLS DSP | HLS FF | HLS LUT | HLS URAM | Routed WNS | Power | Readout |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| C3b PAR16/MEM16 board candidate | `37,508,072` | `332` | `604` | `59,505` | `126,506` | `64` | `4.415 ns` | `3.476 W` | Low-latency high-DSP board-loadable candidate |

Artifact hashes:

- IP export zip: `a471fed1d43d2323efd22cc3581cfd7ee1c1cb5a0cfbc69ba57b2877da527e38`.
- Bitstream: `989e77836341467da05d0ccb86f0b4764d38f4711af8fbcb567290b78fe3076d`.
- HWH: `8ca659da3bbcc061f7299ac18314c00b8ee2112a274990d969635b3f6acf3c90`.

Remaining choices:

- C3b-board-smoke: run physical ZCU104 AXIS/DMA smoke.
- C2: run PAR32 HLS stress if maximum DSP pressure is worth higher LUT/timing risk.
- C3c: add and validate an intermediate memory-bank factor only if we want another latency/LUT tradeoff point.
- E: still pending.

## C3b Board-Smoke Bundle Choice Result

Status: C3b PAR16/MEM16 AXIS/DMA PYNQ transfer bundle completed on 2026-06-10 KST.

Decision taken:

- Continued the C3b board path far enough to make physical ZCU104 smoke easy to run.
- Added `--variant c3b-mem16` to the AXIS/DMA smoke CLI instead of duplicating a second board-smoke script.
- Packaged C3b bit/HWH, Python helpers, and the exported packed Q4 weight binary into a self-contained transfer bundle.
- Kept C2 PAR32 stress and E pending.

Measured result:

- Bundle directory: `hardware/generated/pynq/e2e_axis_dma_c3b_mem16_smoke_bundle`.
- Bundle tarball: `hardware/generated/pynq/e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz`.
- Bundle SHA256: `3de505a6627409b4717ea95e5e76d74f0667c61b601fd96db1e495319adda712`.
- Bundle size: `592,249` bytes.
- Board command expects output raw `[32, -13, 26, -6, 14, -11]` and `runtime_state=2`.

Remaining choices:

- C3b-board-smoke: copy/extract the bundle on ZCU104 PYNQ and run `./run_e2e_axis_dma_c3b_mem16_file_smoke.sh`.
- C2: run PAR32 HLS stress only if a larger DSP/LUT/timing risk is acceptable.
- C3c: add and validate another memory-bank factor only if another LUT/latency tradeoff point is needed.
- E: still pending.

## C3b Bundle Preflight Gate Result

Status: C3b bundle readiness is now enforced by board-ready preflight on 2026-06-10 KST.

Decision taken:

- Did not start C2/PAR32 or C3c without another explicit branch choice.
- Strengthened the selected C3b board-smoke path by making the bundle a checked artifact, not only a documented artifact.
- `board-ready` now fails if C3b bit/HWH exist but the matching PYNQ smoke bundle is missing or inconsistent.

Measured result:

- Board-ready preflight now checks C3b bundle directory, manifest, tarball, run script, variant, command, expected output, required contents, and tar SHA256.
- Current board-ready preflight: `ok=54`, `warn=6`, `fail=0`.
- Durable JSON: `docs/resources/third_goal_preflight_board_c3b_bundle_2026_06_10.json`.

Remaining choices:

- C3b-board-smoke: run physical ZCU104 smoke from the checked bundle.
- C2: run PAR32 HLS stress if maximum DSP/latency exploration is selected.
- C3c: add another memory-bank factor only if a new latency/LUT tradeoff point is selected.
- E: still pending.

## C3b Physical Smoke Result Gate

Status: host-side physical-result validator and preflight hook completed on 2026-06-10 KST.

Decision taken:

- Did not open C2/PAR32 or C3c because those remain separate choices.
- Added a result validator so the next physical ZCU104 smoke run can be checked against the same expected CSim-mirrored output.
- Board-ready preflight now warns when no copied-back C3b physical result JSON exists and fails if a copied-back result exists but is wrong.

Expected result:

- When ZCU104 run passes, copy `e2e_axis_dma_c3b_mem16_file_smoke.json` back to `hardware/pynq/hgtxr/` or the generated bundle directory.
- Run `python3 tools/validate_pynq_smoke_result.py <json> --preset axis-c3b-mem16`.
- Board-ready preflight should then replace the current physical-smoke warning with an `ok` result.

Current measured result:

- Validator unit tests pass.
- Current board-ready preflight: `ok=54`, `warn=7`, `fail=0`.
- New warning is expected: C3b physical smoke result has not been captured yet.
- Durable JSON: `docs/resources/third_goal_preflight_board_c3b_result_gate_2026_06_10.json`.

Remaining choices:

- C3b-board-smoke: run physical ZCU104 smoke and copy the JSON result back for validation.
- C2: run PAR32 HLS stress if selected.
- C3c: add another memory-bank factor only if selected.
- E: still pending.

## C3b Self-Validating Bundle Result

Status: C3b bundle now includes local result validation on 2026-06-10 KST.

Decision taken:

- Kept the selected C3b board-smoke path active.
- Repacked the C3b bundle with `tools/validate_pynq_smoke_result.py`.
- Added `validate_e2e_axis_dma_c3b_mem16_file_smoke.sh` so the result JSON can be checked directly inside the extracted bundle.
- Kept C2/PAR32, C3c, and E pending.

Measured result:

- New bundle SHA256: `3de505a6627409b4717ea95e5e76d74f0667c61b601fd96db1e495319adda712`.
- New bundle size: `593,198` bytes.
- Bundle manifest now includes `validation_command`.
- Bundle contents now include `tools/validate_pynq_smoke_result.py` and `validate_e2e_axis_dma_c3b_mem16_file_smoke.sh`.

Remaining choices:

- C3b-board-smoke: run smoke script, then validation script, on ZCU104 PYNQ.
- C2: run PAR32 HLS stress if selected.
- C3c: add another memory-bank factor only if selected.
- E: still pending.

## C3b Transfer Package Precheck

Status: C3b bundle tar/package integrity is now checked before board transfer on 2026-06-10 KST.

Decision taken:

- Continued the selected C3b board-smoke preparation path.
- Added a host-side package validator instead of opening a new C2/PAR32 or C3c branch.
- Made board-ready preflight verify not only manifest metadata, but also actual bundle files and tar members.

Expected result:

- Missing validator script, stale manifest SHA, broken tarball, or tar/member mismatch should fail before the bundle is copied to ZCU104.
- The physical board smoke remains a separate pending step; this check only proves transfer package integrity.

Measured result:

- New validator: `hardware/tools/validate_pynq_bundle_package.py`.
- New tests: `hardware/tests/test_validate_pynq_bundle_package.py`.
- Current board-ready preflight: `ok=57`, `warn=7`, `fail=0`.
- Durable JSON: `docs/resources/third_goal_preflight_board_c3b_result_gate_2026_06_10.json`.

Remaining choices:

- C3b-board-smoke: run smoke script, then validation script, on ZCU104 PYNQ.
- C2: run PAR32 HLS stress only if selected.
- C3c: add another memory-bank factor only if selected.
- E: still pending.

## C3b Physical Result Import Precheck

Status: C3b board-result copy-back now has a host-side import gate on 2026-06-10 KST.

Decision taken:

- Continued the selected C3b board-smoke path.
- Added a host-side result importer instead of starting C2/PAR32, C3c, or E.
- The importer validates the board-produced JSON before copying it into the canonical preflight path.

Expected result:

- A valid C3b smoke JSON can be imported with one command and then board-ready preflight should convert the physical-smoke warning into `ok`.
- An invalid JSON should fail validation and must not overwrite an existing canonical result.

Measured result:

- New importer: `hardware/tools/import_pynq_smoke_result.py`.
- New tests: `hardware/tests/test_import_pynq_smoke_result.py`.
- Canonical C3b copy-back path: `hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json`.

Board-result import command:

```sh
python3 tools/import_pynq_smoke_result.py /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --preset axis-c3b-mem16 --json-out /tmp/hgtxr_c3b_smoke_import.json --validation-out /tmp/hgtxr_c3b_smoke_validation.json
```

Remaining choices:

- C3b-board-smoke: run smoke script and import the copied-back JSON.
- C2: run PAR32 HLS stress only if selected.
- C3c: add another memory-bank factor only if selected.
- E: still pending.

## C3b ZCU104 Smoke Session Runbook

Status: C3b board-smoke session now has generated JSON/Markdown runbooks on 2026-06-10 KST.

Decision taken:

- Continued C3b-board-smoke preparation.
- Added a session generator that validates the bundle first, then emits exact board and host copy-back commands.
- Did not start C2/PAR32, C3c, or E.

Expected result:

- Board executor can use one runbook to transfer, extract, run, validate, copy back, import, and rerun preflight.
- Bundle validation errors are surfaced before board execution.

Measured result:

- New tool: `hardware/tools/prepare_zcu104_smoke_session.py`.
- New tests: `hardware/tests/test_prepare_zcu104_smoke_session.py`.
- Session JSON: `hardware/generated/pynq/e2e_axis_dma_c3b_mem16_smoke_session.json`.
- Session Markdown: `hardware/generated/pynq/e2e_axis_dma_c3b_mem16_smoke_session.md`.
- Session status: `pass`.
- Tarball SHA256: `3de505a6627409b4717ea95e5e76d74f0667c61b601fd96db1e495319adda712`.

Remaining choices:

- C3b-board-smoke: execute the generated session on ZCU104.
- C2: run PAR32 HLS stress only if selected.
- C3c: add another memory-bank factor only if selected.
- E: still pending.

## Final Signoff Gate

Status: final completion now has a strict preflight mode on 2026-06-10 KST.

Decision taken:

- Added `--mode final-signoff` to distinguish "ready for board execution" from "goal can be closed".
- Kept `board-ready` permissive for the still-open physical board result.
- Did not start C2/PAR32, C3c, or E.

Expected result:

- `board-ready` should stay green for the host-side prepared bundle/runbook state.
- `final-signoff` should fail until the physical ZCU104 C3b smoke JSON is imported.
- Missing requested `PAPER_PRJXR` image and `XR-VITs` sibling should block final signoff, but not board-ready.

Measured result:

- `board-ready`: `ok=61`, `warn=7`, `fail=0`.
- `final-signoff`: `ok=61`, `warn=4`, `fail=3`.
- Blocking failures: C3b physical smoke result missing, `PAPER_PRJXR` DeiT image missing, `XR-VITs` sibling missing.

Remaining choices:

- C3b-board-smoke: execute the generated session on ZCU104 and import the JSON.
- C2: run PAR32 HLS stress only if selected.
- C3c: add another memory-bank factor only if selected.
- E: still pending.

## Final Signoff Audit Artifact

Status: final-signoff blocker state is now captured as JSON/Markdown on 2026-06-10 KST.

Decision taken:

- Added an audit writer instead of weakening the final-signoff gate.
- Kept missing `PAPER_PRJXR` and `XR-VITs` as blockers.
- Kept physical C3b smoke as a blocker until real ZCU104 output is imported.

Expected result:

- A next session can read one file and know exactly why the goal is not complete.
- Board executor can use the audit's command list to run and import C3b smoke.

Measured result:

- `hardware/generated/signoff/final_signoff_audit_2026_06_10.json`: `status=blocked`, `blocker_count=3`, `warning_count=4`.
- `hardware/generated/signoff/final_signoff_audit_2026_06_10.md`: blocker summary plus board/import commands.

Remaining choices:

- C3b-board-smoke: execute the generated session on ZCU104 and import the JSON.
- Reference inputs: restore requested `PAPER_PRJXR` and `XR-VITs`, or explicitly approve replacement sources.
- C2: run PAR32 HLS stress only if selected.
- C3c: add another memory-bank factor only if selected.
- E: still pending.

## Reference Input Audit

Status: requested reference inputs were audited on 2026-06-10 KST.

Decision taken:

- Added an audit tool for `PAPER_PRJXR` and `XR-VITs` blockers.
- Did not silently approve `HGPIPE`, `XR_Accel`, `analysis/XR_Accel`, or `ViT_Accel` as replacements.
- Did not create symlinks or fake checkouts outside the hardware workspace.

Expected result:

- The next decision can be made from concrete evidence: restore requested paths, or explicitly approve candidate replacements.
- Final-signoff remains strict until that decision is made.

Measured result:

- Requested `PAPER_PRJXR` image: missing.
- Candidate image: `HGPIPE/DeiT-Tiny C-Syn Results.png`, SHA256 `90495791e6103be30268320d3b17dce2eaa52d003dd6f341393262745c1c3b79`.
- Requested `XR-VITs`: missing.
- Candidate HLS dirs: `XR_Accel`, `analysis/XR_Accel`, `ViT_Accel`.

Remaining choices:

- Restore exact requested paths.
- Approve candidate replacements and update final-signoff policy accordingly.
- Keep strict blocker state until external reference inputs are restored.

## PAPER_PRJXR Image Restore

Status: the requested PAPER_PRJXR image path is restored on 2026-06-10 KST.

Decision taken:

- Used the existing HGPIPE image as the source because it has the exact expected file name.
- Restored the exact requested PAPER_PRJXR path rather than weakening final-signoff.
- Kept `XR-VITs` strict because no exact requested checkout exists.

Measured result:

- Restored image path exists.
- Source/restored SHA256 match: `90495791e6103be30268320d3b17dce2eaa52d003dd6f341393262745c1c3b79`.
- Final-signoff improved from `fail=3` to `fail=2`.

Remaining choices:

- Restore `/home/kjm26/project/PRJXR/XR-VITs`, or explicitly approve one of `XR_Accel`, `analysis/XR_Accel`, `ViT_Accel` as replacement reference.
- Execute/import C3b physical ZCU104 smoke.

## XR-VITs Replacement Candidate

Status: replacement candidates were ranked on 2026-06-10 KST.

Decision taken:

- Added candidate audit instead of silently treating a different folder as `XR-VITs`.
- Recommended `XR_Accel` as the strongest candidate by evidence.
- Kept final-signoff strict: `approved_replacement=false`.

Candidate outcomes:

- `XR_Accel`: score `99`; best match. Has ZCU104 configs, cyclic/DeiT configs, C/C++ HLS files, LayerNorm/GELU/Softmax/Quant sources, Vivado ZCU104 board script, cyclic top files.
- `ViT_Accel`: score `90`; strong generic ViT candidate, but less cyclic/ZCU104-specific for this HGTXR continuation.
- `analysis/XR_Accel`: score `0`; useful as analysis notes, not implementation source.

Remaining choices:

- Approve `XR_Accel` as replacement reference and update final-signoff policy.
- Restore exact `/home/kjm26/project/PRJXR/XR-VITs` checkout.
- Keep blocker until exact path or explicit approval exists.

## C3b Smoke Session Preflight Gate

Status: generated C3b smoke-session JSON/Markdown are now board-ready preflight inputs on 2026-06-10 KST.

Decision taken:

- Continued the C3b-board-smoke preparation path.
- Added a preflight gate for the generated session files.
- Did not start C2/PAR32, C3c, or E.

Expected result:

- Missing or stale session JSON/Markdown should fail `board-ready` before a board run.
- Session tar SHA, expected output, board commands, host import command, and canonical result path are checked from the same preflight command used for final board readiness.

Measured result:

- `hardware/tools/check_third_goal_preflight.py` now validates C3b smoke session JSON/Markdown.
- `hardware/tests/test_check_third_goal_preflight.py` covers complete-session success and missing-session failure.
- Current board-ready preflight: `ok=61`, `warn=7`, `fail=0`.

Remaining choices:

- C3b-board-smoke: execute the generated session on ZCU104.
- C2: run PAR32 HLS stress only if selected.
- C3c: add another memory-bank factor only if selected.
- E: still pending.

## XR-VITs Replacement Policy Gate

Status: replacement-policy support was added on 2026-06-10 KST, but no replacement is approved yet.

Decision taken:

- Kept the exact requested `/home/kjm26/project/PRJXR/XR-VITs` path as the strict final-signoff requirement.
- Added a policy mechanism instead of silently treating `XR_Accel` as `XR-VITs`.
- Added only an inactive template: `docs/resources/xr_vits_replacement_policy.template.json`.
- Did not create active `docs/resources/xr_vits_replacement_policy.json`.

Expected result:

- If the exact `XR-VITs` checkout is restored, final-signoff can pass this reference gate without policy.
- If `XR_Accel` is explicitly approved, copying/filling the template as `xr_vits_replacement_policy.json` can satisfy this gate.
- If neither happens, final-signoff remains blocked.

Measured result:

- `XR_Accel` remains recommended by audit with score `99`.
- Current final-signoff preflight: `ok=62`, `warn=4`, `fail=2`.
- Current blockers: missing C3b physical smoke result; missing `XR-VITs` with no approved replacement policy.

Remaining choices:

- Restore exact `/home/kjm26/project/PRJXR/XR-VITs`.
- Approve `XR_Accel` using active `docs/resources/xr_vits_replacement_policy.json`.
- Keep blocker state until reference policy is explicitly decided.

## XR-VITs Policy Creation Helper

Status: a guarded policy helper was added on 2026-06-10 KST.

Decision taken:

- Added a helper command to create active replacement policy only after explicit approval.
- Kept the current state unapproved: no active `docs/resources/xr_vits_replacement_policy.json` was written.
- Verified `XR_Accel` can pass policy validation in dry-run mode.

Expected result:

- Approval path is now reproducible and less error-prone.
- Accidental replacement approval is prevented by required `--approve`, `--approved-by`, and `--reason`.
- Final-signoff remains strict until active policy exists or exact `XR-VITs` is restored.

Measured result:

- Dry-run policy for `/home/kjm26/project/PRJXR/XR-VIT/XR_Accel` passed.
- Active policy file remains absent.

Remaining choices:

- Explicitly approve `XR_Accel` and run the helper without `--dry-run`.
- Restore exact `/home/kjm26/project/PRJXR/XR-VITs`.
- Keep blocker state.

## Third Goal Completion Audit

Status: a requirement-by-requirement completion audit was added on 2026-06-10 KST.

Decision taken:

- Added a completion audit instead of claiming the active third goal is complete.
- Kept the original objective scope intact.
- Classified each objective item as `pass`, `partial`, or `blocked` from current evidence.

Measured result:

- Overall status: `blocked`.
- Pass: `7`.
- Partial: `4`.
- Blocked: `2`.
- Blocked `(11)`: exact `XR-VITs` is missing and no active replacement policy exists.
- Blocked `final`: final-signoff still fails C3b physical smoke and `XR-VITs` gate.

Remaining choices:

- Approve `XR_Accel` replacement policy or restore exact `XR-VITs`.
- Execute/import C3b physical ZCU104 smoke.
- Keep C2 PAR32 and E pending.

## Third Goal Unblock Checklist

Status: a concrete unblock checklist was added on 2026-06-10 KST.

Decision taken:

- Converted the two remaining final-signoff blockers into three ordered steps.
- Preserved the user choice on the `XR-VITs` gate: exact restore or explicit `XR_Accel` approval.
- Preserved physical-smoke integrity: no synthetic board JSON is generated.

Measured result:

- Checklist status: `pending-unblock`.
- Final blockers: `2`.
- Steps: `3`.

Remaining choices:

- `B1a`: restore exact `/home/kjm26/project/PRJXR/XR-VITs`.
- `B1b`: approve `XR_Accel` as replacement with `create_xr_vits_replacement_policy.py`.
- `B2`: run/import physical ZCU104 C3b smoke.
- `B3`: rerun final host signoff.

## C3b Smoke Transfer Manifest

Status: B2 physical-smoke transfer manifest was added on 2026-06-10 KST.

Decision taken:

- Added a SHA256 transfer manifest for the existing C3b PYNQ smoke bundle.
- Did not fabricate or import a physical smoke result.
- Kept B2 as pending until the bundle is run on actual ZCU104 hardware.

Measured result:

- Transfer manifest status: `pass`.
- Bundle SHA256: `3de505a6627409b4717ea95e5e76d74f0667c61b601fd96db1e495319adda712`.
- Board-side verify command: `sha256sum -c e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz.sha256`.

Remaining choices:

- Transfer bundle + SHA file to ZCU104 and run B2.
- Defer physical board smoke and keep final-signoff blocked.

## C3b Board Smoke Readiness Gate

Status: B2 host-side board-readiness gate was added on 2026-06-10 KST.

Decision taken:

- Added a readiness checker after the transfer manifest so package readiness and physical proof are separated.
- Treat host/package pass plus missing physical result as `ready-for-board`, not as final completion.
- Kept E pending and did not start C2 PAR32/C3c.

Measured result:

- Readiness status: `ready-for-board`.
- Ready: `true`.
- Errors: `0`.
- Warnings: `1`.
- Warning: physical C3b board smoke result is not present yet.
- Bundle SHA256: `3de505a6627409b4717ea95e5e76d74f0667c61b601fd96db1e495319adda712`.

Remaining choices:

- Run/import physical ZCU104 C3b smoke result.
- Resolve `XR-VITs` by exact restore or explicit `XR_Accel` replacement approval.
- Keep final-signoff blocked until both are done.

## Final Signoff Runner Gate

Status: final evidence regeneration runner was added on 2026-06-11 KST.

Decision taken:

- Added one host-side runner instead of manually re-running five tools after each unblock attempt.
- Runner regenerates C3b readiness, final preflight, final signoff audit, completion audit, and unblock checklist.
- Runner mirrors generated evidence into `docs/resources` for tracked handoff.
- Runner does not approve `XR_Accel` and does not fabricate physical smoke JSON.

Measured result:

- Runner status: `blocked`.
- Final preflight: `ok=62`, `warn=4`, `fail=2`.
- Readiness status: `ready-for-board`.
- Remaining blockers: `requested XR-VITs sibling`, `C3b AXIS/DMA physical smoke result`.

Remaining choices:

- Restore exact `/home/kjm26/project/PRJXR/XR-VITs` or explicitly approve `XR_Accel`.
- Run/import C3b ZCU104 physical smoke.
- Re-run `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR` without `--allow-blocked` after both blockers are cleared.

## ZCU104 Remote C3b Smoke Runner Gate

Status: remote physical-smoke runner was added on 2026-06-11 KST.

Decision taken:

- Added a host-side SSH/SCP runner for the C3b physical smoke blocker.
- Default mode is dry-run; no network command executes unless `--execute` is passed.
- Runner validates local tarball/SHA256 before generating or running board commands.
- Runner fetches board-produced JSON and imports it through `import_pynq_smoke_result.py` when executed.

Measured result:

- Dry-run status: `dry-run`.
- Local input errors: `0`.
- Default target: `xilinx@zcu104.local`.
- Default remote dir: `/home/xilinx/hgtxr_c3b_smoke`.
- Canonical imported result path: `hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json`.

Remaining choices:

- Provide a reachable ZCU104 host/user/identity and run the tool with `--execute`.
- Keep physical smoke pending until board access exists.

## XR-VITs Unblock Packet

Status: user-choice packet for the XR-VITs gate was added on 2026-06-11 KST.

Decision taken:

- Added a packet generator that converts the XR-VITs gate into two explicit choices.
- Choice 1: restore exact `/home/kjm26/project/PRJXR/XR-VITs`.
- Choice 2: explicitly approve `XR_Accel` via `create_xr_vits_replacement_policy.py`.
- Did not create active `docs/resources/xr_vits_replacement_policy.json`.

Measured result:

- Packet status: `pending-user-choice`.
- Exact requested path exists: `false`.
- Active policy exists: `false`.
- Active policy approved: `false`.
- Recommended replacement: `XR_Accel`, score `99`.

Remaining choices:

- Restore exact `XR-VITs`.
- Approve `XR_Accel` with explicit approver/reason.
- Keep final-signoff blocked.

## Integrated Final Signoff Runner

Status: final signoff runner now includes both blocker-support artifacts as of 2026-06-11 KST.

Decision taken:

- Integrated ZCU104 remote dry-run generation into the final runner.
- Integrated XR-VITs unblock packet generation into the final runner.
- Kept final pass/fail judgment tied to final preflight, final audit, and completion audit.

Measured result:

- Runner status: `blocked`.
- Final preflight: `ok=62`, `warn=4`, `fail=2`.
- ZCU104 remote status: `dry-run`.
- XR-VITs packet status: `pending-user-choice`.
- C3b readiness status: `ready-for-board`.

Remaining choices:

- Execute physical C3b smoke on ZCU104.
- Restore exact XR-VITs or explicitly approve XR_Accel.

## Final Unblock Dry-Run Rehearsal Command

Status: closure-readiness output now includes a no-side-effect rehearsal command
for supplied final unblock candidates on 2026-06-11 KST.

Decision:

- Keep final-signoff strict.
- Do not create a fake physical C3b smoke result.
- Do not create active `docs/resources/xr_vits_replacement_policy.json` without explicit approval.
- Add a dry-run command that operators can run before the active final unblock command.

Options:

| Option | Action | Expected result | Risk |
|---|---|---|---|
| R0: rehearsal first | Run `dry_run_final_runner_command` after supplying a real C3b smoke JSON and XR-VITs exact/replacement inputs. | Validates import and replacement inputs without writing canonical smoke JSON or active policy. Runner may remain `blocked` by design, but candidate closure evidence should show readiness. | Does not complete final signoff because no canonical unblock input is written. |
| R1: active unblock after rehearsal | Run `final_runner_command` after R0 passes. | Imports the real C3b smoke JSON, writes approved XR-VITs replacement policy when selected, and should clear both final blockers. | Side effects are intentional; requires real board JSON and explicit approval. |
| R2: defer | Do not supply external inputs yet. | Current evidence remains reproducible and `blocked` with two known blockers. | Goal remains incomplete. |

Implementation:

- `hardware/tools/check_final_blocker_closure_readiness.py` now emits `dry_run_final_runner_command`.
- Dry-run command uses `--dry-run-import-c3b-smoke`, `--dry-run-xr-vits-replacement` when applicable, and `--allow-blocked`.
- Generated active and dry-run final runner commands now forward custom `--xr-vits-replacement-path`.

Current recommendation: use R0 before R1 once the real C3b board JSON and XR-VITs decision are available.

## Final Unblock Intake Artifact

Status: final unblock candidate intake was added on 2026-06-11 KST.

Decision:

- Keep final blocker closure candidate validation in one no-side-effect artifact.
- Use the same candidate-audit and closure-readiness logic as the final runner.
- Hash the intake artifact through the closeout packet and final evidence manifest.

Options:

| Option | Action | Expected result | Risk |
|---|---|---|---|
| I0: no candidates supplied | Run intake with default final runner inputs. | `status=blocked`; it lists the two missing blocker inputs and emits no active command. | No final unblock progress until external inputs exist. |
| I1: supplied C3b JSON plus exact XR-VITs | Run intake with `--c3b-smoke-json` and `--xr-vits-mode exact`. | `status=ready-for-active-unblock` if the board JSON is valid and exact `/home/kjm26/project/PRJXR/XR-VITs` exists. | Requires exact checkout to exist. |
| I2: supplied C3b JSON plus XR_Accel approval | Run intake with `--c3b-smoke-json`, `--xr-vits-mode replacement`, approver, reason, and optional replacement path. | `status=ready-for-active-unblock` if the board JSON is valid and replacement approval inputs match the candidate audit. | Active command intentionally writes canonical smoke result and active replacement policy. |

Implementation:

- New tool: `hardware/tools/write_final_unblock_intake.py`.
- New test: `hardware/tests/test_write_final_unblock_intake.py`.
- Runner step: `final-unblock-intake`.
- Docs resources: `docs/resources/final_unblock_intake_2026_06_10.json/.md`.
- Evidence manifest now requires `42/42` stable artifacts and `19` consistency checks.

Current recommendation: after receiving a real C3b board JSON, run intake first, then run the generated dry-run command, then run the generated active command only if both pass.

## C3b Smoke Result Contract

Status: C3b physical smoke JSON contract was added on 2026-06-11 KST.

Decision:

- Do not create a fake `hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json`.
- Define the exact board JSON fields that a real ZCU104 run must produce.
- Include both dry-run import and active import commands so the operator can rehearse before writing canonical inputs.
- Hash the contract through the closeout packet and final evidence manifest.

Options:

| Option | Action | Expected result | Risk |
|---|---|---|---|
| K0: contract only | Generate `c3b_smoke_result_contract_2026_06_10.json/.md`. | Makes required physical-smoke JSON shape explicit; final remains blocked. | No final unblock without real board run. |
| K1: dry-run import | Run validator/import with `--dry-run` against a supplied board JSON. | Confirms board JSON matches `axis-c3b-mem16` without copying to canonical path. | Still blocked by design because canonical result is not written. |
| K2: active import | Run import without `--dry-run` after K1 passes. | Copies valid board JSON to canonical path and clears the C3b physical-smoke blocker. | Intentional side effect; must use real board output only. |

Implementation:

- New tool: `hardware/tools/write_c3b_smoke_result_contract.py`.
- New test: `hardware/tests/test_write_c3b_smoke_result_contract.py`.
- Runner step: `c3b-smoke-result-contract`.
- Docs resources: `docs/resources/c3b_smoke_result_contract_2026_06_10.json/.md`.
- Evidence manifest now requires `44/44` stable artifacts and `22` consistency checks.

Current recommendation: use K1 before K2 once the real ZCU104 board JSON exists.

## Contract-Aware Final Unblock Commands

Status: command card and closeout validation now use the C3b contract directly on 2026-06-11 KST.

Decision:

- Keep final-signoff blocked until real external inputs arrive.
- Expose C3b contract details inside the operator command card.
- Validate contract content in the closeout packet, not just presence and hash.
- Raise closeout-validation consistency threshold to `29` checks.

Options:

| Option | Action | Expected result | Risk |
|---|---|---|---|
| L0: read contract-aware card | Use `final_unblock_commands_2026_06_10.json/.md` before board run. | Operator sees exact validate/dry-run/active import commands and expected C3b output. | Still requires manual board output. |
| L1: closeout validation first | Inspect `final_unblock_closeout_packet_validation_2026_06_10.json`. | Confirms contract status, preset, variant, canonical path, output, and commands are consistent. | Validation is host-side only. |
| L2: active unblock later | After real C3b JSON and XR-VITs decision exist, run dry-run then active command from card/intake. | Should clear both remaining blockers if inputs are valid. | Active command intentionally writes canonical board JSON and possibly replacement policy. |

Implementation:

- `write_final_unblock_commands.py` now accepts `--c3b-contract-json`.
- `run_third_goal_final_signoff.py` passes generated contract JSON to command-card step.
- `write_final_unblock_closeout_packet.py` includes `c3b_smoke_contract`.
- `validate_final_unblock_closeout_packet.py` validates contract content.
- Closeout validation now passes `29/29`.

Current recommendation: keep using L0 and L1 as the operator preflight; run L2 only after the real board JSON and XR-VITs decision are available.

## E2E Resource Policy Audit

Status: DSP/URAM/LUTRAM/parallelism policy audit was added on 2026-06-11 KST.

Decision:

- Keep C3b as the current board-smoke candidate because it preserves C1 latency/DSP gain while lowering LUT and using 16 memory banks.
- Validate resource policy from both source pragmas/macros and generated report metrics.
- Do not run a new HLS/Vivado build in this step.

Options:

| Option | Action | Expected result | Risk |
|---|---|---|---|
| M0: audit current policy | Generate `e2e_resource_policy_audit_2026_06_10.json/.md`. | Confirms DSP/URAM/LUTRAM/PAR evidence is present and consistent. | Host-side evidence only; no new synthesis. |
| M1: continue C3b board path | Use C3b PAR16 MEM16 bit/hwh for physical smoke. | Uses current best-fit candidate: DSP `604`, LUT `126506`, URAM `64`, latency `37508072`. | Still needs real ZCU104 execution. |
| M2: run another synthesis sweep | Launch new HLS/Vivado sweep after changing pragmas or PAR. | Produces fresher resource data if design changes. | Long runtime; may stall; should be queued after current final unblock inputs. |

Implementation:

- New tool: `hardware/tools/write_e2e_resource_policy_audit.py`.
- New test: `hardware/tests/test_write_e2e_resource_policy_audit.py`.
- Runner step: `resource-policy-audit`.
- Docs resources: `docs/resources/e2e_resource_policy_audit_2026_06_10.json/.md`.
- Evidence manifest now requires `46/46` stable artifacts.

Current recommendation: M1. The host evidence supports C3b; next meaningful external action is real C3b ZCU104 smoke.

## Selected Path Execution Audit

Status: selected A2/A1/C/E-pending plan was locked as a machine-readable audit on 2026-06-11 KST.

Decision:

- Preserve user-selected Path 1 as `A2 then A1`.
- Preserve user-selected Path 2 as `C`, with C3b PAR16/MEM16 as the current board-smoke candidate.
- Keep `E` pending.
- Do not run new HLS/Vivado or board smoke in this audit step.

Options:

| Option | Action | Expected result | Risk |
|---|---|---|---|
| N0: audit current selection | Generate `selected_path_execution_audit_2026_06_10.json/.md`. | Confirms selected branches and evidence consistency with no side effects. | Host-side evidence only. |
| N1: continue C3b board smoke | Use the C3b bundle and import real board JSON. | Clears the C3b physical-smoke blocker if output matches. | Requires real ZCU104 execution. |
| N2: revisit E | Reopen the E branch later. | Allows another performance/architecture direction. | User explicitly kept E pending for now. |

Implementation:

- New tool: `hardware/tools/write_selected_path_execution_audit.py`.
- New test: `hardware/tests/test_write_selected_path_execution_audit.py`.
- Runner step: `selected-path-execution-audit`.
- Docs resources: `docs/resources/selected_path_execution_audit_2026_06_10.json/.md`.
- Evidence manifest now requires `48/48` stable artifacts.

Current recommendation: N1 only after real board access; keep E pending.

## Spec/Plan Conformance Audit

Status: planning/spec conformance was locked as a machine-readable audit on 2026-06-11 KST.

Decision:

- Keep spec-kit status explicit: `spec-kit`/`specify` are unavailable, so the project spec is maintained manually.
- Validate that the manual plan/spec artifacts still cover the active 3차목표 requirements and selected A2/A1/C/E-pending plan.
- Do not treat this audit as hardware completion; it only proves planning and trace consistency.

Options:

| Option | Action | Expected result | Risk |
|---|---|---|---|
| O0: audit current manual spec | Generate `spec_plan_conformance_audit_2026_06_10.json/.md`. | Confirms planning/spec/trace documents are internally consistent. | Does not install spec-kit or clear external blockers. |
| O1: install or restore spec-kit later | Re-run spec generation with actual spec-kit when available. | Better compliance with the requested spec-kit workflow. | Requires tool availability outside current host evidence. |
| O2: continue physical unblock | Keep current docs and run/import real C3b board result plus XR-VITs resolution. | Moves final signoff blockers directly. | Requires external board/source inputs. |

Implementation:

- New tool: `hardware/tools/write_spec_plan_conformance_audit.py`.
- New test: `hardware/tests/test_write_spec_plan_conformance_audit.py`.
- Runner step: `spec-plan-conformance-audit`.
- Docs resources: `docs/resources/spec_plan_conformance_audit_2026_06_10.json/.md`.
- Evidence manifest now requires `50/50` stable artifacts.

Current recommendation: O2 for final signoff; O0 is complete and keeps the manual spec path auditable.
