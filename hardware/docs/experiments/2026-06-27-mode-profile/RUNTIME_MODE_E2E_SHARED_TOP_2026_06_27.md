# Runtime-Mode E2E Shared Top Evidence - 2026-06-27

## Goal

Configure one E2E AXIS top so Search and Track select their workload at runtime while using the same controller-level ATTN/MLP execution path, then verify the ZCU104 HLS latency targets:

- Search target: <= 4.000 ms.
- Track target: <= 1.000 ms.

## Implementation Summary

- Top: `hls/src/hgtxr_e2e_axis_top.cpp`.
- Shared runtime helpers and compute path: `hls/include/hgtxr_e2e_vit.hpp`.
- Runtime mode selection:
  - `HGTXR_MODE_SEARCH` when `num_pixels` is not Track mode/control.
  - `HGTXR_MODE_TRACK` when `num_pixels == HGTXR_MODE_TRACK` or Track frame size.
- Runtime workload selection:
  - Search: `HGTXR_SEARCH_H/W`, `HGTXR_SEARCH_TOKENS`, `HGTXR_SEARCH_DEPTH`.
  - Track: `HGTXR_TRACK_H/W`, `HGTXR_TRACK_TOKENS`, `HGTXR_TRACK_CUT_DEPTH`.
- The fast runtime-mode profile uses one top-level controller and alternates the same templated fast ATTN/MLP unit pair for both modes.
- `runtime-mode-par32-search` and `runtime-mode-par32-track` are separate board-smoke/latency invocations, not separate accelerator builds; both package and run the same `hgtxr_e2e_axis_dma_par32_runtime_mode_mem16` `.bit/.hwh` pair.
- MLP fast projection was split into a normalized wrapper and a no-norm core path so MLP performs one layernorm before fc1/gelu/fc2 rather than repeating layernorm inside both projections.
- `HGTXR_E2E_OMIT_WEIGHT_AXI=1` is enabled for this fast runtime-mode profile, so the unused external weight memory master is removed from generated RTL/IP/HWH. The AXI-Lite `weights` pointer-control fields remain only as ABI/dummy control fields.

## Commands

```bash
bash -n hardware/scripts/run/run_e2e_q4w8a_no_board.sh
bash -n hardware/scripts/run/run_e2e_axis_dma_vivado_no_board.sh
timeout 900 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par32_runtime_mode_mem16
timeout 3600 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_mode_mem16
timeout 7200 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh cosim par32_runtime_mode_mem16
timeout 1800 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh package par32_runtime_mode_mem16
timeout 7200 sh hardware/scripts/run/run_e2e_axis_dma_vivado_no_board.sh par32_runtime_mode_mem16
```

## CSim Evidence

Profile: `par32_runtime_mode_mem16`.

- Search CSim passed:
  - `search_runtime_state=0 expected=0`
  - output count `6`
  - TLAST asserted on final state word.
- Track CSim passed:
  - `track_runtime_state=1 expected=1`
  - output count `6`
  - TLAST asserted on final state word.
- Vitis HLS reported: `CSim done with 0 errors`.

## CSynth Evidence

Report:

`hardware/generated/hgtxr_e2e_axis_par32_runtime_mode_mem16_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`

Target:

- Device: `xczu7ev-ffvc1156-2-e`.
- Clock target: `5.00 ns`.
- Estimated clock: `3.691 ns`.

Latency:

| Case | Cycles | Time @ 5 ns | Target | Status |
|---|---:|---:|---:|---|
| Track-bound min | 76,150 | 0.381 ms | <= 1.000 ms | pass |
| Search-bound max | 594,840 | 2.974 ms | <= 4.000 ms | pass |

Main instance latency:

| Instance | Min cycles | Max cycles | Max time |
|---|---:|---:|---:|
| `hgtxr_axis_read_frame` | 4,100 | 16,388 | 81.940 us |
| `hgtxr_e2e_fast_patch_embedding` | 195 | 771 | 3.855 us |
| `hgtxr_e2e_controller_run` | 71,523 | 577,637 | 2.888 ms |
| `hgtxr_e2e_fast_state_head` | 520 | 808 | 4.040 us |

Resource estimate:

| Resource | Used | Available | Utilization |
|---|---:|---:|---:|
| BRAM_18K | 68 | 624 | 10% |
| DSP | 400 | 1728 | 23% |
| FF | 39,696 | 460,800 | 8% |
| LUT | 88,246 | 230,400 | 38% |
| URAM | 0 | 96 | 0% |

Interface check:

- `m_axi_gmem_e2e_runtime` remains present.
- `m_axi_gmem_e2e_weights` / `gmem_e2e_weights` is absent from the packaged IP, generated RTL, overlay `.hwh`, and copied PYNQ `.hwh`.

## HW Emulation / RTL Co-Simulation Attempt

Flow:

- Added `cosim` mode to `hardware/scripts/run/run_e2e_q4w8a_no_board.sh`.
- Added `hardware/vivado/scripts/run_e2e_q4w8a_cosim.tcl`.
- The cosim Tcl reuses the same csynth setup, then runs `cosim_design -rtl verilog -trace_level none`.
- `runtime_mode_active64_b8_ff768` now enables `HGTXR_E2E_TEST_RUNTIME_MODES=1`, so the cosim C testbench covers both Search and Track transactions.

Command:

```bash
timeout 7200 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh cosim par32_runtime_mode_mem16
```

Completed before RTL run:

- C testbench instrumentation completed.
- Search C testbench transaction passed:
  - output words `15, 15, 15, 15, 15, 15`
  - TLAST asserted on word `5`
  - `search_runtime_state=0 expected=0 count=6 last=1 failures=0`
- Track C testbench transaction passed:
  - output words `9, 9, 9, 9, 9, 9`
  - TLAST asserted on word `5`
  - `track_runtime_state=1 expected=1 count=6 last=1`
- XSIM RTL compile/elaboration built snapshot `hgtxr_e2e_axis_top`.

Current blocker:

- HLS reports C/RTL co-simulation `FAIL` because XSIM fails while loading the generated snapshot:

```text
xsim {hgtxr_e2e_axis_top} -autoloadwcfg -tclbatch {hgtxr_e2e_axis_top.tcl}
ERROR: unexpected exception when evaluating tcl command
```

Negative checks already tried:

- Set `TERMINFO=/lib/terminfo` and tried `TERM=dumb`, `vt100`, `xterm`, `xterm-256color`, and `linux`.
- Added Vivado/Vitis runtime library paths for `libtinfo.so.5` and XSIM libraries.
- Temporarily disabled the generated `.wcfg` file.
- Tried direct `xsim ... -R` and direct `xsimk` execution. `xsimk` can load and exit the snapshot, but does not produce the HLS cosim PASS/latency report.

Evidence files:

- HLS cosim report: `hardware/generated/hgtxr_e2e_axis_par32_runtime_mode_mem16_no_board/solution_e2e_q4w8a/sim/report/hgtxr_e2e_axis_top_cosim.rpt`
- HLS cosim log: `hardware/generated/hgtxr_e2e_axis_par32_runtime_mode_mem16_no_board/solution_e2e_q4w8a/sim/report/verilog/hgtxr_e2e_axis_top.log`
- XSIM log: `hardware/generated/hgtxr_e2e_axis_par32_runtime_mode_mem16_no_board/solution_e2e_q4w8a/sim/verilog/xsim.log`

## Vivado Route Evidence

Overlay:

`hardware/generated/build/vivado/overlay/hgtxr_e2e_axis_dma_par32_runtime_mode_mem16_overlay/`

Artifacts:

- `hgtxr_e2e_axis_dma_par32_runtime_mode_mem16.bit`
- `hgtxr_e2e_axis_dma_par32_runtime_mode_mem16.hwh`

The same `.bit/.hwh` pair was copied to `hardware/pynq/hgtxr/`.

Board smoke preparation:

- Search bundle: `hardware/generated/pynq/e2e_axis_dma_par32_runtime_mode_search_smoke_bundle.tar.gz`
- Search session: `hardware/generated/pynq/e2e_axis_dma_par32_runtime_mode_search_smoke_session.{json,md}`
- Track bundle: `hardware/generated/pynq/e2e_axis_dma_par32_runtime_mode_track_smoke_bundle.tar.gz`
- Track session: `hardware/generated/pynq/e2e_axis_dma_par32_runtime_mode_track_smoke_session.{json,md}`
- Search runner uses `--variant runtime-mode-par32 --weights-mode zero --mode-profile search --expect-runtime-state 0`.
- Track runner uses `--variant runtime-mode-par32 --weights-mode zero --mode-profile track --expect-runtime-state 1`.
- Runtime-mode validation presets enforce the board accelerator-latency targets from the returned JSON:
  - Search preset `axis-runtime-mode-par32-search`: every `accelerator_latency_ms_samples[]` value and `accelerator_latency_ms_summary.max` must be `<= 4.000 ms`.
  - Track preset `axis-runtime-mode-par32-track`: every `accelerator_latency_ms_samples[]` value and `accelerator_latency_ms_summary.max` must be `<= 1.000 ms`.
- Remote dry-run plans were generated for both profiles:
  - `hardware/generated/signoff/zcu104_runtime_mode_par32_search_smoke_remote_run_2026_06_10.{json,md}`
  - `hardware/generated/signoff/zcu104_runtime_mode_par32_track_smoke_remote_run_2026_06_10.{json,md}`
- Combined board-latency completion gate:
  - `hardware/generated/signoff/runtime_mode_board_latency_gate_2026_06_28.{json,md}`
  - Current status is `missing` because the canonical Search/Track board result JSON files have not been copied back yet.
- Combined Search/Track remote-run wrapper:
  - `hardware/tools/run_runtime_mode_board_latency.py`
  - Dry-run output: `hardware/generated/signoff/runtime_mode_board_latency_run_2026_06_28.{json,md}`
  - Current status is `dry-run` with board-latency gate `missing`.
- Execute attempt:
  - `hardware/generated/signoff/runtime_mode_board_latency_run_2026_06_28_execute.{json,md}`
  - Current status is `host-unresolved`; host preflight failed for `zcu104.local`, so SSH/SCP was not attempted.
  - No Search/Track board result JSON was imported.

Route/timing:

| Metric | Value |
|---|---:|
| WNS | 3.038 ns |
| TNS | 0.000 ns |
| WHS | 0.010 ns |
| THS | 0.000 ns |
| Timing constraints | met |
| Fully routed nets | 50,076 / 50,076 |
| Nets with routing errors | 0 |
| Bitgen | pass |

Placed utilization:

| Resource | Used | Available | Utilization |
|---|---:|---:|---:|
| CLB LUTs | 22,412 | 230,400 | 9.73% |
| CLB Registers | 21,105 | 460,800 | 4.58% |
| Block RAM Tile | 37 | 312 | 11.86% |
| DSP | 401 | 1,728 | 23.21% |
| URAM | 0 | 96 | 0.00% |

Routed power estimate:

| Metric | Value |
|---|---:|
| Total on-chip power | 3.736 W |
| Dynamic power | 3.042 W |
| Device static power | 0.694 W |
| PS static | 0.099 W |
| PL static | 0.595 W |

## Evidence Boundary

- Latency is HLS CSim/csynth evidence for the ZCU104 part. The matching AXIS/DMA overlay now routes, bitgen completes, Search/Track board-smoke bundles are prepared, and board JSON validation will enforce `4 ms` / `1 ms`; however, board-measured latency JSON has not been captured.
- HW Emulation at the HLS RTL co-simulation level has been wired into the runner and reaches C testbench pass plus XSIM snapshot build. It is not a completed C/RTL PASS yet because XSIM fails on the generated `-autoloadwcfg` snapshot-load command in this host environment.
- The current runtime-mode fastpath is a mode-profile compute path inside the E2E AXIS top. It proves shared runtime scheduling and HLS latency, but it is not yet a full exact weight-consuming ViT datapath proof.
- Search/Track profiles are kept separate only to enforce different runtime controls, frame sizes, and board latency limits. They do not imply separate Search and Track ATTN/MLP hardware blocks.
- The previous dangling `gmem_e2e_weights` memory-master warning is cleaned up for this profile by guarding the unused weight AXI master. The remaining `weights` AXI-Lite control fields are only pointer-control ABI, not a DDR master interface.

## Current Judgment

The HLS-level latency target is achieved for the single runtime-mode E2E AXIS top, and the matching ZCU104 AXIS/DMA overlay routes with positive timing slack and successful bitgen. Remaining signoff work is executing the combined Search/Track board-latency runner against a resolvable ZCU104 host or IP, importing both board JSON files, and re-running the completion gate.
