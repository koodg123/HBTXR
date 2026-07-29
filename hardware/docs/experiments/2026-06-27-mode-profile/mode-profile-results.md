# Mode-Specific No-Board HLS Profile Results

Date: 2026-06-27  
Scope: Host/HLS/IP-package only. Board inference is excluded.

## 1. Objective

Measure mode-specific HLS latency against the active target:

| Mode | Target | Evidence level |
|---|---:|---|
| Search | <= 4.000 ms | Vitis HLS CSim, csynth, IP package, Vivado route |
| Track | <= 1.000 ms | Vitis HLS CSim, csynth, IP package, Vivado route |

This is not a full E2E AXIS board-runtime measurement. The previous E2E top always runs the full E2E path, so separate mode-profile tops were added for no-board target gating.

## 2. Implemented Artifacts

| Artifact | Purpose |
|---|---|
| `hls/src/hgtxr_mode_profile_top.cpp` | Separate Search and Track HLS profile tops |
| `hls/tb/tb_hgtxr_mode_profile_top.cpp` | CSim smoke for both profile tops |
| `vivado/scripts/run_mode_profile_q4w8a_csim.tcl` | Mode-profile CSim Tcl |
| `vivado/scripts/run_mode_profile_q4w8a_csynth.tcl` | Mode-profile csynth Tcl |
| `vivado/scripts/package_mode_profile_ip.tcl` | Mode-profile IP package Tcl |
| `vivado/scripts/build_mode_profile_bitstream.tcl` | Mode-profile Vivado BD route/bitstream Tcl |
| `scripts/run/run_mode_profile_no_board.sh` | `csim`, `csynth`, `package` wrapper |
| `scripts/run/run_mode_profile_vivado_no_board.sh` | Mode-profile Vivado route wrapper |
| `scripts/report/collect_no_board_reports.py` | Collector now includes mode profile tops and target pass/fail columns |

## 3. Top-Level Results

| Mode | Profile | Top | Target cycles @ 200 MHz | Worst cycles | Worst latency | Target met | Est. clock |
|---|---|---|---:|---:|---:|---|---:|
| Search | `search_par32` | `hgtxr_search_profile_top` | 800,000 | 772,268 | 3.861 ms | yes | 3.836 ns |
| Track | `track_par32` | `hgtxr_track_profile_top` | 200,000 | 99,449 | 0.497 ms | yes | 3.819 ns |

## 4. ZCU104 HLS Resource Fit

| Mode | LUT | LUT % | DSP | DSP % | FF | FF % | BRAM_18K | BRAM % | URAM | URAM % | Fit note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Search | 65,557 | 28.45 | 294 | 17.01 | 23,131 | 5.02 | 8 | 1.28 | 96 | 100.00 | HLS resource fit, but no URAM margin |
| Track | 69,728 | 30.26 | 302 | 17.48 | 25,735 | 5.58 | 51 | 8.17 | 64 | 66.67 | HLS resource fit with URAM margin |

## 5. Latency Breakdown

| Mode | Block | Worst cycles | Latency @ 5 ns |
|---|---|---:|---:|
| Search | active frame patch embed | 17,921 | 0.090 ms |
| Search | active transformer profile | 753,745 | 3.769 ms |
| Search | active pool | 387 | 0.002 ms |
| Search | search head | 208 | 0.001 ms |
| Track | active event patch embed | 4,481 | 0.022 ms |
| Track | active transformer profile | 94,441 | 0.472 ms |
| Track | cached feature copy loop | 9 | 0.000 ms |
| Track | active pool | 99 | 0.000 ms |
| Track | fusion loop | 195 | 0.001 ms |
| Track | track head | 217 | 0.001 ms |

The transformer profile dominates both modes. Search still has the narrowest timing/resource margin because it uses all 96 URAMs in the HLS estimate.

## 6. IP Package Evidence

| Mode | IP package status | Component XML | Export ZIP | Vivado route |
|---|---|---|---|---|
| Search | pass | `generated/hgtxr_mode_search_par32_no_board/solution_mode_q4w8a/impl/ip/component.xml` | `generated/hgtxr_mode_search_par32_no_board/solution_mode_q4w8a/impl/export.zip` | pass |
| Track | pass | `generated/hgtxr_mode_track_par32_no_board/solution_mode_q4w8a/impl/ip/component.xml` | `generated/hgtxr_mode_track_par32_no_board/solution_mode_q4w8a/impl/export.zip` | pass |

## 7. Vivado Route Evidence

| Mode | WNS | TNS | WHS | THS | Failing setup/hold endpoints | Fully routed nets | Route errors | Bitgen | Power |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| Search | 2.638 ns | 0.000 ns | 0.010 ns | 0.000 ns | 0 / 0 | 39,455 / 39,455 | 0 | pass | 3.917 W |
| Track | 1.481 ns | 0.000 ns | 0.010 ns | 0.000 ns | 0 / 0 | 47,137 / 47,137 | 0 | pass | 3.778 W |

| Mode | Post-route LUT | LUT % | Post-route FF | FF % | DSP | DSP % | BRAM_18K | BRAM % | URAM | URAM % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Search | 18,825 | 8.17 | 15,171 | 3.29 | 294 | 17.01 | 2 | 0.32 | 96 | 100.00 |
| Track | 22,245 | 9.65 | 20,826 | 4.52 | 302 | 17.48 | 42 | 6.73 | 64 | 66.67 |

Vivado route outputs:

| Mode | Bitstream | Handoff | Reports |
|---|---|---|---|
| Search | `generated/build/vivado/overlay/hgtxr_mode_search_par32_overlay/hgtxr_mode_search_par32.bit` | `generated/build/vivado/overlay/hgtxr_mode_search_par32_overlay/hgtxr_mode_search_par32.hwh` | `generated/build/vivado/overlay/hgtxr_mode_search_par32_overlay/reports/` |
| Track | `generated/build/vivado/overlay/hgtxr_mode_track_par32_overlay/hgtxr_mode_track_par32.bit` | `generated/build/vivado/overlay/hgtxr_mode_track_par32_overlay/hgtxr_mode_track_par32.hwh` | `generated/build/vivado/overlay/hgtxr_mode_track_par32_overlay/reports/` |

Power is Vivado vectorless routed estimate. Board power, sensor I/O, runtime DMA bandwidth, p95/p99 latency, and search/track invocation distribution remain out of scope for this no-board step.

## 8. Evidence Boundary

| Item | Status |
|---|---|
| CSim functional smoke | pass for Search and Track |
| Mode-specific csynth target | pass for Search and Track |
| Mode-specific IP package | pass for Search and Track |
| Mode-specific Vivado routed overlay | pass for Search and Track |
| Board latency / DMA / p95 / p99 | out of scope for this no-board step |
| Runtime search/track invocation distribution | not measured; profile tops are synthetic mode-isolated tops |

## 9. Next Required Work

| Priority | Work | Reason |
|---|---|---|
| P0 | Reduce Search URAM from 96/96 to a routed-margin target | Search routes, but has zero URAM slack |
| P1 | Add collector gate that fails when Search/Track mode target is false or a mode profile exceeds ZCU104 HLS capacity | completed via `collect_no_board_reports.py --enforce-p0`; current reports pass |
| P1 | Add optional mode counters to the E2E AXIS top | Connect synthetic profile evidence to runtime mode dispatch later |
