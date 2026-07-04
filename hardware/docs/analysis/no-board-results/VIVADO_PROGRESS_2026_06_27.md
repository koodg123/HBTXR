# P0 PAR32 Vivado Progress

Date: 2026-06-27 01:30:23 KST

Board inference is excluded. This note tracks the no-board HLS package and Vivado
route attempt for the current best PAR32 candidate.

## Prompt Brief

| Field | Content |
|---|---|
| Goal | Continue P0 toward Search 4 ms / Track 1 ms, ZCU104 fit, and Vivado-level evidence without board inference. |
| Inputs | `analysis/no-board-results/P0_PROGRESS_2026_06_26.md`, HLS `csynth.xml`, `scripts/run/run_e2e_q4w8a_no_board.sh`, `scripts/run/run_e2e_axis_dma_vivado_no_board.sh`, Vivado BD Tcl. |
| Constraints | Do not overwrite C3b as protected baseline; do not claim board/runtime evidence; route only after HLS IP `component.xml` exists. |
| Expected outputs | Packaged PAR32 HLS IP, Vivado overlay route result, route utilization/timing/power evidence if implementation completes. |
| Acceptance | `component.xml` exists, Vivado wrapper launches, routed `.bit/.hwh` and timing/utilization reports are produced, or blocker is recorded. |

## Expert Roles

| Role | Responsibility |
|---|---|
| HLS/Vivado execution manager | Package current PAR32 stream candidate and launch route when legal. |
| FPGA resource modeler | Keep DSP/LUT/BRAM/URAM and ZCU104 fit gates explicit. |
| Verification lead | Separate HLS estimates, Vivado route evidence, and unavailable board/runtime metrics. |

## Sub-Agent Usage

| Task | Runtime | Status | Result |
|---|---|---|---|
| `VIVADO-A` | GPT5.3-Codex-Spark sub-agent | completed | Confirmed exact package gate and route wrapper: PAR32 route requires `generated/hgtxr_e2e_axis_par32_dsp_mixed_stream_mem16_no_board/solution_e2e_q4w8a/impl/ip/component.xml` before Vivado can launch. |
| `MODE-B` | GPT5.3-Codex-Spark sub-agent | failed | Remote compact failed from context exhaustion. Main agent completed local mode-latency audit. |

## Execution DAG

| Step | Command / Gate | Status |
|---|---|---|
| 1 | `scripts/run/run_e2e_q4w8a_no_board.sh package par32_dsp_mixed_stream_mem16` | completed |
| 2 | Check `generated/hgtxr_e2e_axis_par32_dsp_mixed_stream_mem16_no_board/solution_e2e_q4w8a/impl/ip/component.xml` | completed |
| 3 | `scripts/run/run_e2e_axis_dma_vivado_no_board.sh par32_dsp_mixed_stream_mem16` | completed |
| 4 | Collect routed timing/utilization/power artifacts | completed |

## Current Candidate

| Metric | C3b baseline | PAR32 stream candidate | Interpretation |
|---|---:|---:|---|
| HLS latency cycles | 37,508,072 | 26,139,644 | PAR32 improves HLS latency by 30.31%. |
| HLS latency @ 5 ns | 187.540 ms | 130.698 ms | Still far above mode targets if interpreted as full E2E frame path. |
| DSP | 604 / 1,728, 34.95% | 1,148 / 1,728, 66.44% | DSP utilization target direction improved. |
| LUT | 126,506 / 230,400, 54.91% | 193,546 / 230,400, 84.00% | Main route risk. |
| BRAM_18K | 332 / 624, 53.21% | 332 / 624, 53.21% | Fits csynth capacity. |
| URAM | 64 / 96, 66.67% | 64 / 96, 66.67% | Fits csynth capacity. |

## Package Status

| Item | Value |
|---|---|
| tmux session | `hgtxr_par32_stream_package_20260627` |
| observed stage | package completed |
| package output gate | `generated/hgtxr_e2e_axis_par32_dsp_mixed_stream_mem16_no_board/solution_e2e_q4w8a/impl/ip/component.xml` |
| current gate status | present as of 2026-06-27 01:32 KST |
| route status | completed; bitgen completed successfully |

## Vivado Route Command

Run only after `component.xml` exists:

```sh
scripts/run/run_e2e_axis_dma_vivado_no_board.sh par32_dsp_mixed_stream_mem16
```

Launch note: the first route attempt exited immediately because the wrapper did
not have the executable bit. `chmod +x scripts/run/run_e2e_axis_dma_vivado_no_board.sh`
fixed it, and the route session then reached Vivado BD creation/validation.

Expected outputs:

| Artifact | Path |
|---|---|
| bitstream | `generated/build/vivado/overlay/hgtxr_e2e_axis_dma_par32_dsp_mixed_stream_mem16_overlay/hgtxr_e2e_axis_dma_par32_dsp_mixed_stream_mem16.bit` |
| handoff | `generated/build/vivado/overlay/hgtxr_e2e_axis_dma_par32_dsp_mixed_stream_mem16_overlay/hgtxr_e2e_axis_dma_par32_dsp_mixed_stream_mem16.hwh` |
| PYNQ copy | `pynq/hgtxr/hgtxr_e2e_axis_dma_par32_dsp_mixed_stream_mem16.{bit,hwh}` |

## Routed Result

| Metric | Value | Evidence |
|---|---:|---|
| WNS | 3.885 ns | `generated/build/vivado/hgtxr_e2e_axis_dma_par32_dsp_mixed_stream_mem16_overlay/hgtxr_e2e_axis_dma_par32_dsp_mixed_stream_mem16_overlay.runs/impl_1/hgtxr_e2e_axis_dma_system_wrapper_timing_summary_routed.rpt` |
| TNS | 0.000 ns | same |
| WHS | 0.010 ns | same |
| THS | 0.000 ns | same |
| Setup failing endpoints | 0 | same |
| Hold failing endpoints | 0 | same |
| Fully routed nets | 22,142 / 22,142 | `.../hgtxr_e2e_axis_dma_system_wrapper_route_status.rpt` |
| Routing errors | 0 | same |
| Bitgen | completed successfully | `.../impl_1/runme.log` |
| DRC at bitgen | 0 errors, 7 warnings | `.../impl_1/runme.log` |

Routed artifacts:

| Artifact | Size | Path |
|---|---:|---|
| overlay bitstream | 19 MB | `generated/build/vivado/overlay/hgtxr_e2e_axis_dma_par32_dsp_mixed_stream_mem16_overlay/hgtxr_e2e_axis_dma_par32_dsp_mixed_stream_mem16.bit` |
| overlay handoff | 492 KB | `generated/build/vivado/overlay/hgtxr_e2e_axis_dma_par32_dsp_mixed_stream_mem16_overlay/hgtxr_e2e_axis_dma_par32_dsp_mixed_stream_mem16.hwh` |
| PYNQ bitstream copy | 19 MB | `pynq/hgtxr/hgtxr_e2e_axis_dma_par32_dsp_mixed_stream_mem16.bit` |
| PYNQ handoff copy | 492 KB | `pynq/hgtxr/hgtxr_e2e_axis_dma_par32_dsp_mixed_stream_mem16.hwh` |

## Power / Utilization Evidence Boundary

Vivado routed power was generated by vector-less `report_power`; this is a
no-board estimate, not measured board power.

| Power metric | Value |
|---|---:|
| Total on-chip power | 3.477 W |
| Dynamic | 2.785 W |
| Device static | 0.692 W |
| PS8 dynamic contribution | 2.647 W |
| PL static | 0.594 W |

Vivado top-level placed utilization reports the HLS IP mostly as generated/OOC
integration hierarchy and under-reports the actual HLS IP arithmetic/memory use
(`DSP=0`, `URAM=0` in wrapper placed report). Therefore:

| Resource evidence | Use for decision |
|---|---|
| HLS `csynth.xml` | Official candidate resource policy: DSP `1,148`, LUT `193,546`, BRAM_18K `332`, URAM `64`. |
| Vivado routed wrapper utilization | Integration sanity only; not used to claim HLS block resource totals. |

## Search / Track Latency Mapping

| Requirement | Current evidence | Status |
|---|---|---|
| Search mode <= 4 ms | `hgtxr_search_profile_top`: `772,268` cycles / `3.861 ms @ 5 ns`; routed WNS `2.638 ns`, route errors `0`, bitgen pass. | no-board HLS/IP/Vivado route proven |
| Track mode <= 1 ms | `hgtxr_track_profile_top`: `99,449` cycles / `0.497 ms @ 5 ns`; routed WNS `1.481 ns`, route errors `0`, bitgen pass. | no-board HLS/IP/Vivado route proven |
| Mode distribution | No board/runtime invocation counters in current no-board flow. | unavailable |

Local code audit:

- `hls/src/hgtxr_e2e_axis_top.cpp` ignores `num_pixels` and always executes the same full E2E path.
- Existing non-E2E `hgtxr_top.cpp` has search/track concepts, but current PAR32 E2E AXIS top does not expose mode-specific latency or routing counters.
- Separate mode-profile tops in `hls/src/hgtxr_mode_profile_top.cpp` now provide no-board Search/Track latency and routed integration evidence.

## Mode-Profile Routed Result

| Mode | Routed WNS | Routed TNS | Route errors | Fully routed nets | Post-route LUT | DSP | BRAM_18K | URAM | Power |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Search | 2.638 ns | 0.000 ns | 0 | 39,455 / 39,455 | 18,825 | 294 | 2 | 96 | 3.917 W |
| Track | 1.481 ns | 0.000 ns | 0 | 47,137 / 47,137 | 22,245 | 302 | 42 | 64 | 3.778 W |

Evidence:

- `generated/build/vivado/overlay/hgtxr_mode_search_par32_overlay/reports/hgtxr_mode_search_par32_timing_summary.rpt`
- `generated/build/vivado/overlay/hgtxr_mode_search_par32_overlay/reports/hgtxr_mode_search_par32_route_status.rpt`
- `generated/build/vivado/overlay/hgtxr_mode_track_par32_overlay/reports/hgtxr_mode_track_par32_timing_summary.rpt`
- `generated/build/vivado/overlay/hgtxr_mode_track_par32_overlay/reports/hgtxr_mode_track_par32_route_status.rpt`

## Next Work

| Priority | Work | Expected result |
|---|---|---|
| P0-B | Reduce Search URAM pressure or record zero-margin exception | Search currently routes but uses `96/96` URAM. |
| P0-C | Generate PAR32-LUT-reduced or PAR24 fallback only if mode-specific profiling shows this full E2E candidate is the wrong optimization target | Keeps DSP improvement while reducing LUT/URAM pressure. |
| P0-D | Reconcile HLS IP resource totals with Vivado hierarchy reporting | Prevents wrapper-level `DSP=0`/`URAM=0` integration reports from being mistaken for actual HLS core resources. |
