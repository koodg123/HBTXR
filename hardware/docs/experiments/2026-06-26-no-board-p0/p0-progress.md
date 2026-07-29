# P0 No-Board Progress

Date: 2026-06-26

## Scope

Board inference is excluded. This progress note covers P0 host/HLS work only.

## Completed

| P0 item | Status | Output |
|---|---|---|
| `NB-P0-00` report collector | completed | `scripts/report/collect_no_board_reports.py` |
| Existing HLS report collection | completed | `analysis/no-board-results/p0_report_collection_2026_06_26/hls_modules.{json,csv}` |
| Top variant table | completed | `analysis/no-board-results/p0_report_collection_2026_06_26/hls_top_variants.csv` |
| Markdown summary | completed | `analysis/no-board-results/p0_report_collection_2026_06_26/summary.md` |
| C3b baseline reproduction | completed | latency `37,508,072`, LUT `126,506`, DSP `604`, BRAM_18K `332`, URAM `64` |

## Collector Result

| Metric | Value |
|---|---:|
| Parsed HLS module reports | 407 |
| Parsed top variants | 11 |
| C3b latency cycles | 37,508,072 |
| C3b latency ms @ 5ns HLS target | 187.540 |
| C3b LUT | 126,506 |
| C3b DSP | 604 |
| C3b BRAM_18K | 332 |
| C3b URAM | 64 |

After the PAR32 rerun and report refresh, the collector parsed `515` HLS module
reports and `14` top variants. The earlier table is retained as the original
C3b baseline lock snapshot.

## Observed Existing Top Variants

| Variant | Latency cycles | LUT | DSP | BRAM_18K | URAM | Note |
|---|---:|---:|---:|---:|---:|---|
| `hgtxr_e2e_axis_hls` | 71,316,968 | 81,168 | 332 | 298 | 64 | Lower resource, slower baseline |
| `hgtxr_e2e_axis_par16_c3_mem8` | 48,598,922 | 113,124 | 604 | 292 | 64 | PAR16 with MEM8 |
| `hgtxr_e2e_axis_par16_c3b_mem16` | 37,508,072 | 126,506 | 604 | 332 | 64 | Protected C3b baseline |
| `hgtxr_e2e_axis_par16_hls` | 37,508,072 | 127,916 | 604 | 338 | 64 | PAR16 reference |
| `hgtxr_e2e_axis_vref_p0_softmax_input_x2_qkv_uram_csynth` | 498,485 | 43,236 | 128 | 114 | 40 | Successor-style reduced test path; not equivalent to full C3b signoff |

## Immediate Next P0 Actions

| Next ID | Action | Required before run |
|---|---|---|
| `P1-PAR32-CSYNTH` / `NB-P0-01` | Generate or select PAR32 exploratory csynth path | Confirm compile-time knobs for PAR factor and generated output directory suffix |
| `P3-DSP-DENSITY` / `NB-P0-02` | Audit dense path multiplier mapping | Identify QKV/MHA/MLP multiplication loops and HLS reports to compare |
| `P3-URAM-BANKING` / `NB-P0-03` | Sweep URAM banking/layout | Enumerate large buffers and current primitive mapping |
| `P3-LUTRAM-SMALLBUF` / `NB-P0-04` | Sweep small-buffer LUTRAM threshold | Enumerate small score/prob/scratch/scale tables |

## P0-A/P0-B/P0-C Agent Results

| Task | Status | Result |
|---|---|---|
| `P0-A` PAR32 path | completed | `HGTXR_E2E_PAR=32` and `HGTXR_E2E_MEM_BANK_PAR=16/32` are supported by existing E2E Q4W8A Tcl; project output is `generated/<project>/solution_e2e_q4w8a/` |
| `P0-B` DSP density audit | completed | QKV, attention score/value, WO, MLP FC1/FC2, and final head use `hgtxr_e2e_dsp_mul`; `kDensePar` controls lane unroll width |
| `P0-C` memory binding audit | completed | Large buffers `tokens/norm/q/k/v/attn/hidden` are macro-controlled URAM/BRAM candidates; small `pooled/score/prob/exp_raw` buffers use `HGTXR_E2E_SMALL_MEM_LUTRAM`; RMU/SMU `score/prob` already use LUTRAM |

## P0 No-Board Run Wrapper

| Item | Value |
|---|---|
| Wrapper | `scripts/run/run_e2e_q4w8a_no_board.sh` |
| Modes | `csim`, `csynth`, `package` |
| Default profile | `par32_dsp_uram_mem16` |
| Vitis fallback path | `/tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls` |
| Vitis library path fix | `LD_LIBRARY_PATH` prepends `/tools/Xilinx/Vitis_HLS/2023.2/lib/lnx64.o/Rhel/9` to satisfy `libtinfo.so.5` |
| Expected PAR32 csynth report | `generated/hgtxr_e2e_axis_par32_dsp_uram_mem16_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt` |

## P0 Vivado-Level Continuation

| Item | Status | Evidence |
|---|---|---|
| PAR32 HLS IP package | completed | `generated/hgtxr_e2e_axis_par32_dsp_mixed_stream_mem16_no_board/solution_e2e_q4w8a/impl/ip/component.xml` |
| PAR32 package gate | passed | packaged HLS IP exists |
| PAR32 Vivado route | completed | WNS `3.885 ns`, TNS `0.000 ns`, route errors `0`, bitgen completed |
| Detailed status note | updated | `analysis/no-board-results/VIVADO_PROGRESS_2026_06_27.md` |

Route command after package completion:

```sh
scripts/run/run_e2e_axis_dma_vivado_no_board.sh par32_dsp_mixed_stream_mem16
```

Routed outputs:

| Artifact | Path |
|---|---|
| bitstream | `generated/build/vivado/overlay/hgtxr_e2e_axis_dma_par32_dsp_mixed_stream_mem16_overlay/hgtxr_e2e_axis_dma_par32_dsp_mixed_stream_mem16.bit` |
| handoff | `generated/build/vivado/overlay/hgtxr_e2e_axis_dma_par32_dsp_mixed_stream_mem16_overlay/hgtxr_e2e_axis_dma_par32_dsp_mixed_stream_mem16.hwh` |
| PYNQ copy | `pynq/hgtxr/hgtxr_e2e_axis_dma_par32_dsp_mixed_stream_mem16.{bit,hwh}` |

## Runnable P0 Profiles

| Profile | PAR | MEM_BANK_PAR | Resource policy | Purpose | Expected outcome |
|---|---:|---:|---|---|---|
| `par16_c3b_recheck` | 16 | 16 | `dsp_uram` | Baseline repro in new namespace | Confirms wrapper reproduces C3b-like resource policy |
| `par32_dsp_uram_mem16` | 32 | 16 | `dsp_uram` | Raise dense/attention parallelism with C3b memory bank count | Higher DSP use and lower latency if memory ports do not bottleneck |
| `par32_dsp_uram_mem32` | 32 | 32 | `dsp_uram` | Raise parallelism and bank count together | Lower bank conflict risk, higher routing/resource pressure |
| `par32_dsp_mixed_stream_mem16` | 32 | 16 | `dsp_mixed_stream` | Keep Q/K/V/attention in URAM but move hidden back to BRAM | Intended follow-up after `dsp_mixed` remains above URAM capacity |
| `par32_dsp_mixed_mem16` | 32 | 16 | `dsp_mixed` | Keep PAR32 but place only Q/K/V/attention/hidden in URAM | Intended follow-up after full URAM exceeds device capacity |
| `par32_dsp_mixed_mem32` | 32 | 32 | `dsp_mixed` | Put Q/K/V/attention/hidden in URAM selectively | Tests whether selective large-buffer URAM reduces pressure vs full URAM |

## Direct Commands

```sh
scripts/run/run_e2e_q4w8a_no_board.sh csim par32_dsp_uram_mem16
scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_dsp_uram_mem16
scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_dsp_uram_mem32
scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_dsp_mixed_stream_mem16
scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_dsp_mixed_mem16
scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_dsp_mixed_mem32
```

## Run Results

| Run | Status | Latency cycles | LUT | DSP | BRAM_18K | URAM | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| `P1-PAR32-CSYNTH` / `par32_dsp_uram_mem16` | completed | 26,139,644 | 193,546 | 1,148 | 124 | 160 | resource-fail: URAM `160/96` exceeds ZCU104 capacity |
| `P1-PAR32-CSYNTH` / `par32_dsp_mixed_mem16` | completed | 26,139,644 | 193,546 | 1,148 | 204 | 112 | resource-fail: URAM `112/96` exceeds ZCU104 capacity |
| `P1-PAR32-CSYNTH` / `par32_dsp_mixed_stream_mem16` | completed | 26,139,644 | 193,546 | 1,148 | 332 | 64 | csynth-fit candidate: latency and DSP improve; URAM fits; LUT is high at `84.00%`, so route risk remains |

## Best Current No-Board Candidate

| Field | C3b baseline | Best PAR32 no-board candidate | Delta |
|---|---:|---:|---:|
| Variant | `par16_c3b_mem16` | `par32_dsp_mixed_stream_mem16` | n/a |
| Latency cycles | 37,508,072 | 26,139,644 | -30.31% |
| HLS latency @ 5 ns | 187.540 ms | 130.698 ms | -56.842 ms |
| DSP | 604 / 1,728, 34.95% | 1,148 / 1,728, 66.44% | +544 DSP |
| LUT | 126,506 / 230,400, 54.91% | 193,546 / 230,400, 84.00% | +67,040 LUT |
| BRAM_18K | 332 / 624, 53.21% | 332 / 624, 53.21% | 0 |
| URAM | 64 / 96, 66.67% | 64 / 96, 66.67% | 0 |
| Estimated clock | 3.953 ns | 3.697 ns | no csynth timing regression |
| Decision | protected baseline | routed no-board candidate | ZCU104 route passes, but mode targets remain unproven |

## PAR32 Routed Result

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
| Total on-chip power estimate | 3.477 W | `.../hgtxr_e2e_axis_dma_system_wrapper_power_routed.rpt` |

Vivado placed wrapper utilization is treated as integration sanity only because
the generated/OOC HLS IP hierarchy under-reports arithmetic/memory totals
(`DSP=0`, `URAM=0`). The HLS `csynth.xml` resource totals above remain the
resource-policy authority for the candidate.

## Mode-Specific Latency Boundary

| Requirement | Current no-board evidence | Status |
|---|---|---|
| Search mode <= 4 ms | Mode-profile top `hgtxr_search_profile_top` reports `772,268` cycles / `3.861 ms @ 5 ns`; routed WNS `2.638 ns`, route errors `0`, bitgen pass. | HLS/IP/Vivado route proven |
| Track mode <= 1 ms | Mode-profile top `hgtxr_track_profile_top` reports `99,449` cycles / `0.497 ms @ 5 ns`; routed WNS `1.481 ns`, route errors `0`, bitgen pass. | HLS/IP/Vivado route proven |
| Search/Track invocation distribution | Current no-board flow has no runtime mode counters. | unavailable |

The original `hls/src/hgtxr_e2e_axis_top.cpp` still ignores `num_pixels` and always executes the same full E2E path. The new evidence is from separate no-board mode-profile tops in `hls/src/hgtxr_mode_profile_top.cpp`, not from a deployed mode-dispatching E2E AXIS top.

## Mode-Profile HLS/IP Package Results

| Mode | Profile | Worst cycles | Latency @ 5 ns | Target | Target met | LUT | DSP | BRAM_18K | URAM | IP package |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| Search | `search_par32` | 772,268 | 3.861 ms | 4.000 ms | yes | 65,557 | 294 | 8 | 96 | pass |
| Track | `track_par32` | 99,449 | 0.497 ms | 1.000 ms | yes | 69,728 | 302 | 51 | 64 | pass |

## Mode-Profile Vivado Route Results

| Mode | WNS | TNS | WHS | THS | Route errors | Fully routed nets | Bit/hwh | Post-route LUT | DSP | BRAM_18K | URAM | Power |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| Search | 2.638 ns | 0.000 ns | 0.010 ns | 0.000 ns | 0 | 39,455 / 39,455 | pass | 18,825 | 294 | 2 | 96 | 3.917 W |
| Track | 1.481 ns | 0.000 ns | 0.010 ns | 0.000 ns | 0 | 47,137 / 47,137 | pass | 22,245 | 302 | 42 | 64 | 3.778 W |

Detailed evidence is saved in `analysis/no-board-results/MODE_PROFILE_RESULTS_2026_06_27.md` and the refreshed collector output under `analysis/no-board-results/p0_mode_profile_2026_06_27/`.

Search reaches the HLS and Vivado route targets but consumes `96/96` URAM, so resource margin is still a P0 risk. Track has a cleaner routed resource margin with `64/96` URAM.

The no-board collector now supports a hard P0 gate:
`python3 scripts/report/collect_no_board_reports.py --output-dir /tmp/hgtxr_p0_collect_check --enforce-p0`.
This gate passed on the current report set and fails if Search/Track mode-profile targets are not met or if either mode profile exceeds ZCU104 HLS capacity.

## Evidence Boundary

- Existing `csynth.xml` files were parsed.
- A no-board Vitis HLS wrapper was added; launch status should be tracked per generated project directory.
- `P1-PAR32-CSYNTH` was launched in detached tmux sessions after adding the Vitis `libtinfo.so.5` library path.
- PAR32 HLS IP package completed, Vivado route completed, and routed `.bit/.hwh` artifacts were generated and copied for PYNQ packaging.
- Mode-specific Search/Track HLS profile tops were added, passed CSim/csynth, packaged as HLS IP, and routed through a m_axi-only ZCU104 Vivado BD flow.
- No board/PYNQ inference was launched.
