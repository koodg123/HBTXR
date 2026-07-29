# AQ2 Search/Track Measurement Report - 2026-06-30

## Scope

This note reports the current AQ2 HLS-level latency/DMA/throughput metrics and
the post-implementation Vivado timing/resource/power metrics for the E2E AXIS
runtime top. The AQ2 profile is identified by the generated project names
containing `attntok2` / `aq2`:

- Search-only:
  `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq2_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/`
- Track-only:
  `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq2_300_mem16_track_only_no_board/solution_e2e_q4w8a/syn/report/`

All time-derived values below are normalized to the requested 300 MHz target
clock unless otherwise noted. Vitis HLS reports Track absolute latency using
the estimated 3.473 ns clock; this document also gives the 300 MHz normalized
value for apples-to-apples AQ2 comparison.

## Methodology

- Clock: 300 MHz target, 3.333 ns period.
- AXIS bus width: 256 bit, 32 B/beat.
- Ideal DMA bandwidth: `256 bit * 300 MHz / 8 = 9.6 GB/s`.
- Search input frame: `128 x 128 = 16,384` AXIS input beats.
- Track input frame: `64 x 64 = 4,096` AXIS input beats.
- Output state: `HGTXR_STATE = 6` AXIS output beats.
- Wire-transfer effective bandwidth counts full 256-bit AXIS beats.
- Useful-payload lower bound counts the low-byte payload used by the testbench.
- Latency percentiles are not runtime histogram measurements. CSynth provides a
  deterministic estimate/range, so P95/P99 are reported as the conservative
  HLS max bound for each mode.
- Vivado timing/resource/power are from implemented routed overlays generated
  on 2026-07-01 KST. Search meets strict Vivado timing. Track has WNS
  `-0.017 ns`, which fails strict Vivado timing but is inside the user-approved
  `-0.5 ns` WNS tolerance.
- Vivado power is vector-less `report_power` at implemented/routed state. It
  includes the PS8 estimate in the Vivado model and PL on-chip static/dynamic
  estimates, but it does not separately measure external DDR or sensor I/O
  board rails.
- GOPS uses the analytical model:
  `ops = 2 * (blocks * (4*T*E^2 + 2*T*E*FF + 2*T^2*E) + T*E*PATCH^2 + STATE*E)`.
  Here `E=192`, `FF=768`, `PATCH=16`, Search `T=64`, Track `T=16`,
  Search body iterations `blocks=4`, Track body iterations `blocks=2`.
  One MAC is counted as two operations.

## Summary

| Metric | Search AQ2 | Track AQ2 |
|---|---:|---:|
| Target clock | 300 MHz | 300 MHz |
| HLS estimated clock | 2.777 ns | 3.473 ns |
| Latency min | 1,709,763 cycles / 5.699210 ms | 268,710 cycles / 0.895700 ms |
| Latency max | 1,709,783 cycles / 5.699277 ms | 268,710 cycles / 0.895700 ms |
| Mean latency, HLS deterministic conservative | 5.699277 ms | 0.895700 ms |
| Median latency, HLS deterministic conservative | 5.699277 ms | 0.895700 ms |
| P95 latency, HLS deterministic conservative | 5.699277 ms | 0.895700 ms |
| P99 latency, HLS deterministic conservative | 5.699277 ms | 0.895700 ms |
| Initial interval | 1,709,784 cycles / 5.699280 ms | 268,711 cycles / 0.895703 ms |
| Invocation throughput from II | 175.46 inv/s | 1,116.44 inv/s |
| Analytical compute | 0.245369 GOP | 0.030280 GOP |
| Throughput | 43.05 GOPS | 33.81 GOPS |
| AXIS/AXI data width | 256 bit | 256 bit |
| Ideal DMA bandwidth | 9.600 GB/s | 9.600 GB/s |
| Wire bytes per inference | 524,480 B | 131,264 B |
| Effective DMA bandwidth, wire | 0.0920 GB/s | 0.1465 GB/s |
| Useful-payload bytes per inference | 16,390 B | 4,102 B |
| Effective DMA bandwidth, useful lower bound | 0.00288 GB/s | 0.00458 GB/s |

For a 10% Search / 90% Track analytical mixture, using the mode-specific HLS
latencies above:

| Metric | Hybrid 10/90 analytical value |
|---|---:|
| Mean latency | 1.376058 ms |
| Median latency | 0.895700 ms |
| Min latency | 0.895700 ms |
| Max latency | 5.699277 ms |
| P95 latency | 5.699277 ms |
| P99 latency | 5.699277 ms |
| Mean initial interval | 1.376061 ms |
| Invocation throughput from mean II | 726.71 inv/s |

## Latency Breakdown

Search-only top instance breakdown:

| Major block | Latency cycles | Latency @300 MHz |
|---|---:|---:|
| AXIS read frame | 16,386 | 54.620 us |
| Frame conv patch embedding | 258,052 | 0.860173 ms |
| Global buffer load | 12,292 | 40.973 us |
| Controller run | 1,397,902-1,397,922 | 4.659673-4.659740 ms |
| MLP head | 25,114 | 83.713 us |
| Top end-to-end | 1,709,763-1,709,783 | 5.699210-5.699277 ms |

Search controller breakdown:

| Controller block | Latency cycles | Latency @300 MHz |
|---|---:|---:|
| Weight dispatch/prefetch | 6,928-6,933 | 23.093-23.110 us |
| Shared attention unit | 131,388 | 0.437960 ms |
| Shared MLP unit | 211,151 | 0.703837 ms |
| Dispatch loop, trip 4 | 27,724-27,744 | 92.413-92.480 us |
| Body loop, trip 4 | 1,370,176 | 4.567253 ms |

Track-only top instance breakdown:

| Major block | Latency cycles | Latency @300 MHz |
|---|---:|---:|
| AXIS read frame | 4,109 | 13.697 us |
| Event conv patch embedding | 86,020 | 0.286733 ms |
| Global buffer load | 3,076 | 10.253 us |
| Controller run | 168,806 | 0.562687 ms |
| MLP head | 6,682 | 22.273 us |
| Top end-to-end | 268,710 | 0.895700 ms |

Track controller breakdown:

| Controller block | Latency cycles | Latency @300 MHz |
|---|---:|---:|
| Weight dispatch/prefetch | 6,926 | 23.087 us |
| Shared attention unit | 25,257 | 84.190 us |
| Shared MLP unit | 52,212 | 0.174040 ms |
| Dispatch loop, trip 2 | 13,858 | 46.193 us |
| Body loop, trip 2 | 154,946 | 0.516487 ms |

## Resource Utilization

ZCU104 target device in the HLS reports: `xczu7ev-ffvc1156-2-e`.
Available resources used for percentages: BRAM_18K 624, DSP 1,728,
FF 460,800, LUT 230,400, URAM 96.

Search top total:

| Resource | Used | Available | Utilization |
|---|---:|---:|---:|
| BRAM_18K | 361 | 624 | 57.85% |
| DSP | 1,288 | 1,728 | 74.54% |
| FF | 133,107 | 460,800 | 28.89% |
| LUT | 264,616 | 230,400 | 114.85% |
| URAM | 92 | 96 | 95.83% |

Search major block instance breakdown:

| Major block | BRAM_18K | DSP | FF | LUT | URAM |
|---|---:|---:|---:|---:|---:|
| control_s_axi | 0 / 0.00% | 0 / 0.00% | 406 / 0.09% | 744 / 0.32% | 0 / 0.00% |
| gmem_e2e_runtime_m_axi | 4 / 0.64% | 0 / 0.00% | 818 / 0.18% | 678 / 0.29% | 0 / 0.00% |
| AXIS read frame | 0 / 0.00% | 0 / 0.00% | 47 / 0.01% | 185 / 0.08% | 0 / 0.00% |
| Frame conv patch embedding | 0 / 0.00% | 30 / 1.74% | 1,907 / 0.41% | 5,171 / 2.24% | 0 / 0.00% |
| Controller run | 229 / 36.70% | 1,258 / 72.80% | 128,591 / 27.91% | 246,125 / 106.82% | 92 / 95.83% |
| MLP head | 0 / 0.00% | 0 / 0.00% | 928 / 0.20% | 4,490 / 1.95% | 0 / 0.00% |
| Global buffer load | 0 / 0.00% | 0 / 0.00% | 275 / 0.06% | 499 / 0.22% | 0 / 0.00% |
| Top local memories | 128 / 20.51% | 0 / 0.00% | 0 / 0.00% | 0 / 0.00% | 0 / 0.00% |

Search controller internal major blocks:

| Controller block | BRAM_18K | DSP | FF | LUT | URAM |
|---|---:|---:|---:|---:|---:|
| Shared attention unit | 79 / 12.66% | 725 / 41.96% | 62,600 / 13.59% | 129,779 / 56.33% | 64 / 66.67% |
| Shared MLP unit | 86 / 13.78% | 405 / 23.44% | 53,539 / 11.62% | 90,289 / 39.19% | 0 / 0.00% |
| Weight dispatch/prefetch | 0 / 0.00% | 0 / 0.00% | 1,612 / 0.35% | 5,607 / 2.43% | 0 / 0.00% |
| Controller local memories | 64 / 10.26% | 0 / 0.00% | 0 / 0.00% | 0 / 0.00% | 28 / 29.17% |

Track top total:

| Resource | Used | Available | Utilization |
|---|---:|---:|---:|
| BRAM_18K | 359 | 624 | 57.53% |
| DSP | 1,412 | 1,728 | 81.71% |
| FF | 145,179 | 460,800 | 31.51% |
| LUT | 260,389 | 230,400 | 113.02% |
| URAM | 71 | 96 | 73.96% |

Track major block instance breakdown:

| Major block | BRAM_18K | DSP | FF | LUT | URAM |
|---|---:|---:|---:|---:|---:|
| control_s_axi | 0 / 0.00% | 0 / 0.00% | 406 / 0.09% | 744 / 0.32% | 0 / 0.00% |
| gmem_e2e_runtime_m_axi | 4 / 0.64% | 0 / 0.00% | 818 / 0.18% | 678 / 0.29% | 0 / 0.00% |
| AXIS read frame | 0 / 0.00% | 0 / 0.00% | 429 / 0.09% | 409 / 0.18% | 0 / 0.00% |
| Event conv patch embedding | 0 / 0.00% | 34 / 1.97% | 2,893 / 0.63% | 9,930 / 4.31% | 0 / 0.00% |
| Controller run | 225 / 36.06% | 1,378 / 79.75% | 139,308 / 30.23% | 236,547 / 102.67% | 71 / 73.96% |
| MLP head | 0 / 0.00% | 0 / 0.00% | 924 / 0.20% | 4,478 / 1.94% | 0 / 0.00% |
| Global buffer load | 0 / 0.00% | 0 / 0.00% | 266 / 0.06% | 488 / 0.21% | 0 / 0.00% |
| Top local memories | 130 / 20.83% | 0 / 0.00% | 0 / 0.00% | 0 / 0.00% | 0 / 0.00% |

Track controller internal major blocks:

| Controller block | BRAM_18K | DSP | FF | LUT | URAM |
|---|---:|---:|---:|---:|---:|
| Shared attention unit | 79 / 12.66% | 721 / 41.72% | 63,060 / 13.69% | 129,603 / 56.25% | 64 / 66.67% |
| Shared MLP unit | 82 / 13.14% | 529 / 30.61% | 63,992 / 13.89% | 84,723 / 36.77% | 0 / 0.00% |
| Weight dispatch/prefetch | 0 / 0.00% | 0 / 0.00% | 15 / 0.00% | 100 / 0.04% | 7 / 7.29% |
| Controller local memories | 64 / 10.26% | 0 / 0.00% | 0 / 0.00% | 0 / 0.00% | 0 / 0.00% |

## Vivado Implementation Status

| Metric | Search AQ2 implemented | Track AQ2 implemented |
|---|---:|---:|
| Bitstream generation | pass | pass |
| WNS | 0.000 ns | -0.017 ns |
| TNS | 0.000 ns | -0.086 ns |
| TNS failing endpoints | 0 / 424,960 | 8 / 422,971 |
| WHS | 0.007 ns | 0.001 ns |
| THS | 0.000 ns | 0.000 ns |
| Strict Vivado timing | pass | fail |
| User WNS tolerance, >= -0.5 ns | pass | pass |
| Fully routed nets | 166,085 / 166,085 | 164,929 / 164,929 |
| Routing errors | 0 | 0 |

## Vivado Implemented Resource Utilization

These are post-implementation Vivado utilization numbers. They supersede the
HLS resource estimates for fit checks. The HLS estimates above remain useful for
major-block attribution inside the HLS top, but they over-predict LUT pressure.

| Resource | Search used | Search util | Track used | Track util |
|---|---:|---:|---:|---:|
| CLB LUTs | 77,494 / 230,400 | 33.63% | 80,565 / 230,400 | 34.97% |
| LUT as Logic | 69,675 / 230,400 | 30.24% | 72,749 / 230,400 | 31.58% |
| LUT as Distributed RAM | 5,278 / 101,760 | 5.19% | 5,252 / 101,760 | 5.16% |
| CLB Registers | 78,687 / 460,800 | 17.08% | 83,427 / 460,800 | 18.10% |
| BRAM tile | 191.5 / 312 | 61.38% | 185.5 / 312 | 59.46% |
| RAMB36/FIFO | 93 / 312 | 29.81% | 70 / 312 | 22.44% |
| RAMB18 | 197 / 624 | 31.57% | 231 / 624 | 37.02% |
| URAM | 92 / 96 | 95.83% | 64 / 96 | 66.67% |
| DSP | 1,257 / 1,728 | 72.74% | 1,270 / 1,728 | 73.50% |
| PS8 | 1 / 1 | 100.00% | 1 / 1 | 100.00% |

Implemented top-level hierarchy breakdown:

| Mode | Major block | LUT | Logic LUT | LUTRAM | SRL | FF | RAMB36 | RAMB18 | URAM | DSP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Search | AXI control | 757 | 756 | 0 | 1 | 891 | 0 | 0 | 0 | 0 |
| Search | DMA input | 385 | 359 | 0 | 26 | 619 | 1 | 1 | 0 | 0 |
| Search | DMA output | 1,579 | 1,494 | 32 | 53 | 2,355 | 4 | 1 | 0 | 0 |
| Search | AXI memory/interconnect | 3,499 | 2,434 | 1,024 | 41 | 5,767 | 0 | 0 | 0 | 0 |
| Search | E2E accelerator IP | 71,282 | 64,640 | 4,222 | 2,420 | 69,055 | 88 | 195 | 92 | 1,257 |
| Search | PSU | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Track | AXI control | 863 | 862 | 0 | 1 | 905 | 0 | 0 | 0 | 0 |
| Track | DMA input | 388 | 362 | 0 | 26 | 619 | 1 | 1 | 0 | 0 |
| Track | DMA output | 1,584 | 1,499 | 32 | 53 | 2,355 | 4 | 1 | 0 | 0 |
| Track | AXI memory/interconnect | 3,497 | 2,432 | 1,024 | 41 | 5,777 | 0 | 0 | 0 | 0 |
| Track | E2E accelerator IP | 74,240 | 67,601 | 4,196 | 2,443 | 73,771 | 65 | 229 | 64 | 1,270 |
| Track | PSU | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Power Breakdown

Vivado `report_power` top-level hierarchy reports dynamic power by BD block.
Internal conv/attention/MLP/head power is not separable from the current
hierarchy-preserved power report; for those blocks, use the HLS resource
breakdown above as attribution evidence, not as measured power.

| Power item | Search AQ2 | Track AQ2 | Notes |
|---|---:|---:|---|
| Total on-chip power | 7.032 W | 6.777 W | Vivado vector-less estimate |
| Dynamic power | 6.304 W | 6.054 W | On-chip dynamic |
| Device static | 0.728 W | 0.723 W | PS static + PL static |
| PS static | 0.103 W | 0.102 W | Vivado PS8 static estimate |
| PL static | 0.625 W | 0.620 W | Vivado PL static estimate |
| PS8 dynamic | 2.675 W | 2.684 W | Top-level hierarchy `psu` |
| AXI control dynamic | 0.020 W | 0.023 W | AXI-Lite control block |
| DMA input dynamic | 0.006 W | 0.006 W | AXIS DMA input path |
| DMA output dynamic | 0.018 W | 0.018 W | AXIS DMA output path |
| AXI memory/interconnect dynamic | 0.192 W | 0.194 W | Runtime AXI memory path |
| AXI/DMA/memory subtotal dynamic | 0.236 W | 0.241 W | control + DMA + AXI memory |
| E2E accelerator IP dynamic | 3.393 W | 3.129 W | Includes conv, shared ATTN/MLP, head, internal memories |
| External DDR/DRAM rail | n/a | n/a | Not separately measured by this Vivado report |
| Sensor I/O rail | n/a | n/a | Not present in this overlay/power report |

On-chip component view:

| Component | Search AQ2 | Track AQ2 |
|---|---:|---:|
| Clocks | 0.531 W | 0.543 W |
| CLB Logic | 0.727 W | 0.840 W |
| Signals | 0.976 W | 0.835 W |
| Block RAM | 0.200 W | 0.191 W |
| URAM | 0.200 W | 0.128 W |
| DSPs | 0.998 W | 0.847 W |
| PS8 | 2.671 W | 2.671 W |

## Evidence Files

- Search top CSynth:
  `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq2_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`
- Search controller CSynth:
  `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq2_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_controller_run_csynth.rpt`
- Track top CSynth:
  `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq2_300_mem16_track_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`
- Track controller CSynth:
  `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq2_300_mem16_track_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_controller_run_csynth.rpt`
- Search Vivado implemented timing/utilization/power:
  `hardware/generated/build/vivado/hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq2_300_mem16_search_only_overlay/hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq2_300_mem16_search_only_overlay.runs/impl_1/`
- Track Vivado implemented timing/utilization/power:
  `hardware/generated/build/vivado/hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq2_300_mem16_track_only_overlay/hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq2_300_mem16_track_only_overlay.runs/impl_1/`
- Shape/config sources:
  `hardware/hls/include/config.h`,
  `hardware/hls/include/cyclic_config.h`,
  `hardware/configs/zcu104_e2e_q4w8a_defines.h`,
  `hardware/hls/src/hgtxr_e2e_axis_top.cpp`,
  `hardware/hls/tb/tb_hgtxr_e2e_axis_top.cpp`

## Open Gaps

- AQ2 is over LUT capacity in HLS estimates, but the implemented Vivado
  overlays fit the device: Search 33.63% CLB LUT, Track 34.97% CLB LUT.
- Track HLS estimated clock is 3.473 ns and final implemented WNS is
  `-0.017 ns`; this fails strict Vivado timing but passes the user-approved
  `-0.5 ns` WNS tolerance.
- No AQ2 runtime histogram exists, so mean/median/P95/P99 are deterministic HLS
  conservative estimates, not measured runtime distributions.
- Vivado power is vector-less, not SAIF-annotated or board-measured.
- Internal conv/attention/MLP/head dynamic power is not separated by the current
  Vivado hierarchy; only top-level E2E accelerator IP power is available.
- External DDR/DRAM and sensor I/O rail power are not separately measured.
