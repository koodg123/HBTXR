# AQ2 Search/Track Metric Checklist - 2026-07-01

## 2026-07-01 Refresh

This document was rechecked for the requested AQ2 metric set on 2026-07-01.
The saved values below are still the current available evidence for AQ2:

- HLS Search/Track latency, initial interval, deterministic percentile
  envelope, DMA bandwidth estimate, and GOPS are derived from the AQ2
  Search-only and Track-only CSynth reports.
- Resource utilization is reported in two layers: HLS major-block attribution
  for Conv/Controller/Head-level analysis, and implemented Vivado utilization
  for physical ZCU104 fit.
- Power breakdown is the implemented Vivado vector-less hierarchy/component
  estimate. It separates PS8, AXI/DMA/interconnect, and E2E accelerator IP, but
  does not split internal Conv/ATTN/MLP/Head dynamic power.
- Board-sampled mean/median/P95/P99 latency, board DMA counters, external DDR
  rail power, and sensor I/O power remain unavailable for AQ2.

## 2026-07-01 Current Recheck

The AQ2 report values were re-read from the saved HLS and Vivado reports during
the 2026-07-01 KST work session. No new board-runtime JSON, SAIF/VCD activity,
or external rail-power capture was found, so the available AQ2 metric state is
unchanged:

- Search HLS latency envelope: `1,709,763-1,709,783 cycles`, normalized
  `5.699210-5.699277 ms` at 300 MHz.
- Search initial interval: `1,709,784 cycles`, normalized `5.699280 ms`.
- Track HLS latency envelope: `268,710 cycles`, normalized `0.895700 ms`.
- Track initial interval: `268,711 cycles`, normalized `0.895703 ms`.
- Search/Track deterministic HLS-envelope mean, median, P95, and P99 are the
  max-bound values above until repeated board samples are collected.
- Effective DMA bandwidth remains analytical, derived from AXIS transfer size
  and HLS latency: Search wire `0.0920 GB/s`, Track wire `0.1465 GB/s`.
- Implemented vector-less power remains Search `7.032 W` and Track `6.777 W`.

Additional same-session DSE note:

- A post-AQ2 Search-only `tokbank4` probe was synthesized to check whether
  token-dimension cyclic banking reduces the current Search bottleneck. This is
  not an AQ2 replacement because it is Search-only and not physically
  implemented.
- `tokbank4` CSim passed, but CSynth regressed Search II to `1,294,276 cycles`
  / `4.314253 ms` at 300 MHz and worsened resources to BRAM_18K
  `545/624 = 87%`, DSP `2,410/1,728 = 139%`, LUT
  `365,042/230,400 = 158%`, URAM `28/96 = 29%`.
- The scheduler still reports memory-port II violations inside the attention
  score path and MLP output path, so AQ2 requested metrics below remain the
  current documented AQ2 measurement set.

## Scope

This document restates the AQ2 Search/Track measurements in the exact metric
groups requested on 2026-07-01. It uses the AQ2 Search-only and Track-only HLS
reports plus the implemented Vivado utilization/power reports.

AQ2 evidence roots:

- Search HLS:
  `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq2_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/`
- Track HLS:
  `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq2_300_mem16_track_only_no_board/solution_e2e_q4w8a/syn/report/`
- Search Vivado implemented:
  `hardware/generated/build/vivado/hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq2_300_mem16_search_only_overlay/hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq2_300_mem16_search_only_overlay.runs/impl_1/`
- Track Vivado implemented:
  `hardware/generated/build/vivado/hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq2_300_mem16_track_only_overlay/hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq2_300_mem16_track_only_overlay.runs/impl_1/`

All normalized latency/bandwidth values below use the requested 300 MHz PL
clock, i.e. 3.333 ns/cycle. The Track CSynth report also prints absolute time
using its HLS estimated 3.473 ns clock; the normalized 300 MHz value is used for
Search/Track comparison.

Verification status on 2026-07-01:

- HLS Search and Track latency/resource values were re-read from the AQ2
  `hgtxr_e2e_axis_top_csynth.rpt` reports.
- Implemented utilization and power values were re-read from the AQ2 Vivado
  implemented reports.
- Mean, median, P95, and P99 are not board-sampled runtime statistics in this
  document. They are deterministic HLS-envelope values until repeated
  hardware-runtime samples are collected.

## DMA Bandwidth

Assumptions:

- AXIS data width: 256 bit = 32 B/beat.
- Ideal DMA bandwidth: `256 bit * 300 MHz / 8 = 9.600 GB/s`.
- Search input beats: `128 * 128 = 16,384`; output beats: `6`.
- Track input beats: `64 * 64 = 4,096`; output beats: `6`.
- Wire bytes count full 32 B AXIS beats.
- Useful bytes count only the low-byte payload used by the current testbench.

| Metric | Search AQ2 | Track AQ2 |
|---|---:|---:|
| AXIS/AXI data width | 256 bit | 256 bit |
| Ideal DMA bandwidth | 9.600 GB/s | 9.600 GB/s |
| Wire bytes per inference | 524,480 B | 131,264 B |
| Effective DMA bandwidth, wire | 0.0920 GB/s | 0.1465 GB/s |
| Useful-payload bytes per inference | 16,390 B | 4,102 B |
| Effective DMA bandwidth, useful lower bound | 0.00288 GB/s | 0.00458 GB/s |

## Latency And Initial Interval

CSynth provides deterministic min/max latency and initial interval, not a
runtime histogram. Therefore mean/median/P95/P99 are reported as deterministic
HLS estimates for this AQ2 report. Board or SAIF-backed repeated runs are still
required for measured runtime percentiles.

| Metric | Search AQ2 | Track AQ2 |
|---|---:|---:|
| Latency min | 1,709,763 cycles / 5.699210 ms | 268,710 cycles / 0.895700 ms |
| Latency max | 1,709,783 cycles / 5.699277 ms | 268,710 cycles / 0.895700 ms |
| Mean latency | 5.699277 ms | 0.895700 ms |
| Median latency | 5.699277 ms | 0.895700 ms |
| Min latency | 5.699210 ms | 0.895700 ms |
| Max latency | 5.699277 ms | 0.895700 ms |
| P95 latency | 5.699277 ms | 0.895700 ms |
| P99 latency | 5.699277 ms | 0.895700 ms |
| Initial interval | 1,709,784 cycles / 5.699280 ms | 268,711 cycles / 0.895703 ms |
| Invocation throughput from II | 175.46 inv/s | 1,116.44 inv/s |

## Throughput

Analytical operation model:

`ops = 2 * (blocks * (4*T*E^2 + 2*T*E*FF + 2*T^2*E) + T*E*PATCH^2 + STATE*E)`

Parameters: `E=192`, `FF=768`, `PATCH=16`, `STATE=6`,
Search `T=64, blocks=4`, Track `T=16, blocks=2`. One MAC is counted as two
operations.

| Metric | Search AQ2 | Track AQ2 |
|---|---:|---:|
| Analytical compute | 0.245369 GOP | 0.030280 GOP |
| Throughput | 43.05 GOPS | 33.81 GOPS |

## HLS Major-Block Latency Breakdown

Search-only top:

| Major block | Latency cycles | Latency at 300 MHz |
|---|---:|---:|
| AXIS read frame | 16,386 | 54.620 us |
| Frame conv patch embedding | 258,052 | 0.860173 ms |
| Global buffer load | 12,292 | 40.973 us |
| Controller run | 1,397,902-1,397,922 | 4.659673-4.659740 ms |
| MLP head | 25,114 | 83.713 us |
| Top end-to-end | 1,709,763-1,709,783 | 5.699210-5.699277 ms |

Track-only top:

| Major block | Latency cycles | Latency at 300 MHz |
|---|---:|---:|
| AXIS read frame | 4,109 | 13.697 us |
| Event conv patch embedding | 86,020 | 0.286733 ms |
| Global buffer load | 3,076 | 10.253 us |
| Controller run | 168,806 | 0.562687 ms |
| MLP head | 6,682 | 22.273 us |
| Top end-to-end | 268,710 | 0.895700 ms |

Controller internal latency:

| Controller block | Search AQ2 | Track AQ2 |
|---|---:|---:|
| Weight dispatch/prefetch | 6,928-6,933 cycles | 6,926 cycles |
| Shared attention unit | 131,388 cycles | 25,257 cycles |
| Shared MLP unit | 211,151 cycles | 52,212 cycles |
| Dispatch loop | 27,724-27,744 cycles, trip 4 | 13,858 cycles, trip 2 |
| Body loop | 1,370,176 cycles, trip 4 | 154,946 cycles, trip 2 |

## Resource Utilization

ZCU104 HLS target resources: BRAM_18K 624, DSP 1,728, FF 460,800,
LUT 230,400, URAM 96.

HLS Search major-block resource attribution:

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
| Top total | 361 / 57.85% | 1,288 / 74.54% | 133,107 / 28.89% | 264,616 / 114.85% | 92 / 95.83% |

HLS Track major-block resource attribution:

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
| Top total | 359 / 57.53% | 1,412 / 81.71% | 145,179 / 31.51% | 260,389 / 113.02% | 71 / 73.96% |

Implemented Vivado top resources supersede HLS estimates for physical fit:

| Resource | Search implemented | Track implemented |
|---|---:|---:|
| CLB LUTs | 77,494 / 230,400 = 33.63% | 80,565 / 230,400 = 34.97% |
| CLB registers | 78,687 / 460,800 = 17.08% | 83,427 / 460,800 = 18.10% |
| Block RAM tile | 191.5 / 312 = 61.38% | 185.5 / 312 = 59.46% |
| URAM | 92 / 96 = 95.83% | 64 / 96 = 66.67% |
| DSP | 1,257 / 1,728 = 72.74% | 1,270 / 1,728 = 73.50% |

Implemented top-level hierarchy resource split:

| Mode | Major block | LUT | FF | RAMB36 | RAMB18 | URAM | DSP |
|---|---|---:|---:|---:|---:|---:|---:|
| Search | AXI control | 757 | 891 | 0 | 0 | 0 | 0 |
| Search | DMA input | 385 | 619 | 1 | 1 | 0 | 0 |
| Search | DMA output | 1,579 | 2,355 | 4 | 1 | 0 | 0 |
| Search | AXI memory/interconnect | 3,499 | 5,767 | 0 | 0 | 0 | 0 |
| Search | E2E accelerator IP | 71,282 | 69,055 | 88 | 195 | 92 | 1,257 |
| Track | AXI control | 863 | 905 | 0 | 0 | 0 | 0 |
| Track | DMA input | 388 | 619 | 1 | 1 | 0 | 0 |
| Track | DMA output | 1,584 | 2,355 | 4 | 1 | 0 | 0 |
| Track | AXI memory/interconnect | 3,497 | 5,777 | 0 | 0 | 0 | 0 |
| Track | E2E accelerator IP | 74,240 | 73,771 | 65 | 229 | 64 | 1,270 |

## Power Breakdown

Power values are Vivado implemented vector-less estimates. They are not
SAIF/VCD-annotated and not board rail measurements. The current Vivado hierarchy
separates PS, AXI/DMA/interconnect, and the E2E accelerator IP; it does not
separate internal Conv/ATTN/MLP/Head dynamic power.

| Power item | Search AQ2 | Track AQ2 |
|---|---:|---:|
| Total on-chip power | 7.032 W | 6.777 W |
| Dynamic power | 6.304 W | 6.054 W |
| Device static | 0.728 W | 0.723 W |
| PS static | 0.103 W | 0.102 W |
| PL static | 0.625 W | 0.620 W |
| PS8 dynamic | 2.675 W | 2.684 W |
| AXI control dynamic | 0.020 W | 0.023 W |
| DMA input dynamic | 0.006 W | 0.006 W |
| DMA output dynamic | 0.018 W | 0.018 W |
| AXI memory/interconnect dynamic | 0.192 W | 0.194 W |
| AXI/DMA/memory subtotal dynamic | 0.236 W | 0.241 W |
| E2E accelerator IP dynamic | 3.393 W | 3.129 W |
| External DDR/DRAM rail | n/a | n/a |
| Sensor I/O rail | n/a | n/a |

On-chip component view:

| Component | Search AQ2 | Track AQ2 |
|---|---:|---:|
| Clocks | 0.531 W | 0.543 W |
| CLB logic | 0.727 W | 0.840 W |
| Signals | 0.976 W | 0.835 W |
| Block RAM | 0.200 W | 0.191 W |
| URAM | 0.200 W | 0.128 W |
| DSPs | 0.998 W | 0.847 W |
| PS8 | 2.671 W | 2.671 W |

## Measurement Gaps

- Repeated board-runtime samples are not available for AQ2, so mean, median,
  P95, and P99 are HLS deterministic estimates rather than measured runtime
  statistics.
- Measured DMA bandwidth is not available from board counters. Effective DMA
  bandwidth above is computed from HLS latency and AXIS transfer size.
- Internal Conv/ATTN/MLP/Head dynamic power is not separable from the current
  Vivado top-level power hierarchy. The available major-block power split is
  PS8, AXI/DMA/interconnect, and E2E accelerator IP.
- External DDR/DRAM rail power and sensor I/O power are not measured by the
  current Vivado report.

## Verification Commands

Commands run from repository root:

```sh
sed -n '25,90p' hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq2_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt
sed -n '25,90p' hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq2_300_mem16_track_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt
sed -n '25,95p' hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq2_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_controller_run_csynth.rpt
sed -n '25,95p' hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq2_300_mem16_track_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_controller_run_csynth.rpt
rg -n "CLB LUTs|CLB Registers|Block RAM Tile|URAM|DSPs|Total On-Chip Power|Dynamic|Device Static|PS Static|PL Static|axi_dma|axi_mem|hgtxr_e2e_axis_top_0|psu" hardware/generated/build/vivado/hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq2_300_mem16_*_overlay/*/impl_1/*_{utilization,power}_implemented.rpt
git diff --check -- hardware/docs/track/PROGRESS.md
git diff --no-index --check /dev/null hardware/docs/track/AQ2_SEARCH_TRACK_METRIC_CHECKLIST_2026_07_01.md
```

`git diff --check` passed for `PROGRESS.md`. The no-index check for this new
document printed no whitespace errors; exit code `1` is expected because
`/dev/null` and the new file differ.
