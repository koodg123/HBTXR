# AQ2 Requested Search/Track Metrics - 2026-07-01

## Scope

This document records the requested AQ2 Search/Track metric set from the current
HGTXR hardware artifacts. AQ2 here means the force-mode profiles whose generated
project names contain `attntok2_attnbram_patch32_mtok4_dtok4_aq2_300_mem16`.

Evidence roots:

- Search HLS:
  `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq2_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/`
- Track HLS:
  `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq2_300_mem16_track_only_no_board/solution_e2e_q4w8a/syn/report/`
- Search Vivado:
  `hardware/generated/build/vivado/hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq2_300_mem16_search_only_overlay/.../impl_1/`
- Track Vivado:
  `hardware/generated/build/vivado/hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq2_300_mem16_track_only_overlay/.../impl_1/`

All latency, bandwidth, and throughput values below are normalized to the
requested 300 MHz PL clock, i.e. 3.333 ns/cycle. Track HLS prints 0.933 ms
using its estimated 3.473 ns clock; the normalized value used here is
0.895700 ms.

## Sub-Agent Note

No real sub-agent was launched for this refresh. The current callable tool
surface does not expose a safe native sub-agent spawn for this user request, so
the main agent performed the extraction and documentation directly.

## Measurement Boundary

- HLS CSynth gives deterministic latency/interval estimates, not runtime sample
  histograms. Therefore mean/median/P95/P99 below are HLS-envelope proxies, not
  board-measured statistics.
- No AQ2 board-runtime JSON with repeated samples was found in the current
  workspace.
- Vivado power is vector-less routed `report_power` with medium confidence. It
  includes the PS8 model and PL on-chip estimates, but it does not measure
  external DDR rails or sensor I/O rails.
- Vivado power hierarchy separates PSU, AXI control, DMA, AXI memory, and the
  E2E accelerator IP. It does not split the E2E IP dynamic power into internal
  Conv/ATTN/MLP/Head blocks. For those internal blocks, HLS resource attribution
  is the available evidence.

## DMA Bandwidth

Assumptions:

- AXIS/AXI data width: 256 bit = 32 B/beat.
- Ideal bandwidth: `256 bit * 300 MHz / 8 = 9.600 GB/s`.
- Search input beats: `128 * 128 = 16,384`; output beats: `6`.
- Track input beats: `64 * 64 = 4,096`; output beats: `6`.
- Wire bytes count full 32 B AXIS beats.
- Useful bytes count the low-byte payload consumed by the current testbench.

| Metric | Search AQ2 | Track AQ2 |
|---|---:|---:|
| AXIS/AXI data width | 256 bit | 256 bit |
| Ideal DMA bandwidth | 9.600 GB/s | 9.600 GB/s |
| Wire bytes/inference | 524,480 B | 131,264 B |
| Effective DMA bandwidth, wire | 0.0920 GB/s | 0.1465 GB/s |
| Useful-payload bytes/inference | 16,390 B | 4,102 B |
| Effective DMA bandwidth, useful lower bound | 0.00288 GB/s | 0.00458 GB/s |

## Latency And Initiation Interval

| Metric | Search AQ2 | Track AQ2 |
|---|---:|---:|
| Latency min | 1,709,763 cycles / 5.699210 ms | 268,710 cycles / 0.895700 ms |
| Latency max | 1,709,783 cycles / 5.699277 ms | 268,710 cycles / 0.895700 ms |
| Mean latency | 5.699277 ms, HLS-envelope proxy | 0.895700 ms, HLS-envelope proxy |
| Median latency | 5.699277 ms, HLS-envelope proxy | 0.895700 ms, HLS-envelope proxy |
| P95 latency | 5.699277 ms, HLS-envelope proxy | 0.895700 ms, HLS-envelope proxy |
| P99 latency | 5.699277 ms, HLS-envelope proxy | 0.895700 ms, HLS-envelope proxy |
| Initial/initiation interval | 1,709,784 cycles / 5.699280 ms | 268,711 cycles / 0.895703 ms |
| Invocation rate from II | 175.46 inv/s | 1,116.44 inv/s |

Status against targets:

| Mode | Target | AQ2 status |
|---|---:|---|
| Search | <= 4.000 ms | fail, 5.699277 ms |
| Track | <= 1.000 ms | pass, 0.895700 ms |

## Throughput

Analytical operation model:

`ops = 2 * (blocks * (4*T*E^2 + 2*T*E*FF + 2*T^2*E) + T*E*PATCH^2 + STATE*E)`

Parameters: `E=192`, `FF=768`, `PATCH=16`, `STATE=6`. Search uses
`T=64, blocks=4`; Track uses `T=16, blocks=2`. One MAC is counted as two
operations.

| Metric | Search AQ2 | Track AQ2 |
|---|---:|---:|
| Analytical work | 0.245369 GOP | 0.030280 GOP |
| Throughput | 43.05 GOPS | 33.81 GOPS |

## HLS Latency Breakdown

| Major block | Search cycles | Search @300 MHz | Track cycles | Track @300 MHz |
|---|---:|---:|---:|---:|
| AXIS read frame | 16,386 | 54.620 us | 4,109 | 13.697 us |
| Conv/patch embedding | 258,052 frame conv | 0.860173 ms | 86,020 event conv | 0.286733 ms |
| Global buffer load | 12,292 | 40.973 us | 3,076 | 10.253 us |
| Controller run | 1,397,902-1,397,922 | 4.659673-4.659740 ms | 168,806 | 0.562687 ms |
| MLP head | 25,114 | 83.713 us | 6,682 | 22.273 us |
| Top end-to-end | 1,709,763-1,709,783 | 5.699210-5.699277 ms | 268,710 | 0.895700 ms |

Controller internal latency:

| Controller block | Search AQ2 | Track AQ2 |
|---|---:|---:|
| Weight dispatch/prefetch | 6,928-6,933 cycles | 6,926 cycles |
| Shared attention unit | 131,388 cycles | 25,257 cycles |
| Shared MLP unit | 211,151 cycles | 52,212 cycles |
| Dispatch loop | 27,724-27,744 cycles, trip 4 | 13,858 cycles, trip 2 |
| Body loop | 1,370,176 cycles, trip 4 | 154,946 cycles, trip 2 |

## HLS Resource Utilization By Major Block

ZCU104 HLS target resources: BRAM_18K 624, DSP 1,728, FF 460,800, LUT
230,400, URAM 96.

Search AQ2:

| Major block | BRAM_18K | DSP | FF | LUT | URAM |
|---|---:|---:|---:|---:|---:|
| AXIS read frame | 0 / 0.00% | 0 / 0.00% | 47 / 0.01% | 185 / 0.08% | 0 / 0.00% |
| Frame conv patch embedding | 0 / 0.00% | 30 / 1.74% | 1,907 / 0.41% | 5,171 / 2.24% | 0 / 0.00% |
| Global buffer load | 0 / 0.00% | 0 / 0.00% | 275 / 0.06% | 499 / 0.22% | 0 / 0.00% |
| Controller run | 229 / 36.70% | 1,258 / 72.80% | 128,591 / 27.91% | 246,125 / 106.82% | 92 / 95.83% |
| MLP head | 0 / 0.00% | 0 / 0.00% | 928 / 0.20% | 4,490 / 1.95% | 0 / 0.00% |
| Top local memories | 128 / 20.51% | 0 / 0.00% | 0 / 0.00% | 0 / 0.00% | 0 / 0.00% |
| Top total | 361 / 57.85% | 1,288 / 74.54% | 133,107 / 28.89% | 264,616 / 114.85% | 92 / 95.83% |

Search controller internal blocks:

| Controller block | BRAM_18K | DSP | FF | LUT | URAM |
|---|---:|---:|---:|---:|---:|
| Shared attention unit | 79 / 12.66% | 725 / 41.96% | 62,600 / 13.59% | 129,779 / 56.33% | 64 / 66.67% |
| Shared MLP unit | 86 / 13.78% | 405 / 23.44% | 53,539 / 11.62% | 90,289 / 39.19% | 0 / 0.00% |
| Weight dispatch/prefetch | 0 / 0.00% | 0 / 0.00% | 1,612 / 0.35% | 5,607 / 2.43% | 0 / 0.00% |
| Controller local memories | 64 / 10.26% | 0 / 0.00% | 0 / 0.00% | 0 / 0.00% | 28 / 29.17% |

Track AQ2:

| Major block | BRAM_18K | DSP | FF | LUT | URAM |
|---|---:|---:|---:|---:|---:|
| AXIS read frame | 0 / 0.00% | 0 / 0.00% | 429 / 0.09% | 409 / 0.18% | 0 / 0.00% |
| Event conv patch embedding | 0 / 0.00% | 34 / 1.97% | 2,893 / 0.63% | 9,930 / 4.31% | 0 / 0.00% |
| Global buffer load | 0 / 0.00% | 0 / 0.00% | 266 / 0.06% | 488 / 0.21% | 0 / 0.00% |
| Controller run | 225 / 36.06% | 1,378 / 79.75% | 139,308 / 30.23% | 236,547 / 102.67% | 71 / 73.96% |
| MLP head | 0 / 0.00% | 0 / 0.00% | 924 / 0.20% | 4,478 / 1.94% | 0 / 0.00% |
| Top local memories | 130 / 20.83% | 0 / 0.00% | 0 / 0.00% | 0 / 0.00% | 0 / 0.00% |
| Top total | 359 / 57.53% | 1,412 / 81.71% | 145,179 / 31.51% | 260,389 / 113.02% | 71 / 73.96% |

Track controller internal blocks:

| Controller block | BRAM_18K | DSP | FF | LUT | URAM |
|---|---:|---:|---:|---:|---:|
| Shared attention unit | 79 / 12.66% | 721 / 41.72% | 63,060 / 13.69% | 129,603 / 56.25% | 64 / 66.67% |
| Shared MLP unit | 82 / 13.14% | 529 / 30.61% | 63,992 / 13.89% | 84,723 / 36.77% | 0 / 0.00% |
| Weight dispatch/prefetch | 0 / 0.00% | 0 / 0.00% | 15 / 0.00% | 100 / 0.04% | 7 / 7.29% |
| Controller local memories | 64 / 10.26% | 0 / 0.00% | 0 / 0.00% | 0 / 0.00% | 0 / 0.00% |

## Implemented Vivado Resource Utilization

These values supersede HLS estimates for physical fit. HLS remains the
available major-block attribution source.

| Resource | Search implemented | Track implemented |
|---|---:|---:|
| CLB LUTs | 77,494 / 230,400 = 33.63% | 80,565 / 230,400 = 34.97% |
| LUT as Logic | 69,675 / 230,400 = 30.24% | 72,749 / 230,400 = 31.58% |
| LUT as Distributed RAM | 5,278 / 101,760 = 5.19% | 5,252 / 101,760 = 5.16% |
| CLB registers | 78,687 / 460,800 = 17.08% | 83,427 / 460,800 = 18.10% |
| Block RAM tile | 191.5 / 312 = 61.38% | 185.5 / 312 = 59.46% |
| URAM | 92 / 96 = 95.83% | 64 / 96 = 66.67% |
| DSP | 1,257 / 1,728 = 72.74% | 1,270 / 1,728 = 73.50% |

Implemented top-level hierarchy:

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

## Timing

| Metric | Search AQ2 | Track AQ2 |
|---|---:|---:|
| Bitstream | pass | pass |
| Final implemented WNS/TNS | 0.000 ns / 0.000 ns | -0.017 ns / -0.086 ns |
| Final implemented WHS/THS | 0.007 ns / 0.000 ns | 0.001 ns / 0.000 ns |
| Strict timing | pass | fail |
| User tolerance, WNS >= -0.5 ns | pass | pass |
| Routed intermediate WNS/TNS | 0.000 ns / 0.000 ns | -0.033 ns / -0.200 ns |
| Fully routed nets | 166,085 / 166,085 | 164,929 / 164,929 |
| Routing errors | 0 | 0 |

## Power Breakdown

Vivado vector-less routed power:

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

On-chip component power:

| Component | Search AQ2 | Track AQ2 |
|---|---:|---:|
| Clocks | 0.531 W | 0.542 W |
| CLB logic | 0.727 W | 0.840 W |
| Signals | 0.976 W | 0.835 W |
| Block RAM | 0.200 W | 0.191 W |
| URAM | 0.200 W | 0.128 W |
| DSPs | 0.998 W | 0.847 W |
| PS8 | 2.671 W | 2.671 W |

## Unavailable Without Additional Experiments

The following requested values cannot be honestly reported as measured AQ2 data
from the current artifacts:

- Board-measured mean/median/min/max/P95/P99 latency from repeated samples.
- Measured DMA bandwidth from board DMA counters.
- Mode-specific SAIF/VCD-annotated dynamic power.
- External DDR/DRAM rail power and sensor I/O rail power.
- Internal Conv/ATTN/MLP/Head dynamic power split inside the E2E IP.

Required next experiments:

1. Run the AQ2 Search and Track PYNQ smoke runners on ZCU104 with repeated
   samples and save JSON outputs.
2. Add board-side byte counters/timestamps around MM2S/S2MM DMA transactions.
3. Generate SAIF/VCD activity for Search and Track, then rerun Vivado
   `report_power` with activity annotation.
4. Preserve deeper E2E IP hierarchy or add separate power reporting hooks if
   Conv/ATTN/MLP/Head dynamic power must be reported directly.
