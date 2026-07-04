# C3b Resource, Power, and Latency Tables

Date: 2026-06-23

## 1. Evidence Scope

| Category | Evidence source | Status |
|---|---|---|
| ZCU104 resource utilization | `generated/signoff/e2e_resource_matrix_2026_06_10.{json,md}` and HLS csynth reports | available |
| Major block resource breakdown | `generated/hgtxr_e2e_axis_par16_c3b_mem16/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt` and `.autopilot/db/*.verbose.rpt` | available |
| Routed timing | `generated/build/vivado/hgtxr_e2e_axis_dma_c3b_mem16_overlay/.../impl_1/*timing_summary_routed.rpt` | available |
| Power | `generated/build/vivado/hgtxr_e2e_axis_dma_c3b_mem16_overlay/.../impl_1/*power_routed.rpt` | available, Vivado estimate only |
| Board runtime latency / p95 / p99 / invocation distribution | canonical C3b smoke JSON under `pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json`; measurement attempt in `docs/reports/c3b_board_measurement_attempt_2026_06_23.md` | instrumented, blocked by ZCU104 host resolution |

## 2. ZCU104 Resource Utilization

Resource authority: HLS `csynth.xml`/resource matrix. Denominators are ZCU104 target device resources from HLS report.

| Resource | Used | Available | Utilization |
|---|---:|---:|---:|
| BRAM_18K | 332 | 624 | 53.21% |
| DSP | 604 | 1,728 | 34.95% |
| FF | 59,505 | 460,800 | 12.91% |
| LUT | 126,506 | 230,400 | 54.91% |
| URAM | 64 | 96 | 66.67% |

## 3. Major Block Resource Breakdown

### 3.1 Top-Level HLS Blocks

| Major block | Latency cycles | BRAM_18K | BRAM % | DSP | DSP % | FF | FF % | LUT | LUT % | URAM | URAM % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hgtxr_axis_read_frame` | 65,538 | 0 | 0.00% | 0 | 0.00% | 19 | <0.01% | 87 | 0.04% | 0 | 0.00% |
| `hgtxr_conv_patch_embedding` | 87,208 | 0 | 0.00% | 0 | 0.00% | 83 | 0.02% | 1,290 | 0.56% | 0 | 0.00% |
| `hgtxr_global_buffer_load` | 49,154 | 0 | 0.00% | 0 | 0.00% | 53 | 0.01% | 264 | 0.11% | 0 | 0.00% |
| `hgtxr_e2e_controller_run` | 37,296,061 | 250 | 40.06% | 572 | 33.10% | 53,901 | 11.70% | 114,988 | 49.91% | 64 | 66.67% |
| `hgtxr_e2e_mlp_head` | 75,633 | 0 | 0.00% | 32 | 1.85% | 2,996 | 0.65% | 5,584 | 2.42% | 0 | 0.00% |
| `hgtxr_axis_write_state` | 8 | 0 | 0.00% | 0 | 0.00% | 7 | <0.01% | 81 | 0.04% | 0 | 0.00% |
| AXI/control interfaces | n/a | 34 | 5.45% | 0 | 0.00% | 2,233 | 0.48% | 2,203 | 0.96% | 0 | 0.00% |
| HLS top local memory bucket | n/a | 48 | 7.69% | 0 | 0.00% | 0 | 0.00% | 0 | 0.00% | 0 | 0.00% |
| HLS top total | 37,508,072 | 332 | 53.21% | 604 | 34.95% | 59,505 | 12.91% | 126,506 | 54.91% | 64 | 66.67% |

### 3.2 Controller Internal Breakdown

| Controller component | BRAM_18K | BRAM % | DSP | DSP % | FF | FF % | LUT | LUT % | URAM | URAM % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `attn_unit<0>` | 45 | 7.21% | 207 | 11.98% | 16,800 | 3.65% | 34,434 | 14.95% | 0 | 0.00% |
| `attn_unit<1>` | 45 | 7.21% | 207 | 11.98% | 16,800 | 3.65% | 34,434 | 14.95% | 0 | 0.00% |
| `mlp_unit<0>` | 0 | 0.00% | 79 | 4.57% | 10,140 | 2.20% | 16,998 | 7.38% | 0 | 0.00% |
| `mlp_unit<1>` | 0 | 0.00% | 79 | 4.57% | 10,140 | 2.20% | 16,998 | 7.38% | 0 | 0.00% |
| Controller memory bucket | 160 | 25.64% | 0 | 0.00% | 0 | 0.00% | 0 | 0.00% | 64 | 66.67% |
| Controller misc register/mux/expression | 0 | 0.00% | 0 | 0.00% | 21 | <0.01% | 12,124 | 5.26% | 0 | 0.00% |
| Controller total | 250 | 40.06% | 572 | 33.10% | 53,901 | 11.70% | 114,988 | 49.91% | 64 | 66.67% |

### 3.3 Logical Transformer Block Breakdown

One logical block is `attention + MLP`. The C3b controller executes 6 logical blocks but physically instantiates two attention units and two MLP units.

| Logical block component | Latency cycles | BRAM_18K | BRAM % | DSP | DSP % | FF | FF % | LUT | LUT % | URAM | URAM % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Attention unit | 2,247,180 | 45 | 7.21% | 207 | 11.98% | 16,800 | 3.65% | 34,434 | 14.95% | 0 | 0.00% |
| MLP unit | 3,968,826 | 0 | 0.00% | 79 | 4.57% | 10,140 | 2.20% | 16,998 | 7.38% | 0 | 0.00% |
| Logical block subtotal | 6,216,006 | 45 | 7.21% | 286 | 16.55% | 26,940 | 5.85% | 51,432 | 22.32% | 0 | 0.00% |

### 3.4 Attention/QKV/MLP Breakdown

| Block | Latency cycles | BRAM_18K | BRAM % | DSP | DSP % | FF | FF % | LUT | LUT % | URAM | URAM % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Attention LayerNorm | 117,619 | 0 | 0.00% | 15 | 0.87% | 2,750 | 0.60% | 3,160 | 1.37% | 0 | 0.00% |
| QKV projection | 453,359 | 45 | 7.21% | 96 | 5.56% | 5,914 | 1.28% | 12,715 | 5.52% | 0 | 0.00% |
| Attention core | 1,186,585 | 0 | 0.00% | 64 | 3.70% | 6,086 | 1.32% | 7,094 | 3.08% | 0 | 0.00% |
| Output projection | 489,609 | 0 | 0.00% | 32 | 1.85% | 1,883 | 0.41% | 8,070 | 3.50% | 0 | 0.00% |
| MLP LayerNorm | 117,619 | 0 | 0.00% | 15 | 0.87% | 2,750 | 0.60% | 3,160 | 1.37% | 0 | 0.00% |
| MLP W1/GELU pipeline | 204 | 0 | 0.00% | 32 | 1.85% | 2,783 | 0.60% | 2,941 | 1.28% | 0 | 0.00% |
| MLP W2 pipeline | 9,230 | 0 | 0.00% | 32 | 1.85% | 4,180 | 0.91% | 5,476 | 2.38% | 0 | 0.00% |
| Attention `score/prob` local memory | n/a | 0 | 0.00% | 0 | 0.00% | 32 | 0.01% | 50 | 0.02% | 0 | 0.00% |
| MLP LUTRAM table | n/a | 0 | 0.00% | 0 | 0.00% | 16 | <0.01% | 2 | <0.01% | 0 | 0.00% |

## 4. Power Breakdown

### 4.1 Measurement Methodology

| Item | Value |
|---|---|
| Method | Vivado `report_power` / Vivado Power Estimator style estimate |
| Report file | `generated/build/vivado/hgtxr_e2e_axis_dma_c3b_mem16_overlay/.../impl_1/hgtxr_e2e_axis_dma_c3b_mem16_system_wrapper_power_routed.rpt` |
| Design state | routed |
| Measurement type | estimate, not board power measurement |
| Activity source | no simulation activity file; internal node confidence is medium |
| Recommended paper wording | use PL-only value for accelerator claims; provide full on-chip system estimate separately |

### 4.2 PL-Only Power View

| Scope | Power W | Notes |
|---|---:|---|
| PL dynamic, resource-type total excluding PS8 | 0.137 | Clocks + CLB + signals + BRAM |
| PL static | 0.594 | Reported as `PL Static` |
| PL-only estimated on-chip total | 0.731 | PL dynamic + PL static |
| Accelerator IP dynamic hierarchy, `hgtxr_e2e_axis_top_0` | 0.040 | HGTXR kernel IP hierarchy only |
| AXI/control/DMA/interconnect dynamic hierarchy | 0.094 | `axi_ctrl + axi_dma_in + axi_dma_out + axi_mem` |

### 4.3 Full On-Chip Power Summary

| Component | Power W | Used | Available | Utilization |
|---|---:|---:|---:|---:|
| Clocks | 0.040 | 3 | n/a | n/a |
| CLB Logic | 0.059 | 32,204 | n/a | n/a |
| LUT as Distributed RAM | 0.036 | 1,204 | 101,760 | 1.18% |
| LUT as Logic | 0.019 | 9,416 | 230,400 | 4.09% |
| Register | 0.003 | 15,936 | 460,800 | 3.46% |
| Signals | 0.031 | 21,967 | n/a | n/a |
| Block RAM | 0.007 | 7 | 312 | 2.24% |
| PS8 | 2.647 | 1 | n/a | n/a |
| Device static | 0.692 | n/a | n/a | n/a |
| Total on-chip | 3.476 | n/a | n/a | n/a |

### 4.4 Dynamic Power By Hierarchy

| Hierarchy | Power W | Share of dynamic |
|---|---:|---:|
| `hgtxr_e2e_axis_dma_c3b_mem16_system_wrapper` | 2.784 | 100.00% |
| `axi_ctrl` | 0.007 | 0.25% |
| `axi_dma_in` | 0.002 | 0.07% |
| `axi_dma_out` | 0.007 | 0.25% |
| `axi_mem` | 0.078 | 2.80% |
| `hgtxr_e2e_axis_top_0` | 0.040 | 1.44% |
| `psu` | 2.651 | 95.22% |

### 4.5 Full System Coverage

| System item | Covered by current report? | Current value | Limitation |
|---|---|---:|---|
| PL accelerator IP | yes | 0.040 W dynamic | Vivado estimate only |
| PL shell/DMA/interconnect | yes | 0.094 W dynamic | Vivado estimate only |
| PL static | yes | 0.594 W | Vivado estimate only |
| PS8 | yes | 2.647 W dynamic, 0.099 W static | Vivado estimate only |
| PS DDR interface rails | partially | available in power-supply table | not equivalent to external DRAM chip power |
| External DRAM device power | no | n/a | board measurement required |
| Sensor I/O power | no | n/a | sensor/board measurement required |
| Full board power | no | n/a | rail or wall power measurement required |

## 5. Latency Breakdown

### 5.1 HLS Latency

| Scope | Latency cycles | Latency @ 200 MHz HLS target | Throughput estimate |
|---|---:|---:|---:|
| E2E top | 37,508,072 | 0.188 s | 5.33 frames/s |
| Controller only | 37,296,061 | 0.186 s | 5.36 controller runs/s |
| Logical block, module sum | 6,216,006 | 31.080 ms | n/a |
| Logical block, observed controller iteration | 6,216,010 | 31.080 ms | n/a |

### 5.2 Top-Level Stage Latency

| Stage | Latency cycles | Latency @ 200 MHz | Share of E2E top |
|---|---:|---:|---:|
| AXIS frame read | 65,538 | 0.328 ms | 0.17% |
| Conv patch embedding | 87,208 | 0.436 ms | 0.23% |
| Global buffer load | 49,154 | 0.246 ms | 0.13% |
| Transformer controller | 37,296,061 | 186.480 ms | 99.43% |
| MLP head | 75,633 | 0.378 ms | 0.20% |
| AXIS state write | 8 | <0.001 ms | <0.01% |

### 5.3 DMA / Batch / Runtime Metrics

| Metric | Current value | Evidence level | Notes |
|---|---:|---|---|
| Batch size | 1 frame/invocation | source/runtime wrapper | no batching loop in PYNQ smoke runner |
| Input DMA frame buffer allocation | 2,097,152 bytes | PYNQ wrapper | `256*256*8 uint32`; only low byte carries pixel value |
| Useful input image payload | 65,536 bytes | PYNQ wrapper | `256*256` one-byte pixels |
| Output DMA buffer allocation | 192 bytes | PYNQ wrapper | `6*8 uint32`; one 16-bit value used per state element |
| AXIS width | 256 bit | HLS config | `HGTXR_BUS_WIDTH=256` |
| Vivado PL clock constraint | 100 MHz | power report | `clk_pl_0` constraint 10 ns |
| Theoretical AXIS DMA peak @ 100 MHz | 3.2 GB/s | derived | not measured |
| HLS target clock | 200 MHz | HLS csynth | 5 ns target |
| Theoretical AXIS bandwidth @ 200 MHz | 6.4 GB/s | derived | only if clocked at HLS target |
| Measured DMA bandwidth | not captured | instrumented but blocked | next board run emits `dma_bandwidth_Bps_summary`; current SSH target `zcu104.local` does not resolve |

### 5.4 Mode-Specific / Worst-Case / Percentile Latency

| Metric | Current value | Status | Required next evidence |
|---|---|---|---|
| Mode-specific latency, search | not captured | runner supports mode label | run board smoke with `--mode-label search --repeat N` |
| Mode-specific latency, track | not captured | runner supports mode label | run board smoke with `--mode-label track --repeat N` |
| Worst-case latency | HLS min=max top latency: 37,508,072 cycles / 0.188 s | HLS-only | repeated board timing run |
| 95th percentile latency | not captured | instrumented but blocked | next board run emits `latency_ms_summary.p95` |
| 99th percentile latency | not captured | instrumented but blocked | next board run emits `latency_ms_summary.p99` |
| Throughput | 5.33 fps HLS estimate | HLS-only | board-measured end-to-end throughput |
| Search invocation distribution | not captured | runner-label only | C3b hardware does not expose internal search/track counters; runner emits `search_track_invocation_distribution` |
| Track invocation distribution | not captured | runner-label only | C3b hardware does not expose internal search/track counters; runner emits `search_track_invocation_distribution` |
| Current smoke runtime state | expected `2` | ready-for-board bundle only | canonical board result JSON missing |

## 6. Reporting Caveats

| Caveat | Impact |
|---|---|
| C3b canonical board-smoke result is missing | no board-measured latency, DMA bandwidth, percentiles, or invocation distribution can be claimed yet; latest attempt failed because `zcu104.local` did not resolve |
| Vivado power report has medium confidence for internal nodes | power numbers are estimates, not physical measurements |
| Vivado placed wrapper utilization shows shell-level resources and does not match HLS kernel resource accounting | use HLS resource matrix as resource authority for kernel utilization |
| Full system power is not measured | PS/PL on-chip estimates are available, but external DRAM/sensor/board rail power is not |
