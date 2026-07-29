# Eight-Question Hybrid Runtime Report - 2026-06-28

## Scope

This document answers the eight requested reviewer-style questions for the current
HGTXR hardware state, using the full learned-parameter AXI weight path profile.

Target objective:

- Single E2E AXIS cyclic accelerator.
- Shared ATTN/MLP compute blocks.
- Runtime Search/Track mode branch.
- ZCU104 target latency goals: Search <= 4 ms, Track <= 1 ms.
- Invocation mix assumption: Search 10%, Track 90%.

Current profile under analysis:

- HLS profile: `par32_runtime_full_axi_mem16`
- HLS project: `hardware/generated/hgtxr_e2e_axis_par32_runtime_full_axi_mem16_no_board`
- Vivado overlay project: `hardware/generated/build/vivado/hgtxr_e2e_axis_dma_par32_runtime_full_axi_mem16_overlay`
- PYNQ artifacts:
  - `hardware/pynq/hgtxr/hgtxr_e2e_axis_dma_par32_runtime_full_axi_mem16.bit`
  - `hardware/pynq/hgtxr/hgtxr_e2e_axis_dma_par32_runtime_full_axi_mem16.hwh`

## Evidence Sources

- HLS csynth:
  `hardware/generated/hgtxr_e2e_axis_par32_runtime_full_axi_mem16_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`
- Vivado routed timing:
  `hardware/generated/build/vivado/hgtxr_e2e_axis_dma_par32_runtime_full_axi_mem16_overlay/hgtxr_e2e_axis_dma_par32_runtime_full_axi_mem16_overlay.runs/impl_1/hgtxr_e2e_axis_dma_system_wrapper_timing_summary_routed.rpt`
- Vivado routed power:
  `hardware/generated/build/vivado/hgtxr_e2e_axis_dma_par32_runtime_full_axi_mem16_overlay/hgtxr_e2e_axis_dma_par32_runtime_full_axi_mem16_overlay.runs/impl_1/hgtxr_e2e_axis_dma_system_wrapper_power_routed.rpt`
- Vivado route status:
  `hardware/generated/build/vivado/hgtxr_e2e_axis_dma_par32_runtime_full_axi_mem16_overlay/hgtxr_e2e_axis_dma_par32_runtime_full_axi_mem16_overlay.runs/impl_1/hgtxr_e2e_axis_dma_system_wrapper_route_status.rpt`
- Vivado placed utilization:
  `hardware/generated/build/vivado/hgtxr_e2e_axis_dma_par32_runtime_full_axi_mem16_overlay/hgtxr_e2e_axis_dma_par32_runtime_full_axi_mem16_overlay.runs/impl_1/hgtxr_e2e_axis_dma_system_wrapper_utilization_placed.rpt`
- HWH interface map:
  `hardware/pynq/hgtxr/hgtxr_e2e_axis_dma_par32_runtime_full_axi_mem16.hwh`

## Additional Experiment Performed

### Experiment

Run the full AXI learned-weight path through the Vivado AXIS DMA overlay build.

Command:

```sh
timeout 7200 sh hardware/scripts/run/run_e2e_axis_dma_vivado_no_board.sh par32_runtime_full_axi_mem16
```

### Required Script Change

The Vivado no-board build script did not previously expose the full AXI profile.
The following profile was added to
`hardware/scripts/run/run_e2e_axis_dma_vivado_no_board.sh`:

```sh
par32_runtime_full_axi_mem16)
  PROJECT_NAME="hgtxr_e2e_axis_dma_par32_runtime_full_axi_mem16_overlay"
  ARTIFACT_NAME="hgtxr_e2e_axis_dma_par32_runtime_full_axi_mem16"
  HLS_IP_REPO="generated/hgtxr_e2e_axis_par32_runtime_full_axi_mem16_no_board/solution_e2e_q4w8a/impl/ip"
  ;;
```

### Result

The Vivado build completed successfully:

- HWH generated and copied.
- Bitstream generated and copied.
- Route status: 0 routing errors.
- Timing constraints met.
- PYNQ artifacts available under `hardware/pynq/hgtxr/`.

## Current Measured And Estimated Numbers

### HLS Latency

HLS target clock: 5.00 ns, 200 MHz.

| Mode interpretation | Cycles | Latency | Interval cycles | Effective rate |
|---|---:|---:|---:|---:|
| Track-bound min | 627,117 | 3.136 ms | 627,118 | 318.9 events/s |
| Search-bound max | 5,101,975 | 25.510 ms | 5,101,976 | 39.2 events/s |
| 90% Track / 10% Search expected | 1,074,603 | 5.373 ms | n/a | 186.1 events/s |

Formula:

```text
Hybrid expected latency =
  0.9 * 3.136 ms + 0.1 * 25.510 ms
= 5.373 ms
```

The current full learned-weight path does not meet the target latencies:

- Track target 1 ms: current HLS estimate is 3.136 ms.
- Search target 4 ms: current HLS estimate is 25.510 ms.

### Vivado Timing And Route

Routed wrapper clock:

- `clk_pl_0`: 10.000 ns, 100.000 MHz.

Routed timing:

- WNS: 4.741 ns.
- TNS: 0.000 ns.
- Setup failing endpoints: 0.
- WHS: 0.010 ns.
- THS: 0.000 ns.
- Hold failing endpoints: 0.

Route status:

- Logical nets: 54,235.
- Routable nets: 21,125.
- Fully routed nets: 21,125.
- Routing errors: 0.

Important clock caveat:

- HLS latency was estimated at 200 MHz.
- Vivado wrapper currently constrains PL clock at 100 MHz.
- If the deployed accelerator is driven at 100 MHz without changing the clock,
  HLS cycle-derived latency doubles approximately:
  - Track: about 6.271 ms.
  - Search: about 51.020 ms.

### Vivado Routed Power

Vivado routed `report_power` result:

| Power item | Value |
|---|---:|
| Total on-chip power | 3.459 W |
| Dynamic power | 2.767 W |
| Device static power | 0.692 W |
| PS static | 0.099 W |
| PL static | 0.594 W |
| Confidence | Medium |

Dynamic power by hierarchy:

| Hierarchy item | Power |
|---|---:|
| `psu` | 2.650 W |
| `axi_mem` | 0.071 W |
| `hgtxr_e2e_axis_top_0` | 0.030 W |
| `axi_ctrl` | 0.007 W |
| `axi_dma_in` | 0.002 W |
| `axi_dma_out` | 0.007 W |

PL dynamic by component class:

| Component | Power |
|---|---:|
| Clocks | 0.038 W |
| CLB logic | 0.053 W |
| Signals | 0.025 W |
| Block RAM | 0.004 W |
| DSPs | <0.001 W |
| Approximate PL dynamic subtotal excluding PS8 | about 0.120 W |

DDR-related rail estimate from the power supply table:

| Rail | Total | Dynamic | Static |
|---|---:|---:|---:|
| `VCC_PSINTFP_DDR` | about 0.611 W | about 0.607 W | about 0.003 W |
| `VCCO_PSDDR_504` | about 0.744 W | about 0.703 W | about 0.041 W |
| `VCC_PSDDR_PLL` | about 0.002 W | 0.000 W | about 0.002 W |
| DDR-related subtotal | about 1.357 W | about 1.310 W | about 0.046 W |

DDR caveat:

- This is a Vivado rail estimate for the PS DDR interface related supplies.
- It is not a physical board measurement of external DDR memory chips.
- It is not a sensor I/O power measurement.

### Resource Utilization

#### Vivado routed/placed wrapper utilization

This is the implementation-level report for the DMA overlay wrapper:

| Resource | Used | Available | Utilization |
|---|---:|---:|---:|
| CLB LUTs | 10,442 | 230,400 | 4.53% |
| CLB registers | 15,195 | 460,800 | 3.30% |
| Block RAM tile | 6 | 312 | 1.92% |
| URAM | 0 | 96 | 0.00% |
| DSP | 1 | 1,728 | 0.06% |

#### Vivado OOC HLS IP synthesis utilization

This is the synthesized HLS IP instance inside the Vivado project:

| Resource | Used | Available | Utilization |
|---|---:|---:|---:|
| CLB LUTs | 3,768 | 230,400 | 1.64% |
| CLB registers | 5,068 | 460,800 | 1.10% |
| Block RAM tile | 1 | 312 | 0.32% |
| URAM | 0 | 96 | 0.00% |
| DSP | 1 | 1,728 | 0.06% |

The HLS IP synthesis report has no black boxes listed, so this is a real
Vivado synthesis result, not a black-box stub.

#### HLS csynth resource estimate

This is the HLS pre-implementation estimate. It is much higher than Vivado
OOC/placed utilization and should be treated as a conservative HLS estimate,
not the final implemented resource number.

| Resource | Used | Available | Utilization |
|---|---:|---:|---:|
| BRAM_18K | 332 | 624 | 53% |
| DSP | 1,148 | 1,728 | 66% |
| FF | 86,123 | 460,800 | 18% |
| LUT | 193,627 | 230,400 | 84% |
| URAM | 64 | 96 | 66% |

HLS major block estimate:

| Block | BRAM_18K | DSP | FF | LUT | URAM |
|---|---:|---:|---:|---:|---:|
| `gmem_e2e_runtime_m_axi` | 4 | 0 | 818 | 678 | 0 |
| `gmem_e2e_weights_m_axi` | 30 | 0 | 1,201 | 1,165 | 0 |
| `hgtxr_conv_patch_embedding` | 0 | 0 | 75 | 1,294 | 0 |
| `hgtxr_e2e_controller_run` | 250 | 1,084 | 77,625 | 177,328 | 64 |
| `hgtxr_e2e_mlp_head` | 0 | 64 | 5,934 | 10,226 | 0 |

### Interface And Memory Path

The HWH confirms:

- AXIS input: `axis_in`, 256-bit `TDATA`.
- AXIS output: `axis_out`, 256-bit `TDATA`.
- Weight AXI master: `m_axi_gmem_e2e_weights`, 256-bit data.
- Runtime AXI master: `m_axi_gmem_e2e_runtime`, 32-bit data.
- Both weight and runtime AXI masters map to PS DDR through
  `S_AXI_HPC0_FPD` / `HPC0_DDR_LOW`.

This means the selected parameter path is external AXI/DDR, not persistent
ROM/BRAM/URAM-resident parameters.

## Eight Questions

### 1. HW Top 전체 자원 사용량, Hybrid Inference Latency, Hybrid Inference Power Breakdown

#### Question

Search 10%, Track 90% 호출 분포에서 hybrid HW top의 전체 자원 사용량,
hybrid inference latency, hybrid inference power breakdown을 보고한다.

#### Plan

1. Use Vivado routed/placed utilization as the implementation-level top result.
2. Keep HLS csynth resource estimates as a separate conservative reference.
3. Compute hybrid expected latency from HLS Search/Track estimates.
4. Use Vivado routed power for PS/PL/AXI/DMA/IP dynamic breakdown.
5. Mark board-only metrics as pending.

#### Progress

- HLS CSim/csynth completed.
- Full AXI learned-weight IP packaged.
- Vivado DMA overlay route/bitgen completed.
- Power/timing/utilization/HWH reports parsed.

#### Results

Implementation-level wrapper utilization:

- CLB LUTs: 10,442 / 230,400 = 4.53%.
- CLB registers: 15,195 / 460,800 = 3.30%.
- Block RAM tile: 6 / 312 = 1.92%.
- URAM: 0 / 96 = 0.00%.
- DSP: 1 / 1,728 = 0.06%.

HLS conservative estimate:

- BRAM_18K: 332 / 624 = 53%.
- DSP: 1,148 / 1,728 = 66%.
- FF: 86,123 / 460,800 = 18%.
- LUT: 193,627 / 230,400 = 84%.
- URAM: 64 / 96 = 66%.

Hybrid expected latency at HLS 200 MHz:

- 5.373 ms for 90% Track / 10% Search.

Hybrid worst-case latency:

- 25.510 ms, because a Search invocation is the worst case.

Hybrid power breakdown from routed Vivado report:

- Total on-chip: 3.459 W.
- Dynamic: 2.767 W.
- Static: 0.692 W.
- PS dynamic hierarchy: 2.650 W.
- Approximate PL dynamic excluding PS8: about 0.120 W.
- AXI memory interconnect dynamic: 0.071 W.
- DMA dynamic: 0.009 W total (`axi_dma_in` 0.002 W + `axi_dma_out` 0.007 W).
- HLS IP dynamic hierarchy: 0.030 W.

Remaining gap:

- This is not mode-specific measured power. It is one routed design estimate
  without mode-specific switching activity.

### 2. Search Mode Inference Latency And Power Breakdown

#### Question

Search Mode 단독 실행 시 inference latency와 power breakdown을 보고한다.

#### Plan

1. Use HLS max latency as Search-bound latency.
2. Use Vivado power as the current design-level power estimate.
3. Generate mode-specific switching activity later for true Search power.

#### Progress

- Search-bound HLS latency is available.
- Search-specific board or SAIF/VCD power is not available yet.

#### Results

Search latency at HLS 200 MHz:

- 5,101,975 cycles.
- 25.510 ms.
- Interval: 5,101,976 cycles.
- Search-only max rate: about 39.2 events/s.

If the routed wrapper PL clock remains 100 MHz:

- Search latency is approximately 51.020 ms.

Search power:

- Current available estimate is the same design-level routed estimate:
  total 3.459 W, dynamic 2.767 W, static 0.692 W.
- Search-specific dynamic power cannot be separated without Search-only
  switching activity or board power measurement.

Target status:

- Search target is <= 4 ms.
- Current Search HLS estimate is 25.510 ms, so the target is not met.

### 3. Search Mode Active Block And Path

#### Question

Search Mode에서 활성화되는 block과 data/control path를 명확히 보고한다.

#### Plan

1. Use HLS instance latency table for logical block path.
2. Use HWH for AXIS/AXI/DDR connection evidence.
3. Separate shared compute blocks from mode-specific runtime control.

#### Progress

- HLS logical stages identified.
- HWH confirms AXIS input/output and AXI weight/runtime paths.

#### Results

Search active path:

1. PS DDR input buffer.
2. AXI DMA MM2S.
3. 256-bit `axis_in`.
4. `hgtxr_axis_read_frame`.
5. `hgtxr_conv_patch_embedding`.
6. `hgtxr_global_buffer_load`.
7. Shared cyclic controller: `hgtxr_e2e_controller_run`.
8. Shared ATTN/MLP units inside the controller.
9. `hgtxr_e2e_mlp_head`.
10. 256-bit `axis_out`.
11. AXI DMA S2MM.
12. PS DDR output buffer.

Parameter and state paths:

- Learned weights are read through `m_axi_gmem_e2e_weights`, 256-bit AXI.
- Runtime state is accessed through `m_axi_gmem_e2e_runtime`, 32-bit AXI.
- Both connect to PS DDR through `S_AXI_HPC0_FPD`.

Search uses the same physical ATTN/MLP blocks as Track. The difference is the
runtime mode profile and active sequence/depth bounds, not a separate Search
accelerator instance.

### 4. Track Mode Inference Latency And Power Breakdown

#### Question

Track Mode 단독 실행 시 inference latency와 power breakdown을 보고한다.

#### Plan

1. Use HLS min latency as Track-bound latency.
2. Use Vivado power as current design-level power estimate.
3. Generate mode-specific switching activity later for true Track power.

#### Progress

- Track-bound HLS latency is available.
- Track-specific board or SAIF/VCD power is not available yet.

#### Results

Track latency at HLS 200 MHz:

- 627,117 cycles.
- 3.136 ms.
- Interval: 627,118 cycles.
- Track-only max rate: about 318.9 events/s.

If the routed wrapper PL clock remains 100 MHz:

- Track latency is approximately 6.271 ms.

Track power:

- Current available estimate is the same design-level routed estimate:
  total 3.459 W, dynamic 2.767 W, static 0.692 W.
- Track-specific dynamic power cannot be separated without Track-only
  switching activity or board power measurement.

Target status:

- Track target is <= 1 ms.
- Current Track HLS estimate is 3.136 ms, so the target is not met.

### 5. Track Mode Active Block And Path

#### Question

Track Mode에서 활성화되는 block과 data/control path를 명확히 보고한다.

#### Plan

1. Use the same shared physical top path as Search.
2. Mark Track-specific behavior as runtime mode selection.
3. Confirm that the weight/runtime AXI path is still shared.

#### Progress

- HLS CSim validates runtime state output differs for Search and Track.
- HWH confirms one shared IP instance, not separate Search/Track IPs.

#### Results

Track active path:

1. PS DDR input buffer.
2. AXI DMA MM2S.
3. 256-bit `axis_in`.
4. `hgtxr_axis_read_frame`.
5. `hgtxr_conv_patch_embedding`.
6. `hgtxr_global_buffer_load`.
7. Shared cyclic controller: `hgtxr_e2e_controller_run`.
8. Shared ATTN/MLP units inside the controller.
9. `hgtxr_e2e_mlp_head`.
10. 256-bit `axis_out`.
11. AXI DMA S2MM.
12. PS DDR output buffer.

Track uses the same physical ATTN/MLP/Head path as Search, with runtime mode
selecting the Track profile and shorter active work bound.

### 6. Search/Track Only vs Hybrid Throughput, II, Max Frame/Event Rate, Worst-Case Latency

#### Question

Search-only, Track-only, Hybrid 각각의 처리량, initiation interval, 최대
frame/event rate, worst-case latency를 보고한다.

#### Plan

1. Use HLS interval cycles as initiation interval between transactions.
2. Convert HLS cycles to event rates at 200 MHz.
3. Compute hybrid expected rate from weighted expected latency.
4. Report worst-case as Search latency.

#### Progress

- HLS interval values are available.
- Board DMA sustained throughput is not measured yet.

#### Results

At HLS 200 MHz:

| Case | Latency | Interval cycles | Max rate | Worst-case latency |
|---|---:|---:|---:|---:|
| Track-only | 3.136 ms | 627,118 | 318.9 events/s | 3.136 ms |
| Search-only | 25.510 ms | 5,101,976 | 39.2 events/s | 25.510 ms |
| Hybrid 90/10 expected | 5.373 ms | n/a | 186.1 events/s | 25.510 ms |

The top is not transaction-pipelined in the HLS report, so interval is
approximately one full transaction latency plus one cycle.

At routed wrapper 100 MHz, cycle-derived rates would be roughly half of the
200 MHz HLS rates unless the PL clock is increased or the latency is remeasured
on board.

### 7. Power Measurement Methodology And Runtime Measurement Metadata

#### Question

Power measurement methodology, PS/PL/DRAM/sensor I/O inclusion, DMA bandwidth,
batch size, throughput, mode-specific latency, worst-case latency, p95/p99,
and Search/Track invocation distribution을 제공한다.

#### Plan

1. State the current methodology honestly.
2. Separate Vivado estimate from board measurement.
3. State inclusion/exclusion boundaries.
4. Provide computed p95/p99 only as an analytic estimate for the assumed
   two-point distribution, not as measured percentiles.
5. Define the additional board experiment required for final numbers.

#### Progress

- Vivado routed power estimate is available.
- No SAIF/VCD mode-specific activity is available.
- No ZCU104 rail measurement is available.
- No board-side latency distribution log is available.

#### Results

Current methodology:

- Power source: Vivado 2023.2 `report_power` on routed design.
- Confidence: Medium.
- Switching activity: no simulation activity file; internal node confidence is
  Medium because fewer than 25% of internal nodes have user-specified activity.
- Therefore current power is an implementation estimate, not physical
  measurement.

Inclusion status:

| Component | Current status |
|---|---|
| PL fabric | Included in Vivado on-chip estimate |
| PS | Included as `PS8` / `psu` estimate |
| PS DDR interface rails | Included as DDR-related rail estimates |
| External DDR memory chips | Not physically measured |
| Sensor I/O | Not included |
| Board regulators/fans/peripherals | Not included |

Runtime metadata:

- Batch size: 1 transaction/event per accelerator invocation.
- Distribution assumption: Search 10%, Track 90%.
- Mode-specific latency: HLS Search 25.510 ms, HLS Track 3.136 ms at 200 MHz.
- Worst-case latency: 25.510 ms at 200 MHz.
- p95/p99 under a deterministic two-point 90% Track / 10% Search model:
  - p95 = Search = 25.510 ms.
  - p99 = Search = 25.510 ms.
- p95/p99 are not measured yet.

DMA bandwidth:

- AXIS data width is 256 bits.
- Weight AXI data width is 256 bits.
- Runtime AXI data width is 32 bits.
- The HWH confirms the masters use PS DDR through `S_AXI_HPC0_FPD`.
- Sustained DMA bandwidth has not been measured on ZCU104 yet.

Additional experiment needed:

- Run board-side Search/Track loop on ZCU104.
- Timestamp each invocation in software and/or PL counters.
- Log at least thousands of invocations with the 90/10 distribution.
- Measure rails with PMBus/INA/power monitor if available.
- Produce measured throughput, bandwidth, p50/p95/p99, and mode-specific power.

### 8. Resource Utilization Absolute And Percentage, Major Block Breakdown

#### Question

ZCU104 device 기준 resource utilization을 absolute value와 percentage 모두로
보고하고, major block별로 분해한다.

#### Plan

1. Report implementation-level Vivado placed utilization.
2. Report Vivado OOC HLS IP synthesis utilization.
3. Report HLS major block estimates separately.
4. Do not mix HLS estimates with Vivado implemented utilization as if they were
   the same measurement.

#### Progress

- Vivado wrapper utilization parsed.
- Vivado OOC HLS IP utilization parsed.
- HLS major block resource estimates parsed.

#### Results

Implementation-level overlay wrapper:

| Resource | Used | Available | Utilization |
|---|---:|---:|---:|
| CLB LUTs | 10,442 | 230,400 | 4.53% |
| CLB registers | 15,195 | 460,800 | 3.30% |
| Block RAM tile | 6 | 312 | 1.92% |
| URAM | 0 | 96 | 0.00% |
| DSP | 1 | 1,728 | 0.06% |

Vivado OOC HLS IP:

| Resource | Used | Available | Utilization |
|---|---:|---:|---:|
| CLB LUTs | 3,768 | 230,400 | 1.64% |
| CLB registers | 5,068 | 460,800 | 1.10% |
| Block RAM tile | 1 | 312 | 0.32% |
| URAM | 0 | 96 | 0.00% |
| DSP | 1 | 1,728 | 0.06% |

HLS major block estimate:

| Block | BRAM_18K | DSP | FF | LUT | URAM |
|---|---:|---:|---:|---:|---:|
| AXI runtime port | 4 | 0 | 818 | 678 | 0 |
| AXI weight port | 30 | 0 | 1,201 | 1,165 | 0 |
| Conv patch embedding | 0 | 0 | 75 | 1,294 | 0 |
| Shared E2E controller / ATTN / MLP | 250 | 1,084 | 77,625 | 177,328 | 64 |
| MLP head | 0 | 64 | 5,934 | 10,226 | 0 |

Interpretation:

- Vivado implementation utilization is the current implementation answer.
- HLS major-block estimates are useful for architectural attribution but are not
  final placed utilization.
- The large gap between HLS estimate and Vivado synthesis should be treated as
  an optimization/reporting discrepancy that needs follow-up if the paper needs
  one canonical resource table.

## Answerability Matrix

| Requested item | Current answer status | Additional experiment needed |
|---|---|---|
| Full top resource | Answered with Vivado placed utilization and HLS estimate | Optional reconciliation of HLS vs Vivado utilization |
| Hybrid latency | Answered analytically from HLS Search/Track | Board measurement needed for final |
| Hybrid power | Partially answered with routed estimate | Mode-mix activity or board rail measurement needed |
| Search latency | Answered with HLS max latency | Board measurement needed |
| Search power | Partially answered with design-level estimate | Search-only activity/power needed |
| Search active path | Answered | No |
| Track latency | Answered with HLS min latency | Board measurement needed |
| Track power | Partially answered with design-level estimate | Track-only activity/power needed |
| Track active path | Answered | No |
| Throughput/II/rate/worst-case | Answered from HLS cycles | Board DMA sustained throughput needed |
| Methodology/inclusion | Answered | Board measurement needed for final physical power |
| DMA bandwidth | Interface width answered; sustained bandwidth pending | ZCU104 DMA benchmark needed |
| p95/p99 latency | Analytic two-point estimate only | Runtime latency log needed |
| Major block resource breakdown | HLS estimate answered; Vivado major-block implementation pending | Hierarchical implemented utilization extraction or manual preservation needed |

## Concrete Next Experiments

1. Board latency sweep:
   - Load `hgtxr_e2e_axis_dma_par32_runtime_full_axi_mem16.bit`.
   - Run Search-only, Track-only, and 90/10 hybrid invocation loops.
   - Log per-call latency, mode, input bytes, output bytes, and DMA transfer time.

2. Board bandwidth benchmark:
   - Measure MM2S/S2MM payload throughput separately.
   - Measure full accelerator end-to-end payload throughput.
   - Record PS DDR buffer placement and cache maintenance policy.

3. Mode-specific power:
   - Search-only run: record idle, DMA-only, and full accelerator power.
   - Track-only run: record idle, DMA-only, and full accelerator power.
   - Hybrid 90/10 run: record average and peak.

4. Activity-based Vivado power:
   - Generate VCD/SAIF for Search-only and Track-only RTL/co-sim when feasible.
   - Re-run `report_power` with activity annotation.
   - Compare with board rail measurements.

5. Utilization reconciliation:
   - Decide whether the paper table should use Vivado implemented utilization
     or HLS architectural estimate.
   - If major-block implemented utilization is required, preserve hierarchy or
     generate hierarchical utilization before optimization removes attribution.

## Current Bottom Line

The full learned-parameter AXI path now builds through HLS, packages IP, and
routes/bitstreams as a ZCU104-style AXIS DMA overlay. Functionally and
physically, the profile is usable for the next ZCU104 board experiment.

However, the current latency target is not met:

- Track is 3.136 ms at HLS 200 MHz, above the 1 ms target.
- Search is 25.510 ms at HLS 200 MHz, above the 4 ms target.
- Hybrid 90/10 expected latency is 5.373 ms at HLS 200 MHz.

The current power answer is a routed Vivado estimate, not a measured
mode-specific board power result.
