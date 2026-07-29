# HGTXR Design and Experiment Integration

Date: 2026-06-26

## 1. Integration Goal

Use the integrated reference analysis to define bounded HGTXR experiments without mixing static reference claims with measured HGTXR results. The immediate hardware objective remains:

- Increase DSP utilization in QKV, attention, and MLP compute.
- Increase URAM utilization for deep FIFO, large activation, score, and value buffers.
- Use LUTRAM only for small memories, routing tables, and calibrated lookup tables.
- Reduce LUT pressure by avoiding LUT-implemented arithmetic unless explicitly measured and justified.

## 2. Experiment Queue

| ID | Path | Source basis | Work description | Expected result | Required gate |
|---|---|---|---|---|---|
| HXR-R0 | Resource rebalance | `HG-PIPE`, `MSD-FCCM23`, `efficient-transformer-accelerator` | Increase QKV/MLP parallel lanes; force MAC-heavy paths toward DSP | Higher DSP%, lower LUT arithmetic pressure | csynth resource breakdown by RMU/SMU/MHA/MLP |
| HXR-R1 | Memory binding | `AURA`, `TATAA`, `HG-PIPE` | Bind large buffers/deep FIFOs to URAM, mid buffers to BRAM, tiny tables to LUTRAM | Higher URAM utilization and cleaner LUT usage | HLS storage binding report and block-level utilization |
| HXR-Q0 | Quantization calibration | `P2-ViT`, `FQ-ViT`, `PTQ4ViT`, HLS4ML/mase | Export weights, fixed-point scales, calibration metadata, and bit-accurate checks | Traceable quantized inference path | pre/post quant accuracy and bit-accurate simulation |
| HXR-A0 | Attention exact baseline hardening | `HG-PIPE`, `TATAA`, `Transformer-Accelerator-Based-on-FPGA` | Keep exact attention path as reference while refactoring block reporting | Stable baseline for approximation comparison | csim/csynth match against current reference |
| HXR-A1 | Optional attention approximation | `ViTALiTy`, `ViTCoD`, `AURA` | Add sparse/linear/tiled attention as a compile-time variant | Possible latency/memory reduction | approximation error, accuracy degradation, fallback proof |
| HXR-M0 | Search/track mode distribution | `M3ViT`, `Edge-MoE`, `UbiMoE` | Add software/HLS counters for mode invocation and worst-case route | Average latency analysis grounded in real distribution | mode counts, p95/p99 latency, worst-case latency |
| HXR-T0 | Report automation | `Prometheus`, `FlexCNN`, `Sextans` | Normalize HLS/Vivado utilization, latency, and power reports | Faster experiment comparison and fewer manual errors | generated summary tables from real reports |

## 3. Priority and Sequencing

| Priority | Experiments | Reason |
|---|---|---|
| P0 | HXR-R0, HXR-R1, HXR-T0 | Directly addresses current DSP/URAM/LUT imbalance and improves evidence quality |
| P1 | HXR-Q0, HXR-A0, HXR-M0 | Needed for quantization and mode-specific reporting before new performance claims |
| P2 | HXR-A1 | Useful but changes algorithmic behavior; requires explicit approximation/accuracy evidence |

## 4. Block-Level Reporting Template

| Block | Metrics to report | Source inspiration | Notes |
|---|---|---|---|
| RMU | LUT, FF, DSP, BRAM, URAM, latency, II | Current HGTXR block reporting | Include routing/control memory split |
| SMU | LUT, FF, DSP, BRAM, URAM, latency, II | Current HGTXR block reporting | Distinguish scalar tables from large buffers |
| MHA/QKV | LUT, FF, DSP, BRAM, URAM, QK/softmax/RV latency | HG-PIPE, TATAA, AURA | QKV/MLP are first targets for DSP parallelism |
| MLP | LUT, FF, DSP, BRAM, URAM, MAC lane count | HG-PIPE, MSD-FCCM23, efficient-transformer-accelerator | Avoid LUT MAC expansion |
| DMA/AXI | bandwidth, batch size, transfer bytes, overlap policy | REMOT-FPGA-22, ViT-FPGA-TPU, Sextans | Only claim measured values when board/host data exists |

## 5. Measurement Contract

| Measurement | Static analysis status | Required evidence before claim |
|---|---|---|
| Resource utilization | Can be planned from references | HLS/Vivado reports for HGTXR variant |
| Throughput/FPS | Not claimable from references | cycles, frequency, batch size, end-to-end path definition |
| p95/p99 latency | Not claimable from deterministic csynth alone | repeated simulation or board/host measurements |
| DMA bandwidth | Not claimable without transfer traces | measured bytes/time or board-side counters |
| Accuracy/error distribution | Not claimable without HGTXR dataset run | mean/median/p95, subject split, motion split, CI if repeated |
| Quantization degradation | Not claimable without float-vs-quant run | same dataset, same checkpoint, scale manifest |

## 6. Concrete Next Actions

| Step | Action | Output |
|---|---|---|
| 1 | Add automated report collector for current generated variants | block-level resource CSV/Markdown |
| 2 | Create `par16/par32` HLS variants with explicit DSP binding | csynth reports and resource matrix |
| 3 | Add URAM/LUTRAM binding variants for buffer classes | memory binding comparison |
| 4 | Add quantization export manifest | weights/scales/calibration trace |
| 5 | Add mode counters before MoE/routing changes | search/track distribution evidence |

