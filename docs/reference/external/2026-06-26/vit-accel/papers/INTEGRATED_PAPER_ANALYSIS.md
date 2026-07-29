# Integrated Paper Analysis: Existing ViT-Accel Reports

Source: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/papers`

## 1. Scope

This document integrates the existing paper analyses into a hardware-planning view. Each underlying paper report still contains the detailed sections for problem, proposed method, algorithm, hardware architecture, experiment setup, datasets, results, options, and HGTXR applicability.

## 2. Paper-to-Design Map

| Paper/report | Problem addressed | Proposed technique class | HGTXR use | Priority |
|---|---|---|---|---|
| `HG-PIPE` | ViT pipeline imbalance and buffer overhead | Hybrid-grained pipeline, operator-level partitioning | C-path HLS hygiene, attention/MLP split, FIFO policy | P0 |
| `P2-ViT` | Quantized ViT scale cost and hardware unfriendly arithmetic | PoT/fixed-point quantization | P0 quantization and shift-friendly requantization | P0 |
| `EfficientViT-FPGA` | Hybrid conv-transformer accelerators underutilize EfficientViT structure | Reconfigurable architecture and time-multiplexed pipeline | Conv/local-feature ablation, MAT-style DSP mapping | P2 |
| `ViTCoD` | Dense attention wastes work in ViT | Sparse attention co-design | Attention sparsity ablation after exact baseline | P1/P2 |
| `ViTALiTy` | Softmax attention and dense ViT dataflow cost | Linear/Taylor attention, hardware dataflow | Approximation study only with explicit error reporting | P1/P2 |
| `TATAA` | Attention data movement and softmax overhead | Token/attention transformation architecture | QK/softmax/RV block comparison | P1 |
| `Edge-MoE`, `M3ViT`, `UbiMoE`, `CoQMoE` | Conditional compute and expert routing cost | Sparse expert routing and co-design | Search/track mode-aware execution experiments | P1 |
| `AHCPTQ`, `LUT-GEMM`, `LUT-LLM` | Nonlinear/low-bit approximation cost | LUT or approximate arithmetic | Negative/limited controls under LUT policy | P2 |
| `FlexLLM`, `FlightLLM`, `TerEffic` | Large-model memory and scheduling bottlenecks | Memory hierarchy and scheduling | Indirect memory/buffer scheduling references | P2 |

## 3. Existing Method Problems Across Papers

| Problem class | Papers | HGTXR translation |
|---|---|---|
| Dense attention memory traffic | ViTCoD, ViTALiTy, TATAA, AURA-related analyses | Prioritize QK/softmax/RV buffering and stream-width balance |
| Low DSP utilization from conservative parallelism | HG-PIPE, EfficientViT-FPGA, systolic/integer reports | Increase QKV/MLP parallel lanes with DSP binding evidence |
| LUT pressure from nonlinear or approximate operators | AHCPTQ, LUT-GEMM, LUT-LLM | Restrict LUT-based methods to calibrated small tables |
| DDR traffic and repeated activation loads | ME-ViT-style memory findings, HG-PIPE | Use URAM-backed large buffers and double buffering |
| Dynamic routing overhead | Edge-MoE, M3ViT, UbiMoE | Measure search/track invocation distribution before hardware routing |

## 4. Experiment Options to Carry Forward

| Option axis | Values | Required measurements |
|---|---|---|
| Parallelism factor | 8, 16, 32, resource-constrained variants | DSP/LUT/FF/BRAM/URAM, II, latency |
| Memory binding | URAM large buffers, BRAM mid buffers, LUTRAM small tables | Utilization by block and timing |
| Attention mode | Exact, sparse, linear/Taylor, FlashAttention-style tiled | Accuracy degradation, approximation error, latency |
| Quantization | Baseline fixed-point, PoT scale, per-channel/per-token calibration | Accuracy before/after, bit-accurate simulation |
| Mode policy | Unified path, search/track split, fixed routing | Invocation distribution and p95/p99 latency |

## 5. Integration Judgment

The strongest immediate path is still C3b-compatible resource rebalance: DSP-bound QKV/MLP, URAM for large buffers, and LUTRAM only for small memories. Paper-derived algorithmic changes should be staged after C3b evidence, because approximation or routing changes can affect paper scope and accuracy claims.

