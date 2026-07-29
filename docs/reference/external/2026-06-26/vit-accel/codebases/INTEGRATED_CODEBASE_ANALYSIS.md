# Integrated Codebase Analysis: Existing ViT-Accel Reports

Source: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/codebases`

## 1. Scope

This document integrates the 30 existing per-codebase reports. It does not replace them. It condenses their hardware-design implications for HGTXR C3b successor work and third-goal experiment planning.

## 2. Codebase Groups

| Group | Codebases | Main value for HGTXR | Hardware implication |
|---|---|---|---|
| Pipeline and ViT FPGA baselines | `HG-PIPE`, `Trio-ViT`, `ViT-FPGA-TPU`, `Transformer-Accelerator-Based-on-FPGA`, `transformer-hls-thesis` | Full-network or block-level Transformer/ViT FPGA structure | Use as reference for attention/MLP decomposition, stream widths, and HLS project partitioning |
| Quantization and integer dataflow | `P2-ViT`, `AHCPTQ`, `ViTCoD`, `efficient-transformer-accelerator` | Scale handling, integer-only path, sparse/quantized operators | Keep DSP-bound MACs; move only small lookup/control structures to LUTRAM |
| Attention and dataflow variants | `ViTALiTy`, `TATAA`, `AURA-FlashAttention-AISC-Accelerator` | Alternative attention mapping, FlashAttention-like SRAM reuse, Taylor/linear approximation | Treat exact attention replacement as an explicit scope decision; start with software/HLS ablation |
| MoE and routing | `Edge-MoE`, `CoQMoE`, `M3ViT`, `UbiMoE` | Search/track asymmetry, routing policy, expert reuse | Useful for mode-aware execution, but dynamic routing increases verification risk |
| LLM-oriented accelerators | `FlexLLM`, `HLS-Acceleration-of-LLaMA2`, `LLM_FPGA`, `LUT-LLM`, `flightllm_test_demo`, `llama-fpga`, `ternaryLLM`, `TinyTransformer` | Quantization, memory hierarchy, kernel orchestration patterns | Reuse memory and scheduler ideas; do not import LUT-heavy arithmetic into HGTXR default path |
| GEMM and low-level operator references | `hls-fpga-accelerators`, `lut-gemm`, `TeraFly`, `Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280`, `HLSTransformation` | Operator-level mapping and toolflow examples | Useful for validation harnesses and systolic/kernel comparison |

## 3. HGTXR-Relevant Design Rules

| Rule | Evidence source family | HGTXR action |
|---|---|---|
| Keep MAC-heavy operators DSP-bound | HG-PIPE, P2-ViT, efficient-transformer-accelerator | Increase parallelism factor in QKV/MLP before adding LUT arithmetic |
| Split large buffers from small control buffers | HG-PIPE, ME-ViT-style memory analysis, Edge-MoE | Bind activation/global/deep FIFO buffers to URAM or BRAM; bind small scalar/state tables to LUTRAM |
| Preserve exact baseline before attention replacement | ViTALiTy, TATAA, AURA, ViTCoD | Gate Taylor/linear/sparse attention behind an explicit experiment define |
| Treat dynamic routing as software-first | Edge-MoE, CoQMoE, M3ViT, UbiMoE | Validate accuracy and invocation distribution before HLS integration |
| Separate claims from evidence | All previous reports | Do not promote static analysis into resource/performance claims without csynth/routed/board artifacts |

## 4. Priority Conversion to Experiments

| Priority | Experiment class | Candidate sources | Expected outcome | Gate |
|---|---|---|---|---|
| P0 | C3b resource rebalance | HG-PIPE, efficient-transformer-accelerator, P2-ViT | Higher DSP use, lower LUT pressure, URAM use for large buffers | csynth resource table and timing estimate |
| P0 | PoT/shift-friendly quantization | P2-ViT, AHCPTQ | Reduce requantization cost while keeping accuracy | SW accuracy + bit-accurate HLS check |
| P1 | Attention dataflow ablation | TATAA, AURA, ViTALiTy, ViTCoD | Lower memory traffic or latency for attention block | Exact-attention non-regression or explicit paper-scope decision |
| P1 | Mode-aware search/track execution | Edge-MoE, M3ViT, UbiMoE | Lower average latency by mode distribution | Invocation counters and distribution report |
| P2 | LLM-derived memory scheduler ideas | FlexLLM, LLM_FPGA, HLS-Acceleration-of-LLaMA2 | Better tiling and double-buffer policy | HLS-only microbenchmark first |

## 5. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| LUT-heavy approximation imported directly | Violates current DSP/URAM utilization goal | Allow LUTRAM for small tables only; keep arithmetic DSP-bound |
| Approximate attention changes paper scope | Accuracy or method mismatch | Add exact fallback and document approximation error |
| MoE routing creates non-deterministic resource/control paths | HLS verification complexity | Software-only first, then fixed-top-k HLS variant |
| Reference reports have heterogeneous board targets | Cross-paper ranking can be misleading | Compare only normalized operator-level or same-board evidence |

