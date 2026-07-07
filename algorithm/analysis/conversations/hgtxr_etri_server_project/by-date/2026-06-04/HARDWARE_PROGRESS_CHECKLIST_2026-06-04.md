# HGTXR Hardware Progress Checklist - 2026-06-04

## Plan vs Progress

| Step | Status | Notes |
|---|---:|---|
| Normalize paper hardware goal | Done | Goal is HGTXR paper-style hardware path, not only software scaffold. |
| Confirm execution environment | Done | WSL Shell worked with approval-based execution after sandbox helper failure. |
| Inspect HGTXR hardware scaffold | Done | `config`, `cyclic_config`, `common`, `top`, attention/MLP/matmul/nonlinear/head files reviewed. |
| Inspect `impl_repos` HLS references | Done | XR_Accel matmul, layernorm, softmax, gelu, attn, mlp, cyclic_vit reviewed. |
| Inspect `REF` HLS references | Done | ESDA HLS TCL and Transformer/LLM precision/matmul reference categories sampled. |
| Implement real nonlinear stages | Done | LayerNorm, softmax, GeLU changed from placeholders to approximation kernels. |
| Implement tiled matmul | Done | Basic matmul replaced by tiled local-accumulation form. |
| Implement RMU/SMU stages | Done | Placeholder increments replaced with projection/relation surrogate kernels. |
| Implement attention residual stage | Done | RMU-SMU-RMU plus residual merge added. |
| Implement MLP residual stage | Done | LN-projection-GeLU-projection plus residual merge added. |
| Implement feature-based search head | Done | Constant output replaced with pooled-feature bbox/confidence logic. |
| Implement feature-based track head | Done | Residual state update based on frame/event fused features. |
| Connect top-level FSM to new signals | Pending | `hgtxr_top.cpp` update was blocked by command approval usage limit. |
| Run host HLS C++ smoke | Pending | Must be run after top-level update or immediately to catch compile errors. |
| Run Python tests | Pending | Required after HLS changes because HW/SW compare may be affected. |
| Run HW/SW compare | Pending | Required after recompilation and reference export. |
| Run Vitis HLS csim/csynth | Pending | Depends on local Vitis installation. |
| Update main docs | Partial | New tracking docs created; README/Validation/PROGRESS still need integration update. |

## Implementation Completeness

| Area | Current Level | Target Level |
|---|---|---|
| Patch embedding | Scaffold | Quantized trained patch projection |
| Attention | Structural surrogate | Quantized Q/K/V, QK, softmax, RV, output projection with goldens |
| MLP | Structural surrogate | Quantized FC1/GELU/FC2 with goldens |
| RMU/SMU | Structural surrogate | Paper-accurate arithmetic and scheduling |
| Search head | Feature-dependent surrogate | Trained quantized search head |
| Track head | Feature-dependent surrogate | Trained quantized residual tracking head |
| Runtime FSM | Existing scaffold | Connected confidence/similarity/event-density gates |
| HLS scripts | Existing scaffold | Board-specific csim/csynth/export verified |
| FPGA integration | TCL scaffold | Implemented Vivado block design, bitstream, runtime driver |

## Blockers

- `hgtxr_top.cpp` FSM reconnection is not yet applied.
- Latest HLS source changes have not been compiled.
- Final paper weights and quantization artifacts are unavailable.
- Vitis/Vivado availability has not been confirmed in this turn.

## Acceptance Checklist

Minimum for "implementation pass complete":

- [ ] `hgtxr_top.cpp` connects runtime FSM to real computed signals.
- [ ] `g++` host compile passes.
- [ ] HLS top testbench produces finite state outputs.
- [ ] `.venv/bin/python -m pytest -q -s` passes.
- [ ] HW/SW compare report is regenerated.
- [ ] `docs/Validation.md` records new evidence.
- [ ] `docs/track/PROGRESS.md` is synchronized.

Minimum for "paper-equivalent claim":

- [ ] Final trained weights exist.
- [ ] Quantization scales/tables/LUTs exist.
- [ ] Per-layer golden vectors exist.
- [ ] HLS csim matches per-layer goldens.
- [ ] HLS csynth meets target latency/II/resource constraints.
- [ ] Vivado implementation meets timing.
- [ ] FPGA runtime output matches expected end-to-end behavior.

