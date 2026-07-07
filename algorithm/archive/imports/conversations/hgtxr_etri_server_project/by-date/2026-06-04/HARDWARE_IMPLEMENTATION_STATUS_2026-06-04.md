# HGTXR Hardware Implementation Status - 2026-06-04

## Summary

This document records the current state of the HGTXR hardware implementation work requested in the current conversation. It follows the project/global guidance by preserving scope, assumptions, evidence, changed files, verification status, and remaining risks.

## Goal

Implement the paper-aligned HGTXR hardware path in `/home/user/project/PRJXR/HGTXR`, using the existing HGTXR scaffold plus relevant HLS patterns from:

- `/home/user/project/PRJXR/impl_repos`
- `/home/user/project/PRJXR/REF`

The implementation target is the end-to-end path:

```text
training
  -> quantization/export
  -> HLS C/C++ kernels
  -> HLS C simulation
  -> HLS synthesis
  -> IP packaging
  -> Vivado board integration
  -> FPGA runtime validation
```

## Expert Perspectives

Role: Hardware Microarchitecture / HLS Implementation / HW-SW Verification

- Hardware Microarchitecture: map paper concepts into RMU/SMU, cyclic ViT, buffering, runtime FSM, and head pipelines.
- HLS Implementation: keep kernels C-sim compatible while adding HLS-friendly loops, storage pragmas, tiling, residual paths, and deterministic fixed-shape interfaces.
- HW-SW Verification: preserve software/HLS comparability, test-vector export, smoke C++ compile, and future Vitis HLS csim/csynth hooks.

## Prompt Brief

- Goal: strengthen the HGTXR hardware implementation so it is closer to the paper hardware, not just a placeholder scaffold.
- Inputs: current HGTXR source, `impl_repos/XR_Accel` HLS kernels, `REF/_classified` HLS/TCL assets, existing HGTXR docs.
- Assumptions: final paper weights, quantization tables, LUTs, RTL signoff reports, and per-layer golden vectors are not available in the workspace.
- Constraints: WSL Shell was used for analysis and edits; `apply_patch` could not read existing WSL files, so verified WSL Shell file writes were used for modified source files.
- Expected outputs: updated HLS source files, progress/checklist docs, and clear remaining work.
- Acceptance criteria: host C++ smoke compile, Python tests, HW/SW compare, and Vitis HLS csim/csynth where tools are available.

## Sub-Agent Workflow Status

Project instructions require a sub-agent workflow for non-trivial work. The available multi-agent tool explicitly restricts spawning to cases where the user explicitly asks for sub-agents/delegation. The user requested implementation and WSL Shell work, not sub-agent delegation, so no real sub-agent was spawned.

Fallback used:

- Main agent performed planning, repo/reference analysis, implementation, and progress documentation.
- Task Cards are documented below for traceability.

## Task Cards

```yaml
task_card:
  task_id: T-001
  sub_agent: "main-agent-fallback"
  role: "research"
  objective: "Identify HLS patterns in impl_repos and REF relevant to HGTXR hardware."
  file_ownership: []
  assigned_skill:
    - algorithm-hardware-codesign-expert
    - algo2fpga
    - attention-kernel-mapper
  inputs:
    - impl_repos/XR_Accel/workspace/hardware/src
    - REF/_classified
  outputs:
    - "Reference pattern summary in this status document"
  validation:
    - "WSL Shell file scans and targeted source reads"
  dependencies: []
```

```yaml
task_card:
  task_id: T-002
  sub_agent: "main-agent-fallback"
  role: "implementer"
  objective: "Replace shallow placeholder HLS stages with paper-oriented kernels."
  file_ownership:
    - HGTXR/hardware/hls/src/nonlinear.cpp
    - HGTXR/hardware/hls/src/matmul.cpp
    - HGTXR/hardware/hls/src/rmu_smu.cpp
    - HGTXR/hardware/hls/src/attention.cpp
    - HGTXR/hardware/hls/src/mlp.cpp
    - HGTXR/hardware/hls/src/search_head.cpp
    - HGTXR/hardware/hls/src/track_head.cpp
  assigned_skill:
    - algo2fpga
    - algorithm-hardware-codesign-expert
  inputs:
    - HGTXR existing HLS scaffold
    - XR_Accel Matmul/LayerNorm/Softmax/GELU/Attn/MLP patterns
  outputs:
    - "Updated HLS source files"
  validation:
    - "Pending: host C++ smoke compile"
    - "Pending: pytest -s"
    - "Pending: Vitis HLS csim/csynth if available"
  dependencies:
    - T-001
```

```yaml
task_card:
  task_id: T-003
  sub_agent: "main-agent-fallback"
  role: "evaluator"
  objective: "Verify implementation and document gaps."
  file_ownership:
    - HGTXR/docs/track
  assigned_skill:
    - algo2fpga
  inputs:
    - Changed source files
    - Existing test scripts
  outputs:
    - "Progress checklist"
    - "Change log"
    - "Known gaps and verification plan"
  validation:
    - "Pending due command approval usage limit during final edit attempt"
  dependencies:
    - T-002
```

## Reference Sources Reviewed

### impl_repos

The following relevant HLS files were read or sampled:

- `impl_repos/XR_Accel/workspace/hardware/src/matmul.h`
- `impl_repos/XR_Accel/workspace/hardware/src/layernorm.h`
- `impl_repos/XR_Accel/workspace/hardware/src/softmax.h`
- `impl_repos/XR_Accel/workspace/hardware/src/gelu.h`
- `impl_repos/XR_Accel/workspace/hardware/src/attn.h`
- `impl_repos/XR_Accel/workspace/hardware/src/mlp.h`
- `impl_repos/XR_Accel/workspace/hardware/src/cyclic_vit.h`
- `impl_repos/XR_Accel/workspace/hardware/case_cyclic_zcu104/README.md`

Key patterns extracted:

- Matmul uses adapter/dataflow style with input window caching and MAC stages.
- LayerNorm uses multi-pass mean/variance/rsqrt-table style.
- Softmax uses max, exp-table/sum, reciprocal-table/requant stages.
- GeLU uses LUT cursoring.
- Attention composes LN, Q/K/V projection, head split, QK relation, softmax, RV, output projection, and residual merge.
- MLP composes LN, FC1, GeLU, FC2, and residual merge.
- Cyclic ViT defines mode-dependent token counts, depths, and search/track paths.

### REF

The following relevant reference categories were sampled:

- `REF/_classified/Hardware/Transformer-Accel/.../ESDA/.../hls.tcl`
- `REF/_classified/Hardware/Transformer-Accel/.../ESDA/.../hls_conf.tcl`
- `REF/_classified/Software/XR/Pacakges/ESDA/.../eventNet/.../prj/*.tcl`
- `REF/_classified/Software/LLM/task1-llm-npu/packages/40_Precision/HLS/template-hls-float/...`

Key patterns extracted:

- TCL project flow: `open_project`, `set_top`, `add_files`, `open_solution`, `set_part`, `create_clock`, `csim_design`, `csynth_design`, `cosim_design`, `export_design`.
- Precision HLS examples include dot/matmul benchmark families and explicit type sweeps.
- ESDA event-oriented references are useful for event stream/HLS project organization, but are not a direct drop-in for HGTXR transformer blocks.

## Changed Hardware Files

### `hardware/hls/src/nonlinear.cpp`

Changed from:

- Mean-only LayerNorm
- Softmax placeholder
- ReLU-like GeLU placeholder

Changed to:

- Mean/variance LayerNorm with reciprocal square-root approximation
- Three-pass token-axis softmax approximation
- tanh-polynomial GeLU approximation

Purpose:

- Aligns HGTXR non-linear operators with the paper/XR_Accel style: multi-pass, fixed-shape, LUT-friendly HLS operators.

### `hardware/hls/src/matmul.cpp`

Changed from:

- Basic `m x n x k` triple loop.

Changed to:

- Tiled matrix multiply with local accumulation tile.
- HLS-friendly loop structure and array partitioning.

Purpose:

- Mirrors the XR_Accel window-cache/MAC separation in a simpler array-based implementation compatible with the existing HGTXR scaffold.

### `hardware/hls/src/rmu_smu.cpp`

Changed from:

- Add-small-constant placeholder stages.

Changed to:

- RMU-like layernorm plus low-rank channel mixing.
- SMU-like token relation stage with saliency scoring, normalized token weighting, context accumulation, and feature injection.

Purpose:

- Provides a concrete hardware dataflow approximation of the paper's relational memory/update structure while final trained weights are unavailable.

### `hardware/hls/src/attention.cpp`

Changed from:

- Sequential calls without residual semantics.

Changed to:

- Residual snapshot.
- RMU projection surrogate.
- SMU relation surrogate.
- Output RMU projection surrogate.
- Residual merge.

Purpose:

- Makes attention stage structurally match transformer block behavior and the paper's cyclic execution intent.

### `hardware/hls/src/mlp.cpp`

Changed from:

- LN/RMU/GELU/RMU placeholder sequence.

Changed to:

- Residual snapshot.
- LN.
- RMU projection as FC1 surrogate.
- GeLU.
- RMU projection as FC2 surrogate.
- Residual merge.

Purpose:

- Makes MLP stage structurally match transformer FFN block behavior.

### `hardware/hls/src/search_head.cpp`

Changed from:

- Mostly constant bbox/confidence output.

Changed to:

- Pooled feature mean/spread/asymmetry driven bbox and confidence output.

Purpose:

- Produces feature-dependent search states for HW/SW comparison and future quantized head weight replacement.

### `hardware/hls/src/track_head.cpp`

Changed from:

- Mean delta applied only to x/y.

Changed to:

- Frame/event feature residuals for x/y/size/confidence/angle.
- Similarity-like confidence estimate.

Purpose:

- Produces feature-dependent residual tracking states that better match the stage2 hybrid/event residual tracking concept.

## Attempted But Not Completed

### `hardware/hls/src/hgtxr_top.cpp`

Planned change:

- Replace constant FSM inputs with computed values:
  - event density
  - frame/event feature similarity
  - track confidence
  - track quality
  - closed-eye proxy

Status:

- Not applied. The WSL Shell edit command was rejected by the approval reviewer due usage limit.

Impact:

- The updated heads and kernels exist, but the top-level runtime decision path still needs to be connected to the new confidence/similarity signals.

## Current Verification Status

Previously verified before this hardware-deepening pass:

- `uv venv .venv --python python3`
- `uv pip install -r requirements.txt`
- `python -m pytest -q -s`: 12 passed
- smoke stage1 training
- smoke eval
- smoke infer/runtime trace
- HW reference export
- HW/SW compare
- HLS host C++ smoke

Verification after this hardware-deepening pass:

- Not completed yet.
- Required next checks:
  - host C++ smoke compile
  - HGTXR Python tests
  - HW/SW compare
  - Vitis HLS csim if available
  - Vitis HLS csynth if available

## Known Risks

- `확실하지 않음`: final paper-equivalent weights, quantization scales, LUTs, and per-layer golden vectors are not available.
- `확실하지 않음`: the modified HLS files have not yet been recompiled after the latest changes.
- The current implementation is paper-structured but not bit-exact paper reproduction.
- `hgtxr_top.cpp` still requires FSM input reconnection.
- Exact FPGA resource/latency claims cannot be made until Vitis HLS/Vivado reports are produced.

## Next Required Actions

1. Update `hgtxr_top.cpp` to compute event density, similarity, and track quality.
2. Run host C++ compile:

```bash
cd /home/user/project/PRJXR/HGTXR
g++ -std=c++17 -Ihardware/hls/include hardware/hls/tb/tb_hgtxr_top.cpp hardware/hls/src/*.cpp -o /tmp/hgtxr_top_tb
/tmp/hgtxr_top_tb
```

3. Run Python tests:

```bash
cd /home/user/project/PRJXR/HGTXR
.venv/bin/python -m pytest -q -s
```

4. Run HW/SW reference flow:

```bash
cd /home/user/project/PRJXR/HGTXR
PYTHON_BIN=.venv/bin/python sh hardware/scripts/export_hw_refs.sh
PYTHON_BIN=.venv/bin/python sh hardware/scripts/run_hw_sw_compare.sh
```

5. Run Vitis HLS if installed:

```bash
cd /home/user/project/PRJXR/HGTXR
sh hardware/scripts/run_hls_csim.sh
sh hardware/scripts/run_hls_csynth.sh
```

