# HGTXR Master Plan: ZCU104 Cyclic Transformer Accelerator

Date: 2026-06-05

## Normalized Goal

Finish the HGTXR path from HLS to PYNQ while moving the implementation toward the paper hardware:

1. PYNQ-friendly AXI interface on `hgtxr_top.cpp`.
2. Re-run `csynth -> export IP -> Vivado block design -> bit/hwh -> PYNQ driver`.
3. Fit on ZCU104.
4. Implement a paper-aligned Transformer Block with cyclic accelerator reuse.
5. Parameterize tiling, parallelism, bus width, bit width, buffer size, and FIFO depth.
6. Use the DeiT-Tiny C-synthesis result image and `impl_repos` experiment results as resource references.

## Assumptions

- Toolchain is Vitis/Vivado 2023.2 under `/tools/Xilinx`.
- Target board is ZCU104.
- PYNQ overlay delivery requires both `.bit` and `.hwh`.
- Full paper equivalence is not proven until functional vectors, board execution, and benchmark comparisons exist.

## Expert Council

| Role | Focus |
|---|---|
| HLS/FPGA Architect | Transformer Block, cyclic scheduler, tiling, fixed-point design |
| Vivado/PYNQ Integrator | AXI interface, IP packaging, BD, bit/hwh, driver compatibility |
| Verification Lead | Simulation, synthesis reports, ZCU104 fit, paper benchmark evidence |

## Execution DAG

```text
Baseline AXI top
  -> HLS csynth
  -> HLS IP export
  -> Vivado BD
  -> bit/hwh packaging
  -> PYNQ driver smoke
  -> parameter header
  -> cyclic mac_tile
  -> QKV/WO/MLP integration
  -> attention online softmax
  -> csynth sweep
  -> ZCU104 implementation candidate
  -> board/runtime validation
  -> paper benchmark comparison
```

## Work Packages

| ID | Work Package | Output |
|---|---|---|
| WP0 | Recover current bit/hwh packaging after Tcl copy failure | overlay artifacts copied to build and PYNQ package |
| WP1 | Resource reference extraction | markdown table from DeiT-Tiny image and `impl_repos` |
| WP2 | Parameterized hardware config | HLS parameter header and build-time config |
| WP3 | Cyclic Transformer Block | reusable HLS modules and scheduler |
| WP4 | Sweep infrastructure | matrix runner and result manifest |
| WP5 | ZCU104 final candidate | implemented bitstream with utilization/timing report |
| WP6 | PYNQ runtime validation | driver smoke and board execution logs |
| WP7 | Paper reproduction audit | gap table and benchmark comparison |

## Current Constraint

At the time this plan was written, the Codex runner could patch files through `apply_patch` but could not launch WSL shell commands. Therefore build and report validation must resume when WSL command execution is restored.

