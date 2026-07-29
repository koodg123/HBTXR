# Pipeline Overview

This document explains the end-to-end accelerator generation pipeline implemented by `ViT_Accel`.

## 1. Configuration Model

The repository is driven by layered JSON configuration files:

- model config
- target config
- design config
- experiment config

The experiment file is the user-facing starting point.

Example:

```text
configs/experiments/vck190_full_deit_tiny_fit.json
```

This resolves to:

- model: `configs/models/deit_tiny_baseline.json`
- target: `configs/targets/vck190.json`
- design: `configs/designs/full_deit_tiny_fit.json`

The resolved configuration then drives both the legacy step-based wrappers and the newer config-driven `hgpipe` flow.

## 2. Step-Based Baseline Flow

The legacy baseline remains the easiest way to understand the repository.

### Step0. Case Generation

Entry:

```bash
workspace/flow/scripts/targets/vck190/readme.sh step0
```

What happens:

- `workspace/hardware/statistics/type.npy` is loaded
- layer-specific C++ case sources are generated for:
  - `PATCH_EMBED`
  - `ATTN0..ATTN11`
  - `MLP0..MLP11`
  - `HEAD`

Output:

- generated case sources under `workspace/hardware/case`

### Step1. HLS Flow

Entry:

```bash
workspace/flow/scripts/targets/vck190/readme.sh step1
```

What happens:

- HLS subprojects are created under `workspace/artifacts/instances`
- per-case Tcl scripts are emitted
- Vitis HLS runs:
  - `csim`
  - `csynth`
  - `cosim`
  - optional `export_design -flow syn`

Important outputs:

- HLS Verilog
- resource estimates
- latency estimates
- generated `.dat` memory initializers for some flows

### Step2. Resource Aggregation

Entry:

```bash
workspace/flow/scripts/targets/vck190/readme.sh step2
```

What happens:

- the HLS subproject outputs are scanned
- a consolidated resource summary is written

Outputs:

- `workspace/artifacts/reports/<target>/<model>/<design>/step2_hls_resource_summary.md`
- `workspace/artifacts/reports/<target>/<model>/<design>/step2_hls_resource_summary.csv`
- `workspace/artifacts/reports/<target>/<model>/<design>/step2_hls_resource_summary.json`

### Step3. Spinal Runtime Flow

Entry:

```bash
workspace/flow/scripts/targets/vck190/readme.sh step3-all
```

What happens:

- HLS-generated block Verilog is staged into the Spinal runtime flow
- the Spinal socket server is launched or reused
- per-block simulation is executed
- block latencies are printed

Optional:

- full-network simulation through the Spinal project

### Step4 Export

Entry:

```bash
workspace/flow/scripts/targets/vck190/readme.sh step4-export
```

What happens:

- Spinal generates the whole-network Verilog
- `to_vivado.py` exports the bundle needed by later Vivado integration

Outputs:

- `workspace/hardware/SPINAL/BlockSequence.v`
- `workspace/hardware/SPINAL/BlockSequence_bb.v`
- `workspace/hardware/SPINAL/vivado/*`

### Step4 OOC

Entry:

```bash
workspace/flow/scripts/targets/vck190/readme.sh step4-ooc
```

What happens:

- the generated top is packaged into an accelerator-only implementation flow
- board Tcl is sourced for the chosen target
- Vivado runs in batch mode for out-of-context implementation

Output root:

- `workspace/artifacts/build/<target>/<experiment>/impl`

## 3. Config-Driven Flow

The newer path is driven through:

```bash
python3 workspace/flow/entrypoints/python/hgpipe_flow.py <command>
```

Main commands:

- `baseline-report`
- `generate-top`
- `prepare-sources`
- `export-refs`
- `vivado-sim`
- `vivado-package`
- `board-impl`
- `board-bitstream`
- `spinal-board-bitstream`
- `experiment-sweep`

This path is the preferred internal API surface because it writes structured build bundles and JSON plans.

## 4. ZynqMP Vivado-Only Flow

Entry:

```bash
workspace/flow/scripts/targets/zynqmp/vivado.sh step4-prepare zcu102
workspace/flow/scripts/targets/zynqmp/vivado.sh step4-bitstream zcu102
```

What happens:

- the generated `HGPIPE_VIVADO_TOP` is treated as a module reference
- a ZynqMP block design is assembled
- DMA and MMIO base addresses are sourced from target configuration
- Vivado validation, synthesis, or full bitstream runs are prepared or executed

Supported targets in the merged tree:

- `zu15eg`
- `zcu102`
- `zcu104`
- `ultra96v2`

## 5. ZynqMP Spinal-Backed Flow

Entry:

```bash
workspace/flow/scripts/targets/zynqmp/spinal.sh step4-export
workspace/flow/scripts/targets/zynqmp/spinal.sh step4-prepare zcu102
workspace/flow/scripts/targets/zynqmp/spinal.sh step4-bitstream zcu102
```

What happens:

- instead of the generated `HGPIPE_VIVADO_TOP`, the flow consumes the exported Spinal `BlockSequence` bundle
- the block design still uses the same target runtime configuration
- this path depends on the exported Spinal `.dat` memory files being present

For a host-friendly `vck190`-style README entrypoint on ZynqMP, use:

```bash
workspace/flow/scripts/targets/zu15eg/host.sh baseline
workspace/flow/scripts/targets/zu15eg/host.sh all
```

This pins the flow to `configs/experiments/zu15eg_full_deit_tiny_fit.json`, which uses the same `deit_tiny_baseline` model and `full_deit_tiny_fit` design as the VCK190 baseline, but routes through the non-Spinal ZynqMP Vivado-only path.

For Docker use, split the ZU15EG flow on purpose:

```bash
workspace/flow/scripts/targets/zu15eg/docker.sh nonspinal all
workspace/flow/scripts/targets/zu15eg/docker.sh spinal all
```

Execution policy:

- Docker path:
  - SpinalHDL and non-SpinalHDL preparation flows are both allowed
  - Vivado execution stages are blocked
- Host path:
  - non-SpinalHDL flow only
  - Vivado execution stages run here

## 6. Artifact Locations

The main execution artifacts are grouped by purpose.

- `workspace/artifacts/instances`
  - HLS subprojects and per-case outputs
- `workspace/artifacts/reports`
  - consolidated summaries and matrices
- `workspace/artifacts/build`
  - prepared simulation, packaging, implementation, and bitstream bundles
- `workspace/artifacts/logs`
  - execution logs for shell wrappers and Vivado runs

## 7. Current Validation Boundary

As of the current integration snapshot:

- baseline HLS and Spinal export stages are working
- ZynqMP prepare flow is working
- the remaining risk is concentrated in heavy Vivado implementation stages, not in the basic pipeline wiring
