# Project Structure

This document explains the final intended structure of `ViT_Accel`.

## Top-Level Layout

```text
ViT_Accel/
├── automation/
├── configs/
├── docker/
├── docs/
├── workspace/
├── README.md
├── requirements.txt
└── .gitignore
```

## `automation/`

The `automation/` package is the control plane for the repository.

Subdirectories:

- `automation/cli`
  - argument parsers and command entrypoints
- `automation/flows`
  - orchestration for HLS, Spinal, Vivado simulation, packaging, implementation, and bitstream generation
- `automation/legacy`
  - legacy HLS and report-generation helpers ported from earlier branch layouts
- `automation/utils`
  - shared helpers for runtime paths, cases, layouts, and supporting logic

Key files:

- `automation/config.py`
- `automation/cli/hgpipe.py`
- `automation/cli/compat.py`
- `automation/flows/reconstruction.py`
- `automation/flows/legacy_hls.py`
- `automation/flows/spinal_runtime.py`
- `automation/vivado.py`

## `configs/`

The `configs/` tree describes the accelerator through composable JSON files.

Subdirectories:

- `configs/models`
  - network and quantization assumptions
- `configs/targets`
  - FPGA part, board part, clock, and MMIO runtime addresses
- `configs/designs`
  - design-level variants such as `full_deit_tiny_fit`
- `configs/experiments`
  - full experiment tuples that bind model + target + design together

Example resolution chain:

```text
configs/experiments/vck190_full_deit_tiny_fit.json
  -> configs/models/deit_tiny_baseline.json
  -> configs/targets/vck190.json
  -> configs/designs/full_deit_tiny_fit.json
```

## `docker/`

The `docker/` directory packages the host-mounted Xilinx workflow.

Important files:

- `docker/Dockerfile.host-ubuntu22.04`
- `docker/Dockerfile.host-centos7`
- `docker/README.md`
- `docker/xilinx/webtalksettings`

The image intentionally expects a host Xilinx installation to be mounted at runtime instead of bundling Vivado into the image.

## `docs/`

The `docs/` directory contains both merge-history documents and the operator runbooks for the final integrated tree.

Recommended entrypoint:

- `docs/README.md`
  - the main documentation index
  - groups setup, runbooks, architecture notes, and validation/status docs
  - points to `docs/experiments/README.md` for long-lived experiment reports

Important categories:

- merge status and integration planning
- local setup and Docker usage
- VCK190 and ZynqMP runbooks
- `src/` / `case/` architecture notes
  - especially `docs/SRC_CASE_MODULE_GUIDE.md`, which is the detailed module-centric guide for `workspace/hardware/src/` and `workspace/hardware/case/`

## `workspace/`

`workspace/` is the canonical execution root for the merged repository.

### `workspace/flow/`

This subtree holds user-facing launchers and wrapper scripts.

Subdirectories:

- `workspace/flow/entrypoints/python`
  - Python compatibility wrappers like `step1_hls_flow.py`
- `workspace/flow/entrypoints/tcl`
  - Tcl-side entrypoints used by some build flows
- `workspace/flow/scripts/targets`
  - target-organized user-facing wrappers for `vck190`, `zynqmp`, `zu15eg`, `zcu102`, `zcu104`, and `ultra96`
- `workspace/flow/scripts/host`
  - host-only repro and Vivado launch helpers
- `workspace/flow/scripts/ops`
  - maintenance and recovery wrappers
- `workspace/flow/scripts/run`
  - backend runners for VCK190 and ZynqMP flows
- `workspace/flow/scripts/env`
  - environment setup and WebTalk hardening
- `workspace/flow/scripts/docker`
  - Docker launcher and attach helpers
- `workspace/flow/scripts/utils`
  - utilities such as Xilinx mount diagnostics

Compatibility note:

- the historical top-level wrappers under `workspace/flow/scripts/*.sh` are still kept as compatibility shims
- the organized homes now live under `targets/`, `host/`, `ops/`, `docker/`, `env/`, and `run/`

### `workspace/hardware/`

This subtree contains the accelerator source and board-facing hardware assets.

Subdirectories:

- `workspace/hardware/case`
  - shared case templates, generated case C++ files, and reference data
- `workspace/hardware/case_vck190`
  - VCK190-dedicated baseline case root used by the VCK190 bash wrappers
  - keeps VCK190 `cpp` and `cpp.template` generation separate from the shared case tree
- `workspace/hardware/case_zcu102`
  - ZCU102-dedicated baseline case root used by the ZCU102 bash wrappers
- `workspace/hardware/case_zcu104`
  - ZCU104-dedicated baseline case root used by the ZCU104 bash wrappers
- `workspace/hardware/case_zu15eg`
  - ZU15EG-dedicated baseline case root used by the ZU15EG bash wrappers
- `workspace/hardware/case_ultra96`
  - Ultra96-V2-dedicated baseline case root used by the Ultra96 bash wrappers
- `workspace/hardware/src`
  - reusable HLS kernels and utility headers
- `workspace/hardware/statistics`
  - quantization/type statistics used by the case generator
- `workspace/hardware/SPINAL`
  - legacy SpinalHDL simulation and export project
- `workspace/hardware/vivado`
  - common Tcl flow scripts and board-specific wrappers

### `workspace/board/`

This subtree is reserved for board-side notebooks and MMIO-oriented experiments.

### `workspace/artifacts/`

This subtree is the output area for generated content.

Subdirectories:

- `workspace/artifacts/instances`
  - generated HLS subprojects
- `workspace/artifacts/build`
  - prepared Vivado build bundles
- `workspace/artifacts/reports`
  - Step2 reports, baseline reports, experiment matrices
- `workspace/artifacts/logs`
  - shell, HLS, Spinal, and Vivado execution logs

### `docs/experiments/`

This subtree stores curated experiment markdown reports that were promoted out of the runtime artifact area.

Examples:

- multi-board PnR status summaries
- board-specific detailed implementation reports
- selected Step2 and Step5 markdown summaries that are useful as long-lived documentation


## Compatibility Wrappers

The repository keeps a thin compatibility layer at both script and Python levels.

Examples:

- `workspace/flow/scripts/targets/vck190/readme.sh`
  - README-friendly wrapper
- `workspace/flow/scripts/targets/vck190/docker.sh`
  - Docker-safe VCK190 wrapper for `step0` through `step4-export`
- `workspace/flow/scripts/targets/vck190/up.sh`
  - VCK190-targeted wrapper around the shared Docker run helper
- `workspace/flow/scripts/targets/vck190/connect.sh`
  - VCK190-targeted wrapper around the shared Docker exec helper
- `workspace/flow/scripts/targets/vck190/pynq_handoff.sh`
  - dedicated VCK190 step5 Bash wrapper
- `workspace/flow/scripts/run/vck190_readme_backend.sh`
  - backend implementation
- `workspace/flow/entrypoints/python/step4_board_impl_flow.py`
  - compatibility wrapper around `hgpipe board-impl`

This lets older workflows remain recognizable while the actual implementation logic moves into `automation/`.
