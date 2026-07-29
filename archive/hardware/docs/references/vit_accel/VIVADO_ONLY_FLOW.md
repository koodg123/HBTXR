# Vivado-Only Flow

This guide merges:

- the `workspace/`-centered layout from `codex/home-wsl`
- the ZynqMP Vivado-only and ZynqMP Spinal-backed extensions from `exp`

The command paths below reflect the current merged tree in `ViT_Accel`.

## What Was Added

- config-driven experiment loading under `configs/`
- `workspace/flow/entrypoints/python/hgpipe_flow.py` as the user-facing Python entrypoint
- generated Vivado-only top RTL (`HGPIPE_VIVADO_TOP`)
- RTL bundle preparation that prefers `workspace/artifacts/instances/<target>/<model>/<design>/` HLS outputs and falls back to checked-in `workspace/hardware/SPINAL/src/main/verilog/*`
- reference export from `workspace/hardware/case/refs/*.txt` to `.mem`
- systemVerilog testbench generation
- Vivado batch Tcl generation for:
  - simulation preparation
  - RTL packaging
  - target implementation
  - ZynqMP block-design bitstream preparation

## New Entry Points

Baseline report:

```bash
python3 workspace/flow/entrypoints/python/hgpipe_flow.py baseline-report
```

Vivado-only simulation layout:

```bash
python3 workspace/flow/entrypoints/python/hgpipe_flow.py vivado-sim \
  --experiment configs/experiments/vck190_full_deit_tiny_fit.json
```

Vivado packaging layout:

```bash
python3 workspace/flow/entrypoints/python/hgpipe_flow.py vivado-package \
  --experiment configs/experiments/vck190_full_deit_tiny_fit.json
```

Target implementation layout:

```bash
python3 workspace/flow/entrypoints/python/hgpipe_flow.py board-impl \
  --experiment configs/experiments/vck190_full_deit_tiny_fit.json
```

ZynqMP board-level bitstream layout:

```bash
python3 workspace/flow/entrypoints/python/hgpipe_flow.py board-bitstream \
  --experiment configs/experiments/zcu102_full_deit_tiny_fit.json
```

ZynqMP board-level bitstream layout from the exported Spinal bundle:

```bash
python3 workspace/flow/entrypoints/python/hgpipe_flow.py spinal-board-bitstream \
  --experiment configs/experiments/zcu102_full_deit_tiny_fit.json
```

Experiment matrix:

```bash
python3 workspace/flow/entrypoints/python/hgpipe_flow.py experiment-sweep
```

## Generated Layout

Each experiment is intended to write into:

```text
workspace/artifacts/build/<target>/<experiment>/
```

Main subdirectories:

- `generated/`
  - generated top RTL
  - `source_manifest.tcl`
  - `clock.xdc`
- `rtl/`
  - collected block RTL bundles
- `refs/`
  - `.mem` files for simulation
- `sim/`
  - generated testbenches
  - xsim launch helpers
- `package/`
  - package flow Tcl
- `impl/`
  - board-specific implementation Tcl
- `bitstream/`
  - ZynqMP block-design bitstream Tcl
- `spinal_bitstream/`
  - ZynqMP block-design Tcl for the exported Spinal bundle
- `spinal_rtl/`
  - staged Spinal export sources plus rewritten `.dat` memory paths

## Current Scope

The current implementation direction remains centered on accelerator-only, out-of-context closure first.

That means:

- resource and timing feasibility can be checked per target
- the Vivado-only top reduces dependency on the live Spinal runtime
- full PS/DMA/DDR board automation remains a follow-on integration layer

## Oracle And Parity

The checked-in Spinal baseline remains the current oracle:

- `workspace/hardware/SPINAL/latency/`
- `workspace/hardware/SPINAL/src/main/verilog/*`
- `workspace/hardware/SPINAL/BlockSequence.v`

The new flow prepares artifacts needed to compare the generated Vivado-only path against that oracle.

## Recent Hardening Notes

The merged documentation keeps the hardening intent from the `codex/home-wsl` line:

- generated top signal collisions must be avoided
- relative `--build-dir` values should be normalized before batch execution
- Vivado batch helpers should disable WebTalk aggressively
- local-user-data state should be redirected away from the default shared path

This is especially important for:

- `board-impl`
- `board-bitstream`
- `spinal-board-bitstream`

## ZynqMP Bitstream Flow

The new `board-bitstream` command targets:

- `zu15eg`
- `zcu102`
- `zcu104`
- `ultra96v2`

The generic wrapper is:

```bash
workspace/flow/scripts/targets/zynqmp/vivado.sh step0
workspace/flow/scripts/targets/zynqmp/vivado.sh step1
workspace/flow/scripts/targets/zynqmp/vivado.sh step2
workspace/flow/scripts/targets/zynqmp/vivado.sh step4-prepare
workspace/flow/scripts/targets/zynqmp/vivado.sh step4-validate zcu102
workspace/flow/scripts/targets/zynqmp/vivado.sh step4-bitstream
```

Alias examples:

```bash
workspace/flow/scripts/targets/zynqmp/vivado.sh hls
workspace/flow/scripts/targets/zynqmp/vivado.sh prepare
workspace/flow/scripts/targets/zynqmp/vivado.sh bitstream
```

Recommended default experiment suffix:

```bash
HGPIPE_EXPERIMENT_SUFFIX=full_deit_tiny_fit
```

Recommended host-only wrappers for the board-specific Vivado path:

```bash
workspace/flow/scripts/targets/zu15eg/host.sh all-host
workspace/flow/scripts/targets/zcu102/host.sh all-host
workspace/flow/scripts/targets/zcu104/host.sh all-host
workspace/flow/scripts/targets/ultra96/host.sh all-host
```

Equivalent alias:

```bash
workspace/flow/scripts/targets/zu15eg/host.sh vivado-only
workspace/flow/scripts/targets/zcu102/host.sh vivado-only
workspace/flow/scripts/targets/zcu104/host.sh vivado-only
workspace/flow/scripts/targets/ultra96/host.sh vivado-only
```

Those host wrappers execute the Vivado-only path:

```text
step0 -> step1 -> step2 -> step4-bitstream
```

without the Spinal-specific `step3-*` / `step4-export` stages.

## Spinal-Backed ZynqMP Flow

The same ZynqMP bitstream shell can also be driven from the legacy Spinal export instead of the generated `HGPIPE_VIVADO_TOP`.

Expected exported sources:

- `workspace/hardware/SPINAL/vivado/BlockSequence_replaced.v`
- `workspace/hardware/SPINAL/vivado/BlockSequence_bb_replaced.v`
- the `.dat` files referenced by those exported Verilog sources

Primary helper command:

```bash
python3 workspace/flow/entrypoints/python/hgpipe_flow.py spinal-board-bitstream \
  --experiment configs/experiments/zcu102_full_deit_tiny_fit.json
```

README-style shell wrapper:

```bash
workspace/flow/scripts/targets/zynqmp/spinal.sh step0
workspace/flow/scripts/targets/zynqmp/spinal.sh step1
workspace/flow/scripts/targets/zynqmp/spinal.sh step2
workspace/flow/scripts/targets/zynqmp/spinal.sh step3-all
workspace/flow/scripts/targets/zynqmp/spinal.sh step4-export
workspace/flow/scripts/targets/zynqmp/spinal.sh step4-prepare zcu102
workspace/flow/scripts/targets/zynqmp/spinal.sh step4-bitstream zcu102
```

Alias examples:

```bash
workspace/flow/scripts/targets/zynqmp/spinal.sh hls
workspace/flow/scripts/targets/zynqmp/spinal.sh export
workspace/flow/scripts/targets/zynqmp/spinal.sh prepare
workspace/flow/scripts/targets/zynqmp/spinal.sh bitstream
```

The backend implementation is intended to live at:

```bash
workspace/flow/scripts/run/zynqmp_spinal_backend.sh
```

`spinal-board-bitstream` should stage the exported Spinal Verilog and `.dat` files into the `spinal_rtl/` build area and rewrite `$readmemh` paths so Vivado can consume a self-contained bundle.

If the exported `workspace/hardware/SPINAL/vivado/` bundle is incomplete, the prepare step should fail early and direct the user to regenerate the export.

The merged tree now also includes a helper for the legacy HLS-to-SPINAL `.dat` recovery path:

```bash
cd /home/kjm26/project/PRJXR/HG_PIPE_MERGE/ViT_Accel
bash workspace/flow/scripts/ops/spinal_dat_recover.sh HEAD ATTN0
```

That helper reruns `step1` only when a case still lacks real HLS `.dat` files, imports the resulting files into `workspace/hardware/SPINAL/src/main/verilog/`, and refreshes the exported `workspace/hardware/SPINAL/vivado/` bundle.
By default it disables `csim` and `cosim` during `step1` so the recovery flow can focus on generating real RTL and `.dat` outputs.

For smoke-test verification only, the merged tree now also supports:

```bash
HGPIPE_ALLOW_PLACEHOLDER_SPINAL_DAT=1 \
workspace/flow/scripts/targets/zynqmp/spinal.sh step4-prepare zcu102
```

That opt-in path creates placeholder `.dat` files only in the staged `spinal_rtl/` build area so the prepare step can be exercised without claiming that the resulting hardware image has valid initialized weights.

## Status Note

The code and script paths described here now exist in `ViT_Accel`.

Current validation status:

- `board-bitstream` and `spinal-board-bitstream` are exposed through `hgpipe_flow.py`
- the ZynqMP Spinal wrapper path is runnable
- `workspace/flow/scripts/targets/zynqmp/vivado.sh step4-prepare zcu102` successfully prepares a board bitstream bundle
- `workspace/flow/scripts/targets/zynqmp/spinal.sh step4-export` now regenerates the top-level Spinal Verilog successfully
- the checked-in/exported Spinal bundle still lacks real `.dat` files because those come from HLS instance outputs, not from the Spinal generator
- `workspace/flow/scripts/ops/spinal_dat_recover.sh HEAD ATTN0 ATTN1 ATTN2 ATTN10 ATTN11 MLP0 PATCH_EMBED` has already recovered 116 real `.dat` files into the SPINAL tree and reduced strict `step4-prepare` missing-file count from `343` to `227`
- `HGPIPE_ALLOW_PLACEHOLDER_SPINAL_DAT=1 workspace/flow/scripts/targets/zynqmp/spinal.sh step4-prepare zcu102` now succeeds for smoke-test bundle preparation

Without that opt-in flag, the early failure is still expected and desirable.
