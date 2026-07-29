# Merge Status

This document records the current merge state for `ViT_Accel`.

## Structural Base

The target repository structure is based on:

- `HG-PIPE-codex-home-wsl`

That means the final canonical layout is expected to use:

- `workspace/flow/`
- `workspace/hardware/`
- `workspace/board/`
- `workspace/artifacts/`
- `automation/`

## Imported Documentation Themes

### From `HG-PIPE-codex-home-wsl`

- clustered `workspace/` repository structure
- `automation/` control-plane layout
- `workspace/artifacts/` default artifact layout
- local setup guide
- multi-board validation guide
- VCK190 Docker baseline path written against clustered paths
- Vivado/WebTalk hardening notes

### From `HG-PIPE-main`

- SRC/case documentation set
- VCK190 helper flow documentation for:
  - ATTN/MLP DSP override cases
  - block-subset HLS flow
  - PATCH/HEAD DSP override flow
  - Xilinx mount diagnostics

### From `HG-PIPE-exp`

- ZynqMP Spinal-backed runner documentation
- `board-bitstream` / `spinal-board-bitstream` usage notes
- ZynqMP README-style helper sequences

## Current Scope

Merged so far:

- `README.md`
- `docs/LOCAL_SETUP.md`
- `docs/MULTI_BOARD_VALIDATION.md`
- `docs/VCK190_DOCKER_BASELINE.md`
- `docs/VIVADO_ONLY_FLOW.md`
- `docs/SRC_CASE_ANALYSIS.md`
- `docs/SRC_CASE_DIAGRAMS.md`
- `docs/SRC_CASE_MICROARCH.md`
- `docs/SRC_CASE_MODULE_GUIDE.md`
- `automation/*`
- `configs/*`
- `workspace/flow/entrypoints/python/*`
- `workspace/flow/entrypoints/tcl/*`
- `workspace/flow/scripts/*`
- `workspace/hardware/{SPINAL,case,src,statistics,vivado}/*`

Merged and adapted for the clustered layout:

- `main` VCK190 helper scripts
- `main` VCK190 baseline runner DSP profile commands (`step1-dsp`, `step2-dsp`, `baseline-dsp`)
- `main` case override generators
- `main` residual FIFO URAM override in `src/attn.h` and `src/mlp.h`
- `exp` ZynqMP Spinal runner flow
- `exp` `board-bitstream` / `spinal-board-bitstream` command support
- `main` Xilinx mount diagnostics wiring in environment and container entrypoints
- `main` host-side Docker/Xilinx root validation logic in the Docker launcher
- Ubuntu-host Docker runtime now defaults to host UID/GID mapping to avoid `root:root` ownership on mounted files

Still pending or partial:

- full Docker-backed HLS regeneration for every block to recreate the real `.dat` memory bundle
- end-to-end Vivado synthesis / bitstream execution on the regenerated Spinal bundle
- replacing the current smoke-test placeholder `.dat` path with real HLS-produced memory initializers

## Documentation Conventions In This Tree

- Prefer `workspace/...` paths over old flat root paths
- Treat `ViT_Accel` as the target merged repository, not a copy of any single branch
- Preserve `codex/home-wsl` path conventions unless a later code merge proves a better canonical location

## Verification Snapshot

- `bash -n` passed for the newly added and modified shell runners
- `python3 -m py_compile` passed for the modified Python modules and new helper scripts
- `automation.cli.hgpipe` parser now exposes:
  - `board-bitstream`
  - `spinal-board-bitstream`
- helper runner smoke tests completed:
  - `run_vck190_attn_mlp_dsp_flow.sh prepare`
  - `run_vck190_patch_head_dsp_flow.sh prepare`
  - `VITIS_HLS_BIN=/bin/true HGPIPE_STEP1_MAX_THREADS=1 run_vck190_block_subset_flow.sh step1`
  - `VITIS_HLS_BIN=/bin/true HGPIPE_STEP1_MAX_THREADS=1 workspace/flow/scripts/targets/vck190/readme.sh step1-dsp`
- Docker-backed VCK190 smoke test completed with real runtime environment:
  - `workspace/flow/scripts/targets/vck190/readme.sh step0`
  - `run_vck190_attn_mlp_dsp_flow.sh prepare`
  - `run_vck190_block_subset_flow.sh prepare`
- ZynqMP Spinal smoke test completed:
  - `workspace/flow/scripts/targets/zynqmp/spinal.sh env`
  - `workspace/flow/scripts/targets/zynqmp/spinal.sh step4-export` now succeeds inside Docker and regenerates:
    - `workspace/hardware/SPINAL/BlockSequence.v`
    - `workspace/hardware/SPINAL/BlockSequence_bb.v`
  - the regenerated export still does not include real `.dat` files because those originate from HLS instance outputs, not from the Spinal generator itself
  - `HGPIPE_ALLOW_PLACEHOLDER_SPINAL_DAT=1 workspace/flow/scripts/targets/zynqmp/spinal.sh step4-prepare zcu102` now succeeds for smoke-test bundle preparation
  - real HLS `.dat` recovery has now been verified for:
    - `HEAD` (5 real `.dat` files imported)
    - `ATTN1` (20 real `.dat` files imported)
    - `ATTN2` (20 real `.dat` files imported)
    - `ATTN0` (20 real `.dat` files imported)
    - `ATTN10` (20 real `.dat` files imported)
    - `ATTN11` (20 real `.dat` files imported)
    - `MLP0` (8 real `.dat` files imported)
    - `PATCH_EMBED` (3 real `.dat` files imported)
  - after re-exporting the Spinal Vivado bundle, strict `step4-prepare zcu102` missing-file count dropped from `343` to `227`
  - `workspace/flow/scripts/ops/spinal_dat_recover.sh` now replays the same recovery path end-to-end and refreshes `step4-export`
  - the helper now defaults to `HGPIPE_RECOVER_STEP1_DO_CSIM=0` and `HGPIPE_RECOVER_STEP1_DO_COSIM=0` so case-by-case `.dat` recovery does not spend extra time on simulations that are not needed for the export bundle
- ZynqMP Vivado-only smoke test completed:
  - `workspace/flow/scripts/targets/zynqmp/vivado.sh step4-prepare zcu102`
- top-level wrapper smoke test completed:
  - `workspace/flow/scripts/targets/vck190/readme.sh env`
- Docker runtime verification completed:
  - Ubuntu 22.04 image rebuilt successfully as `vit-accel:ubuntu22`
  - daemon mode now stays alive by default
  - `workspace/flow/scripts/docker/container_exec.sh` attaches as the host UID/GID instead of `root`
  - inside the container:
    - `id` resolves to `uid=1001 gid=1001`
    - `HOME=/tmp/hgpipe-home-1001`
    - `LM_LICENSE_FILE` and `XILINXD_LICENSE_FILE` are forwarded correctly
  - files created from inside Docker now land on the mounted repository as `1001:1001`, not `root:root`
  - host-identity mitigation is now available through:
    - `--network`
    - `--hostname`
    - `--mac-address`
    - optional host `/run/udev` mount
    - optional host `machine-id` mount
  - the latest `vck190 step4-ooc` retry under the host-identity profile progresses beyond the old immediate license-checkout crash and enters long-running OOC synthesis
- SPINAL metadata cleanup completed:
  - removed `.idea/`
  - removed `.bsp/`

Imported file counts:

- `configs/*`: 27 files
- `automation/*`: 48 files
- `workspace/flow/entrypoints/*`: 17 files
- `workspace/flow/scripts/*`: 34 files
- `workspace/hardware/*`: 1195 files

## Notes

- Because `codex/home-wsl` has no merge-base with `main`, this integration is being treated as a staged replay/port effort rather than a direct `git merge`.
- The documentation in this tree began as a forward-looking target description, but the repository now includes the first full code-port pass as well.
- The remaining real blocker for the Spinal-backed ZynqMP flow is not `sbt` anymore. It is the absence of HLS-produced `.dat` files under the instance outputs that older flows copied into `workspace/hardware/SPINAL/src/main/verilog/`.
- The repository now includes `workspace/flow/scripts/ops/spinal_dat_recover.sh` to automate that legacy recovery path case-by-case.
- Current recovered real `.dat` inventory in `workspace/hardware/SPINAL/vivado/`: `116` files.
- For smoke testing only, `HGPIPE_ALLOW_PLACEHOLDER_SPINAL_DAT=1` now stages placeholder `.dat` files into the generated `spinal_rtl/` directory so `step4-prepare` can be verified without pretending the final hardware image is functionally correct.
