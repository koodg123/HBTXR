# Project Checklist

This checklist summarizes the current implementation and validation status for `ViT_Accel`.

## Repository and Merge

- [x] Analyze source branches under `HG_PIPE_MERGE`
- [x] Select `HG-PIPE-codex-home-wsl` as the structural base
- [x] Convert the merge strategy from direct merge to staged replay/port
- [x] Create `ViT_Accel` as the integrated target tree
- [x] Port `main` helper flows
- [x] Port `exp` ZynqMP Spinal-backed flows
- [x] Preserve the `codex/home-wsl` clustered path layout

## Documentation

- [x] Merge top-level README
- [x] Write local setup guide
- [x] Write VCK190 Docker guide
- [x] Write ZynqMP/Vivado-only guide
- [x] Write SRC/case analysis guides
- [x] Write merge status document
- [x] Write `HISTORY.md`
- [x] Write `CHECKLIST.md`
- [x] Write `PLAN.md`
- [x] Write project structure guide
- [x] Write pipeline overview
- [x] Write training/evaluation/metrics guide
- [x] Write function call stack guide
- [x] Write deployment run script guide
- [x] Write VCK190/ZU15EG parameter comparison
- [x] Write 2026-04-15 progress snapshot

## Code Port and Structure

- [x] Create `workspace/` skeleton
- [x] Port `automation/*`
- [x] Port `configs/*`
- [x] Port `workspace/flow/entrypoints/*`
- [x] Port `workspace/flow/scripts/*`
- [x] Port `workspace/hardware/*`
- [x] Add Ubuntu 22.04 Dockerfile
- [x] Add CentOS 7 Dockerfile

## Runtime and Docker

- [x] Forward Xilinx license variables into Docker
- [x] Keep daemon-mode container alive
- [x] Attach as host UID/GID instead of `root`
- [x] Fix mounted file ownership
- [x] Add Docker host-identity options
- [x] Add optional host `udev` mount
- [x] Add optional host `machine-id` mount
- [x] Add optional fixed hostname / network / MAC support

## HLS and Artifact Recovery

- [x] Re-enable real HLS regeneration in the merged tree
- [x] Add `.dat` recovery helper scripts
- [x] Recover real `.dat` files for `HEAD`
- [x] Recover real `.dat` files for `PATCH_EMBED`
- [x] Recover real `.dat` files for `ATTN0`
- [x] Recover real `.dat` files for `ATTN1`
- [x] Recover real `.dat` files for `ATTN2`
- [x] Recover real `.dat` files for `ATTN10`
- [x] Recover real `.dat` files for `ATTN11`
- [x] Recover real `.dat` files for `MLP0`
- [x] Reduce strict missing `.dat` count from `343` to `227`

## Verified Flows

- [x] `workspace/flow/scripts/targets/vck190/readme.sh step0`
- [x] `workspace/flow/scripts/targets/vck190/readme.sh step1`
- [x] `workspace/flow/scripts/targets/vck190/readme.sh step2`
- [x] `workspace/flow/scripts/targets/vck190/readme.sh step3-all`
- [x] `workspace/flow/scripts/targets/vck190/readme.sh step4-export`
- [x] `run_vck190_attn_mlp_dsp_flow.sh prepare`
- [x] `run_vck190_block_subset_flow.sh prepare`
- [x] `run_vck190_patch_head_dsp_flow.sh prepare`
- [x] `workspace/flow/scripts/targets/zynqmp/vivado.sh step4-prepare zcu102`
- [x] `workspace/flow/scripts/targets/zynqmp/spinal.sh env`
- [x] `workspace/flow/scripts/targets/zynqmp/spinal.sh step4-export`
- [x] placeholder-assisted `workspace/flow/scripts/targets/zynqmp/spinal.sh step4-prepare zcu102`

## Current Patch Snapshot

- [x] Add VCK190 `reuse-ooc-dcp` / `rebuild-ip` board-export modes
- [x] Sanitize Vivado batch environment for VCK190 DDRMC / NoC IP generation
- [x] Patch VCK190 OOC DCP pre-impl hook to find `HGPIPE_VIVADO_0` robustly
- [x] Add ZU15EG-specific `full_deit_tiny_fit` design config
- [x] Align ZU15EG HLS and board clock policy to `3.0 ns`
- [x] Add storage-style macros for reshaper and ATTN FIFO binding
- [x] Apply ZU15EG URAM storage policy without reducing `ATTN QKV/R/A`
- [x] Keep generated Vivado / HLS / Spinal artifacts out of the source commit

## Active Validation

- [ ] `vck190 step4-board-export` retry in `reuse-ooc-dcp` mode after pre-impl hook patch
- [ ] `zu15eg` targeted HLS check for `PATCH_EMBED,ATTN8,MLP0` at `3.0 ns`
- [ ] `zu15eg` full `step1` / `step2` report regeneration
- [ ] `zu15eg step4-synth` resource DRC check
- [ ] `zcu102 step4-bitstream` retry under the host-identity profile

## Remaining Work

- [ ] Recover the remaining real `.dat` inventory for all Spinal-backed cases
- [ ] Bring strict Spinal export missing-file count to `0`
- [ ] Confirm final `zcu102` bitstream completion
- [ ] Confirm final `vck190` board export and device-image completion
- [ ] Confirm final `zu15eg` resource-fit completion
- [ ] Push the committed tree to the configured remote repository
