# Project History

This document records the merge and validation history of `ViT_Accel`.

## Repository Target

- Target local tree: `ViT_Accel`
- Target remote repository: `https://github.com/koodg123/ViT_Accel`
- Structural base: `HG-PIPE-codex-home-wsl`
- Functional sources ported on top:
  - `HG-PIPE-main`
  - `HG-PIPE-exp`
  - `HG-PIPE-codex-vivado-flow-hardening-docs`

## Phase 1. Branch Analysis

The first task was to analyze the four sibling worktrees under `HG_PIPE_MERGE` and determine whether a direct merge was safe.

Findings:

- `main`, `exp`, and `codex-vivado-flow-hardening-docs` shared a common history.
- `codex/home-wsl` had no merge-base with `main`.
- `codex/home-wsl` provided the most maintainable final layout because it already reorganized the repository around:
  - `workspace/flow`
  - `workspace/hardware`
  - `workspace/board`
  - `workspace/artifacts`
  - `automation`

Conclusion:

- The integration should not be treated as a normal `git merge`.
- The repository should be rebuilt as a staged replay/port on top of the `codex/home-wsl` layout.

## Phase 2. Docs-First Integration

The merge started with documentation to stabilize terminology, paths, and operator guidance before code movement.

Integrated documents included:

- merged `README.md`
- `docs/LOCAL_SETUP.md`
- `docs/MULTI_BOARD_VALIDATION.md`
- `docs/VCK190_DOCKER_BASELINE.md`
- `docs/VIVADO_ONLY_FLOW.md`
- `docs/SRC_CASE_ANALYSIS.md`
- `docs/SRC_CASE_DIAGRAMS.md`
- `docs/SRC_CASE_MICROARCH.md`
- `docs/SRC_CASE_MODULE_GUIDE.md`
- `docs/MERGE_STATUS.md`

## Phase 3. Skeleton and Layout Creation

After the documentation pass, the actual clustered repository layout was created in `ViT_Accel`.

Created or normalized areas:

- `workspace/flow/entrypoints`
- `workspace/flow/scripts`
- `workspace/hardware/{SPINAL,case,src,statistics,vivado}`
- `workspace/board/notebooks`
- `workspace/artifacts/{build,instances,reports,logs}`
- `automation/{cli,flows,legacy,utils}`
- `configs/{models,targets,designs,experiments}`
- `docker/`

This phase established the path conventions that later code ports had to follow.

## Phase 4. Code Port

The first full code-port pass then moved the functional parts of the branch family into `ViT_Accel`.

Imported content:

- `automation/*`
- `configs/*`
- `workspace/flow/entrypoints/*`
- `workspace/flow/scripts/*`
- `workspace/hardware/*`

Feature ports that mattered most:

- `main` VCK190 helper flows
- `main` DSP override helpers and case preparation tools
- `main` residual FIFO URAM changes in `attn.h` and `mlp.h`
- `exp` ZynqMP Spinal-backed runners
- config-driven `board-bitstream` and `spinal-board-bitstream` CLI support

## Phase 5. Docker Runtime Stabilization

The repository then moved from static merge work into execution bring-up.

Problems observed:

- container sessions frequently entered as `root`
- mounted repository files were created as `root:root`
- Xilinx license variables were not forwarded reliably
- daemon-mode containers sometimes exited immediately

Fixes applied:

- host UID/GID mapping in `workspace/flow/scripts/docker/container_run.sh`
- non-root attach helper in `workspace/flow/scripts/docker/container_exec.sh`
- license forwarding for:
  - `LM_LICENSE_FILE`
  - `XILINXD_LICENSE_FILE`
  - `XILINX_LICENSE_FILE`
- daemon keepalive behavior
- safer entrypoint handling for `spinal_bin_refs`

Result:

- Docker-created files now land as the host user instead of `root`
- container `HOME` and Xilinx local state are isolated under `/tmp/hgpipe-home-<uid>`

## Phase 6. Spinal `.dat` Recovery Work

The next blocker was not merge structure but generated artifact completeness.

What was missing:

- the checked-in Spinal export did not include the real `.dat` memory initializers needed by the legacy exported bundle

What was added:

- `workspace/flow/scripts/ops/spinal_dat_recover.sh`
- `workspace/flow/scripts/run/spinal_dat_recover_backend.sh`
- recovery-oriented `HGPIPE_STEP1_DO_CSIM` / `HGPIPE_STEP1_DO_COSIM` controls in the legacy HLS path

Recovered cases verified during the work:

- `HEAD`
- `PATCH_EMBED`
- `ATTN0`
- `ATTN1`
- `ATTN2`
- `ATTN10`
- `ATTN11`
- `MLP0`

This reduced the strict missing `.dat` count for the exported Spinal bundle from `343` to `227`.

## Phase 7. Baseline Flow Validation

The repository was then validated inside Docker instead of only by static inspection.

Validated successfully:

- VCK190 README flow through `step4-export`
- HLS `step2` summary generation with `26/26 COSIM_PASS`
- Spinal `step4-export`
- ZynqMP `step4-prepare zcu102`

Important generated outputs included:

- `workspace/artifacts/reports/<target>/deit_tiny_baseline/full_deit_tiny_fit/step2_hls_resource_summary.*`
- `workspace/hardware/SPINAL/BlockSequence.v`
- `workspace/hardware/SPINAL/BlockSequence_bb.v`

## Phase 8. Vivado Crash Investigation

The remaining failures were concentrated in the heavy Vivado implementation stages.

Original failures:

- `zcu102 step4-bitstream` failed near `launch_runs synth_1`
- `vck190 step4-ooc` failed immediately after synthesis license checkout

The common signature pointed to Docker host identity and Vivado runtime interaction:

- `libudev`
- `libXil_lmgr11`
- WebTalk / host-info / license checkout paths

Mitigations added:

- `--network`
- `--hostname`
- `--mac-address`
- optional host `/run/udev` mount
- optional host `machine-id` mount

Latest observed result at the time of writing:

- the host-identity retry of `vck190 step4-ooc` no longer dies immediately at license checkout
- it now advances into a long-running OOC synthesis stage

## Current State

At the end of this history snapshot:

- the merged repository structure is in place
- the main helper flows are ported
- Docker ownership and launch behavior are stabilized
- baseline VCK190 flow is validated through `step4-export`
- ZynqMP prepare flow is validated
- Spinal export regeneration is working
- the main open technical item is final Vivado completion for:
  - `vck190 step4-ooc`
  - `zcu102 step4-bitstream`
