# Deployment and Run Scripts

This document summarizes the scripts you actually use to build, validate, and deploy flows in `ViT_Accel`.

For the full documentation map, see:

- `docs/README.md`

For full board-by-board copy-paste command sequences, see:

- `docs/BOARD_RUNBOOKS.md`

## Script Layout

`workspace/flow/scripts` is now organized by role.

- `workspace/flow/scripts/targets/vck190`
  - VCK190 user-facing wrappers
- `workspace/flow/scripts/targets/zynqmp`
  - generic ZynqMP user-facing wrappers
- `workspace/flow/scripts/targets/zu15eg`
  - ZU15EG-specific host and Docker wrappers
- `workspace/flow/scripts/targets/zcu102`
  - ZCU102-specific host and Docker wrappers
- `workspace/flow/scripts/targets/zcu104`
  - ZCU104-specific host and Docker wrappers
- `workspace/flow/scripts/targets/ultra96`
  - Ultra96-V2-specific host and Docker wrappers
- `workspace/flow/scripts/host`
  - host-only repro and Vivado launch helpers
- `workspace/flow/scripts/ops`
  - maintenance and recovery helpers
- `workspace/flow/scripts/docker`
  - Docker runtime helpers
- `workspace/flow/scripts/env`
  - environment setup and WebTalk hardening
- `workspace/flow/scripts/run`
  - backend implementations

## 1. Docker Lifecycle

### Build the Ubuntu host image

```bash
docker build -f docker/Dockerfile.host-ubuntu22.04 -t vit-accel:ubuntu22 .
```

### Launch the Docker runtime

Recommended host-identity profile:

```bash
export LM_LICENSE_FILE=2100@<license-host>
export XILINXD_LICENSE_FILE=2100@<license-host>

sudo --preserve-env=LM_LICENSE_FILE,XILINXD_LICENSE_FILE,XILINX_LICENSE_FILE \
  bash workspace/flow/scripts/docker/container_run.sh \
  --daemon \
  --name vit-accel-ubuntu22 \
  --image vit-accel:ubuntu22 \
  --network host \
  --hostname "$(hostname)"
```

### Attach to the running container

```bash
sudo --preserve-env=LM_LICENSE_FILE,XILINXD_LICENSE_FILE,XILINX_LICENSE_FILE \
  bash workspace/flow/scripts/docker/container_exec.sh \
  --name vit-accel-ubuntu22
```

Preferred VCK190 attach wrapper:

```bash
bash workspace/flow/scripts/targets/vck190/connect.sh
```

Optional ZynqMP Docker mode wrappers:

```bash
workspace/flow/scripts/targets/zu15eg/nonspinal.sh
workspace/flow/scripts/targets/zu15eg/spinal.sh
workspace/flow/scripts/targets/zcu102/nonspinal.sh
workspace/flow/scripts/targets/zcu102/spinal.sh
workspace/flow/scripts/targets/zcu104/nonspinal.sh
workspace/flow/scripts/targets/zcu104/spinal.sh
workspace/flow/scripts/targets/ultra96/nonspinal.sh
workspace/flow/scripts/targets/ultra96/spinal.sh
```

## 2. VCK190 Baseline Scripts

Main wrapper:

```bash
workspace/flow/scripts/targets/vck190/readme.sh
```

Preferred Docker launch wrapper:

```bash
workspace/flow/scripts/targets/vck190/up.sh
```

Preferred Docker-safe wrapper:

```bash
workspace/flow/scripts/targets/vck190/docker.sh
```

Preferred Docker attach wrapper:

```bash
workspace/flow/scripts/targets/vck190/connect.sh
```

Recommended VCK190 Docker launch:

```bash
export LM_LICENSE_FILE=2100@<license-host>
export XILINXD_LICENSE_FILE=2100@<license-host>

bash workspace/flow/scripts/targets/vck190/up.sh \
  --daemon \
  --network host \
  --hostname "$(hostname)"
```

Common commands:

```bash
workspace/flow/scripts/targets/vck190/readme.sh env
workspace/flow/scripts/targets/vck190/readme.sh step0
workspace/flow/scripts/targets/vck190/readme.sh step1
workspace/flow/scripts/targets/vck190/readme.sh step2
workspace/flow/scripts/targets/vck190/readme.sh step3-all
workspace/flow/scripts/targets/vck190/readme.sh step4-export
workspace/flow/scripts/targets/vck190/readme.sh step4-ooc
workspace/flow/scripts/targets/vck190/readme.sh step4-board-export
workspace/flow/scripts/targets/vck190/readme.sh step5
workspace/flow/scripts/targets/vck190/readme.sh baseline
```

Purpose:

- baseline end-to-end HLS -> Spinal -> export flow
- final VCK190 accelerator-only OOC implementation
- optional VCK190 full board export from either routed OOC DCP reuse or packaged-IP rebuild

Dedicated step5 wrapper:

```bash
workspace/flow/scripts/targets/vck190/pynq_handoff.sh
workspace/flow/scripts/targets/vck190/pynq_handoff.sh --bundle-dir /tmp/vck190_step5 --report-path /tmp/vck190_step5.md
workspace/flow/scripts/targets/vck190/pynq_handoff.sh --run-package
workspace/flow/scripts/targets/vck190/pynq_handoff.sh --run-board-export --reuse-ooc-dcp
workspace/flow/scripts/targets/vck190/pynq_handoff.sh --run-board-export --rebuild-ip
```

Docker-safe VCK190 commands:

```bash
workspace/flow/scripts/targets/vck190/docker.sh env
workspace/flow/scripts/targets/vck190/docker.sh step0
workspace/flow/scripts/targets/vck190/docker.sh step1
workspace/flow/scripts/targets/vck190/docker.sh step2
workspace/flow/scripts/targets/vck190/docker.sh step3-all
workspace/flow/scripts/targets/vck190/docker.sh step4-export
workspace/flow/scripts/targets/vck190/docker.sh baseline
```

Recommended VCK190 board-export examples:

```bash
workspace/flow/scripts/targets/vck190/readme.sh step4-board-export
HGPIPE_STEP4_BOARD_EXPORT_MODE=reuse-ooc-dcp workspace/flow/scripts/targets/vck190/readme.sh step4-board-export
HGPIPE_STEP5_EXPORT_MODE=rebuild-ip workspace/flow/scripts/targets/vck190/pynq_handoff.sh --run-board-export --run-package
```

Mode summary:

- `step4-board-export`
  - straight-through board export helper
  - defaults to `rebuild-ip`
- `step5 --run-board-export`
  - handoff/report + board export helper
  - defaults to `reuse-ooc-dcp`

Host-only VCK190 commands:

```bash
workspace/flow/scripts/targets/vck190/readme.sh step4-ooc
workspace/flow/scripts/targets/vck190/readme.sh step4-board-export
workspace/flow/scripts/targets/vck190/pynq_handoff.sh --run-board-export --reuse-ooc-dcp
```

## 3. VCK190 Helper Scripts

### DSP-focused ATTN/MLP helper

```bash
workspace/flow/scripts/targets/vck190/attn_mlp_dsp.sh prepare
workspace/flow/scripts/targets/vck190/attn_mlp_dsp.sh step1
workspace/flow/scripts/targets/vck190/attn_mlp_dsp.sh step2
```

### Block subset helper

```bash
workspace/flow/scripts/targets/vck190/block_subset.sh prepare
workspace/flow/scripts/targets/vck190/block_subset.sh step1
workspace/flow/scripts/targets/vck190/block_subset.sh step2
```

### PATCH/HEAD DSP helper

```bash
workspace/flow/scripts/targets/vck190/patch_head_dsp.sh prepare
workspace/flow/scripts/targets/vck190/patch_head_dsp.sh step1
workspace/flow/scripts/targets/vck190/patch_head_dsp.sh step2
```

Supporting utilities:

```bash
python3 workspace/flow/scripts/utils/make_attn_mlp_dsp_cases.py --help
python3 workspace/flow/scripts/utils/make_case_subset.py --help
bash workspace/flow/scripts/utils/check_xilinx_mounts.sh
```

## 4. ZynqMP Vivado-Only Scripts

Main wrapper:

```bash
workspace/flow/scripts/targets/zynqmp/vivado.sh
```

Common commands:

```bash
workspace/flow/scripts/targets/zynqmp/vivado.sh env
workspace/flow/scripts/targets/zynqmp/vivado.sh step0
workspace/flow/scripts/targets/zynqmp/vivado.sh step1
workspace/flow/scripts/targets/zynqmp/vivado.sh step2
workspace/flow/scripts/targets/zynqmp/vivado.sh step4-prepare zcu102
workspace/flow/scripts/targets/zynqmp/vivado.sh step4-validate zcu102
workspace/flow/scripts/targets/zynqmp/vivado.sh step4-synth zcu102
workspace/flow/scripts/targets/zynqmp/vivado.sh step4-bitstream zcu102
```

`HGPIPE_BUILD_ROOT` note:

- default build layout is `workspace/artifacts/build/<target>/<experiment>/`
- you may point `HGPIPE_BUILD_ROOT` to either the parent build root or the final experiment directory
- the backend now avoids appending `<experiment>` twice when the final directory is already supplied

Supported targets:

- `zu15eg`
- `zcu102`
- `zcu104`
- `ultra96v2`

## 5. ZynqMP Spinal-Backed Scripts

Main wrapper:

```bash
workspace/flow/scripts/targets/zynqmp/spinal.sh
```

Common commands:

```bash
workspace/flow/scripts/targets/zynqmp/spinal.sh env
workspace/flow/scripts/targets/zynqmp/spinal.sh step0
workspace/flow/scripts/targets/zynqmp/spinal.sh step1
workspace/flow/scripts/targets/zynqmp/spinal.sh step2
workspace/flow/scripts/targets/zynqmp/spinal.sh step3-all
workspace/flow/scripts/targets/zynqmp/spinal.sh step4-export
workspace/flow/scripts/targets/zynqmp/spinal.sh step4-prepare zcu102
workspace/flow/scripts/targets/zynqmp/spinal.sh step4-bitstream zcu102
```

The same `HGPIPE_BUILD_ROOT` behavior applies to the Spinal-backed wrapper.

Important note:

- this path requires the exported Spinal `.dat` memory files to exist unless placeholder smoke-test mode is explicitly enabled

## 6. ZU15EG README-Style Host Flow

Main wrapper:

```bash
workspace/flow/scripts/targets/zu15eg/host.sh
```

This wrapper pins the generic ZynqMP Vivado-only flow to:

- target: `zu15eg`
- HLS experiment seed: `configs/experiments/zu15eg_full_deit_tiny_fit.json`
- experiment suffix: `full_deit_tiny_fit`

It is intentionally host-friendly and does not require `sbt` / SpinalHDL.

Quick start on host:

```bash
cd /home/kjm26/project/PRJXR/HG_PIPE_MERGE/ViT_Accel

export LM_LICENSE_FILE=2100@<license-host>
export XILINXD_LICENSE_FILE=2100@<license-host>
export XILINX_ROOT=/tools/Xilinx
export XILINX_VERSION=2023.2

bash workspace/flow/scripts/targets/zu15eg/host.sh step0
bash workspace/flow/scripts/targets/zu15eg/host.sh step1
bash workspace/flow/scripts/targets/zu15eg/host.sh step2
bash workspace/flow/scripts/targets/zu15eg/host.sh step4-prepare
bash workspace/flow/scripts/targets/zu15eg/host.sh step4-validate
bash workspace/flow/scripts/targets/zu15eg/host.sh step4-synth
bash workspace/flow/scripts/targets/zu15eg/host.sh step4-bitstream
```

Common commands:

```bash
workspace/flow/scripts/targets/zu15eg/host.sh env
workspace/flow/scripts/targets/zu15eg/host.sh step0
workspace/flow/scripts/targets/zu15eg/host.sh step1
workspace/flow/scripts/targets/zu15eg/host.sh step2
workspace/flow/scripts/targets/zu15eg/host.sh step4-prepare
workspace/flow/scripts/targets/zu15eg/host.sh step4-validate
workspace/flow/scripts/targets/zu15eg/host.sh step4-synth
workspace/flow/scripts/targets/zu15eg/host.sh step4-bitstream
workspace/flow/scripts/targets/zu15eg/host.sh step4-ooc
workspace/flow/scripts/targets/zu15eg/host.sh baseline
workspace/flow/scripts/targets/zu15eg/host.sh all
```

Notes:

- This is the closest host-side analogue to `workspace/flow/scripts/targets/vck190/readme.sh` for the ZynqMP family.
- `zu15eg_full_deit_tiny_fit.json` already uses the same model/design pair as `vck190_full_deit_tiny_fit.json`; only the target file changes.
- `step4-ooc` is accepted as a convenience alias and maps to the full ZU15EG `step4-bitstream` board flow.
- `baseline` is accepted as a convenience alias and maps to `all`.
- `step3-all` and `step4-export` are intentionally unsupported here. Use `workspace/flow/scripts/targets/zynqmp/spinal.sh` only when you explicitly want the Spinal-backed path.

## 7. ZU15EG Docker Flow

Main wrapper:

```bash
workspace/flow/scripts/targets/zu15eg/docker.sh
```

Modes:

- `nonspinal`
- `spinal`

Quick start in Docker:

```bash
cd /workspace

bash workspace/flow/scripts/targets/zu15eg/docker.sh nonspinal step0
bash workspace/flow/scripts/targets/zu15eg/docker.sh nonspinal step1
bash workspace/flow/scripts/targets/zu15eg/docker.sh nonspinal step2
bash workspace/flow/scripts/targets/zu15eg/docker.sh nonspinal step4-prepare

bash workspace/flow/scripts/targets/zu15eg/docker.sh spinal step3-all
bash workspace/flow/scripts/targets/zu15eg/docker.sh spinal step4-export
bash workspace/flow/scripts/targets/zu15eg/docker.sh spinal step4-prepare
```

Then move back to the host for Vivado execution:

```bash
cd /home/kjm26/project/PRJXR/HG_PIPE_MERGE/ViT_Accel

bash workspace/flow/scripts/targets/zu15eg/host.sh step4-validate
bash workspace/flow/scripts/targets/zu15eg/host.sh step4-synth
bash workspace/flow/scripts/targets/zu15eg/host.sh step4-bitstream
```

Docker-safe non-Spinal examples:

```bash
workspace/flow/scripts/targets/zu15eg/docker.sh nonspinal env
workspace/flow/scripts/targets/zu15eg/docker.sh nonspinal step0
workspace/flow/scripts/targets/zu15eg/docker.sh nonspinal step1
workspace/flow/scripts/targets/zu15eg/docker.sh nonspinal step2
workspace/flow/scripts/targets/zu15eg/docker.sh nonspinal step4-prepare
workspace/flow/scripts/targets/zu15eg/docker.sh nonspinal all
```

Docker-safe Spinal examples:

```bash
workspace/flow/scripts/targets/zu15eg/docker.sh spinal env
workspace/flow/scripts/targets/zu15eg/docker.sh spinal step0
workspace/flow/scripts/targets/zu15eg/docker.sh spinal step1
workspace/flow/scripts/targets/zu15eg/docker.sh spinal step2
workspace/flow/scripts/targets/zu15eg/docker.sh spinal step3-all
workspace/flow/scripts/targets/zu15eg/docker.sh spinal step4-export
workspace/flow/scripts/targets/zu15eg/docker.sh spinal step4-prepare
workspace/flow/scripts/targets/zu15eg/docker.sh spinal baseline
workspace/flow/scripts/targets/zu15eg/docker.sh spinal all
```

Host-only Vivado commands:

- `step4-validate`
- `step4-synth`
- `step4-bitstream`
- `step4-ooc`

## 8. Other Board-Specific ZynqMP Wrappers

These wrappers mirror the `zu15eg` split:

- host path: non-Spinal, host-only Vivado stages
- docker path: `nonspinal` and `spinal` preparation flows, but no Vivado execution

### ZCU102

```bash
workspace/flow/scripts/targets/zcu102/host.sh
workspace/flow/scripts/targets/zcu102/docker.sh
```

### ZCU104

```bash
workspace/flow/scripts/targets/zcu104/host.sh
workspace/flow/scripts/targets/zcu104/docker.sh
```

### Ultra96-V2

```bash
workspace/flow/scripts/targets/ultra96/host.sh
workspace/flow/scripts/targets/ultra96/docker.sh
```

Note:

- the wrapper directory is `ultra96`
- the internal target / experiment key remains `ultra96v2`

## 9. Config-Driven Commands

Primary CLI:

```bash
python3 workspace/flow/entrypoints/python/hgpipe_flow.py --help
```

Common commands:

```bash
python3 workspace/flow/entrypoints/python/hgpipe_flow.py baseline-report
python3 workspace/flow/entrypoints/python/hgpipe_flow.py experiment-sweep
python3 workspace/flow/entrypoints/python/hgpipe_flow.py vivado-package --experiment configs/experiments/vck190_full_deit_tiny_fit.json
python3 workspace/flow/entrypoints/python/hgpipe_flow.py board-impl --experiment configs/experiments/vck190_full_deit_tiny_fit.json --run
python3 workspace/flow/entrypoints/python/hgpipe_flow.py board-bitstream --experiment configs/experiments/zcu102_full_deit_tiny_fit.json --run
python3 workspace/flow/entrypoints/python/hgpipe_flow.py spinal-board-bitstream --experiment configs/experiments/zcu102_full_deit_tiny_fit.json --run
```

Use this surface when you want structured build bundles and a cleaner internal API than the legacy wrappers.

## 9. Recovery Scripts

To rebuild missing Spinal `.dat` files from HLS outputs:

```bash
bash workspace/flow/scripts/ops/spinal_dat_recover.sh
```

or directly:

```bash
bash workspace/flow/scripts/run/spinal_dat_recover_backend.sh
```

This is the operational path for bringing the exported Spinal Vivado bundle back to a fully populated state.

## 10. Environment Knobs You Will Reuse Often

Typical knobs:

```bash
XILINX_ROOT=/tools/Xilinx
XILINX_VERSION=2023.2
HGPIPE_STEP1_MAX_THREADS=16
HGPIPE_STEP4_MAX_THREADS=4
HGPIPE_STEP4_FLATTEN_HIERARCHY=none
HGPIPE_EXPERIMENT_PATH=configs/experiments/vck190_full_deit_tiny_fit.json
HGPIPE_TARGETS="zcu102 zcu104"
HGPIPE_EXPERIMENT_SUFFIX=full_deit_tiny_fit
```

Docker and licensing:

```bash
HGPIPE_DOCKER_IMAGE=vit-accel:ubuntu22
HGPIPE_DOCKER_NETWORK_MODE=host
HGPIPE_DOCKER_HOSTNAME="$(hostname)"
LM_LICENSE_FILE=2100@<license-host>
XILINXD_LICENSE_FILE=2100@<license-host>
```

## 11. Host Vivado Reproduction Scripts

Use these when you want to compare Docker failures against a direct host Vivado run.

### VCK190 `step4-ooc`

```bash
export LM_LICENSE_FILE=2100@<license-host>
export XILINXD_LICENSE_FILE=2100@<license-host>
export XILINX_ROOT=/tools/Xilinx
export XILINX_VERSION=2023.2

bash workspace/flow/scripts/host/vck190_ooc.sh
```

### ZCU102 `step4-synth`

```bash
export LM_LICENSE_FILE=2100@<license-host>
export XILINXD_LICENSE_FILE=2100@<license-host>
export XILINX_ROOT=/tools/Xilinx
export XILINX_VERSION=2023.2

bash workspace/flow/scripts/host/board_synth.sh
```

You can override the target for the second script:

```bash
bash workspace/flow/scripts/host/board_synth.sh zcu104
```
