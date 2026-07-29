# VCK190 Baseline In Docker

This runbook merges:

- the clustered `workspace/` layout from `codex/home-wsl`
- the VCK190 helper experiment notes from `main`

The commands below describe the current merged command surface for `ViT_Accel`.

## 1. Build And Enter The Container

Build the image once:

```bash
bash workspace/flow/scripts/docker/container_run.sh --build
```

Default build behavior:

- by default the launcher builds from `docker/Dockerfile.host-ubuntu22.04`
- otherwise the launcher falls back to `docker/Dockerfile.host-ubuntu22.04`
- override explicitly with `HGPIPE_DOCKERFILE=/abs/path/to/Dockerfile`

Open a shell with the repo and `/tools/Xilinx` mounted:

```bash
bash workspace/flow/scripts/docker/container_run.sh
```

Current default behavior on Ubuntu hosts:

- the container runs with the host UID/GID by default
- files created under the mounted repository should stay owned by the host user
- use `--as-root` only when you intentionally need a root shell inside the container
- if the host already exports `LM_LICENSE_FILE`, `XILINXD_LICENSE_FILE`, or `XILINX_LICENSE_FILE`, the launcher forwards them into the container automatically
- the launcher can optionally expose host identity hints to Vivado with `--network host`, `--hostname`, host `udev`, and host `machine-id` mounts

If you want to restrict how many host CPUs the container can use:

```bash
bash workspace/flow/scripts/docker/container_run.sh --cpus 8
bash workspace/flow/scripts/docker/container_run.sh --cpuset 0-15
```

You can also use environment variables:

```bash
HGPIPE_DOCKER_CPUS=8 bash workspace/flow/scripts/docker/container_run.sh
HGPIPE_DOCKER_CPUSET=0-15 bash workspace/flow/scripts/docker/container_run.sh
```

If Vivado crashes around WebTalk, host-info, or license checkout inside Docker, try the host-identity mitigation profile:

```bash
HGPIPE_DOCKER_NETWORK_MODE=host \
HGPIPE_DOCKER_HOSTNAME="$(hostname)" \
HGPIPE_MOUNT_HOST_UDEV=1 \
HGPIPE_MOUNT_HOST_MACHINE_ID=1 \
bash workspace/flow/scripts/docker/container_run.sh --daemon --name vit-accel-ubuntu22
```

The launcher now adds Docker `--init` by default. This keeps a minimal init process as PID 1 and helps reap exited child processes so repeated `xsim` runs do not accumulate zombie entries. Disable only when needed with `HGPIPE_DOCKER_INIT=0` or `--no-init`.

If your license is tied to a specific MAC address and you do not use host networking:

```bash
HGPIPE_DOCKER_NETWORK_MODE=bridge \
HGPIPE_DOCKER_MAC_ADDRESS="02:42:ac:11:00:02" \
bash workspace/flow/scripts/docker/container_run.sh --daemon --name vit-accel-ubuntu22
```

Preferred VCK190 launcher:

```bash
export LM_LICENSE_FILE=2100@<license-host>
export XILINXD_LICENSE_FILE=2100@<license-host>

bash workspace/flow/scripts/targets/vck190/up.sh \
  --daemon \
  --network host \
  --hostname "$(hostname)"
```

If you launch from a non-interactive IDE task:

```bash
bash workspace/flow/scripts/targets/vck190/up.sh --daemon
bash workspace/flow/scripts/targets/vck190/connect.sh
```

Run a single command from the host:

```bash
bash workspace/flow/scripts/docker/container_run.sh workspace/flow/scripts/targets/vck190/readme.sh env
```

Preferred attach command for the default VCK190 container:

```bash
bash workspace/flow/scripts/targets/vck190/connect.sh
```

## 2. Tool Resolution Inside Docker

Inside the container:

```bash
workspace/flow/scripts/targets/vck190/docker.sh env
```

Expected defaults:

- `XILINX_ROOT=/tools/Xilinx`
- `XILINX_VERSION=2023.2`
- `LANG=en_US.UTF-8`
- `LC_ALL=en_US.UTF-8`

Useful quick checks:

```bash
locale
ls -l /usr/lib/*/libtinfo.so*
ls -ld /usr/include/bits /usr/include/gnu /usr/include/*-linux-gnu/bits /usr/include/*-linux-gnu/gnu
echo "$C_INCLUDE_PATH"
echo "$LIBRARY_PATH"
```

## 3. README-Oriented VCK190 Steps

Step 0:

```bash
workspace/flow/scripts/targets/vck190/docker.sh step0
```

Current default VCK190 case root:

```bash
workspace/hardware/case_vck190/
```

The VCK190 bash wrappers export `HGPIPE_CASE_ROOT=workspace/hardware/case_vck190` automatically so `step0` and `step1` work on the dedicated VCK190 `cpp` and `cpp.template` tree instead of mutating the shared baseline case directory.

If `workspace/hardware/case_vck190/refs` is missing, the helper will reuse `workspace/hardware/case/refs/` or extract `workspace/hardware/case/refs.7z` into the shared location first and then wire it into the VCK190 case root.

Step 1:

```bash
HGPIPE_STEP1_MAX_THREADS=16 workspace/flow/scripts/targets/vck190/docker.sh step1
```

DSP-enabled profile from the baseline runner:

```bash
HGPIPE_STEP1_MAX_THREADS=16 workspace/flow/scripts/targets/vck190/docker.sh step1-dsp
workspace/flow/scripts/targets/vck190/docker.sh step2-dsp
workspace/flow/scripts/targets/vck190/docker.sh baseline-dsp
```

Current baseline expectations:

- Docker Step1 defaults to `csim + csynth + cosim`
- `export_design -flow syn` remains off unless `HGPIPE_STEP1_DO_SYN=1`
- active HLS projects should land under `workspace/artifacts/instances/vck190/<model>/<design>/`

Step 2:

```bash
workspace/flow/scripts/targets/vck190/docker.sh step2
```

Target summary location:

- `workspace/artifacts/reports/vck190/<model>/<design>/`

Step 3 server:

```bash
workspace/flow/scripts/targets/vck190/docker.sh step3-server
workspace/flow/scripts/targets/vck190/docker.sh step3-blocks
```

If port `9966` is already in use:

```bash
HGPIPE_SPINAL_SERVER_PORT=19966 workspace/flow/scripts/targets/vck190/docker.sh step3-server
HGPIPE_SPINAL_SERVER_PORT=19966 workspace/flow/scripts/targets/vck190/docker.sh step3-blocks
```

Step 3 all-in-one:

```bash
workspace/flow/scripts/targets/vck190/docker.sh step3-all
```

With full-top simulation:

```bash
HGPIPE_RUN_FULL_TOP_SIM=1 workspace/flow/scripts/targets/vck190/docker.sh step3-all
```

Step 4 export preparation:

```bash
workspace/flow/scripts/targets/vck190/docker.sh step4-export
```

This prepares the legacy export bundle under `workspace/hardware/SPINAL/vivado/`.

Full Docker-safe baseline:

```bash
workspace/flow/scripts/targets/vck190/docker.sh baseline
```

## 4. Optional Batch OOC Implementation

For accelerator-only out-of-context implementation checks:

```bash
workspace/flow/scripts/targets/vck190/readme.sh step4-ooc
```

To let Vivado use more threads:

```bash
HGPIPE_STEP4_MAX_THREADS=16 workspace/flow/scripts/targets/vck190/readme.sh step4-ooc
```

To experiment with flattening behavior:

```bash
HGPIPE_STEP4_MAX_THREADS=16 \
HGPIPE_STEP4_FLATTEN_HIERARCHY=rebuilt \
workspace/flow/scripts/targets/vck190/readme.sh step4-ooc
```

These steps are intentionally host-only and are blocked by `workspace/flow/scripts/targets/vck190/docker.sh`:

- `step4-ooc`
- `step4-board-export`
- `step5 --run-board-export`

## 5. Board Export Paths

Straight-through board export from the main wrapper:

```bash
workspace/flow/scripts/targets/vck190/readme.sh step4-board-export
```

This command defaults to `rebuild-ip`, meaning the packaged accelerator IP is rebuilt and the board project runs synth/impl/export from RTL/IP inputs.

To force routed OOC checkpoint reuse instead:

```bash
HGPIPE_STEP4_BOARD_EXPORT_MODE=reuse-ooc-dcp \
workspace/flow/scripts/targets/vck190/readme.sh step4-board-export
```

Dedicated Step5 board-export handoff examples:

```bash
workspace/flow/scripts/targets/vck190/pynq_handoff.sh --run-board-export --reuse-ooc-dcp
workspace/flow/scripts/targets/vck190/pynq_handoff.sh --run-board-export --rebuild-ip
workspace/flow/scripts/targets/vck190/pynq_handoff.sh --run-board-export --rebuild-ip --run-package
```

Mode summary:

- `--reuse-ooc-dcp`
  - reuse `step4-ooc` routed `post_route.dcp`
  - keep the accelerator out of board-level re-PnR as much as Vivado allows
- `--rebuild-ip`
  - rebuild the accelerator IP/package and rerun full board synth/impl/export
- `step5 --run-board-export`
  - defaults to `reuse-ooc-dcp`
- `step4-board-export`
  - defaults to `rebuild-ip`

Current artifact layout after the recent path cleanup:

- board builds: `workspace/artifacts/build/vck190/<experiment>/`
- Step5 slim delivery tree: `workspace/artifacts/build/vck190/<experiment>/results/`
- live board-export workspace: `workspace/artifacts/build/vck190/<experiment>/board_export/`

## 6. `main` Branch Helper Experiment Flows

These helpers are already present in the merged tree.

### ATTN / MLP DSP Override Flow

Path:

```bash
HGPIPE_STEP1_MAX_THREADS=16 bash workspace/flow/scripts/run/run_vck190_attn_mlp_dsp_flow.sh step1
bash workspace/flow/scripts/run/run_vck190_attn_mlp_dsp_flow.sh step2
```

Behavior:

- keep the shared baseline `workspace/hardware/case/*.cpp` files untouched
- use the VCK190-dedicated default case root `workspace/hardware/case_vck190/`
- prepare a DSP-enabled override tree
- point Step1 at that override case root

### PATCH / HEAD DSP Override Flow

Path:

```bash
HGPIPE_STEP1_MAX_THREADS=16 bash workspace/flow/scripts/run/run_vck190_patch_head_dsp_flow.sh step1
bash workspace/flow/scripts/run/run_vck190_patch_head_dsp_flow.sh step2
```

Behavior:

- only `PATCH_EMBED` and `HEAD` prefer `USE_DSP=true`
- the rest of the model remains on baseline settings

### Block-Subset Flow

Path:

```bash
bash workspace/flow/scripts/run/run_vck190_block_subset_flow.sh step1
bash workspace/flow/scripts/run/run_vck190_block_subset_flow.sh step2
```

Intended subset:

- `PATCH_EMBED`
- `ATTN0`
- `MLP0`
- `ATTN1`
- `MLP1`
- `HEAD`

### Xilinx Mount Diagnostics

Helper location:

```bash
workspace/flow/scripts/utils/check_xilinx_mounts.sh
```

This helper preserves the diagnostics added in `main` for Docker/Xilinx bind-mount debugging.

## 7. Full Baseline Convenience Command

```bash
workspace/flow/scripts/targets/vck190/readme.sh baseline
```

This performs:

- `step0`
- `step1`
- `step2`
- `step3-all`
- `step4-export`

## Notes

- The README baseline still relies on Spinal for Step 3 and Step 4 export preparation.
- The batch `step4-ooc` path is a useful accelerator-only implementation helper, not a full replacement for the original VCK190 board-design packaging flow.
- The merged tree should preserve the `workspace/` artifact layout even when reintroducing the `main` helper flows.
- Smoke-tested locally in this merge tree:
- `workspace/flow/scripts/targets/vck190/readme.sh env`
- `VITIS_HLS_BIN=/bin/true HGPIPE_STEP1_MAX_THREADS=1 workspace/flow/scripts/targets/vck190/readme.sh step1-dsp`
- `workspace/flow/scripts/targets/vck190/attn_mlp_dsp.sh prepare`
  - `workspace/flow/scripts/targets/vck190/patch_head_dsp.sh prepare`
  - `VITIS_HLS_BIN=/bin/true HGPIPE_STEP1_MAX_THREADS=1 workspace/flow/scripts/targets/vck190/block_subset.sh step1`
- Docker image build was attempted, but this host account cannot access the Docker daemon, and `sudo -n docker ...` also requires an interactive password.
