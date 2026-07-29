# Local Setup

This document reflects the current merged `ViT_Accel` layout.
It is based on the `codex/home-wsl` environment model and keeps the clustered `workspace/` paths as the canonical locations.

## What Gets Installed

- Python 3 packages from `requirements.txt`
- OpenJDK 11
- `sbt`
- Verilator `v4.228`
- `p7zip` for reference archives

Vivado / Vitis HLS is not installed by these repository scripts. If `vitis_hls` is not already on `PATH`, set:

```bash
export VITIS_HLS_BIN=/path/to/vitis_hls
```

## Local Install

The installer path is:

```bash
bash workspace/flow/scripts/env/bootstrap_local.sh
```

That installer is expected to:

- create `.venv/`
- install the Python packages there
- prepare local binary reference files for Spinal simulation

Activate the environment before running the Python step scripts:

```bash
source .venv/bin/activate
```

## Docker

The canonical Docker launcher for the merged tree is expected to be:

```bash
bash workspace/flow/scripts/docker/container_run.sh --build
bash workspace/flow/scripts/docker/container_run.sh
```

On Ubuntu hosts, the launcher now maps the container process to the host UID/GID by default so files created in the mounted repository do not become `root:root`.

If you already export Xilinx license variables on the host, the launcher now forwards these into the container automatically:

- `LM_LICENSE_FILE`
- `XILINXD_LICENSE_FILE`
- `XILINX_LICENSE_FILE`

Typical example:

```bash
export XILINXD_LICENSE_FILE=2100@210.97.15.85
export LM_LICENSE_FILE=2100@210.97.15.85
bash workspace/flow/scripts/docker/container_run.sh
```

If Vivado crashes near WebTalk, host-info, or license checkout inside Docker, retry with host identity exposed more directly:

```bash
export XILINXD_LICENSE_FILE=2100@210.97.15.85
export LM_LICENSE_FILE=2100@210.97.15.85
HGPIPE_DOCKER_NETWORK_MODE=host \
HGPIPE_DOCKER_HOSTNAME="$(hostname)" \
HGPIPE_MOUNT_HOST_UDEV=1 \
HGPIPE_MOUNT_HOST_MACHINE_ID=1 \
bash workspace/flow/scripts/docker/container_run.sh --daemon --name vit-accel-ubuntu22
```

`workspace/flow/scripts/docker/container_run.sh` now uses Docker `--init` by default. This reduces zombie child processes in daemon containers, especially after interrupted `xsim` runs. Override with `HGPIPE_DOCKER_INIT=0` or `--no-init` only when you need to debug the old behavior.

If you need to mimic a fixed MAC-bound license host in bridge mode:

```bash
HGPIPE_DOCKER_NETWORK_MODE=bridge \
HGPIPE_DOCKER_MAC_ADDRESS="02:42:ac:11:00:02" \
bash workspace/flow/scripts/docker/container_run.sh --daemon --name vit-accel-ubuntu22
```

If you explicitly need a root shell, use:

```bash
bash workspace/flow/scripts/docker/container_run.sh --as-root
```

If you start the helper from a non-interactive IDE task:

```bash
bash workspace/flow/scripts/docker/container_run.sh --daemon
bash workspace/flow/scripts/docker/container_exec.sh
```

If you want to use host-installed Vivado / Vitis HLS from inside the container, mount the tool install and point the environment at it:

```bash
docker run --rm -it \
  -v "$(pwd):/workspace" \
  -v /opt/Xilinx:/opt/Xilinx \
  -e VITIS_HLS_BIN=/opt/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls \
  -w /workspace \
  hg-pipe
```

If your Xilinx install is mounted at `/tools/Xilinx`, the merged helper is expected to resolve:

- `XILINX_ROOT=/tools/Xilinx`
- `XILINX_VERSION=2023.2`

## Common Environment Notes

If Vitis HLS reports missing locale, `libtinfo.so.5`, `bits/wordsize.h`, `crt1.o`, `crti.o`, `-lm`, or `-lpthread`, rebuild the image so the latest Dockerfile and environment helper changes are included.

The merged tree is expected to keep:

- Docker image definition under `docker/`
- runner scripts under `workspace/flow/scripts/docker/`
- environment helpers under `workspace/flow/scripts/env/`

## Step3 Binary Reference Files

Spinal simulation is expected to resolve binary references from:

- `HG_PIPE_BIN_REF_DIR` if set
- otherwise `workspace/artifacts/spinal_bin_refs/`

The helper entrypoint is:

```bash
python3 workspace/flow/scripts/utils/ref_txt_to_bin.py
```

The canonical implementation lives under:

```bash
workspace/flow/scripts/utils/ref_txt_to_bin.py
```

## Current Blocker

The code paths in this document now exist in `ViT_Accel`, and Docker-backed verification has already confirmed:

- non-root container execution on the mounted repository
- forwarded Xilinx license variables
- working `sbt` inside the Ubuntu 22.04 Docker image
- successful Spinal top-level export regeneration

The remaining real blocker is narrower:

- the Spinal generator recreates `BlockSequence.v` and `BlockSequence_bb.v`
- but the real `.dat` memory files still come from HLS instance outputs, and those are not yet regenerated in the current workspace

The merged tree now includes a helper to recover those files incrementally:

```bash
cd /home/kjm26/project/PRJXR/HG_PIPE_MERGE/ViT_Accel
bash workspace/flow/scripts/ops/spinal_dat_recover.sh HEAD ATTN0 ATTN1 ATTN2 ATTN10 ATTN11 MLP0 PATCH_EMBED
```

That helper:

- runs `step1` only for cases that do not already have HLS `.dat` outputs
- defaults to `HGPIPE_RECOVER_STEP1_DO_CSIM=0` and `HGPIPE_RECOVER_STEP1_DO_COSIM=0` so recovery focuses on generating the exported RTL/data bundle quickly
- imports the recovered `.dat` files back into `workspace/hardware/SPINAL/src/main/verilog/`
- reruns `workspace/flow/scripts/targets/zynqmp/spinal.sh step4-export`

Verified progress so far:

- `HEAD` contributed 5 real `.dat` files
- `ATTN0` contributed 20 real `.dat` files
- `ATTN1` contributed 20 real `.dat` files
- `ATTN2` contributed 20 real `.dat` files
- `ATTN10` contributed 20 real `.dat` files
- `ATTN11` contributed 20 real `.dat` files
- `MLP0` contributed 8 real `.dat` files
- `PATCH_EMBED` contributed 3 real `.dat` files
- strict `step4-prepare zcu102` missing-file count dropped from `343` to `227`
- `workspace/hardware/SPINAL/vivado/` now contains 116 real `.dat` files

Typical recovery sequence:

```bash
cd workspace/hardware/SPINAL
sbt "test:runMain network.generate_whole_network_verilog"
python3 to_vivado.py
cd /home/kjm26/project/PRJXR/HG_PIPE_MERGE/ViT_Accel
bash workspace/flow/scripts/targets/zynqmp/spinal.sh step4-prepare zcu102
```

For smoke-test bundle preparation only, you can opt in to placeholder `.dat` staging:

```bash
cd /home/kjm26/project/PRJXR/HG_PIPE_MERGE/ViT_Accel
HGPIPE_ALLOW_PLACEHOLDER_SPINAL_DAT=1 \
bash workspace/flow/scripts/targets/zynqmp/spinal.sh step4-prepare zcu102
```

That mode is only for validating the prepare path. It does not produce a functionally correct initialized hardware image.
