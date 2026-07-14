# Board Runbooks

This document collects the end-to-end shell commands for each board target.

Use this as the practical companion to:

- `docs/DEPLOYMENT_RUN_SCRIPTS.md`
- `docs/VCK190_DOCKER_BASELINE.md`
- `docs/VIVADO_ONLY_FLOW.md`

## 1. Common Setup

Host repository root:

```bash
cd /home/kjm26/project/PRJXR/HG_PIPE_MERGE/ViT_Accel
```

Common Docker image build:

```bash
sudo docker build -f docker/Dockerfile.host-ubuntu22.04 -t vit-accel:ubuntu22 .
```

Common host license setup:

```bash
export LM_LICENSE_FILE=2100@<license-host>
export XILINXD_LICENSE_FILE=2100@<license-host>
export XILINX_LICENSE_FILE="${XILINX_LICENSE_FILE:-${XILINXD_LICENSE_FILE}}"
```

Common host Vivado setup:

```bash
export XILINX_ROOT=/tools/Xilinx
export XILINX_VERSION=2023.2
```

If your host requires `sudo` for Docker, prefix the `up.sh` and `connect.sh`
commands with:

```bash
sudo --preserve-env=LM_LICENSE_FILE,XILINXD_LICENSE_FILE,XILINX_LICENSE_FILE
```

Inside the container, the repository root is:

```bash
cd /workspace
```

## 2. Dedicated Case Roots

Current dedicated case roots:

- `workspace/hardware/case_vck190`
- `workspace/hardware/case_zcu102`
- `workspace/hardware/case_zcu104`
- `workspace/hardware/case_zu15eg`
- `workspace/hardware/case_ultra96`

Current status:

- These dedicated case roots are separated per board so future tuning can be
  isolated cleanly.
- `vck190` remains the baseline reference case root.
- `zu15eg`, `zcu102`, `zcu104`, and `ultra96` now carry board-specific tuning
  changes in `PATCH_EMBED`, `ATTN`, and `MLP`.
- The current tuning summary and estimated resource / latency impact live in:
  - `docs/MULTI_BOARD_VALIDATION.md`

Wrapper-level differences currently in effect:

- `VCK190`
  - default case root: `workspace/hardware/case_vck190`
- `ZCU102`
  - `HGPIPE_CASE_ROOT=workspace/hardware/case_zcu102`
  - `HGPIPE_TARGETS=zcu102`
  - `HGPIPE_HLS_TARGET=zcu102`
- `ZCU104`
  - `HGPIPE_CASE_ROOT=workspace/hardware/case_zcu104`
  - `HGPIPE_TARGETS=zcu104`
  - `HGPIPE_HLS_TARGET=zcu104`
- `ZU15EG`
  - `HGPIPE_CASE_ROOT=workspace/hardware/case_zu15eg`
  - `HGPIPE_TARGETS=zu15eg`
  - `HGPIPE_HLS_TARGET=zu15eg`
- `Ultra96`
  - `HGPIPE_CASE_ROOT=workspace/hardware/case_ultra96`
  - `HGPIPE_TARGETS=ultra96v2`
  - `HGPIPE_HLS_TARGET=ultra96v2`

## 3. VCK190

### Docker Launch and Attach

Host:

```bash
cd /home/kjm26/project/PRJXR/HG_PIPE_MERGE/ViT_Accel

export LM_LICENSE_FILE=2100@<license-host>
export XILINXD_LICENSE_FILE=2100@<license-host>

bash workspace/flow/scripts/targets/vck190/up.sh \
  --daemon \
  --network host \
  --hostname "$(hostname)"

bash workspace/flow/scripts/targets/vck190/connect.sh
```

### Docker-Internal Commands

Inside Docker:

```bash
cd /workspace

bash workspace/flow/scripts/targets/vck190/docker.sh env
bash workspace/flow/scripts/targets/vck190/docker.sh step0
bash workspace/flow/scripts/targets/vck190/docker.sh step1
bash workspace/flow/scripts/targets/vck190/docker.sh step1-dsp
bash workspace/flow/scripts/targets/vck190/docker.sh step2
bash workspace/flow/scripts/targets/vck190/docker.sh step2-dsp
bash workspace/flow/scripts/targets/vck190/docker.sh step3-server
bash workspace/flow/scripts/targets/vck190/docker.sh step3-blocks
bash workspace/flow/scripts/targets/vck190/docker.sh step3-full
bash workspace/flow/scripts/targets/vck190/docker.sh step3-all
bash workspace/flow/scripts/targets/vck190/docker.sh step4-export
bash workspace/flow/scripts/targets/vck190/docker.sh baseline
bash workspace/flow/scripts/targets/vck190/docker.sh baseline-dsp
bash workspace/flow/scripts/targets/vck190/docker.sh step5
```

Recommended Docker-side baseline:

```bash
cd /workspace

bash workspace/flow/scripts/targets/vck190/docker.sh env
bash workspace/flow/scripts/targets/vck190/docker.sh step0
bash workspace/flow/scripts/targets/vck190/docker.sh step1
bash workspace/flow/scripts/targets/vck190/docker.sh step2
bash workspace/flow/scripts/targets/vck190/docker.sh step3-all
bash workspace/flow/scripts/targets/vck190/docker.sh step4-export
```

### Host-Only Commands

Host:

```bash
cd /home/kjm26/project/PRJXR/HG_PIPE_MERGE/ViT_Accel

export LM_LICENSE_FILE=2100@<license-host>
export XILINXD_LICENSE_FILE=2100@<license-host>
export XILINX_ROOT=/tools/Xilinx
export XILINX_VERSION=2023.2

bash workspace/flow/scripts/targets/vck190/readme.sh step4-ooc
bash workspace/flow/scripts/targets/vck190/readme.sh step4-board-export
bash workspace/flow/scripts/targets/vck190/pynq_handoff.sh
bash workspace/flow/scripts/targets/vck190/pynq_handoff.sh --run-package
bash workspace/flow/scripts/targets/vck190/pynq_handoff.sh --run-board-export --reuse-ooc-dcp
bash workspace/flow/scripts/targets/vck190/pynq_handoff.sh --run-board-export --rebuild-ip
```

## 4. ZCU102

### Docker Launch and Attach

Host:

```bash
cd /home/kjm26/project/PRJXR/HG_PIPE_MERGE/ViT_Accel

export LM_LICENSE_FILE=2100@<license-host>
export XILINXD_LICENSE_FILE=2100@<license-host>

bash workspace/flow/scripts/targets/zcu102/up.sh \
  --daemon \
  --network host \
  --hostname "$(hostname)"

bash workspace/flow/scripts/targets/zcu102/connect.sh
```

### Docker-Internal Non-Spinal Path

Inside Docker:

```bash
cd /workspace

bash workspace/flow/scripts/targets/zcu102/nonspinal.sh env
bash workspace/flow/scripts/targets/zcu102/nonspinal.sh step0
bash workspace/flow/scripts/targets/zcu102/nonspinal.sh step1
bash workspace/flow/scripts/targets/zcu102/nonspinal.sh step2
bash workspace/flow/scripts/targets/zcu102/nonspinal.sh step4-prepare
```

Shortcut:

```bash
bash workspace/flow/scripts/targets/zcu102/nonspinal.sh all
```

### Docker-Internal Spinal Path

Inside Docker:

```bash
cd /workspace

bash workspace/flow/scripts/targets/zcu102/spinal.sh env
bash workspace/flow/scripts/targets/zcu102/spinal.sh step0
bash workspace/flow/scripts/targets/zcu102/spinal.sh step1
bash workspace/flow/scripts/targets/zcu102/spinal.sh step2
bash workspace/flow/scripts/targets/zcu102/spinal.sh step3-all
bash workspace/flow/scripts/targets/zcu102/spinal.sh step4-export
bash workspace/flow/scripts/targets/zcu102/spinal.sh step4-prepare
```

Shortcut:

```bash
bash workspace/flow/scripts/targets/zcu102/spinal.sh all
```

### Recommended Host-Only Vivado Path

Host:

```bash
cd /home/kjm26/project/PRJXR/HG_PIPE_MERGE/ViT_Accel

export LM_LICENSE_FILE=2100@<license-host>
export XILINXD_LICENSE_FILE=2100@<license-host>
export XILINX_ROOT=/tools/Xilinx
export XILINX_VERSION=2023.2

bash workspace/flow/scripts/targets/zcu102/host.sh all-host
```

Equivalent explicit sequence:

```bash
bash workspace/flow/scripts/targets/zcu102/host.sh step0
bash workspace/flow/scripts/targets/zcu102/host.sh step1
bash workspace/flow/scripts/targets/zcu102/host.sh step2
bash workspace/flow/scripts/targets/zcu102/host.sh step4-prepare
bash workspace/flow/scripts/targets/zcu102/host.sh step4-validate
bash workspace/flow/scripts/targets/zcu102/host.sh step4-synth
bash workspace/flow/scripts/targets/zcu102/host.sh step4-bitstream
```

Notes:

- `step4-ooc` is mapped to `step4-bitstream`
- `baseline` is mapped to `all`
- `all-host` and `vivado-only` are also mapped to the same host-only Vivado flow
- Docker is optional and mainly useful if you explicitly want the Spinal-backed path or want `step4-prepare` generated inside the container
- `step4-validate`, `step4-synth`, and `step4-bitstream` stay host-only

## 5. ZCU104

### Docker Launch and Attach

Host:

```bash
cd /home/kjm26/project/PRJXR/HG_PIPE_MERGE/ViT_Accel

export LM_LICENSE_FILE=2100@<license-host>
export XILINXD_LICENSE_FILE=2100@<license-host>

bash workspace/flow/scripts/targets/zcu104/up.sh \
  --daemon \
  --network host \
  --hostname "$(hostname)"

bash workspace/flow/scripts/targets/zcu104/connect.sh
```

### Docker-Internal Non-Spinal Path

Inside Docker:

```bash
cd /workspace

bash workspace/flow/scripts/targets/zcu104/nonspinal.sh env
bash workspace/flow/scripts/targets/zcu104/nonspinal.sh step0
bash workspace/flow/scripts/targets/zcu104/nonspinal.sh step1
bash workspace/flow/scripts/targets/zcu104/nonspinal.sh step2
bash workspace/flow/scripts/targets/zcu104/nonspinal.sh step4-prepare
```

Shortcut:

```bash
bash workspace/flow/scripts/targets/zcu104/nonspinal.sh all
```

### Docker-Internal Spinal Path

Inside Docker:

```bash
cd /workspace

bash workspace/flow/scripts/targets/zcu104/spinal.sh env
bash workspace/flow/scripts/targets/zcu104/spinal.sh step0
bash workspace/flow/scripts/targets/zcu104/spinal.sh step1
bash workspace/flow/scripts/targets/zcu104/spinal.sh step2
bash workspace/flow/scripts/targets/zcu104/spinal.sh step3-all
bash workspace/flow/scripts/targets/zcu104/spinal.sh step4-export
bash workspace/flow/scripts/targets/zcu104/spinal.sh step4-prepare
```

Shortcut:

```bash
bash workspace/flow/scripts/targets/zcu104/spinal.sh all
```

### Recommended Host-Only Vivado Path

Host:

```bash
cd /home/kjm26/project/PRJXR/HG_PIPE_MERGE/ViT_Accel

export LM_LICENSE_FILE=2100@<license-host>
export XILINXD_LICENSE_FILE=2100@<license-host>
export XILINX_ROOT=/tools/Xilinx
export XILINX_VERSION=2023.2

bash workspace/flow/scripts/targets/zcu104/host.sh all-host
```

Equivalent explicit sequence:

```bash
bash workspace/flow/scripts/targets/zcu104/host.sh step0
bash workspace/flow/scripts/targets/zcu104/host.sh step1
bash workspace/flow/scripts/targets/zcu104/host.sh step2
bash workspace/flow/scripts/targets/zcu104/host.sh step4-prepare
bash workspace/flow/scripts/targets/zcu104/host.sh step4-validate
bash workspace/flow/scripts/targets/zcu104/host.sh step4-synth
bash workspace/flow/scripts/targets/zcu104/host.sh step4-bitstream
```

Notes:

- `step4-ooc` is mapped to `step4-bitstream`
- `baseline` is mapped to `all`
- `all-host` and `vivado-only` are also mapped to the same host-only Vivado flow
- Docker is optional and mainly useful if you explicitly want the Spinal-backed path or want `step4-prepare` generated inside the container
- `step4-validate`, `step4-synth`, and `step4-bitstream` stay host-only

## 6. ZU15EG

Current profile:

- `zu15eg_full_deit_tiny_fit` now uses a `3.0 ns` HLS and board clock target.
- The full-model shape and current `ATTN` / `MLP` compute parallelism are preserved.
- Resource fit is handled by moving ZU15EG `ATTN` reshape / residual / QQ-head buffering and `PATCH_EMBED` weights toward URAM.
- A 3ns timing miss is tracked separately from the first resource-fit acceptance target.

### Docker Launch and Attach

Host:

```bash
cd /home/kjm26/project/PRJXR/HG_PIPE_MERGE/ViT_Accel

export LM_LICENSE_FILE=2100@<license-host>
export XILINXD_LICENSE_FILE=2100@<license-host>

bash workspace/flow/scripts/targets/zu15eg/up.sh \
  --daemon \
  --network host \
  --hostname "$(hostname)"

bash workspace/flow/scripts/targets/zu15eg/connect.sh
```

### Docker-Internal Non-Spinal Path

Inside Docker:

```bash
cd /workspace

bash workspace/flow/scripts/targets/zu15eg/nonspinal.sh env
bash workspace/flow/scripts/targets/zu15eg/nonspinal.sh step0
bash workspace/flow/scripts/targets/zu15eg/nonspinal.sh step1
bash workspace/flow/scripts/targets/zu15eg/nonspinal.sh step2
bash workspace/flow/scripts/targets/zu15eg/nonspinal.sh step4-prepare
```

Shortcut:

```bash
bash workspace/flow/scripts/targets/zu15eg/nonspinal.sh all
```

### Docker-Internal Spinal Path

Inside Docker:

```bash
cd /workspace

bash workspace/flow/scripts/targets/zu15eg/spinal.sh env
bash workspace/flow/scripts/targets/zu15eg/spinal.sh step0
bash workspace/flow/scripts/targets/zu15eg/spinal.sh step1
bash workspace/flow/scripts/targets/zu15eg/spinal.sh step2
bash workspace/flow/scripts/targets/zu15eg/spinal.sh step3-all
bash workspace/flow/scripts/targets/zu15eg/spinal.sh step4-export
bash workspace/flow/scripts/targets/zu15eg/spinal.sh step4-prepare
```

Targeted 3ns resource-fit validation:

```bash
HGPIPE_STEP1_CASES=PATCH_EMBED,ATTN8,MLP0 \
HGPIPE_STEP1_MAX_THREADS=4 \
bash workspace/flow/scripts/targets/zu15eg/spinal.sh step1

bash workspace/flow/scripts/targets/zu15eg/spinal.sh step2
```

Shortcut:

```bash
bash workspace/flow/scripts/targets/zu15eg/spinal.sh all
```

### Recommended Host-Only Vivado Path

Host:

```bash
cd /home/kjm26/project/PRJXR/HG_PIPE_MERGE/ViT_Accel

export LM_LICENSE_FILE=2100@<license-host>
export XILINXD_LICENSE_FILE=2100@<license-host>
export XILINX_ROOT=/tools/Xilinx
export XILINX_VERSION=2023.2

bash workspace/flow/scripts/targets/zu15eg/host.sh all-host
```

Equivalent explicit sequence:

```bash
bash workspace/flow/scripts/targets/zu15eg/host.sh step0
bash workspace/flow/scripts/targets/zu15eg/host.sh step1
bash workspace/flow/scripts/targets/zu15eg/host.sh step2
bash workspace/flow/scripts/targets/zu15eg/host.sh step4-prepare
bash workspace/flow/scripts/targets/zu15eg/host.sh step4-validate
bash workspace/flow/scripts/targets/zu15eg/host.sh step4-synth
bash workspace/flow/scripts/targets/zu15eg/host.sh step4-bitstream
```

Notes:

- `step4-ooc` is mapped to `step4-bitstream`
- `baseline` is mapped to `all`
- `all-host` and `vivado-only` are also mapped to the same host-only Vivado flow
- Docker is optional and mainly useful if you explicitly want the Spinal-backed path or want `step4-prepare` generated inside the container
- `step4-validate`, `step4-synth`, and `step4-bitstream` stay host-only

## 7. Ultra96-V2

### Docker Launch and Attach

Host:

```bash
cd /home/kjm26/project/PRJXR/HG_PIPE_MERGE/ViT_Accel

export LM_LICENSE_FILE=2100@<license-host>
export XILINXD_LICENSE_FILE=2100@<license-host>

bash workspace/flow/scripts/targets/ultra96/up.sh \
  --daemon \
  --network host \
  --hostname "$(hostname)"

bash workspace/flow/scripts/targets/ultra96/connect.sh
```

### Docker-Internal Non-Spinal Path

Inside Docker:

```bash
cd /workspace

bash workspace/flow/scripts/targets/ultra96/nonspinal.sh env
bash workspace/flow/scripts/targets/ultra96/nonspinal.sh step0
bash workspace/flow/scripts/targets/ultra96/nonspinal.sh step1
bash workspace/flow/scripts/targets/ultra96/nonspinal.sh step2
bash workspace/flow/scripts/targets/ultra96/nonspinal.sh step4-prepare
```

Shortcut:

```bash
bash workspace/flow/scripts/targets/ultra96/nonspinal.sh all
```

### Docker-Internal Spinal Path

Inside Docker:

```bash
cd /workspace

bash workspace/flow/scripts/targets/ultra96/spinal.sh env
bash workspace/flow/scripts/targets/ultra96/spinal.sh step0
bash workspace/flow/scripts/targets/ultra96/spinal.sh step1
bash workspace/flow/scripts/targets/ultra96/spinal.sh step2
bash workspace/flow/scripts/targets/ultra96/spinal.sh step3-all
bash workspace/flow/scripts/targets/ultra96/spinal.sh step4-export
bash workspace/flow/scripts/targets/ultra96/spinal.sh step4-prepare
```

Shortcut:

```bash
bash workspace/flow/scripts/targets/ultra96/spinal.sh all
```

### Recommended Host-Only Vivado Path

Host:

```bash
cd /home/kjm26/project/PRJXR/HG_PIPE_MERGE/ViT_Accel

export LM_LICENSE_FILE=2100@<license-host>
export XILINXD_LICENSE_FILE=2100@<license-host>
export XILINX_ROOT=/tools/Xilinx
export XILINX_VERSION=2023.2

bash workspace/flow/scripts/targets/ultra96/host.sh all-host
```

Equivalent explicit sequence:

```bash
bash workspace/flow/scripts/targets/ultra96/host.sh step0
bash workspace/flow/scripts/targets/ultra96/host.sh step1
bash workspace/flow/scripts/targets/ultra96/host.sh step2
bash workspace/flow/scripts/targets/ultra96/host.sh step4-prepare
bash workspace/flow/scripts/targets/ultra96/host.sh step4-validate
bash workspace/flow/scripts/targets/ultra96/host.sh step4-synth
bash workspace/flow/scripts/targets/ultra96/host.sh step4-bitstream
```

Notes:

- `step4-ooc` is mapped to `step4-bitstream`
- `baseline` is mapped to `all`
- `all-host` and `vivado-only` are also mapped to the same host-only Vivado flow
- Docker is optional and mainly useful if you explicitly want the Spinal-backed path or want `step4-prepare` generated inside the container
- `step4-validate`, `step4-synth`, and `step4-bitstream` stay host-only
- wrapper target key is `ultra96v2` even though the user-facing directory is `ultra96`
