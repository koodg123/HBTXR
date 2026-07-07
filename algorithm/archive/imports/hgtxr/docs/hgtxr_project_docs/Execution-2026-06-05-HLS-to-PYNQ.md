# HGTXR Execution Notes

Date: 2026-06-05

## Immediate Resume Steps

When WSL shell execution is restored:

```bash
cd /home/user/project/PRJXR/HGTXR
```

Patch `hardware/vivado/scripts/build_bitstream.tcl` according to:

```text
docs/Patch-Note-2026-06-05-Vivado-HWH-Copy.md
```

Then verify or copy the current artifacts:

```bash
mkdir -p hardware/generated/build/vivado/overlay/hgtxr_overlay
cp hardware/generated/build/vivado/hgtxr_overlay/hgtxr_overlay.runs/impl_1/hgtxr_system_wrapper.bit \
   hardware/generated/build/vivado/overlay/hgtxr_overlay/hgtxr.bit
cp hardware/generated/build/vivado/hgtxr_overlay/hgtxr_overlay.gen/sources_1/bd/hgtxr_system/hw_handoff/hgtxr_system.hwh \
   hardware/generated/build/vivado/overlay/hgtxr_overlay/hgtxr.hwh
cp hardware/generated/build/vivado/overlay/hgtxr_overlay/hgtxr.bit hardware/pynq/hgtxr/hgtxr.bit
cp hardware/generated/build/vivado/overlay/hgtxr_overlay/hgtxr.hwh hardware/pynq/hgtxr/hgtxr.hwh
```

Run local PYNQ helper smoke:

```bash
python3 -m py_compile hardware/pynq/hgtxr/hgtxr_overlay.py hardware/pynq/hgtxr/test_hgtxr_overlay.py
PYTHONPATH=hardware/pynq/hgtxr python3 hardware/pynq/hgtxr/test_hgtxr_overlay.py
```

## HLS-To-Bitstream Rebuild Path

Use this path when a clean rebuild is needed:

```bash
source /tools/Xilinx/Vitis_HLS/2023.2/settings64.sh
vitis_hls -f hardware/vivado/scripts/run_csynth.tcl
vitis_hls -f hardware/vivado/scripts/package_ip.tcl

source /tools/Xilinx/Vivado/2023.2/settings64.sh
vivado -mode batch -source hardware/vivado/scripts/build_bitstream.tcl
```

## Resource Reference Extraction

When file access is available, inspect:

```text
/home/user/project/PRJXR/PAPER_PRJXR/05_RESOURCES/DeiT-Tiny C-Syn Results.png
/home/user/project/PRJXR/impl_repos
```

Required output:

```text
docs/Resource-Reference-2026-06-05-DeiT-Tiny.md
```

The reference table should include LUT, FF, DSP, BRAM, URAM, timing, latency, II, and any reported throughput/power fields visible in the image or experiments.

## Cyclic Transformer Implementation Path

1. Add parameter header.
2. Add `mac_tile`.
3. Add scheduler enum/FSM.
4. Add QKV/WO reuse path.
5. Add online attention accumulation.
6. Add MLP path.
7. Add host/HLS smoke tests.
8. Run Stage S0 and S1 sweeps from `docs/Experiment-Matrix-2026-06-05-ZCU104.md`.

## Current Execution Limitation

The current Codex runner can write with `apply_patch`, but WSL shell command execution is not available in this turn. Build commands above are not yet re-run in this turn.

