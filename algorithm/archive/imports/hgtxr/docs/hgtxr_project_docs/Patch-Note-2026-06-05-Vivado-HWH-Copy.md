# Vivado HWH Copy Patch Note

Date: 2026-06-05
File to patch when WSL shell/file access is restored: `hardware/vivado/scripts/build_bitstream.tcl`

## Problem

The previous Vivado run reportedly completed bitstream generation, then failed in the final artifact-copy step because Tcl in the Vivado environment did not accept:

```tcl
glob -nocomplain -recursive -directory $build_dir *.hwh
```

The failure happened after `write_bitstream` completed, so the expected fix is a post-processing Tcl compatibility patch, not a hardware redesign.

## Required Patch

Replace:

```tcl
set bd_hwh [glob -nocomplain -recursive -directory $build_dir *.hwh]
```

with:

```tcl
set bd_hwh [concat \
    [glob -nocomplain -directory [file join $build_dir "${project_name}.gen" "sources_1" "bd" $bd_name "hw_handoff"] *.hwh] \
    [glob -nocomplain -directory [file join $build_dir "${project_name}.gen" "sources_1" "bd" $bd_name] *.hwh] \
]
```

## Expected Artifact Paths

If the previous run layout is unchanged:

```text
hardware/generated/build/vivado/hgtxr_overlay/hgtxr_overlay.runs/impl_1/hgtxr_system_wrapper.bit
hardware/generated/build/vivado/hgtxr_overlay/hgtxr_overlay.gen/sources_1/bd/hgtxr_system/hw_handoff/hgtxr_system.hwh
```

Copy destinations:

```text
hardware/generated/build/vivado/overlay/hgtxr_overlay/hgtxr.bit
hardware/generated/build/vivado/overlay/hgtxr_overlay/hgtxr.hwh
hardware/pynq/hgtxr/hgtxr.bit
hardware/pynq/hgtxr/hgtxr.hwh
```

## Verification Commands

Run after WSL shell access is restored:

```bash
cd /home/user/project/PRJXR/HGTXR
ls -lh hardware/generated/build/vivado/hgtxr_overlay/hgtxr_overlay.runs/impl_1/*.bit
ls -lh hardware/generated/build/vivado/hgtxr_overlay/hgtxr_overlay.gen/sources_1/bd/hgtxr_system/hw_handoff/*.hwh
ls -lh hardware/generated/build/vivado/overlay/hgtxr_overlay/
ls -lh hardware/pynq/hgtxr/hgtxr.{bit,hwh}
python3 -m py_compile hardware/pynq/hgtxr/hgtxr_overlay.py hardware/pynq/hgtxr/test_hgtxr_overlay.py
PYTHONPATH=hardware/pynq/hgtxr python3 hardware/pynq/hgtxr/test_hgtxr_overlay.py
```

## Status

This patch note exists because the current Codex runner cannot launch the WSL shell and cannot read the Tcl file through `apply_patch` at the time of writing. The actual Tcl source must still be patched and verified when WSL access is restored.

