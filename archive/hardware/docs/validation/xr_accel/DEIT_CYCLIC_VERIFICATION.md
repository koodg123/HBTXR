# DeiT-Tiny Cyclic Verification

This note records the official regression gate for the DeiT-Tiny classification
cyclic hardware path.

## Scope

- Model: DeiT-Tiny baseline, 12 transformer layers, 196 tokens, embed dim 192.
- Official experiment: `configs/experiments/zcu104_deit_tiny_baseline_cyclic.json`.
- Official design: `configs/designs/deit_tiny_cyclic_zcu104.json`.
- Baseline path: `PATCH_EMBED -> ATTN0 -> MLP0 -> ... -> ATTN11 -> MLP11 -> HEAD`.
- Cyclic path: `PATCH_EMBED_IMAGE -> MODE_DEIT cyclic backbone -> DEIT_HEAD`.
- Pass criterion: bit-exact match against the checked-in baseline refs at patch
  embed, every attention layer, every MLP layer, and head output.
- Current milestone scope: DeiT classification only. HBTXR search/track modes
  remain outside the formal acceptance criteria for this note.

## Regression Commands

Vitis HLS C-sim, for Ubuntu 22 style environments compatible with Vitis HLS
2023.2:

```bash
cd /home/user/project/PRJXR/Hardware/XR_Accel
source /tools/Xilinx/Vitis_HLS/2023.2/settings64.sh
vitis_hls -f workspace/flow/scripts/utils/deit_cyclic_csim_only.tcl
python3 workspace/flow/scripts/utils/verify_deit_cyclic_csim.py \
  --log /tmp/xr_accel_deit_cyclic_csim/solution/csim/report/top_csim.log
```

Standalone functional C-sim fallback for hosts where Vitis HLS 2023.2 cannot
link against the system glibc:

```bash
cd /home/user/project/PRJXR/Hardware/XR_Accel
g++ -std=c++14 -Wno-unknown-pragmas \
  -I/tools/Xilinx/Vitis_HLS/2023.2/include \
  -I/tools/Xilinx/Vivado/2023.2/include \
  workspace/hardware/case_cyclic_zcu104/CYCLIC_VIT_TOP.cpp \
  -o /tmp/xr_accel_deit_cyclic_csim_gpp
/tmp/xr_accel_deit_cyclic_csim_gpp > /tmp/xr_accel_deit_cyclic_csim_gpp.log
printf '\nStandalone CSim done with 0 errors\n' >> /tmp/xr_accel_deit_cyclic_csim_gpp.log
python3 workspace/flow/scripts/utils/verify_deit_cyclic_csim.py \
  --log /tmp/xr_accel_deit_cyclic_csim_gpp.log
```

HLS C-synth acceptance for the same top:

```bash
cd /home/user/project/PRJXR/Hardware/XR_Accel
source /tools/Xilinx/Vitis_HLS/2023.2/settings64.sh
vitis_hls -f workspace/flow/scripts/utils/deit_cyclic_csynth_only.tcl
```

## Current Result

The standalone fallback passed on the current host. The generated report is:

```text
workspace/artifacts/reports/zcu104/deit_tiny_baseline_cyclic/deit_tiny_cyclic_zcu104/deit_cyclic_csim_verify.md
```

Summary:

- `MODE_SEARCH`, `MODE_TRACK`, and `MODE_DEIT` smoke checks passed.
- `patch_embed exact match`: 2 passes.
- `attn0..attn11 exact match`: 2 passes each.
- `mlp0..mlp11 exact match`: 2 passes each.
- `head exact match`: 2 passes.
- `MODE_DEIT` is the fixed runtime mode for the official DeiT cyclic design.

## Known Environment Note

On the current Ubuntu 24.04 host, Vitis HLS 2023.2 C-sim compilation can be made
to find multiarch headers with `-I/usr/include/x86_64-linux-gnu`, but its bundled
linker cannot consume the host glibc RELR sections. Run the Vitis C-sim command
inside an Ubuntu 22 compatible environment for the official HLS C-sim artifact.
