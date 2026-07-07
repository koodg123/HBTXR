# HGTXR Progress Log: Cyclic Modules And Sweep Automation

Date: 2026-06-06

## Shell Status

- `exec_command` still fails before `/bin/bash` starts.
- No WSL, Vitis HLS, Vivado, Python, or PYNQ command was executed in this turn.
- Work was limited to `apply_patch` file additions.

## Added This Turn

- `hardware/hls/include/hgtxr_cyclic_mac.hpp`
  - header-only tiled MAC primitive
  - accumulator-to-data conversion helper
  - residual add tile helper
- `hardware/hls/include/hgtxr_cyclic_scheduler.hpp`
  - cyclic Transformer scheduler states
  - scheduler limits/counters
  - next-state and counter-advance helpers
- `hardware/tools/hgtxr_sweep_expand.py`
  - expands sweep YAML into run manifests and compile-time define files
  - supports optional PyYAML and JSON fallback
- `docs/HLS-MAC-Audit-Checklist-2026-06-06.md`
  - HLS review checklist for tiled MAC and later cyclic Transformer kernels

## Spark Sidecars Used

- `T-MAC-TILE-REVIEW`: HLS MAC audit checklist.
- `T-SCHED-SKELETON`: scheduler enum and next-state helper draft.
- `T-SWEEP-SCRIPT`: sweep expansion script design.

## Required Verification When WSL Shell Works

```bash
cd /home/user/project/PRJXR/HGTXR
python3 -m py_compile hardware/tools/hgtxr_sweep_expand.py hardware/tools/extract_resource_metrics.py hardware/tools/repair_hls_to_pynq.py
python3 hardware/tools/hgtxr_sweep_expand.py \
  --sweep-file hardware/configs/sweeps/zcu104_cyclic_transformer_sweep.yaml \
  --manifest /tmp/hgtxr_sweep_manifest.csv \
  --out-dir /tmp/hgtxr_sweep \
  --emit-format cpp \
  --max-combos 16 \
  --overwrite
```

After script checks, compile a small HLS/host test that includes:

- `hgtxr_cyclic_transformer_params.hpp`
- `hgtxr_cyclic_mac.hpp`
- `hgtxr_cyclic_scheduler.hpp`

## Remaining Objective Work

- Patch and verify the Vivado HWH copy Tcl.
- Copy `.bit/.hwh` into overlay and PYNQ package.
- Run PYNQ helper smoke.
- Inspect the DeiT-Tiny C-Syn image and parse `impl_repos`.
- Integrate cyclic modules into `hgtxr_top.cpp` without breaking current AXI ABI.
- Run `csynth -> export IP -> Vivado BD -> bit/hwh -> PYNQ` with evidence.

