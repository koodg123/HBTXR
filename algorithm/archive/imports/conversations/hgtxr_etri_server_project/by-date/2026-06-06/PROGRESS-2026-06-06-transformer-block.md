# HGTXR Progress Log: Transformer Block Wrapper

Date: 2026-06-06

## Shell Status

- `exec_command` still fails before `/bin/bash` starts.
- No WSL, Vitis HLS, Vivado, Python, or PYNQ command was executed in this turn.
- Work was limited to `apply_patch` file additions.

## Added This Turn

- `hardware/hls/include/hgtxr_cyclic_norm.hpp`
  - identity normalization path
  - mean-centering placeholder
  - affine gamma/beta helper
- `hardware/hls/include/hgtxr_cyclic_transformer_block.hpp`
  - pre-norm Transformer block wrapper
  - separated WQ/WK/WV projection
  - tiled attention
  - output projection + residual
  - norm + MLP + residual
- `hardware/tools/static_validate_hgtxr.py`
  - required file checks
  - top AXI pragma marker checks
  - Vivado HWH Tcl patch marker check
  - expected bit/hwh artifact path checks

## Spark Sidecars Used

- `T-LN-PRIM-001`: fixed-point LayerNorm/RMSNorm placeholder guidance.
- `T-BLOCK-WRAP-001`: Transformer block interface review.
- `T-STATIC-VALID-001`: static validation checklist.

## Accuracy And Architecture Notes

- `norm_tile` is not a full paper-equivalent LayerNorm.
- A better next implementation is RMSNorm with bounded `inv_sqrt` approximation, preferably LUT plus one Newton-Raphson refinement.
- `transformer_block_tile` uses separated `wq`, `wk`, `wv`; a packed/fused QKV projection is preferred later for DRAM bandwidth.
- Current attention softmax path remains approximate and must be validated with golden vectors.

## Required Verification When WSL Shell Works

```bash
cd /home/user/project/PRJXR/HGTXR
python3 -m py_compile hardware/tools/static_validate_hgtxr.py
python3 hardware/tools/static_validate_hgtxr.py --root .
g++ -std=c++17 -Ihardware/hls/include hardware/hls/tb/tb_cyclic_primitives.cpp -o /tmp/tb_cyclic_primitives
/tmp/tb_cyclic_primitives
```

Then add a Vitis HLS C simulation target for the cyclic primitive test before integrating into `hgtxr_top.cpp`.

## Remaining Objective Work

- Patch and verify the Vivado HWH copy Tcl.
- Copy `.bit/.hwh` into overlay and PYNQ package.
- Parse the DeiT-Tiny image and `impl_repos` reports.
- Replace placeholder norm/softmax/GELU with validated approximations.
- Integrate `transformer_block_tile` into `hgtxr_top.cpp` behind the existing PYNQ AXI ABI.
- Re-run `csynth -> export IP -> Vivado BD -> bit/hwh -> PYNQ`.

