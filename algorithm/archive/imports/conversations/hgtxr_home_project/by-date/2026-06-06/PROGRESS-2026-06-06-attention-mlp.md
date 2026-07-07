# HGTXR Progress Log: Attention And MLP Primitives

Date: 2026-06-06

## Shell Status

- `exec_command` still fails before `/bin/bash` starts.
- No WSL, Vitis HLS, Vivado, Python, or PYNQ command was executed in this turn.
- Work was limited to `apply_patch` file additions.

## Added This Turn

- `hardware/hls/include/hgtxr_cyclic_math.hpp`
  - fixed-point ReLU
  - hard-sigmoid
  - GELU placeholder approximation
  - bounded score exp approximation
  - row max and approximate softmax helpers
- `hardware/hls/include/hgtxr_cyclic_mlp.hpp`
  - tiled MLP wrapper using `mac_tile`
  - GELU approximation
  - residual add path
- `hardware/hls/include/hgtxr_cyclic_attention.hpp`
  - tiled QK score computation
  - approximate softmax
  - attention-value accumulation
- `hardware/hls/tb/tb_cyclic_primitives.cpp`
  - deterministic host/HLS smoke source for MAC, scheduler, attention, and MLP primitives

## Spark Sidecars Used

- `T-ATTN-PRIM-001`: tiled attention guidance and softmax risks.
- `T-MLP-PRIM-001`: tiled MLP guidance and fixed-point/residual risks.
- `T-SMOKE-PLAN-001`: minimal smoke test plan.

## Required Verification When WSL Shell Works

```bash
cd /home/user/project/PRJXR/HGTXR
g++ -std=c++17 -Ihardware/hls/include hardware/hls/tb/tb_cyclic_primitives.cpp -o /tmp/tb_cyclic_primitives
/tmp/tb_cyclic_primitives
```

If host compile fails because Xilinx HLS headers are unavailable outside Vitis, run after sourcing Vitis HLS:

```bash
source /tools/Xilinx/Vitis_HLS/2023.2/settings64.sh
g++ -std=c++17 -Ihardware/hls/include hardware/hls/tb/tb_cyclic_primitives.cpp -o /tmp/tb_cyclic_primitives
/tmp/tb_cyclic_primitives
```

Then add these files to a Vitis HLS C simulation target before integrating into `hgtxr_top.cpp`.

## Known Risks

- `hgtxr_gelu_approx` is a hardware-friendly placeholder, not paper-equivalent GELU.
- `hgtxr_score_exp_approx` is a coarse piecewise-linear approximation and needs golden-vector validation.
- `softmax_normalize_tile` uses fixed-point division, which can be expensive in HLS; later versions should replace it with reciprocal/LUT or base-2 online recurrence.
- New headers are not yet integrated into the main top or validated by `csynth`.

