# HGTXR Hardware Changelog - 2026-06-04

## Added Reference-Informed HLS Kernel Structure

### Changed

- `hardware/hls/src/nonlinear.cpp`
  - Added variance-aware LayerNorm approximation.
  - Added three-pass softmax approximation.
  - Added tanh-polynomial GeLU approximation.

- `hardware/hls/src/matmul.cpp`
  - Replaced scalar triple-loop implementation with tiled local accumulation.

- `hardware/hls/src/rmu_smu.cpp`
  - Replaced placeholder increments with RMU-like projection and SMU-like relation kernels.

- `hardware/hls/src/attention.cpp`
  - Added residual attention block structure.

- `hardware/hls/src/mlp.cpp`
  - Added residual MLP block structure.

- `hardware/hls/src/search_head.cpp`
  - Changed from constant output to feature-dependent search state generation.

- `hardware/hls/src/track_head.cpp`
  - Changed from simple xy delta to residual tracking state generation.

## Reference Patterns Used

- XR_Accel Matmul: tiled/windowed MAC organization.
- XR_Accel LayerNorm: multi-pass mean/variance/rsqrt approach.
- XR_Accel Softmax: max/exp/sum/reciprocal/requant pass structure.
- XR_Accel GeLU: LUT-friendly non-linear approximation.
- XR_Accel Attn/MLP: residual dataflow composition.
- XR_Accel cyclic ViT: mode-dependent token/depth execution.
- REF ESDA TCL: HLS project flow conventions.
- REF precision HLS: dot/matmul benchmark and type-sweep organization.

## Not Yet Changed

- `hardware/hls/src/hgtxr_top.cpp`
  - Needs event-density/similarity/confidence/quality computation wired into `runtime_fsm`.

- `docs/Validation.md`
  - Needs updated compile/test evidence.

- `docs/track/PROGRESS.md`
  - Needs synchronization with this snapshot.

## Verification

Not rerun after these source changes.

Required commands:

```bash
cd /home/user/project/PRJXR/HGTXR
g++ -std=c++17 -Ihardware/hls/include hardware/hls/tb/tb_hgtxr_top.cpp hardware/hls/src/*.cpp -o /tmp/hgtxr_top_tb
/tmp/hgtxr_top_tb
.venv/bin/python -m pytest -q -s
PYTHON_BIN=.venv/bin/python sh hardware/scripts/export_hw_refs.sh
PYTHON_BIN=.venv/bin/python sh hardware/scripts/run_hw_sw_compare.sh
```

## Notes

- This pass improves structural fidelity but does not establish bit-exact equivalence to the paper.
- `확실하지 않음`: final paper parameters and quantized lookup tables are not present in the current workspace.

