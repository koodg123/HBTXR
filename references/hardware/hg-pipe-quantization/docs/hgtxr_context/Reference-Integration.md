# Reference Integration Report

## Sources Reflected

- `PAPER_PRJXR/10_submission_initial/main.tex`
  - Search/track formulation.
  - Search anchor to runtime pupil-state interface.
  - Event branch as anchor-relative residual track.
  - Reduced-depth event execution at cut point `c`.
  - Runtime scheduler with confidence/quality control.
  - Mode-asymmetric cyclic streaming hardware architecture.

- `PAPER_PRJXR/20_submission_correction/analysis/CURRENT_MANUSCRIPT_ANALYSIS.md`
  - Resolved the previous conflict where event track could look like an independent full-backbone ellipse predictor.
  - Added explicit residual outputs in software: `event/residual`, `track/residual`.
  - Added scheduler anchor memory semantics.

- `PAPER_PRJXR/20_submission_correction/analysis/HG_PIPE_TO_HBTXR_REVISION_PLAN.md`
  - Added hardware stage boundaries for RMU, SMU, LayerNorm, Softmax, GeLU.
  - Added patch/token routing support stages through Global Buffer and NoC scaffolds.

- `repos/XR_Accel/docs/architecture/HBTXR_ARCH_FREEZE.md`
  - Added paper-level architecture components: Global Buffer, NoC, Weight Prefetcher, controller, mode-dependent traversal, reduced-depth track path.

- `repos/XR_Accel/docs/architecture/HBTXR_CYCLIC_IMPLEMENTATION.md`
  - Reflected deployment V1 shape in hardware config: search `1x128x128`, track `2x64x64`, search tokens `64`, track tokens `16`, search depth `8`, track cut depth `4`.

- `repos/XR_Accel/workspace/hardware/src/{layernorm,softmax,gelu,matmul}.h`
  - Reflected nonlinear and matrix-stage decomposition in HLS module boundaries.

- `REF/_classified`
  - `Hardware/Transformer-Accel` and `Hardware/CNN-Accel` are now catalogued and available for future per-reference import.
  - `Software/ViT`, `Software/LLM`, `Software/XR`, and `Software/Compression` are available for software-side reference expansion.

## Code Changes Made From These Sources

| Requirement | Implemented Location | Status |
|---|---|---|
| Full-depth frame search | `software/model.py` | Implemented |
| Reduced-depth event path | `software/modules/hgpipe_backbone.py`, `software/model.py` | Implemented |
| Event residual contract | `software/model.py`, `software/losses.py` | Implemented |
| Anchor memory scheduler | `software/runtime.py`, `software/scheduler.py` | Implemented |
| BBox-to-state decoder helpers | `software/geometry.py` | Implemented |
| Search/track reliability hysteresis | `software/scheduler.py` | Implemented initial policy |
| Global Buffer / NoC / Prefetcher / Controller | `hardware/hls/src/*.cpp` | Implemented scaffold |
| RMU/SMU stage boundaries | `hardware/hls/src/rmu_smu.cpp`, `attention.cpp`, `mlp.cpp` | Implemented scaffold |
| Nonlinear stages | `hardware/hls/src/nonlinear.cpp` | Interface scaffold |
| Paper deployment dimensions | `hardware/configs/vitis_hls.yaml` | Implemented |

## Remaining Gaps

확실하지 않음: final trained HBTXR weights, quantization tables, and golden per-layer outputs are not present in the project. Therefore the implementation now matches the paper-level architecture and algorithm contract, but not final numeric equivalence.

Remaining work required for no-difference reproduction:

1. Export final software checkpoint weights for frame search and event residual track.
2. Export per-layer HLS quantization scales, LUTs, and golden vectors.
3. Replace scaffold RMU/SMU/nonlinear arithmetic with parameterized kernels using those exports.
4. Run Vitis HLS csim/csynth and compare all stage outputs.
5. Run EV-Eye/TimeLens/V2E/Grounded-SAM data pipeline on the final dataset split.
6. Reproduce paper metrics: 0.1812-pixel center error and 0.43 ms average latency.


## 2026-06-09 Software v3 Paper-Reproduction Port

- Ported impl_repos/HBTXR_v3_0/src/hbtxr into HGTXR/software/src/hbtxr as the canonical paper-reproduction software package.
- Ported v3 execution surfaces into HGTXR/software/scripts/v3 and HGTXR/software/configs/v3 without overwriting existing flat-package compatibility scripts.
- Added submitted-version paper config at HGTXR/software/configs/paper/zcu104_option_a.yaml with search (8,192,4), track (4,192,4), frame 1x128x128, event 2x64x64, and paper target tables.
- Added explicit reduced-depth event/track execution and runtime selected-state output in the v3 model/runtime path.
- Added hbtxr.reproduction to emit reproduction_manifest.json with missing-artifact gates for dataset split, checkpoint, quantization tables, nonlinear LUTs, and golden vectors.

Remaining gap: exact submitted-paper metric equality remains blocked until the final dataset/checkpoint/quantization/golden-vector artifacts are available.
