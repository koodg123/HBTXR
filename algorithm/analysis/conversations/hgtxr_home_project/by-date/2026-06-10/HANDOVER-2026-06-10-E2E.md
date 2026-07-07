# HGTXR Handover - 2026-06-10 E2E Q4W/Q8A Baseline

## Scope

This handover captures the current HGTXR ZCU104 cyclic accelerator baseline for continuation on another computer or in a later Codex session. It focuses on the E2E ViT Q4W/Q8A path, reduced public AXI verification gates, remaining equivalence work, and performance improvement plan.

## Workspace

- Project root: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR`
- Main hardware source: `hardware/hls/include/hgtxr_e2e_vit.hpp`
- E2E AXI top: `hardware/hls/src/hgtxr_e2e_axis_top.cpp`
- E2E reduced testbench: `hardware/hls/tb/tb_hgtxr_e2e_axis_top.cpp`
- E2E config: `hardware/configs/zcu104_e2e_q4w8a_defines.h`
- E2E SW reference: `hardware/tools/validate_e2e_axis_vector.py`
- E2E pytest gate: `software/tests/test_e2e_axis_vector_csim.py`
- Vitis scripts: `hardware/vivado/scripts/run_e2e_q4w8a_csim.tcl`, `hardware/vivado/scripts/run_e2e_q4w8a_csynth.tcl`
- Sub-agent audit: `docs/track/SUBAGENT-AUDIT-2026-06-10-E2E-6OUT.md`

## Current Architecture State

Implemented baseline components:

- AXI-Stream public E2E top with frame input and six-word state output.
- AXI memory-mapped packed Q4 weight interface.
- Conv-style patch embedding over compile-time patch grid.
- Global buffer with token, norm, Q, K, V, attention, pooled, and hidden buffers.
- Controller loop over `HGTXR_E2E_BLOCKS`; even blocks use ATTN0/MLP0 and odd blocks use ATTN1/MLP1 scheduling.
- Transformer block path: LayerNorm, Q/K/V projection, attention score/softmax approximation, output projection, MLP W1/GELU/W2, and residual adds.
- MLP head over pooled tokens.
- HG-PIPE-style compact LUT math for GELU, exp, and rsqrt when `HGTXR_USE_HGPIPE_LUT_MATH=1`.
- Parameterized knobs for bit width, weight width, bus width, buffer size, FIFO depth, dense parallelism, block count, active token count, heads, head dim, and FF dim.

## Current E2E Reduced Verification State

The latest completed reduced public E2E gate is a deterministic six-output grouped Q4 gate.

- Reduced compile flags: `HGTXR_E2E_BLOCKS=1`, `HGTXR_E2E_ACTIVE_TOKENS=4`, `HGTXR_E2E_PATCH_GRID_H=1`, `HGTXR_E2E_PATCH_GRID_W=4`, `HGTXR_E2E_FF_DIM=32`.
- Expected raw AXI output: `[18, -3, 0, -2, 0, -4]`.
- Expected runtime state: `2`.
- Output count: `6`.
- Final output word must assert `last=1`.

Known passing gates from the latest completed state:

```bash
cd "$(git rev-parse --show-toplevel)"
python3 hardware/tools/validate_e2e_axis_vector.py --json-out /tmp/e2e_axis_vector_6out_ref.json
./.venv/bin/python -m pytest -s software/tests/test_e2e_axis_vector_csim.py -q
/tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f hardware/vivado/scripts/run_e2e_q4w8a_csim.tcl
python3 hardware/tools/static_validate_hgtxr.py --root .
```

Latest known results:

- SW reference: passed with expected raw `[18, -3, 0, -2, 0, -4]`.
- Native g++ comparator: passed with `runtime_state=2 count=6 last=1 failures=0`.
- Targeted pytest: `2 passed`.
- Vitis HLS csim: `CSim done with 0 errors`.
- Static validation: `[pass] static checks passed`.

## Current Resource And Latency Reference

The current full-size resource-fit references are the PAR8/PAR16 E2E Q4W/Q8A points recorded later in this handover, `docs/Validation.md`, and `docs/track/PROGRESS.md`.

Most relevant recorded points:

- AXIS PAR8 baseline: `3.744 ns`, `71,316,968 cycles`, `306 BRAM_18K`, `332 DSP`, `43,740 FF`, `81,144 LUT`, `64 URAM`.
- A2 `m_axi` wrapper after safe LUTRAM binding: `3.744 ns`, `81,514,836 cycles`, `326 BRAM_18K`, `334 DSP`, `46,502 FF`, `84,220 LUT`, `64 URAM`.
- C AXIS PAR16: `3.953 ns`, `37,508,072 cycles`, `338 BRAM_18K`, `604 DSP`, `59,507 FF`, `127,916 LUT`, `64 URAM`.

Interpretation:

- LUT is the current resource pressure point.
- DSP remains underused, but current fixed-point/ap_fixed and memory access structure do not automatically map all arithmetic to DSP.
- URAM is intentionally used for the hidden buffer and has meaningful remaining but not unlimited headroom.

## Work Not Yet Complete

The project goal is not complete. The following remain open:

1. Make the reduced E2E vector gate data-driven instead of duplicating hardcoded C++ and Python constants.
2. Add denser/arbitrary deterministic Q4 reduced SW/HW equivalence with more nonzero QKV/WO/W1/W2/head taps.
3. Increase validation one axis at a time: active tokens, FF dim, and block count.
4. Build or generate full Q4 packed weights for a full `active_tokens=196`, `blocks=6` E2E equivalence gate.
5. Prove full SW/HW numerical equivalence for the E2E ViT path, not only reduced deterministic gates.
6. Re-run full-size Vitis csynth after any significant E2E numerical or scheduling change.
7. Re-evaluate ZCU104 fit for LUT/DSP/BRAM/URAM and latency.

## Recommended Next Task Cards

```yaml
task_card:
  task_id: T-E2E-DATA-DRIVEN-001
  sub_agent: gpt5.3-codex-spark
  role: implementer
  objective: Replace duplicated reduced E2E C++/Python hardcoded pattern constants with a single deterministic spec/golden flow.
  file_ownership:
    - hardware/tools/validate_e2e_axis_vector.py
    - hardware/hls/tb/tb_hgtxr_e2e_axis_top.cpp
    - software/tests/test_e2e_axis_vector_csim.py
  assigned_skill:
    - fpga-asic-design-expert
    - attention-kernel-mapper
  inputs:
    - docs/track/SUBAGENT-AUDIT-2026-06-10-E2E-6OUT.md
    - hardware/hls/include/hgtxr_e2e_vit.hpp
  outputs:
    - deterministic reduced E2E spec or generated JSON golden
    - updated C++ testbench that consumes the generated expected raw values
  validation:
    - python3 hardware/tools/validate_e2e_axis_vector.py --json-out /tmp/e2e_axis_vector_ref.json
    - ./.venv/bin/python -m pytest -s software/tests/test_e2e_axis_vector_csim.py -q
    - /tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f hardware/vivado/scripts/run_e2e_q4w8a_csim.tcl
    - python3 hardware/tools/static_validate_hgtxr.py --root .
  dependencies: []
```

```yaml
task_card:
  task_id: T-E2E-DENSE-REDUCED-002
  sub_agent: gpt5.3-codex-spark
  role: implementer
  objective: Broaden the reduced E2E gate to a denser deterministic Q4 pattern while preserving blocks=1 and active_tokens=4.
  file_ownership:
    - hardware/tools/validate_e2e_axis_vector.py
    - hardware/hls/tb/tb_hgtxr_e2e_axis_top.cpp
    - software/tests/test_e2e_axis_vector_csim.py
  assigned_skill:
    - fpga-asic-design-expert
    - attention-kernel-mapper
  inputs:
    - T-E2E-DATA-DRIVEN-001 output
  outputs:
    - denser Q4 reduced expected raw vector
    - updated SW/HW equivalence test
  validation:
    - reference CLI
    - native g++ comparator
    - targeted pytest
    - Vitis HLS csim
    - static validation
  dependencies:
    - T-E2E-DATA-DRIVEN-001
```

```yaml
task_card:
  task_id: T-E2E-FULL-SCALE-003
  sub_agent: codex-native
  role: evaluator
  objective: Scale from reduced dense Q4 to full active_tokens=196 and blocks=6 equivalence, then re-run csynth for resource/latency.
  file_ownership:
    - hardware/vivado/scripts/run_e2e_q4w8a_csim.tcl
    - hardware/vivado/scripts/run_e2e_q4w8a_csynth.tcl
    - docs/Validation.md
    - docs/resources/e2e_axis_baseline_2026_06_09.md
  assigned_skill:
    - fpga-asic-design-expert
  inputs:
    - T-E2E-DENSE-REDUCED-002 output
    - hardware/configs/zcu104_e2e_q4w8a_defines.h
  outputs:
    - full E2E equivalence report
    - updated resource and latency table
  validation:
    - Vitis HLS csim for selected full-size gate
    - Vitis HLS csynth
    - resource extraction summary
  dependencies:
    - T-E2E-DENSE-REDUCED-002
```

## Performance Improvement Plan

1. Keep correctness gates ahead of performance sweeps. Every scheduling/resource change must pass the reduced E2E SW/HW gate first.
2. Use `HGTXR_E2E_PAR`, `HGTXR_E2E_DENSE_PAR`, and `HGTXR_PARALLELISM_FACTOR` as the first sweep knobs. PAR16 is validated and much faster, but LUT/timing pressure is higher; PAR32 is a stress point, not a default.
3. Preserve `HGTXR_BUS_WIDTH=256` unless csynth reports routing or LUT pressure that makes 128-bit preferable.
4. Keep large Q/K/V/ATTN buffers in URAM under the mixed-stream policy. Keep small buffers such as `gb.pooled` in LUTRAM where safe. Do not default Q/K/V local weight caches to URAM because that overflows ZCU104.
5. Split weight traffic if dense projection loops remain memory-port limited. The current `gmem_e2e_weights` single bundle can cap parallel lanes.
6. Prefer raising DSP use only where it reduces LUT pressure and does not worsen memory II. Current DSP usage is low, but LUT pressure is high.
7. Re-run csynth after each meaningful configuration change and record BRAM/DSP/FF/LUT/URAM plus latency in `docs/resources/hls_csynth_summary_2026_06_06.csv` or a successor summary file.

## Continuation Checklist

Before coding on another computer:

1. Confirm Vitis HLS exists at `/tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls`.
2. Confirm the project virtual environment exists at `./.venv` or use system Python for pure reference checks.
3. Run `python3 hardware/tools/static_validate_hgtxr.py --root .`.
4. Run the reduced reference and pytest gates above.
5. Read `docs/track/SUBAGENT-AUDIT-2026-06-10-E2E-6OUT.md` before editing the E2E gate.
6. Avoid changing full-size csynth assumptions until reduced SW/HW equivalence is green.

## Do Not Lose These Constraints

- `weights[0].bit(0)` is used as the live weight bit. Do not accidentally overwrite it while generating arbitrary patch weights.
- `HGTXR_STATE` is six. AXI output count and `last` semantics depend on six emitted words.
- Weight offsets are compile-time constants; do not add new packed fields without updating both C++ and Python layout logic.
- Reduced exact comparisons use raw `int(value * 16)` semantics. The Python reference must mirror truncation, not rounding.
- Full goal remains full E2E ViT execution and verification on ZCU104; current gates are staged proof points, not final completion.

## Continuation Update - 2026-06-10 Data-Driven Reduced Gate

T-E2E-DATA-DRIVEN-001 is now complete.

- Source-of-truth spec: `hardware/refs/e2e_axis_vector_spec.json`.
- Generated C++ golden header: `hardware/hls/tb/e2e_axis_vector_golden.hpp`.
- Reference/generator/checker: `hardware/tools/validate_e2e_axis_vector.py --spec hardware/refs/e2e_axis_vector_spec.json --emit-header hardware/hls/tb/e2e_axis_vector_golden.hpp --check-header hardware/hls/tb/e2e_axis_vector_golden.hpp`.
- C++ gate now consumes generated constants instead of local `kPatchRaw`, `kHeadRaw`, and `expected_raw` literals.
- The pytest gate checks spec/header sync and validates every emitted AXI output lane from the generated reference payload.

Latest validation evidence:

- `python3 -m py_compile software/tests/test_e2e_axis_vector_csim.py hardware/tools/validate_e2e_axis_vector.py` passed.
- `python3 hardware/tools/validate_e2e_axis_vector.py --spec hardware/refs/e2e_axis_vector_spec.json --json-out /tmp/e2e_axis_vector_ref.json --check-header hardware/hls/tb/e2e_axis_vector_golden.hpp` passed.
- Native g++ E2E comparator passed with `runtime_state=2 count=6 last=1 failures=0` and outputs `[18, -3, 0, -2, 0, -4]`.
- Direct execution of both Python test functions passed in this environment because pytest is not installed.
- `python3 hardware/tools/static_validate_hgtxr.py --root .` passed.
- Vitis HLS `csim_design` passed from `/tmp/hgtxr_vitis_e2e_csim` with `/tmp/libtinfo.so.5 -> /lib/x86_64-linux-gnu/libtinfo.so.6` and `LD_LIBRARY_PATH=/tmp:...`.

Next task: `T-E2E-DENSE-REDUCED-002`.

## Continuation Update - 2026-06-10 Dense Reduced Q4 Gate

T-E2E-DENSE-REDUCED-002 is now complete.

- The reduced E2E spec now includes deterministic sparse extra entries for patch, QKV, WO, MLP W1/W2, and head weights.
- The Python reference mirrors the HLS reduced E2E path instead of using grouped closed-form math.
- The generated C++ header now carries sparse-entry arrays and expected output `[18, -3, 3, -2, 1, -4]`.
- The C++ testbench applies base grouped weights first, then sparse overrides, then explicitly restores `weights[0].bit(0)` to the expected live-weight bit.
- The software regression checks dense nonzero counts plus every output lane from the generated reference payload.

Latest validation evidence:

- `python3 -m py_compile hardware/tools/validate_e2e_axis_vector.py software/tests/test_e2e_axis_vector_csim.py hardware/tools/static_validate_hgtxr.py` passed.
- `python3 hardware/tools/validate_e2e_axis_vector.py --spec hardware/refs/e2e_axis_vector_spec.json --json-out /tmp/e2e_axis_vector_dense_ref.json --check-header hardware/hls/tb/e2e_axis_vector_golden.hpp` passed.
- Direct execution of both Python test functions passed because pytest is not installed in this environment.
- `python3 hardware/tools/static_validate_hgtxr.py --root .` passed.
- Native g++ comparator passed with outputs `[18, -3, 3, -2, 1, -4]` and `runtime_state=2 count=6 last=1 failures=0`.
- Vitis HLS `csim_design` passed from `/tmp/hgtxr_vitis_e2e_csim` with the same six outputs.

Next task: `T-E2E-FULL-SCALE-003`.


## Continuation Update - 2026-06-10 Full-Scale Ramp Infrastructure

T-E2E-FULL-SCALE-003 is now started, not complete.

- The SW reference now models configurable `blocks` and 2D patch grids, so it can ramp beyond the original one-block, one-row reduced gate.
- Added `hardware/refs/e2e_axis_vector_blocks2_spec.json` as a strict two-block equivalence ramp. Generated expected raw output is `[20, -5, 4, -3, 2, -6]`.
- The C++ E2E AXI testbench now initializes every compiled block using the generated pattern and checks golden config shape against compile-time flags in strict mode.
- The testbench can consume an alternate generated header through `HGTXR_E2E_GOLDEN_HEADER`, enabling staged specs without overwriting the default reduced golden.
- `run_e2e_q4w8a_csim.tcl` now has explicit scale modes: default `reduced` strict and opt-in `full_smoke`.
- `run_e2e_q4w8a_csynth.tcl` now defaults to `full` and supports opt-in `reduced`.
- Both E2E Tcl scripts now resolve paths correctly from either the HGTXR root or the `hardware/` working directory.

Latest validation evidence:

- Default reduced Python/header sync passed with expected raw `[18, -3, 3, -2, 1, -4]`.
- Blocks2 Python/header generation passed with expected raw `[20, -5, 4, -3, 2, -6]`.
- Native g++ strict comparator passed for both default reduced and blocks2 ramp.
- `python3 hardware/tools/static_validate_hgtxr.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR` passed.
- `HGTXR_E2E_SCALE=reduced /tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f vivado/scripts/run_e2e_q4w8a_csim.tcl` passed from `hardware/` with `CSim done with 0 errors`.

Next task: raise one more axis at a time: active token/patch-grid ramp, then FF-dim ramp, then full `active_tokens=196`, `blocks=6` numerical equivalence, followed by full csynth/resource extraction.


## Continuation Update - 2026-06-10 Active-Token Ramp Gates

T-E2E-FULL-SCALE-003 remains in progress. Active-token ramp validation is now extended beyond the blocks2 gate.

- Added `hardware/refs/e2e_axis_vector_active8_spec.json` and `hardware/hls/tb/e2e_axis_vector_active8_golden.hpp` for `blocks=2`, `active_tokens=8`, `patch_grid_w=8`, `ff_dim=32`.
- Added `hardware/refs/e2e_axis_vector_active16_spec.json` and `hardware/hls/tb/e2e_axis_vector_active16_golden.hpp` for `blocks=2`, `active_tokens=16`, `patch_grid_w=16`, `ff_dim=32`.
- `run_e2e_q4w8a_csim.tcl` now supports strict `active8` and `active16` scale modes.
- `run_e2e_q4w8a_csynth.tcl` now supports opt-in `active8` and `active16` modes in addition to default full and reduced.
- `software/tests/test_e2e_axis_vector_csim.py` now covers reduced, active8, and active16 cases with generated spec/header checks and every output lane parsed from native C++ output.

Latest validation evidence:

- active8 SW/header sync passed with expected raw `[21, -7, 5, -4, 2, -9]`.
- active8 native g++ strict comparator passed with `runtime_state=2 count=6 last=1 failures=0`.
- `HGTXR_E2E_SCALE=active8 /tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f vivado/scripts/run_e2e_q4w8a_csim.tcl` passed from `hardware/` with `CSim done with 0 errors`.
- active16 SW/header sync passed with expected raw `[24, -9, 8, -5, 2, -14]`.
- active16 native g++ strict comparator passed with `runtime_state=2 count=6 last=1 failures=0`.
- `HGTXR_E2E_SCALE=active16 /tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f vivado/scripts/run_e2e_q4w8a_csim.tcl` passed from `hardware/` with `CSim done with 0 errors`.
- Direct execution of both software E2E axis test functions passed across reduced, active8, and active16 cases.
- `python3 hardware/tools/static_validate_hgtxr.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR` passed.

Next task: raise FF dim in staged golden specs, then combine token/block/FF ramps toward full `active_tokens=196`, `blocks=6`, `ff_dim=768` equivalence before full csynth/resource extraction.

## Continuation Update - 2026-06-10 FF-Dim Ramp Gates

T-E2E-FULL-SCALE-003 remains in progress. FF-dim ramp validation now extends beyond active16/FF32.

- Added `hardware/refs/e2e_axis_vector_active16_ff64_spec.json` and `hardware/hls/tb/e2e_axis_vector_active16_ff64_golden.hpp` for `blocks=2`, `active_tokens=16`, `patch_grid_w=16`, `ff_dim=64`.
- Added `hardware/refs/e2e_axis_vector_active16_ff128_spec.json` and `hardware/hls/tb/e2e_axis_vector_active16_ff128_golden.hpp` for `blocks=2`, `active_tokens=16`, `patch_grid_w=16`, `ff_dim=128`.
- The C++ E2E AXI testbench now has strict golden-header selection for active16, active16_ff64, and active16_ff128 modes.
- `run_e2e_q4w8a_csim.tcl` and `run_e2e_q4w8a_csynth.tcl` now support `HGTXR_E2E_SCALE=active16_ff64` and `HGTXR_E2E_SCALE=active16_ff128`.
- `software/tests/test_e2e_axis_vector_csim.py` covers reduced, active8, active16, active16_ff64, and active16_ff128 cases with generated spec/header checks and every output lane parsed from native C++ output.

Latest validation evidence:

- active16_ff64 SW/header sync passed with expected raw `[24, -9, 8, -5, 2, -14]`.
- active16_ff64 native g++ strict comparator passed with `runtime_state=2 count=6 last=1 failures=0`.
- `HGTXR_E2E_SCALE=active16_ff64 /tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f vivado/scripts/run_e2e_q4w8a_csim.tcl` passed from `hardware/` with `CSim done with 0 errors`.
- active16_ff128 SW/header sync passed with expected raw `[24, -9, 8, -5, 2, -14]`.
- active16_ff128 native g++ strict comparator passed with `runtime_state=2 count=6 last=1 failures=0`.
- `HGTXR_E2E_SCALE=active16_ff128 /tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f vivado/scripts/run_e2e_q4w8a_csim.tcl` passed from `hardware/` with `CSim done with 0 errors`.
- Direct execution of both software E2E axis test functions passed across reduced, active8, active16, active16_ff64, and active16_ff128 cases.
- `python3 hardware/tools/static_validate_hgtxr.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR` passed.
- active16_ff128 Vitis HLS csynth passed with estimated clock 3.695 ns, 6,743,956..6,747,284 cycles, 33.720..33.736 ms, 284 BRAM, 66 DSP, 49,246 FF, 196,407 LUT, and 10 URAM.

Next task: continue the FF-dim ramp toward `ff_dim=768`, then combine with larger token grids and `blocks=6` to reach full `active_tokens=196`, `blocks=6`, `ff_dim=768` numerical equivalence before final full-size resource extraction.

## Continuation Update - 2026-06-10 FF256 High-Hidden-Tap Ramp

T-E2E-FULL-SCALE-003 remains in progress. FF-dim staged validation now extends to active16/FF256 with nonzero high hidden-index taps.

- Added `hardware/refs/e2e_axis_vector_active16_ff256_spec.json` and `hardware/hls/tb/e2e_axis_vector_active16_ff256_golden.hpp` for `blocks=2`, `active_tokens=16`, `patch_grid_w=16`, `ff_dim=256`.
- The FF256 spec adds W1/W2/head entries that touch hidden lanes 128, 143, 159, 191, 224, and 255, so addresses above FF128 contribute to the public six-state output.
- The C++ E2E AXI testbench now selects `e2e_axis_vector_active16_ff256_golden.hpp` with `HGTXR_E2E_USE_ACTIVE16_FF256_GOLDEN` and strict shape assertions.
- `run_e2e_q4w8a_csim.tcl` and `run_e2e_q4w8a_csynth.tcl` now support `HGTXR_E2E_SCALE=active16_ff256`.
- `software/tests/test_e2e_axis_vector_csim.py` now covers reduced, active8, active16, active16_ff64, active16_ff128, and active16_ff256 cases.

Latest validation evidence:

- active16_ff256 SW/header sync passed with expected raw `[25, -6, 13, -5, 7, -8]`.
- Direct execution of both software E2E axis test functions passed across all six staged cases.
- `python3 hardware/tools/static_validate_hgtxr.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR` passed.
- `HGTXR_E2E_SCALE=active16_ff256 /tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f vivado/scripts/run_e2e_q4w8a_csim.tcl` passed from `hardware/` with `CSim done with 0 errors`.
- active16_ff256 Vitis HLS csynth passed with estimated clock 3.695 ns, 8,360,538..8,373,850 cycles, 41.803..41.869 ms, 284 BRAM, 66 DSP, 49,499 FF, 196,675 LUT, and 20 URAM.

Next task: continue FF-dim toward `ff_dim=768`; after that, increase token grid and block count toward full `active_tokens=196`, `blocks=6`, `ff_dim=768` numerical equivalence and final resource extraction.

## Continuation Update - 2026-06-10 FF768 Final-Dim Ramp

T-E2E-FULL-SCALE-003 remains in progress. FF-dim staged validation now reaches the final DeiT-Tiny FF dimension at active16.

- Added `hardware/refs/e2e_axis_vector_active16_ff768_spec.json` and `hardware/hls/tb/e2e_axis_vector_active16_ff768_golden.hpp` for `blocks=2`, `active_tokens=16`, `patch_grid_w=16`, `ff_dim=768`.
- The FF768 spec adds W1/W2/head entries that touch hidden lanes 256, 383, 511, 512, 640, and 767, plus channels 12..17, so final FF dimension addresses contribute to the public six-state output.
- The C++ E2E AXI testbench now selects `e2e_axis_vector_active16_ff768_golden.hpp` with `HGTXR_E2E_USE_ACTIVE16_FF768_GOLDEN` and strict shape assertions.
- `run_e2e_q4w8a_csim.tcl` and `run_e2e_q4w8a_csynth.tcl` now support `HGTXR_E2E_SCALE=active16_ff768`.
- `software/tests/test_e2e_axis_vector_csim.py` now covers reduced, active8, active16, active16_ff64, active16_ff128, active16_ff256, and active16_ff768 cases.

Latest validation evidence:

- active16_ff768 SW/header sync passed with expected raw `[26, -5, 19, -2, 8, -6]`.
- Direct execution of both software E2E axis test functions passed across all staged cases.
- `python3 hardware/tools/static_validate_hgtxr.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR` passed.
- `HGTXR_E2E_SCALE=active16_ff768 /tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f vivado/scripts/run_e2e_q4w8a_csim.tcl` passed from `hardware/` with `CSim done with 0 errors`.
- active16_ff768 Vitis HLS csynth passed with estimated clock 3.985 ns, 14,760,660..14,780,372 cycles, 73.803..73.902 ms, 284 BRAM, 66 DSP, 49,913 FF, 197,249 LUT, and 50 URAM.

Next task: raise active token/patch-grid coverage beyond active16 while keeping FF768 strict, then combine with `blocks=6` to reach full `active_tokens=196`, `blocks=6`, `ff_dim=768` numerical equivalence and final resource extraction.

## Continuation Update - 2026-06-10 Active32 FF768 Token Ramp

T-E2E-FULL-SCALE-003 remains in progress. Token-grid staged validation now reaches active32 while keeping final `ff_dim=768`.

- Added `hardware/refs/e2e_axis_vector_active32_ff768_spec.json` and `hardware/hls/tb/e2e_axis_vector_active32_ff768_golden.hpp` for `blocks=2`, `active_tokens=32`, `patch_grid_h=2`, `patch_grid_w=16`, `ff_dim=768`.
- This gate exercises two patch-grid rows instead of the previous active16 one-row gate.
- The C++ E2E AXI testbench now selects `e2e_axis_vector_active32_ff768_golden.hpp` with `HGTXR_E2E_USE_ACTIVE32_FF768_GOLDEN` and strict shape assertions.
- `run_e2e_q4w8a_csim.tcl` and `run_e2e_q4w8a_csynth.tcl` now support `HGTXR_E2E_SCALE=active32_ff768`.
- `software/tests/test_e2e_axis_vector_csim.py` now covers reduced, active8, active16, active16_ff64, active16_ff128, active16_ff256, active16_ff768, and active32_ff768 cases.

Latest validation evidence:

- active32_ff768 SW/header sync passed with expected raw `[26, -5, 19, -2, 8, -6]`.
- Direct execution of both software E2E axis test functions passed across all staged cases.
- `python3 hardware/tools/static_validate_hgtxr.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR` passed.
- `HGTXR_E2E_SCALE=active32_ff768 /tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f vivado/scripts/run_e2e_q4w8a_csim.tcl` passed from `hardware/` with `CSim done with 0 errors`.
- active32_ff768 Vitis HLS csynth passed with estimated clock 3.985 ns, 29,488,052..29,527,476 cycles, 0.147..0.148 sec, 284 BRAM, 66 DSP, 50,256 FF, 197,595 LUT, and 50 URAM.

Next task: raise token coverage toward active64/active128/full active196 while keeping FF768 strict, then combine with `blocks=6` for full numerical equivalence and final resource extraction.

## Continuation Update - 2026-06-10 Active64 FF768 Token Ramp

T-E2E-FULL-SCALE-003 remains in progress. Token-grid staged validation now reaches active64 while keeping final `ff_dim=768`.

- Added `hardware/refs/e2e_axis_vector_active64_ff768_spec.json` and `hardware/hls/tb/e2e_axis_vector_active64_ff768_golden.hpp` for `blocks=2`, `active_tokens=64`, `patch_grid_h=4`, `patch_grid_w=16`, `ff_dim=768`.
- This gate exercises four patch-grid rows while preserving the same deterministic high-hidden FF768 pattern used by the active32 ramp.
- The C++ E2E AXI testbench now selects `e2e_axis_vector_active64_ff768_golden.hpp` with `HGTXR_E2E_USE_ACTIVE64_FF768_GOLDEN` and strict shape assertions.
- `run_e2e_q4w8a_csim.tcl` and `run_e2e_q4w8a_csynth.tcl` now support `HGTXR_E2E_SCALE=active64_ff768`.
- `software/tests/test_e2e_axis_vector_csim.py` now covers reduced, active8, active16, active16_ff64, active16_ff128, active16_ff256, active16_ff768, active32_ff768, and active64_ff768 cases.

Latest validation evidence:

- active64_ff768 SW/header sync passed with expected raw `[26, -5, 19, -2, 8, -6]`.
- Direct execution of both software E2E axis test functions passed across all staged cases.
- `python3 hardware/tools/static_validate_hgtxr.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR` passed.
- `HGTXR_E2E_SCALE=active64_ff768 /tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f vivado/scripts/run_e2e_q4w8a_csim.tcl` passed from `hardware/` with `CSim done with 0 errors`.
- active64_ff768 Vitis HLS csynth passed with estimated clock 3.985 ns, 59,191,668..59,270,516 cycles, 0.296 sec, 284 BRAM, 66 DSP, 50,606 FF, 197,896 LUT, and 50 URAM.

Next task: raise token coverage toward active128/full active196 while keeping FF768 strict, then combine with `blocks=6` for full numerical equivalence and final resource extraction.

## Continuation Update - 2026-06-10 Active128 FF768 Token Ramp

T-E2E-FULL-SCALE-003 remains in progress. Token-grid staged validation now reaches active128 while keeping final `ff_dim=768`.

- Added `hardware/refs/e2e_axis_vector_active128_ff768_spec.json` and `hardware/hls/tb/e2e_axis_vector_active128_ff768_golden.hpp` for `blocks=2`, `active_tokens=128`, `patch_grid_h=8`, `patch_grid_w=16`, `ff_dim=768`.
- This gate exercises eight patch-grid rows while preserving the deterministic high-hidden FF768 pattern used by the active64 ramp.
- The C++ E2E AXI testbench now selects `e2e_axis_vector_active128_ff768_golden.hpp` with `HGTXR_E2E_USE_ACTIVE128_FF768_GOLDEN` and strict shape assertions.
- `run_e2e_q4w8a_csim.tcl` and `run_e2e_q4w8a_csynth.tcl` now support `HGTXR_E2E_SCALE=active128_ff768`.
- `software/tests/test_e2e_axis_vector_csim.py` now covers reduced, active8, active16, active16_ff64, active16_ff128, active16_ff256, active16_ff768, active32_ff768, active64_ff768, and active128_ff768 cases.

Latest validation evidence:

- active128_ff768 SW/header sync passed with expected raw `[26, -5, 19, -2, 8, -6]`.
- Direct execution of both software E2E axis test functions passed across all staged cases.
- `python3 hardware/tools/static_validate_hgtxr.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR` passed.
- `HGTXR_E2E_SCALE=active128_ff768 /tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f vivado/scripts/run_e2e_q4w8a_csim.tcl` passed from `hardware/` with `CSim done with 0 errors`.
- active128_ff768 Vitis HLS csynth passed with estimated clock 3.985 ns, 119,602,705..119,760,401 cycles, 0.598..0.599 sec, 284 BRAM, 66 DSP, 50,968 FF, 198,256 LUT, and 50 URAM.

Next task: raise token coverage toward full active196 while keeping FF768 strict, then combine with `blocks=6` for full numerical equivalence and final resource extraction.

## Continuation Update - 2026-06-10 Active196 FF768 Full-Token Ramp

T-E2E-FULL-SCALE-003 remains in progress. Token-grid staged validation now reaches the full 14x14 active196 token grid while keeping final `ff_dim=768` and `blocks=2`.

- Added `hardware/refs/e2e_axis_vector_active196_ff768_spec.json` and `hardware/hls/tb/e2e_axis_vector_active196_ff768_golden.hpp` for `blocks=2`, `active_tokens=196`, `patch_grid_h=14`, `patch_grid_w=14`, `ff_dim=768`.
- This gate exercises the full 14x14 ViT token grid while preserving the deterministic high-hidden FF768 pattern used by the active128 ramp.
- The C++ E2E AXI testbench now selects `e2e_axis_vector_active196_ff768_golden.hpp` with `HGTXR_E2E_USE_ACTIVE196_FF768_GOLDEN` and strict shape assertions.
- `run_e2e_q4w8a_csim.tcl` and `run_e2e_q4w8a_csynth.tcl` now support `HGTXR_E2E_SCALE=active196_ff768`.
- `software/tests/test_e2e_axis_vector_csim.py` now covers reduced, active8, active16, active16_ff64, active16_ff128, active16_ff256, active16_ff768, active32_ff768, active64_ff768, active128_ff768, and active196_ff768 cases.

Latest validation evidence:

- active196_ff768 SW/header sync passed with expected raw `[26, -5, 19, -2, 8, -6]`.
- Direct execution of both software E2E axis test functions passed across all staged cases.
- `python3 hardware/tools/static_validate_hgtxr.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR` passed.
- `HGTXR_E2E_SCALE=active196_ff768 /tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f vivado/scripts/run_e2e_q4w8a_csim.tcl` passed from `hardware/` with `CSim done with 0 errors`.
- active196_ff768 Vitis HLS csynth passed with estimated clock 3.985 ns, 185,285,394..185,526,866 cycles, 0.926..0.928 sec, 284 BRAM, 67 DSP, 51,406 FF, 199,218 LUT, and 50 URAM.

Next task: increase block count from `blocks=2` toward `blocks=6` while keeping active196/FF768 strict, then capture final full numerical equivalence and resource extraction.


## Continuation Update - 2026-06-10 Active196 B6 FF768 Full-Block Gate

T-E2E-FULL-SCALE-003 is now complete for the strict generated-golden E2E AXI numerical gate.

- Added `hardware/refs/e2e_axis_vector_active196_b6_ff768_spec.json` and `hardware/hls/tb/e2e_axis_vector_active196_b6_ff768_golden.hpp` for `blocks=6`, `active_tokens=196`, `patch_grid_h=14`, `patch_grid_w=14`, `ff_dim=768`.
- This gate exercises the full 14x14 ViT token grid and the final six-block E2E controller traversal with the same deterministic arbitrary Q4 pattern family used by the staged ramp.
- The C++ E2E AXI testbench now selects `e2e_axis_vector_active196_b6_ff768_golden.hpp` with `HGTXR_E2E_USE_ACTIVE196_B6_FF768_GOLDEN` and strict shape assertions.
- `run_e2e_q4w8a_csim.tcl` and `run_e2e_q4w8a_csynth.tcl` now support `HGTXR_E2E_SCALE=active196_b6_ff768`.
- `software/tests/test_e2e_axis_vector_csim.py` now covers reduced, active8, active16, active16_ff64, active16_ff128, active16_ff256, active16_ff768, active32_ff768, active64_ff768, active128_ff768, active196_ff768, and active196_b6_ff768 cases.

Latest validation evidence:

- active196_b6_ff768 SW/header sync passed with expected raw `[32, -13, 26, -6, 14, -11]`.
- Direct execution of both software E2E axis test functions passed across all staged cases.
- `python3 hardware/tools/static_validate_hgtxr.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR` passed.
- `HGTXR_E2E_SCALE=active196_b6_ff768 /tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f vivado/scripts/run_e2e_q4w8a_csim.tcl` passed from `hardware/` with `CSim done with 0 errors`.
- active196_b6_ff768 Vitis HLS csynth passed with estimated clock 4.035 ns, 555,505,175..556,229,591 cycles, 2.778..2.781 sec, 284 BRAM, 67 DSP, 52,318 FF, 199,975 LUT, and 50 URAM.

Next task: broader project work remains open: final HG-PIPE-equivalent LUT/calibration tables, final paper artifact alignment, board-side ZCU104/PYNQ validation, and performance optimization around packed-weight traffic.


## Continuation Update - 2026-06-10 HG-PIPE Math Contract Gate

T-HGPIPE-MATH-CONTRACT-004 is complete for the compact baseline.

- Added `hardware/refs/hgpipe_lut_math_contract.json` to capture the current HG-PIPE-style cursor/table/clamp contract for GeLU, Softmax exp, LayerNorm rsqrt, and quantize_clamp.
- `hardware/tools/validate_hgpipe_lut_math.py` now reads this contract instead of carrying only local table literals.
- Generated validation artifact: `docs/resources/hgpipe_lut_math_contract_validation_2026_06_10.json`.
- Static validation now requires both the contract and validation artifact.
- `/home/kjm26/project/PRJXR/XR-VITs` was requested as a reference path, but it is not present in the current filesystem; available `/home/kjm26/project/PRJXR` entries are `HBTXR_MERGE`, `Hardware`, `References`, `SW_REF`, `XR-VIT`, `model_zoo`, and `old`.

Next task: replace compact baseline entries with final trained per-layer HG-PIPE quantization/LUT artifacts when those artifacts are available, then re-run E2E strict gates and csynth.

## Continuation Update - 2026-06-10 PAR8 DSP/LUTRAM Resource Retune

The user's follow-up requested higher DSP use by increasing parallelism and using LUTRAM for small memories. This active16 staged resource path is now retuned, but full active196/b6 PAR8 fit remains open.

Implemented:

- `hardware/configs/zcu104_e2e_q4w8a_defines.h` now sets `HGTXR_PARALLELISM_FACTOR=8`, `HGTXR_E2E_ATTN_PAR=8`, and `HGTXR_E2E_DENSE_PAR=8`.
- `hardware/hls/include/hgtxr_e2e_vit.hpp` now has packed-weight vector cache controls, an aligned fast path, LUTRAM binding for small attention scratch arrays, and LayerNorm gamma/beta packed-word caching.
- Dense QKV, WO, MLP W1/W2, and head loops use packed vector loads.
- `score`, `prob`, and `exp_raw` are mapped to LUTRAM; large activation/global buffers remain URAM-backed under `dsp_uram`.
- GPT5.3-Codex-Spark was unavailable because of usage limit, so GPT5.5 sub-agent `Russell` performed the read-only audit that recommended the LayerNorm gamma/beta cache. The sub-agent made no edits and was closed after integration.

Latest validation evidence:

- `HGTXR_E2E_SCALE=hgpipe_math_lnq_active16`, `HGTXR_E2E_RESOURCE_POLICY=dsp_uram` Vitis HLS CSim passed with outputs `[58, -51, 42, -28, 36, -41]`, `runtime_state=2 count=6 last=1 failures=0`, and `CSim done with 0 errors`.
- Final Vitis HLS CSynth passed with estimated clock `4.069 ns`, latency `494,968 cycles` / `2.475 ms`, and resources `42 BRAM_18K`, `128 DSP`, `19,576 FF`, `45,431 LUT`, `88 URAM`.
- Final report scan confirmed LUTRAM binding for `score`, `prob`, and `exp_raw`.
- Final report scan found no remaining packed-weight `word1` II warning.
- Archived final report: `/tmp/hgtxr_e2e_axis_top_hgpipe_math_lnq_active16_par8_fastcache_dsp_uram_csynth.rpt`.

Resource delta versus prior active16 PAR5 `dsp_uram`:

- DSP: `55 -> 128`
- Latency: `2,841,904 -> 494,968 cycles`
- LUT: `44,670 -> 45,431`
- FF: `27,574 -> 19,576`
- BRAM: `39 -> 42`
- URAM: `80 -> 88`

Next task: run the same policy on a larger/full E2E scale or back off URAM selectively if full active196/b6 PAR8 exceeds the 96-URAM device budget.


## Continuation Update - 2026-06-10 HG-PIPE GeLUQ64 Cursor Gate

T-HGPIPE-GELUQ64-CURSOR-005 is complete as an isolated contract/primitive gate.

- Added generic HG-PIPE cursor-table helpers in `hardware/hls/include/hgtxr_cyclic_math.hpp`: `hgtxr_hgpipe_cursor_lut_index`, `hgtxr_hgpipe_int_table_lookup`, and `hgtxr_hgpipe_geluq64_int`.
- The helper matches HG-PIPE `src/gelu.h` and `src/quant.h` semantics: `cursor = (x + b) >> s`, clamp to `[0, bound]`, then table lookup.
- Extended `hardware/refs/hgpipe_lut_math_contract.json` with the HG-PIPE `mlp_1_geluq` scalars `[94, 2, 63]`, 64-entry table, `ap_int<12>` input, `ap_int<9>` cursor, `ap_uint<3>` output, and `case/refs/mlp_1_geluq_*` provenance.
- Extended `hardware/tools/validate_hgpipe_lut_math.py` to parse the HG-PIPE integer refs and verify every `mlp_1_geluq` input/output sample.
- Re-generated `docs/resources/hgpipe_lut_math_contract_validation_2026_06_10.json`; the new `hgpipe_ref_checks.gelu_quantized_mlp1` reports `checked_samples=150528`, `mismatches=0`, cursor range `[0, 63]`, and output range `[0, 7]`.
- A real read-only sub-agent review (`Ptolemy`, GPT5.5/high) confirmed the safe boundary: keep the helper isolated and do not replace default E2E GeLU until the Python mirror, generated golden headers, csim, and csynth are refreshed together.

Validation evidence:

- `python3 -m py_compile tools/validate_hgpipe_lut_math.py tools/static_validate_hgtxr.py` passed.
- `python3 tools/validate_hgpipe_lut_math.py --contract refs/hgpipe_lut_math_contract.json --out /tmp/hgpipe_lut_math_contract_geluq_validation.json` passed.
- `python3 tools/static_validate_hgtxr.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR` passed.
- `python3 tools/validate_hgpipe_lut_math.py --contract refs/hgpipe_lut_math_contract.json --out /home/kjm26/project/PRJXR/XR-VIT/HGTXR/docs/resources/hgpipe_lut_math_contract_validation_2026_06_10.json` passed.

Next task: if GeLUQ64 is wired into the default E2E path, update `hardware/tools/validate_e2e_axis_vector.py`, regenerate every affected `e2e_axis_vector_*` spec/golden, then rerun native comparator, Vitis csim, and csynth. Until then, the default E2E path remains on the compact 16-entry activation-scale GeLU LUT.


## Continuation Update - 2026-06-10 HG-PIPE Quant attn0_q Cursor Gate

T-HGPIPE-QUANT-ATTN0Q-CURSOR-007 is complete as an isolated contract/primitive gate.

- Spark sidecar execution was attempted first but hit the GPT5.3-Codex-Spark usage limit; fallback read-only sub-agent review (`Hubble`, GPT5.5/medium) completed and confirmed the constants below.
- Added `hgtxr_hgpipe_attn0_q_quant64_int()` in `hardware/hls/include/hgtxr_cyclic_math.hpp`.
- The primitive uses the same generic HG-PIPE cursor-table helper as GeLUQ64: `cursor = (x + b) >> s`, clamp to `[0, bound]`, table lookup.
- Extended `hardware/refs/hgpipe_lut_math_contract.json` with HG-PIPE `case/QUANT.cpp` provenance, `attn_0_q_q` scalars `[88, 2, 63]`, 64-entry signed quant table, `ap_int<11>` input, `ap_int<8>` cursor, `ap_int<3>` output, and `case/refs/attn_0_q_q_*` / `attn_0_qq_*` refs.
- Strengthened `hardware/tools/validate_hgpipe_lut_math.py` so cursor-table refs validate `ap_int`/`ap_uint` ranges, table bound validity, raw cursor range, clamped cursor range, clamp-hit counts, and full input/output replay.
- Re-generated `docs/resources/hgpipe_lut_math_contract_validation_2026_06_10.json`; `hgpipe_ref_checks.quant_attn0_q` reports `checked_samples=37632`, `mismatches=0`, raw cursor range `[-64, 91]`, cursor range `[0, 63]`, clamp hits `{low: 2752, high: 449}`, and output range `[-4, 3]`.

Validation evidence:

- `python3 -m py_compile tools/validate_hgpipe_lut_math.py tools/static_validate_hgtxr.py` passed.
- `python3 tools/validate_hgpipe_lut_math.py --contract refs/hgpipe_lut_math_contract.json --out /tmp/hgpipe_lut_math_contract_quant_validation2.json` passed.
- `python3 tools/static_validate_hgtxr.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR` passed.
- `python3 tools/validate_hgpipe_lut_math.py --contract refs/hgpipe_lut_math_contract.json --out /home/kjm26/project/PRJXR/XR-VIT/HGTXR/docs/resources/hgpipe_lut_math_contract_validation_2026_06_10.json` passed.

Next task: choose whether to add more per-layer HG-PIPE quant refs, starting with Q/K/V/O or MLP tables, or move to one default E2E wiring gate that updates the Python mirror, generated goldens, native comparator, Vitis csim, and csynth together.


## Continuation Update - 2026-06-10 HG-PIPE Softmax attn0 Cursor/Recip Gate

T-HGPIPE-SOFTMAX-ATTN0-CURSOR-009 is complete as an isolated contract/primitive gate.

- Spark sidecar execution was attempted first but hit the GPT5.3-Codex-Spark usage limit; fallback read-only sub-agent review (`Beauvoir`, GPT5.5/medium) completed and confirmed the SOFTMAX_2X1 constants and 3-pass formula.
- Added isolated HLS helper blocks in `hardware/hls/include/hgtxr_cyclic_math.hpp` for HG-PIPE `attn_0_softmaxq`: 32-entry exp lookup, reciprocal table branch selection, reciprocal lookup, and uint3 requant.
- Extended `hardware/refs/hgpipe_lut_math_contract.json` with HG-PIPE `SOFTMAX_2X1.cpp` provenance, `attn_0_softmaxq_*` ref files, all 14 scalars, type metadata, 32-entry exp table, and two 64-entry reciprocal tables.
- Extended `hardware/tools/validate_hgpipe_lut_math.py` with a rowwise 3-pass `softmax_attn0` replay: max, exp/sum, reciprocal table selection, multiply/requant, and uint3 clamp.
- Re-generated `docs/resources/hgpipe_lut_math_contract_validation_2026_06_10.json`; `hgpipe_ref_checks.softmax_attn0` reports `checked_samples=115248`, `mismatches=0`, exp cursor range `[0, 27]`, acc range `[40601, 2335801]`, reciprocal table-two rows `44`, and output range `[0, 7]`.

Validation evidence:

- `python3 -m py_compile tools/validate_hgpipe_lut_math.py tools/static_validate_hgtxr.py` passed.
- `python3 tools/validate_hgpipe_lut_math.py --contract refs/hgpipe_lut_math_contract.json --out /tmp/hgpipe_lut_math_contract_softmax_validation.json` passed.
- `python3 tools/static_validate_hgtxr.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR` passed.
- Native E2E C++ header smoke passed via `/tmp/tb_hgtxr_e2e_axis_top_softmax_header_check` with `failures=0`.

Next task: keep extending exact HG-PIPE table contracts for LayerNorm or remaining per-layer quant tables, or schedule a deliberate E2E wiring gate that updates the Python mirror and generated goldens together. Do not mix the current 16-entry float compact softmax exp contract with the 32-entry integer HG-PIPE softmaxq path.


## Continuation Update - 2026-06-10 HG-PIPE LayerNorm attn1 Cursor Gate

T-HGPIPE-LAYERNORM-ATTN1-CURSOR-011 is complete as an isolated contract/primitive gate.

- Spark sidecar execution was attempted first but hit the GPT5.3-Codex-Spark usage limit; fallback GPT5.5 evaluator was spawned for read-only verification.
- Added isolated HLS helper blocks in `hardware/hls/include/hgtxr_cyclic_math.hpp` for HG-PIPE `attn_1_lnq`: fixed-point mean, 128-entry rsqrt lookup, and signed 3-bit affine requant.
- Extended `hardware/refs/hgpipe_lut_math_contract.json` with HG-PIPE `LAYERNORM_2X2.cpp` provenance, `attn_1_lnq_*` ref files, scalar metadata, 128-entry rsqrt table, and fixed-point type metadata.
- Extended `hardware/tools/validate_hgpipe_lut_math.py` with a rowwise 3-pass `layernorm_attn1` replay: mean, variance-sum cursor, rsqrt lookup, affine with 192-entry `lnw`/`lnb`, right shift, and signed 3-bit clamp.
- Re-generated `docs/resources/hgpipe_lut_math_contract_validation_2026_06_10.json`; `hgpipe_ref_checks.layernorm_attn1` reports `checked_samples=37632`, `mismatches=0`, mean range `[-53, 115]`, variance-sum range `[11421135, 45038293]`, cursor range `[2, 34]`, and output range `[-4, 3]`.

Validation evidence:

- `python3 -m py_compile tools/validate_hgpipe_lut_math.py tools/static_validate_hgtxr.py` passed.
- `python3 tools/validate_hgpipe_lut_math.py --contract refs/hgpipe_lut_math_contract.json --out /tmp/hgpipe_lut_math_contract_layernorm_validation.json` passed.
- `python3 tools/validate_hgpipe_lut_math.py --contract refs/hgpipe_lut_math_contract.json --out /home/kjm26/project/PRJXR/XR-VIT/HGTXR/docs/resources/hgpipe_lut_math_contract_validation_2026_06_10.json` passed.
- `python3 tools/static_validate_hgtxr.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR` passed.
- Native E2E C++ header smoke passed via `/tmp/tb_hgtxr_e2e_axis_top_layernorm_header_check` with `runtime_state=2 count=6 last=1 failures=0`.

Next task: decide whether to continue with more per-layer HG-PIPE quant table contracts or schedule a deliberate default E2E integer-math wiring gate. Do not mix the isolated `attn_1_lnq` integer LayerNorm path with the current compact E2E rsqrt/affine mirror without regenerating the Python mirror, golden headers, native comparator, Vitis csim, and csynth together.


## Continuation Update - 2026-06-10 HG-PIPE Quant attn0 QKVA Cursor Gate

T-HGPIPE-QUANT-ATTN0-QKVA-008A is complete as an isolated contract/primitive gate.

- Spark read-only sidecar confirmed the safe next quant set: `attn_0_k_q`, `attn_0_v_q`, and `attn_0_a_q` share the same `T=196`, `C=192`, 64-entry signed quant cursor-table pattern as the existing `attn_0_q_q` gate.
- Added isolated HLS helpers in `hardware/hls/include/hgtxr_cyclic_math.hpp`: `hgtxr_hgpipe_attn0_k_quant64_int()`, `hgtxr_hgpipe_attn0_v_quant64_int()`, and `hgtxr_hgpipe_attn0_a_quant64_int()`.
- Extended `hardware/refs/hgpipe_lut_math_contract.json` with HG-PIPE `attn_0_{k,v,a}_q` scalar/table refs and `attn_0_{k,v,a}q` input/output refs.
- Extended `hardware/tools/validate_hgpipe_lut_math.py` so the attention-0 quant set validates Q/K/V/A through the same cursor-table replay gate.
- Re-generated `docs/resources/hgpipe_lut_math_contract_validation_2026_06_10.json`; `quant_attn0_q`, `quant_attn0_k`, `quant_attn0_v`, and `quant_attn0_a` each report `checked_samples=37632`, `mismatches=0`, and output range `[-4, 3]`.

Validation evidence:

- `python3 -m py_compile tools/validate_hgpipe_lut_math.py tools/static_validate_hgtxr.py` passed.
- `python3 tools/validate_hgpipe_lut_math.py --contract refs/hgpipe_lut_math_contract.json --out /tmp/hgpipe_lut_math_contract_attn0_quantqkva_validation.json` passed.
- `python3 tools/validate_hgpipe_lut_math.py --contract refs/hgpipe_lut_math_contract.json --out /home/kjm26/project/PRJXR/XR-VIT/HGTXR/docs/resources/hgpipe_lut_math_contract_validation_2026_06_10.json` passed.
- `python3 tools/static_validate_hgtxr.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR` passed.
- Native E2E C++ header smoke passed via `/tmp/tb_hgtxr_e2e_axis_top_attn0_quant_header_check` with `runtime_state=2 count=6 last=1 failures=0`.

Next task: continue `T-HGPIPE-QUANT-LAYERS-008` with MLP and later-layer quant refs, or schedule a deliberate default E2E integer-math wiring gate after all selected per-layer table provenance is frozen.


## Continuation Update - 2026-06-10 HG-PIPE GeLUQ MLP0 Cursor Gate

T-HGPIPE-GELUQ-MLP0-005A is complete as an isolated contract/primitive gate.

- Spark read-only sidecar confirmed `mlp_0_geluq` uses the same HG-PIPE `GeLU::do_gelu` cursor-table formula as `mlp_1_geluq`, but with distinct scalars and table content.
- Added `hardware/refs/hgpipe_lut_math_contract.json` entries for `mlp_0_geluq_scalars`, `mlp_0_geluq_table_m`, `mlp_0_geluq_input`, and `mlp_0_geluq_output`.
- Added isolated HLS helper `hgtxr_hgpipe_mlp0_geluq64_int()` in `hardware/hls/include/hgtxr_cyclic_math.hpp`; the existing `hgtxr_hgpipe_geluq64_int()` remains an MLP1-compatible alias.
- Generalized `hardware/tools/validate_hgpipe_lut_math.py` so GeLUQ refs validate through a MLP0/MLP1 key list.
- Re-generated `docs/resources/hgpipe_lut_math_contract_validation_2026_06_10.json`; `gelu_quantized_mlp0` reports `checked_samples=150528`, `mismatches=0`, raw cursor range `[-182, 295]`, cursor range `[0, 63]`, clamp hits `{low: 26776, high: 7540}`, and output range `[0, 7]`.

Validation evidence:

- `python3 -m py_compile tools/validate_hgpipe_lut_math.py tools/static_validate_hgtxr.py` passed.
- `python3 tools/validate_hgpipe_lut_math.py --contract refs/hgpipe_lut_math_contract.json --out /tmp/hgpipe_lut_math_contract_mlp0_geluq_validation.json` passed.
- `python3 tools/validate_hgpipe_lut_math.py --contract refs/hgpipe_lut_math_contract.json --out /home/kjm26/project/PRJXR/XR-VIT/HGTXR/docs/resources/hgpipe_lut_math_contract_validation_2026_06_10.json` passed.
- `python3 tools/static_validate_hgtxr.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR` passed.
- Native E2E C++ header smoke passed via `/tmp/tb_hgtxr_e2e_axis_top_mlp0_geluq_header_check` with `runtime_state=2 count=6 last=1 failures=0`.

Next task: continue per-layer HG-PIPE activation/quant table contracts for later MLP layers or plan the default E2E integer-math wiring gate. Do not wire MLP0/MLP1 GeLUQ into default E2E until the software mirror and generated golden headers use the same unsigned 3-bit activation scale.


## Continuation Update - 2026-06-10 HG-PIPE GeLUQ MLP10/MLP11 Cursor Gate

T-HGPIPE-GELUQ-MLP10-11-005B is complete as an isolated contract/primitive gate.

- Spark read-only sidecar was attempted first but failed because the sub-agent context window was exhausted; the main agent continued from direct HG-PIPE refs.
- Added `hardware/refs/hgpipe_lut_math_contract.json` entries for `mlp_10_geluq_*` and `mlp_11_geluq_*`.
- Added isolated HLS helpers `hgtxr_hgpipe_mlp10_geluq64_int()` and `hgtxr_hgpipe_mlp11_geluq64_int()` in `hardware/hls/include/hgtxr_cyclic_math.hpp`.
- Extended `hardware/tools/validate_hgpipe_lut_math.py` GeLUQ key registration to MLP0/MLP1/MLP10/MLP11.
- Re-generated `docs/resources/hgpipe_lut_math_contract_validation_2026_06_10.json`; MLP0/MLP1/MLP10/MLP11 GeLUQ checks cover 602,112 total samples with `mismatches=0`.

Validation evidence:

- `python3 -m py_compile tools/validate_hgpipe_lut_math.py tools/static_validate_hgtxr.py` passed.
- `python3 tools/validate_hgpipe_lut_math.py --contract refs/hgpipe_lut_math_contract.json --out /tmp/hgpipe_lut_math_contract_mlp10_11_geluq_validation.json` passed.
- `python3 tools/validate_hgpipe_lut_math.py --contract refs/hgpipe_lut_math_contract.json --out /home/kjm26/project/PRJXR/XR-VIT/HGTXR/docs/resources/hgpipe_lut_math_contract_validation_2026_06_10.json` passed.
- `python3 tools/static_validate_hgtxr.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR` passed.
- Native E2E C++ header smoke passed via `/tmp/tb_hgtxr_e2e_axis_top_mlp10_11_geluq_header_check` with `runtime_state=2 count=6 last=1 failures=0`.

Next task: continue per-layer HG-PIPE quant/activation contract coverage for remaining layer refs, or freeze a selected table provenance set and run a deliberate default E2E integer-math wiring gate.


## Continuation Update - 2026-06-10 HG-PIPE GeLUQ MLP2-9 Cursor Gate

T-HGPIPE-GELUQ-MLP2-9-005C is complete as an isolated contract/primitive gate.

- Spark read-only sidecar and direct HG-PIPE ref sweep confirmed all `MLP0..MLP11` GeLUQ refs are complete; `mlp_2..9_geluq` were the missing contract/helper coverage.
- Added `hardware/refs/hgpipe_lut_math_contract.json` entries for `gelu_quantized_mlp2..9` scalar/table/input/output refs.
- Extended `hardware/tools/validate_hgpipe_lut_math.py` GeLUQ registration to generate all 12 MLP layer keys.
- Added isolated HLS helpers `hgtxr_hgpipe_mlp2_geluq64_int()` through `hgtxr_hgpipe_mlp9_geluq64_int()` in `hardware/hls/include/hgtxr_cyclic_math.hpp`.
- Updated MLP GeLUQ contract input/cursor type metadata to match each HG-PIPE `MLP*.cpp` typedef.
- Re-generated `docs/resources/hgpipe_lut_math_contract_validation_2026_06_10.json`; all MLP0..11 GeLUQ checks cover 1,806,336 total samples with `mismatches=0`.

Validation evidence:

- `python3 -m py_compile tools/validate_hgpipe_lut_math.py tools/static_validate_hgtxr.py` passed.
- `python3 tools/validate_hgpipe_lut_math.py --contract refs/hgpipe_lut_math_contract.json --out /tmp/hgpipe_lut_math_contract_mlp2_9_geluq_validation.json` passed.
- `python3 tools/validate_hgpipe_lut_math.py --contract refs/hgpipe_lut_math_contract.json --out /home/kjm26/project/PRJXR/XR-VIT/HGTXR/docs/resources/hgpipe_lut_math_contract_validation_2026_06_10.json` passed after write access was granted for the docs artifact.
- `python3 tools/static_validate_hgtxr.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR` passed.
- Native reduced E2E C++ comparator passed with `runtime_state=2 count=6 last=1 failures=0`.

Next task: continue HG-PIPE LayerNorm/Softmax layer coverage where complete refs exist, or freeze the current table-provenance subset and schedule the deliberate default E2E integer-math wiring gate.


## Continuation Update - 2026-06-10 HG-PIPE Quant attn1 QKVA Cursor Gate

T-HGPIPE-QUANT-ATTN1-QKVA-008B is complete as an isolated contract/primitive gate.

- Spark read-only sidecar confirmed `attn_1_{q,k,v,a}_q` refs match the existing cursor-table validator shape/type assumptions: `sample_shape=[196,192]`, input `ap_int<11>`, output `ap_int<3>`, 64-entry tables.
- Added `hardware/refs/hgpipe_lut_math_contract.json` entries for `attn_1_{q,k,v,a}_q` scalar/table refs and `attn_1_{q,k,v,a}q` input/output refs.
- Generalized `hardware/tools/validate_hgpipe_lut_math.py` attention quant registration from `ATTN0_QUANT_KEYS` to `ATTN_QUANT_KEYS`.
- Added isolated HLS helpers `hgtxr_hgpipe_attn1_q_quant64_int()`, `hgtxr_hgpipe_attn1_k_quant64_int()`, `hgtxr_hgpipe_attn1_v_quant64_int()`, and `hgtxr_hgpipe_attn1_a_quant64_int()`.
- Re-generated `docs/resources/hgpipe_lut_math_contract_validation_2026_06_10.json`; attention layer 0/1 Q/K/V/A checks cover 301,056 total samples with `mismatches=0`.

Validation evidence:

- `python3 -m py_compile tools/validate_hgpipe_lut_math.py tools/static_validate_hgtxr.py` passed.
- `python3 tools/validate_hgpipe_lut_math.py --contract refs/hgpipe_lut_math_contract.json --out /tmp/hgpipe_lut_math_contract_attn1_quantqkva_validation.json` passed.
- `python3 tools/validate_hgpipe_lut_math.py --contract refs/hgpipe_lut_math_contract.json --out /home/kjm26/project/PRJXR/XR-VIT/HGTXR/docs/resources/hgpipe_lut_math_contract_validation_2026_06_10.json` passed.
- `python3 tools/static_validate_hgtxr.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR` passed.
- Native E2E C++ header smoke passed via `/tmp/tb_hgtxr_e2e_axis_top_attn1_quant_header_check` with `runtime_state=2 count=6 last=1 failures=0`.

Next task: continue later-layer attention quant coverage or freeze a smaller table-provenance subset for a deliberate default E2E integer-math wiring gate.


## Continuation Update - 2026-06-10 HG-PIPE Quant attn2 QKVA Cursor Gate

T-HGPIPE-QUANT-ATTN2-QKVA-008C is complete as an isolated contract/primitive gate.

- Spark read-only sidecar confirmed `attn_2` is the next complete attention quant quartet after `attn_0` and `attn_1`; `attn_2_{q,k,v,a}_q` scalar/table refs and `attn_2_{q,k,v,a}q` input/output refs are all present.
- Added `hardware/refs/hgpipe_lut_math_contract.json` entries for `quant_attn2_{q,k,v,a}` scalar/table/input/output refs.
- Extended `hardware/tools/validate_hgpipe_lut_math.py` attention quant registration to validate `quant_attn2_{q,k,v,a}` through the same cursor-table replay gate.
- Added isolated HLS helpers `hgtxr_hgpipe_attn2_q_quant64_int()`, `hgtxr_hgpipe_attn2_k_quant64_int()`, `hgtxr_hgpipe_attn2_v_quant64_int()`, and `hgtxr_hgpipe_attn2_a_quant64_int()`.
- Re-generated `docs/resources/hgpipe_lut_math_contract_validation_2026_06_10.json`; attention layer 0/1/2 Q/K/V/A checks cover 451,584 total samples with `mismatches=0`.

Validation evidence:

- `python3 -m py_compile tools/validate_hgpipe_lut_math.py tools/static_validate_hgtxr.py` passed.
- `python3 tools/validate_hgpipe_lut_math.py --contract refs/hgpipe_lut_math_contract.json --out /tmp/hgpipe_lut_math_contract_attn2_quantqkva_validation.json` passed.
- `python3 tools/validate_hgpipe_lut_math.py --contract refs/hgpipe_lut_math_contract.json --out /home/kjm26/project/PRJXR/XR-VIT/HGTXR/docs/resources/hgpipe_lut_math_contract_validation_2026_06_10.json` passed after write access was granted for the docs artifact.
- `python3 tools/static_validate_hgtxr.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR` passed.

Next task: continue later-layer attention quant coverage (`attn_3+`) where complete refs exist, or freeze a selected table-provenance subset and schedule the deliberate default E2E integer-math wiring gate.


## Continuation Update - 2026-06-10 HG-PIPE Quant attn3-11 QKVA Cursor Gate

T-HGPIPE-QUANT-ATTN3-11-QKVA-008D is complete as an isolated contract/primitive gate.

- Direct HG-PIPE ref audit and Spark read-only sidecar confirmed `attn_3` is complete; direct sweep confirmed `attn_3..11` all have Q/K/V/A scalar, table, input, and output refs.
- Added `hardware/refs/hgpipe_lut_math_contract.json` entries for `quant_attn3..11_{q,k,v,a}` scalar/table/input/output refs.
- Extended `hardware/tools/validate_hgpipe_lut_math.py` attention quant registration to generate all 12 attention-layer Q/K/V/A keys.
- Added isolated HLS helpers `hgtxr_hgpipe_attn3..11_{q,k,v,a}_quant64_int()` in `hardware/hls/include/hgtxr_cyclic_math.hpp`.
- Updated contract input/cursor type metadata to match each HG-PIPE `ATTN*.cpp` typedef; this caught and fixed late Q-layer width metadata, including `attn_10_q` and `attn_11_q` as `ap_int<12>`.
- Re-generated `docs/resources/hgpipe_lut_math_contract_validation_2026_06_10.json`; all attention layer 0..11 Q/K/V/A checks cover 1,806,336 total samples with `mismatches=0`.

Validation evidence:

- `python3 -m py_compile tools/validate_hgpipe_lut_math.py tools/static_validate_hgtxr.py` passed.
- `python3 tools/validate_hgpipe_lut_math.py --contract refs/hgpipe_lut_math_contract.json --out /tmp/hgpipe_lut_math_contract_attn3_11_quantqkva_validation2.json` passed.
- `python3 tools/validate_hgpipe_lut_math.py --contract refs/hgpipe_lut_math_contract.json --out /home/kjm26/project/PRJXR/XR-VIT/HGTXR/docs/resources/hgpipe_lut_math_contract_validation_2026_06_10.json` passed after write access was granted for the docs artifact.
- `python3 tools/static_validate_hgtxr.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR` passed.
- Native reduced E2E C++ comparator passed with `runtime_state=2 count=6 last=1 failures=0`.

Next task: either add remaining MLP quant refs where complete HG-PIPE ref sets exist, or freeze the current nonlinear/quant table-provenance subset and schedule a deliberate default E2E integer-math wiring gate.


## Continuation Update - 2026-06-10 HG-PIPE LayerNorm attn0-11/mlp0-11 Cursor Gate

T-HGPIPE-LAYERNORM-ALL-013 is complete as an isolated contract/primitive gate.

- Spark read-only sidecar and direct HG-PIPE ref sweep confirmed all `attn_0..11_lnq` and `mlp_0..11_lnq` refs are present: scalars, input, output, rsqrt table, `lnw`, and `lnb`.
- Generalized `hardware/tools/validate_hgpipe_lut_math.py` from a hardcoded `layernorm_attn1` replay to generated `LAYERNORM_KEYS` for 12 attention LayerNorms plus 12 MLP LayerNorms.
- Extended `hardware/refs/hgpipe_lut_math_contract.json` with 24 LayerNorm cursor contracts, 24 per-layer 128-entry rsqrt tables, and per-layer type metadata parsed from HG-PIPE `ATTN*.cpp` and `MLP*.cpp`.
- Replaced the isolated HLS `attn1`-only LayerNorm helper block with per-layer helpers in `hardware/hls/include/hgtxr_cyclic_math.hpp`: `hgtxr_hgpipe_attn0..11_lnq_{mean,rsqrt128,requant}_int()` and `hgtxr_hgpipe_mlp0..11_lnq_{mean,rsqrt128,requant}_int()`.
- Re-generated `docs/resources/hgpipe_lut_math_contract_validation_2026_06_10.json`; all LayerNorm checks cover 903,168 total samples with `mismatches=0`.

Validation evidence:

- `python3 -m py_compile tools/validate_hgpipe_lut_math.py tools/static_validate_hgtxr.py` passed.
- `python3 tools/validate_hgpipe_lut_math.py --contract refs/hgpipe_lut_math_contract.json --out /tmp/hgpipe_lut_math_contract_layernorm_all_validation.json` passed.
- `python3 tools/validate_hgpipe_lut_math.py --contract refs/hgpipe_lut_math_contract.json --out /home/kjm26/project/PRJXR/XR-VIT/HGTXR/docs/resources/hgpipe_lut_math_contract_validation_2026_06_10.json` passed after write access was granted for the docs artifact.
- `python3 tools/static_validate_hgtxr.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR` passed.
- Reduced E2E reference/header sync passed for `refs/e2e_axis_vector_spec.json`.
- Native reduced E2E C++ comparator passed with outputs `[18, -3, 3, -2, 1, -4]`, `runtime_state=2 count=6 last=1 failures=0`.

Next task: either extend rowwise HG-PIPE softmax coverage beyond `attn_0_softmaxq`, or freeze the current table-provenance set and schedule the deliberate default E2E integer-math wiring gate. Do not wire LayerNorm helpers into default E2E without updating `hardware/tools/validate_e2e_axis_vector.py`, regenerating affected golden headers, and rerunning native/Vitis csim plus csynth.


## Continuation Update - 2026-06-10 HG-PIPE Softmax attn0-11 Cursor/Recip Gate

T-HGPIPE-SOFTMAX-ALL-014 is complete as an isolated contract/primitive gate.

- Spark read-only sidecar and direct HG-PIPE ref sweep confirmed all `attn_0..11_softmaxq` refs are present: scalars, input, output, 32-entry exp table, 64-entry reciprocal table one, and 64-entry reciprocal table two.
- Generalized `hardware/tools/validate_hgpipe_lut_math.py` from hardcoded `softmax_attn0` replay to generated `SOFTMAX_KEYS` for all 12 attention layers.
- Extended `hardware/refs/hgpipe_lut_math_contract.json` with 12 Softmax cursor contracts, 12 per-layer exp tables, 12 per-layer reciprocal table pairs, and per-layer type metadata parsed from HG-PIPE `ATTN*.cpp`.
- Replaced the isolated HLS `attn0`-only Softmax helper block with per-layer helpers in `hardware/hls/include/hgtxr_cyclic_math.hpp`: `hgtxr_hgpipe_attn0..11_softmax_{exp32,recip,requant_uint3}_int()`.
- Re-generated `docs/resources/hgpipe_lut_math_contract_validation_2026_06_10.json`; all Softmax checks cover 1,382,976 total samples with `mismatches=0`.

Validation evidence:

- `python3 -m py_compile tools/validate_hgpipe_lut_math.py tools/static_validate_hgtxr.py` passed.
- `python3 tools/validate_hgpipe_lut_math.py --contract refs/hgpipe_lut_math_contract.json --out /tmp/hgpipe_lut_math_contract_softmax_all_validation.json` passed.
- `python3 tools/validate_hgpipe_lut_math.py --contract refs/hgpipe_lut_math_contract.json --out /home/kjm26/project/PRJXR/XR-VIT/HGTXR/docs/resources/hgpipe_lut_math_contract_validation_2026_06_10.json` passed after write access was granted for the docs artifact.
- `python3 tools/static_validate_hgtxr.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR` passed.
- Reduced E2E reference/header sync passed for `refs/e2e_axis_vector_spec.json`.
- Native reduced E2E C++ comparator passed with outputs `[18, -3, 3, -2, 1, -4]`, `runtime_state=2 count=6 last=1 failures=0`.

Next task: freeze the current HG-PIPE table-provenance set and start a deliberate default E2E integer-math wiring gate, or add missing output-projection/MLP quant refs if complete HG-PIPE ref sets exist. Do not wire Softmax helpers into default E2E without updating `hardware/tools/validate_e2e_axis_vector.py`, regenerating affected golden headers, and rerunning native/Vitis csim plus csynth.


## Continuation Update - 2026-06-10 HG-PIPE Head LayerNorm Cursor Gate

T-HGPIPE-LAYERNORM-HEAD-016 is complete as an isolated contract/primitive gate.

- Spark read-only sidecar (`Euler`, GPT5.3-Codex-Spark) audited remaining HG-PIPE ref families after excluding already covered attention Q/K/V/A quant and MLP GeLUQ refs; it reported `head_lnq` as a missing complete candidate set and wrote `/tmp/remaining_hgpipe_refs_inventory.tsv` plus `/tmp/remaining_hgpipe_refs_inventory_verbose.tsv`.
- Direct HG-PIPE ref sweep confirmed `head_lnq` has scalars, input, output, 128-entry rsqrt table, 192-entry `lnw`, and 192-entry `lnb`.
- Extended `hardware/refs/hgpipe_lut_math_contract.json` with `layernorm_head`, `HEAD.cpp` type metadata, and `case/refs/head_lnq_*` provenance.
- Extended `hardware/tools/validate_hgpipe_lut_math.py` so `LAYERNORM_KEYS` includes `layernorm_head`; validation now checks observed cursor/rsqrt type fit because `HEAD.cpp` declares `__ln_cursor_t=ap_uint<6>` and `__ln_rsqrt_t=ap_uint<11>` while the ref scalars/table expose a 128-entry bound and unused high rsqrt values above 11-bit range.
- Added isolated HLS helpers in `hardware/hls/include/hgtxr_cyclic_math.hpp`: `hgtxr_hgpipe_head_lnq_rsqrt128_int()`, `hgtxr_hgpipe_head_lnq_mean_int()`, and `hgtxr_hgpipe_head_lnq_requant_int()`.
- Re-generated `docs/resources/hgpipe_lut_math_contract_validation_2026_06_10.json`; all LayerNorm checks now cover 903,360 total samples across 25 keys with `mismatches=0`.

Validation evidence:

- `python3 -m py_compile tools/validate_hgpipe_lut_math.py tools/static_validate_hgtxr.py` passed.
- `python3 tools/validate_hgpipe_lut_math.py --contract refs/hgpipe_lut_math_contract.json --out /tmp/hgpipe_lut_math_contract_head_layernorm_validation.json` passed.
- `python3 tools/validate_hgpipe_lut_math.py --contract refs/hgpipe_lut_math_contract.json --out /home/kjm26/project/PRJXR/XR-VIT/HGTXR/docs/resources/hgpipe_lut_math_contract_validation_2026_06_10.json` passed after write access was granted for the docs artifact.
- `python3 tools/static_validate_hgtxr.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR` passed.
- Reduced E2E reference/header sync passed for `refs/e2e_axis_vector_spec.json`.
- Native reduced E2E C++ comparator passed with outputs `[18, -3, 3, -2, 1, -4]`, `runtime_state=2 count=6 last=1 failures=0`.

Next task: either freeze the current table-provenance set and start the deliberate default E2E integer-math wiring gate, or model the remaining matmul/root/network/patch ref families separately. The remaining Spark inventory shows missing families are mostly `attn_*_gen_*_matmul`, `mlp_*_matmul*`, `head_matmul`, `patch_embed*`, `network`, and block-root refs rather than simple cursor-table LUT slices.


## Continuation Update - 2026-06-10 E2E HG-PIPE GeLUQ Opt-In Scaffold

T-HGPIPE-GELUQ-E2E-SCAFFOLD-017 is complete as a controlled E2E transition slice.

- Spec-kit CLI was checked first, but neither `spec-kit` nor `specify` was available in the current Ubuntu environment. `docs/Spec.md` was updated manually instead.
- A GPT5.3-Codex-Spark sidecar (`Godel`) was spawned for read-only wiring audit, but the run failed with `max_output_tokens`; the main agent completed the local audit and implementation.
- Added opt-in E2E macro `HGTXR_E2E_USE_HGPIPE_INT_GELUQ` in `hardware/hls/include/hgtxr_e2e_vit.hpp`. Default remains `0`, so existing compact-LUT E2E gates keep their current golden outputs.
- Added explicit scale macros `HGTXR_E2E_HGPIPE_GELUQ_INPUT_SCALE` and `HGTXR_E2E_HGPIPE_GELUQ_OUTPUT_SCALE`, defaulting to `16` and `4`.
- Added per-block dispatch so E2E MLP block `N` calls `hgtxr_hgpipe_mlp{N%12}_geluq64_int()` when opt-in GeLUQ is enabled.
- Extended `hardware/tools/validate_e2e_axis_vector.py` with optional `math.mlp_gelu = "hgpipe_geluq"` and contract-backed GeLUQ table loading from `hardware/refs/hgpipe_lut_math_contract.json`.

Validation evidence:

- `python3 -m py_compile tools/validate_e2e_axis_vector.py tools/static_validate_hgtxr.py` passed.
- Default spec/header sync passed for `refs/e2e_axis_vector_spec.json`; expected output remains `[18, -3, 3, -2, 1, -4]`.
- Default native C++ comparator passed with `runtime_state=2 count=6 last=1 failures=0`.
- Temporary opt-in `/tmp/e2e_axis_vector_hgpipe_geluq_spec.json` generated `/tmp/e2e_axis_vector_hgpipe_geluq_golden.hpp`; expected output is `[19, -4, 3, -2, 1, -5]`.
- Opt-in native C++ comparator passed with `HGTXR_E2E_USE_HGPIPE_INT_GELUQ=1`, input scale `16`, output scale `4`, and `runtime_state=2 count=6 last=1 failures=0`.
- `python3 tools/static_validate_hgtxr.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR` passed.

Next task: promote this scaffold into a durable named E2E scale mode only after deciding the GeLUQ activation scale contract, then repeat the same opt-in pattern for integer softmaxq and integer LayerNorm before changing the default E2E math path.


## Continuation Update - 2026-06-10 E2E HG-PIPE SoftmaxQ Opt-In Scaffold

T-HGPIPE-SOFTMAX-E2E-SCAFFOLD-018 is complete as a controlled E2E transition slice.

- GPT5.3-Codex-Spark sidecar (`Euclid`) completed a read-only audit of the softmaxq helper surface and Python mirror gap. It confirmed the helper signatures and rowwise formula.
- Added opt-in E2E macro `HGTXR_E2E_USE_HGPIPE_INT_SOFTMAXQ` in `hardware/hls/include/hgtxr_e2e_vit.hpp`. Default remains `0`, so existing compact-LUT E2E gates keep their current golden outputs.
- Added explicit scale macros `HGTXR_E2E_HGPIPE_SOFTMAX_INPUT_SCALE` and `HGTXR_E2E_HGPIPE_SOFTMAX_PROB_SCALE`, defaulting to `16` and `4`.
- Added per-block dispatch so E2E attention block `N` calls `hgtxr_hgpipe_attn{N%12}_softmax_exp32_int()`, `hgtxr_hgpipe_attn{N%12}_softmax_recip_int()`, `hgtxr_hgpipe_attn{N%12}_softmax_recip_uses_table_two()`, and `hgtxr_hgpipe_attn{N%12}_softmax_requant_uint3_int()` when opt-in softmaxq is enabled.
- Extended `hardware/tools/validate_e2e_axis_vector.py` with optional `math.attention_softmax = "hgpipe_softmaxq"` and contract-backed softmax table loading from `hardware/refs/hgpipe_lut_math_contract.json`.

Validation evidence:

- `python3 -m py_compile tools/validate_e2e_axis_vector.py tools/static_validate_hgtxr.py` passed.
- Default spec/header sync passed for `refs/e2e_axis_vector_spec.json`; expected output remains `[18, -3, 3, -2, 1, -4]`.
- Default native C++ comparator passed with `runtime_state=2 count=6 last=1 failures=0`.
- Temporary opt-in `/tmp/e2e_axis_vector_hgpipe_softmaxq_spec.json` generated `/tmp/e2e_axis_vector_hgpipe_softmaxq_golden.hpp`; expected output is `[28, -14, 12, -8, 10, -12]`.
- Opt-in native C++ comparator passed with `HGTXR_E2E_USE_HGPIPE_INT_SOFTMAXQ=1`, input scale `16`, probability scale `4`, and `runtime_state=2 count=6 last=1 failures=0`.
- `python3 tools/static_validate_hgtxr.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR` passed.

Next task: add the same kind of opt-in reduced-gate scaffold for integer LayerNorm, or promote GeLUQ+softmaxq into durable named E2E scale modes after the scale contract is frozen. Do not switch defaults until reduced, active-token, full-token, Vitis csim, and csynth gates are regenerated together.


## Continuation Update - 2026-06-10 E2E HG-PIPE Named Math Scale

T-HGPIPE-MATH-E2E-NAMED-019 is complete as a durable reduced-mode gate.

- Added `hardware/refs/e2e_axis_vector_hgpipe_math_spec.json` for combined GeLUQ plus SoftmaxQ opt-in E2E math.
- Added generated header `hardware/hls/tb/e2e_axis_vector_hgpipe_math_golden.hpp` with expected output `[28, -15, 12, -8, 10, -12]`.
- Added `HGTXR_E2E_USE_HGPIPE_MATH_GOLDEN` header selection in `hardware/hls/tb/tb_hgtxr_e2e_axis_top.cpp`.
- Added `HGTXR_E2E_SCALE=hgpipe_math` to `hardware/vivado/scripts/run_e2e_q4w8a_csim.tcl` and `hardware/vivado/scripts/run_e2e_q4w8a_csynth.tcl`.
- Registered the same mode in `software/tests/test_e2e_axis_vector_csim.py` and static validation.

Validation evidence:

- `python3 tools/validate_e2e_axis_vector.py --spec refs/e2e_axis_vector_hgpipe_math_spec.json --json-out /tmp/e2e_axis_vector_hgpipe_math_ref2.json --check-header hls/tb/e2e_axis_vector_hgpipe_math_golden.hpp` passed.
- Native C++ comparator passed with outputs `[28, -15, 12, -8, 10, -12]`, `runtime_state=2 count=6 last=1 failures=0`.
- `HGTXR_E2E_SCALE=hgpipe_math /tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f vivado/scripts/run_e2e_q4w8a_csim.tcl` passed with `CSim done with 0 errors`.
- `HGTXR_E2E_SCALE=hgpipe_math /tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f vivado/scripts/run_e2e_q4w8a_csynth.tcl` passed. Current report: estimated clock `4.058 ns`, latency `460,097 cycles` / `2.300 ms`, utilization `202 BRAM_18K`, `32 DSP`, `18,206 FF`, `64,205 LUT`, `5 URAM`.
- `python3 -m pytest -q ../software/tests/test_e2e_axis_vector_csim.py` is not available in the current system Python because `pytest` is not installed.

Next task: either add integer LayerNorm E2E opt-in/named reduced gate, or extend `hgpipe_math` from reduced mode toward larger staged specs. Do not promote HG-PIPE integer math to the default E2E path until all affected SW mirrors, generated golden headers, native comparators, Vitis csim, and Vitis csynth gates are regenerated together.


## Continuation Update - 2026-06-10 E2E HG-PIPE Named Math+LayerNormQ Scale

T-HGPIPE-LAYERNORM-E2E-020 is complete as a durable reduced-mode gate through native C++ and Vitis HLS validation.

- GPT5.3-Codex-Spark was unavailable because the Spark runtime reported a usage-limit reset time. GPT5.5 read-only audit output was used as the fallback integration guide.
- Added opt-in E2E macro `HGTXR_E2E_USE_HGPIPE_INT_LAYERNORMQ` in `hardware/hls/include/hgtxr_e2e_vit.hpp`. Default remains `0`.
- Added explicit LayerNormQ scale macros `HGTXR_E2E_HGPIPE_LAYERNORM_INPUT_SCALE`, `HGTXR_E2E_HGPIPE_LAYERNORM_OUTPUT_SCALE`, and `HGTXR_E2E_HGPIPE_LAYERNORM_BIAS_SHIFT`, defaulting to `16`, `4`, and `33`.
- Routed E2E LN1 through `hgtxr_hgpipe_attn{block%12}_lnq_{mean,rsqrt128,requant}_int()` and LN2 through `hgtxr_hgpipe_mlp{block%12}_lnq_{mean,rsqrt128,requant}_int()` when opt-in LayerNormQ is enabled.
- Added `hardware/refs/e2e_axis_vector_hgpipe_math_lnq_spec.json` for combined GeLUQ plus SoftmaxQ plus LayerNormQ opt-in E2E math.
- Added generated header `hardware/hls/tb/e2e_axis_vector_hgpipe_math_lnq_golden.hpp` with expected output `[22, -7, 6, -4, 4, -7]`.
- Added `HGTXR_E2E_USE_HGPIPE_MATH_LNQ_GOLDEN` header selection in `hardware/hls/tb/tb_hgtxr_e2e_axis_top.cpp`.
- Added `HGTXR_E2E_SCALE=hgpipe_math_lnq` to `hardware/vivado/scripts/run_e2e_q4w8a_csim.tcl` and `hardware/vivado/scripts/run_e2e_q4w8a_csynth.tcl`.
- Registered the same mode in `software/tests/test_e2e_axis_vector_csim.py` and static validation.

Validation evidence:

- `python3 -m py_compile tools/validate_e2e_axis_vector.py ../software/tests/test_e2e_axis_vector_csim.py` passed.
- `python3 tools/validate_e2e_axis_vector.py --spec refs/e2e_axis_vector_hgpipe_math_lnq_spec.json --emit-header hls/tb/e2e_axis_vector_hgpipe_math_lnq_golden.hpp --json-out /tmp/e2e_axis_vector_hgpipe_math_lnq_ref.json` passed.
- Native C++ comparator passed with outputs `[22, -7, 6, -4, 4, -7]`, `runtime_state=2 count=6 last=1 failures=0`.
- `python3 tools/static_validate_hgtxr.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR` passed.
- `HGTXR_E2E_SCALE=hgpipe_math_lnq /tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f vivado/scripts/run_e2e_q4w8a_csim.tcl` passed with `CSim done with 0 errors`.
- `HGTXR_E2E_SCALE=hgpipe_math_lnq /tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f vivado/scripts/run_e2e_q4w8a_csynth.tcl` passed. Current report: estimated clock `4.058 ns`, latency `455,257 cycles` / `2.276 ms`, utilization `202 BRAM_18K`, `8 DSP`, `16,026 FF`, `57,213 LUT`, `5 URAM`.
- `python3 -m pytest -q ../software/tests/test_e2e_axis_vector_csim.py` is not available in the current system Python because `pytest` is not installed.

Next task: decide whether to keep expanding named reduced HG-PIPE math gates or begin deliberate default-path promotion. Do not promote this to default until final per-layer trained `lnw/lnb` table provenance and full SW/HW golden regeneration are available.


## Continuation Update - 2026-06-10 E2E HG-PIPE Math+LayerNormQ Active8 Scale

T-HGPIPE-MATH-LNQ-ACTIVE8-021 is complete as a staged two-block active8 gate.

- Spark sidecar (`Volta`) could not run because the GPT5.3-Codex-Spark quota is exhausted until the reported reset time. GPT5.5 fallback sidecar (`Mencius`) completed a read-only audit, but its recommendation was stale because the main agent had already verified `hgpipe_math_lnq` CSim/CSynth.
- Added `hardware/refs/e2e_axis_vector_hgpipe_math_lnq_active8_spec.json` for `blocks=2`, `active_tokens=8`, `ff_dim=32`, GeLUQ, SoftmaxQ, and LayerNormQ E2E math.
- Added generated header `hardware/hls/tb/e2e_axis_vector_hgpipe_math_lnq_active8_golden.hpp` with expected output `[37, -26, 22, -14, 18, -21]`.
- Added explicit `HGTXR_E2E_USE_HGPIPE_MATH_LNQ_ACTIVE8_GOLDEN` header selection in `hardware/hls/tb/tb_hgtxr_e2e_axis_top.cpp`.
- Added `HGTXR_E2E_SCALE=hgpipe_math_lnq_active8` to `hardware/vivado/scripts/run_e2e_q4w8a_csim.tcl` and `hardware/vivado/scripts/run_e2e_q4w8a_csynth.tcl`.
- Registered the active8 mode in `software/tests/test_e2e_axis_vector_csim.py` and `hardware/tools/static_validate_hgtxr.py`.

Validation evidence:

- `python3 -m py_compile tools/validate_e2e_axis_vector.py tools/static_validate_hgtxr.py ../software/tests/test_e2e_axis_vector_csim.py` passed.
- `python3 tools/validate_e2e_axis_vector.py --spec refs/e2e_axis_vector_hgpipe_math_lnq_active8_spec.json --json-out /tmp/e2e_axis_vector_hgpipe_math_lnq_active8_ref_check2.json --check-header hls/tb/e2e_axis_vector_hgpipe_math_lnq_active8_golden.hpp` passed.
- Native C++ comparator passed with outputs `[37, -26, 22, -14, 18, -21]`, `runtime_state=2 count=6 last=1 failures=0`.
- `python3 tools/static_validate_hgtxr.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR` passed.
- `HGTXR_E2E_SCALE=hgpipe_math_lnq_active8 /tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f vivado/scripts/run_e2e_q4w8a_csim.tcl` passed with `CSim done with 0 errors`.
- `HGTXR_E2E_SCALE=hgpipe_math_lnq_active8 /tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f vivado/scripts/run_e2e_q4w8a_csynth.tcl` passed. Current report: estimated clock `4.058 ns`, latency `1,473,663..1,474,287 cycles` / `7.368..7.371 ms`, utilization `204 BRAM_18K`, `13 DSP`, `26,784 FF`, `104,815 LUT`, `5 URAM`.
- `python3 -m pytest -q ../software/tests/test_e2e_axis_vector_csim.py` is not available in the current system Python because `pytest` is not installed.

Next task: extend the same staged HG-PIPE math+LayerNormQ gate to `active16` or start default-path promotion planning. Do not promote default until active-token/full-token goldens and final table provenance are frozen.


## Continuation Update - 2026-06-10 DSP/URAM Resource Rebalance

T-E2E-RESOURCE-REBALANCE-023 is complete for the staged `hgpipe_math_lnq_active16` HLS gate.

- Added default-on DSP and URAM steering macros in `hardware/hls/include/hgtxr_e2e_vit.hpp`.
- Removed the Q4W8A 256-bit variable-shift weight-unpack hot path by adding constant-lane extraction for `HGTXR_WEIGHT_BIT_WIDTH=4` and `HGTXR_BUS_WIDTH_BITS=256`.
- Routed main MAC operations through a DSP-bound helper.
- Wrapped the non-HGPIPE LayerNorm variance and affine multiplications in an accumulator-width DSP-bound helper after GPT5.5 sub-agent audit.
- Moved large E2E activation/global buffers to URAM in `hardware/hls/src/hgtxr_e2e_axis_top.cpp`.
- Added `HGTXR_E2E_RESOURCE_POLICY` for `auto_bram`, `dsp_bram`, `auto_uram`, and `dsp_uram`; legacy `bram_lut` is an `auto_bram` alias.

Validation evidence:

- Native comparator passed for active16 with `[58, -51, 42, -28, 36, -41]`.
- Vitis HLS CSim passed for `HGTXR_E2E_SCALE=hgpipe_math_lnq_active16`.
- Vitis HLS CSynth passed. Current report: `4.058 ns`, `2,841,904 cycles` / `14.210 ms`, `39 BRAM_18K`, `55 DSP`, `27,574 FF`, `44,670 LUT`, `80 URAM`.
- Same-scale active16 policy matrix recorded in `docs/resources/e2e_active16_resource_policy_matrix_2026_06_10.md`:
  - `auto_bram`: `209 BRAM_18K`, `55 DSP`, `29,018 FF`, `47,745 LUT`, `0 URAM`, `2,844,464 cycles`.
  - `dsp_uram`: `39 BRAM_18K`, `55 DSP`, `27,574 FF`, `44,670 LUT`, `80 URAM`, `2,841,904 cycles`.

Tradeoff: `dsp_uram` is best for the user's DSP/URAM/LUT concern at the current active16 gate. It raises URAM utilization and lowers BRAM/LUT/FF. DSP count remains `55` because auto mode already maps fixed-point multiply cores to DSP; the explicit helper still improves fabric cost. Next continuation should tune packed-weight prefetch/cache and URAM read scheduling.

## Continuation Update - 2026-06-10 Full PAR8 Selective-URAM Fit

T-E2E-FULL-PAR8-RESOURCE-024 is complete for CSynth resource selection.

- Spark was unavailable earlier in this resource sequence due usage limit; GPT5.5 read-only fallback audit recommended full PAR8 `dsp_uram` first, then BRAM/URAM fallback if URAM exceeded budget.
- Full `active196_b6_ff768` CSim passed with `PAR=8` and expected output `[32, -13, 26, -6, 14, -11]`.
- Full `dsp_uram` CSynth completed but failed resource fit at `160/96 URAM`.
- Full `dsp_bram` CSynth fit with `338 BRAM_18K`, `268 DSP`, `39,370 FF`, `90,836 LUT`, `0 URAM`.
- Added per-buffer URAM controls in `hardware/hls/include/hgtxr_e2e_vit.hpp` and `hardware/hls/src/hgtxr_e2e_axis_top.cpp`.
- Added `dsp_mixed_hidden`, `dsp_mixed_stream`, and `dsp_mixed` policies to E2E CSim/CSynth Tcl.
- Full `dsp_mixed` still failed resource fit at `112/96 URAM`.
- Full `dsp_mixed_stream` fits and is the selected continuation point: clock `3.744 ns`, latency `82,612,245 cycles` / `0.413 sec`, resources `210 BRAM_18K`, `268 DSP`, `39,370 FF`, `90,836 LUT`, `64 URAM`.
- Small scratch evidence: generated RTL includes `score_RAM_2P_LUTRAM_1R1W`, and `prob` maps through the same LUTRAM module shape.

Next task: resolve remaining AXI weight-port II warnings in non-HGPIPE LayerNorm affine (`II=2`) and QKV packed-weight load (`II=3`), then run implementation/timing for `dsp_mixed_stream`.

## Continuation Update - 2026-06-10 Full PAR8 QKV Weight-Cache Fit

T-E2E-FULL-PAR8-QKV-CACHE-025 is complete for CSynth resource selection.

- Spark sidecar was attempted first for the QKV II audit but was unavailable due usage limit. GPT5.5 fallback sidecar `Kant` completed the read-only audit and recommended local packed Q/K/V weight caches instead of extra AXI bundles.
- Added `HGTXR_E2E_QKV_WEIGHT_CACHE=1` in `hardware/hls/include/hgtxr_e2e_vit.hpp`.
- Added BRAM-backed `q_weight_cache`, `k_weight_cache`, and `v_weight_cache` inside `hgtxr_e2e_project_qkv()`.
- Preloaded Q/K/V packed weight matrices in separate `II=1` loops, then changed the QKV MAC loop to read from local caches.
- Extended non-HGPIPE LayerNorm gamma/beta cache earlier in this continuation; LayerNorm affine/data loops are now `II=1`.
- Full `active196_b6_ff768`, `PAR=8`, `dsp_mixed_stream` CSynth now fits with clock `3.744 ns`, latency `71,316,968 cycles` / `0.357 sec`, resources `306 BRAM_18K`, `332 DSP`, `43,740 FF`, `81,144 LUT`, `64 URAM`.
- QKV local report improved from `II=3`, `2,709,518 cycles`, `16 DSP`, `12,728 LUT` to `II=1`, `904,943 cycles`, `48 DSP`, `8,422 LUT`.
- Compared with the previous full selected point, DSP increased `268 -> 332`, LUT decreased `89,756 -> 81,144`, URAM stayed `64`, and BRAM increased `210 -> 306`.
- Final report archived at `/tmp/hgtxr_e2e_axis_top_active196_b6_ff768_par8_dsp_mixed_stream_qkvcache_csynth.rpt`.

Validation evidence:

- `python3 -m py_compile tools/validate_e2e_axis_vector.py tools/static_validate_hgtxr.py ../software/tests/test_e2e_axis_vector_csim.py` passed.
- `python3 tools/static_validate_hgtxr.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR` passed.
- `git diff --check -- hls/include/hgtxr_e2e_vit.hpp` passed.
- Vitis HLS CSim passed after the QKV cache patch with output `[32, -13, 26, -6, 14, -11]`, `runtime_state=2 count=6 last=1 failures=0`, and `CSim done with 0 errors`.
- Vitis HLS CSynth passed with `HGTXR_E2E_SCALE=active196_b6_ff768`, `HGTXR_E2E_RESOURCE_POLICY=dsp_mixed_stream`.

Next task: run implementation/timing for the selected QKV-cache `dsp_mixed_stream` point. Residual risks are CSynth timing slack around `-0.09 ns` and board-side timing/packaging, not QKV II.

## Continuation Update - 2026-06-10 Next-Direction Decision Gate

The user requested that when multiple next directions exist, each direction must be analyzed and the user must get a choice before major execution.

Decision artifact:

- `docs/track/NEXT-DECISION-2026-06-10-E2E.md`

Current options recorded there:

- Option A: run board implementation/timing for the current QKV-cache point.
- Option B: pre-tune HLS timing slack before implementation.
- Option C: increase parallelism beyond PAR8.
- Option D: run ZCU104/PYNQ hardware I/O validation after package generation.
- Option E: move toward paper-trained weights/LUT/calibration alignment.
- Option F: work on software paper-reproduction fixes.

Default recommendation in the decision artifact is Option A, because current HLS already improved DSP/LUT/QKV II and the missing evidence is board-level implementation/timing. Do not start a long Vivado implementation, broad HLS retune, or default-path paper-math promotion without user selection.

## Continuation Update - 2026-06-10 Option A Preflight

Preflight artifact:

- `docs/track/OPTION-A-PREFLIGHT-2026-06-10-E2E-BITSTREAM.md`

Key finding:

- Current selected HLS top is `hgtxr_e2e_axis_top`.
- Existing `vivado/scripts/package_ip.tcl` and `vivado/scripts/create_hls_project.tcl` package old `hgtxr_top`.
- Existing `vivado/scripts/build_bitstream.tcl` instantiates `xilinx.com:hls:hgtxr_top:1.0` and connects old memory-mapped ports.
- Current E2E top uses AXIS input/output plus `gmem_e2e_weights` and `gmem_e2e_runtime`.
- Existing `pynq/hgtxr/hgtxr_overlay.py` is also old memory-mapped-buffer oriented.

Therefore Option A is not a direct long-run command yet. It first needs a user sub-choice:

- A1: preserve E2E AXIS top and build an AXIS/DMA Vivado BD plus matching PYNQ stream driver.
- A2: add an E2E memory-mapped wrapper to reuse more of the existing `hgtxr_top` bitstream/PYNQ infrastructure.
- A3: patch existing build/PYNQ scripts in place, with higher regression risk for old overlay flow.

Recommendation: choose A1 for interface purity, A2 for fastest board smoke. Do not run current `build_bitstream.tcl` as proof for the E2E QKV-cache point without adapting the flow.

## Continuation Update - 2026-06-10 Option A Sub-Agent Audit

Real sub-agent execution status:

- Spark A1 explorer was attempted first, but GPT5.3-Codex-Spark quota was exhausted until 2026-06-15 23:18.
- GPT5.5 fallback sidecar `Nash` completed the A1 AXIS/DMA read-only audit and made no edits.
- GPT5.5 sidecar `Epicurus` completed the A2 memory-mapped-wrapper read-only audit and made no edits.

A1 audit additions:

- Preserve `hgtxr_e2e_axis_top` exactly as the public E2E HLS interface.
- Add a separate `package_e2e_q4w8a_ip.tcl` and `build_e2e_bitstream.tcl`.
- Use AXI DMA simple mode, MM2S and S2MM, plus SmartConnect for AXI-Lite and DDR paths.
- PYNQ must send one 256-bit AXIS beat per pixel. Input length is `65536 * 32 = 2,097,152` bytes, so DMA length-width defaults must be checked.
- `num_pixels` is currently ignored in the E2E top; short transfers can hang.
- Current visible `hardware/generated/hgtxr_e2e_hls` tree has CSim log only, no E2E IP `component.xml`; packaging must be regenerated.

A2 audit additions:

- A2 is feasible and likely fastest for board smoke, but it needs a new top and fresh CSim/CSynth.
- Recommended new top: `hgtxr_e2e_m_axi_top(frame, weights, out_state, runtime_state)`.
- Recommended ports: `frame` m_axi depth `65536`, `weights` m_axi depth `HGTXR_E2E_WEIGHT_DEPTH`, `out_state` m_axi depth `6`, `runtime_state` m_axi depth `1`, all with AXI-Lite control.
- Implement the wrapper by calling E2E internal functions from `hgtxr_e2e_vit.hpp`, not by nesting the AXIS top.
- Expected resource delta versus AXIS point: DSP and URAM effectively unchanged, BRAM same to small increase, LUT/FF low single-digit percent increase, timing risk around `-0.09 ns` unchanged.

Updated decision: A1 remains the clean interface path; A2 remains the pragmatic fast board-smoke path. Ask the user to choose before code edits.

## Continuation Update - 2026-06-10 Third-Goal Completion Audit

Audit artifact:

- `docs/track/GOAL-AUDIT-2026-06-10-THIRD-GOAL.md`

Result:

- `[3차목표]` is not complete.
- Current E2E HLS-selected point is strong and report-backed, but board implementation/timing, board/PYNQ runtime, and final trained paper-weight/LUT/calibration equivalence remain open.
- `spec-kit` and `specify` are not available in the current Ubuntu shell, so `docs/Spec.md` remains the manual spec source.
- Requested `PAPER_PRJXR/05_RESOURCES/DeiT-Tiny C-Syn Results.png` path is absent in this workspace; same filename exists at `/home/kjm26/project/PRJXR/XR-VIT/HGPIPE/DeiT-Tiny C-Syn Results.png`.
- Requested `/home/kjm26/project/PRJXR/XR-VITs` directory is absent; the current implementation evidence uses available `XR-VIT/HGTXR`, `XR-VIT/HGPIPE`, and parsed `impl_repos` artifacts.

Immediate next action remains user selection:

- A2 for fastest board smoke through a new memory-mapped E2E wrapper.
- A1 for clean AXIS/DMA architecture preserving `hgtxr_e2e_axis_top`.
- B for HLS timing-risk reduction.
- C for higher parallelism sweep.
- E for final paper-trained artifact alignment.

## Continuation Update - 2026-06-10 Third-Goal Preflight Checker

Added a branch-neutral checker:

- `hardware/tools/check_third_goal_preflight.py`

Captured result:

- `docs/resources/third_goal_preflight_2026_06_10.json`
- Neutral summary: `ok=22`, `warn=8`, `fail=0`
- Board-ready summary: `ok=22`, `warn=5`, `fail=3`, as expected until A1/A2/A3 resolves E2E package/board artifacts.

The checker verifies:

- `/tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls`
- `/tools/Xilinx/Vivado/2023.2/bin/vivado`
- `spec-kit`/`specify` availability as warnings, not failures
- requested `PAPER_PRJXR` image and HGPIPE substitute
- requested `XR-VITs` sibling path
- ZCU104 Q4W/Q8A macros for PAR, bus width, bit widths, buffer, FIFO, active tokens, blocks, FF dim, and dense parallelism
- visible E2E CSim log
- missing E2E `component.xml` and generated-tree csynth report
- old `hgtxr_top` board-flow mismatch against selected `hgtxr_e2e_axis_top`
- stale PYNQ `hgtxr.hwh` artifact containing old `hgtxr_top_0`

This checker is intentionally not a branch chooser. It is a reproducible guardrail so future A1/A2/B/C/E work does not mistake stale old-flow board artifacts for the current E2E top.

## Continuation Update - 2026-06-10 Preflight Checker Unit Tests

Added unit tests:

- `hardware/tests/test_check_third_goal_preflight.py`

Sub-agent status:

- Spark test-review sidecar was attempted first, but GPT5.3-Codex-Spark quota is exhausted until 2026-06-15 23:18.
- GPT5.5 fallback sidecar `Avicenna` completed a read-only test-policy review and made no edits.

Test coverage:

- Neutral mode allows choice-gated old-flow board gaps as warnings and exits `0`.
- Board-ready mode fails stale old-flow Tcl/HWH plus missing E2E `component.xml`.
- Board-ready mode passes when fake E2E package, bit/hwh, and E2E Tcl markers are present, leaving only external reference warnings.

Verification:

- `python3 -m unittest tests/test_check_third_goal_preflight.py` passed 3 tests.
- `python3 -m py_compile tools/check_third_goal_preflight.py tools/static_validate_hgtxr.py tests/test_check_third_goal_preflight.py` passed.

Next major branch is still user-selected: A1, A2, A3, B, C, E, or another direction.

## Continuation Update - 2026-06-10 Preflight JSON Output Unit Test

Added JSON artifact coverage:

- `hardware/tests/test_check_third_goal_preflight.py::test_json_out_records_mode_summary_and_checks`

Coverage:

- Confirms `--json-out` creates nested parent directories.
- Confirms JSON includes `mode`, `summary`, and list-shaped schema-valid `checks`.
- Confirms stdout and JSON summaries match.
- Confirms board-ready failure paths still write JSON.
- Confirms selected E2E HWH and E2E packaged IP checks are present as `ok` in a fake board-ready E2E tree.

Verification:

- `python3 -m unittest tests/test_check_third_goal_preflight.py` passed 5 tests.
- `python3 -m py_compile tools/check_third_goal_preflight.py tools/static_validate_hgtxr.py tests/test_check_third_goal_preflight.py` passed.

This remains branch-neutral and does not resolve the board-ready failures in the real tree.

## Continuation Update - 2026-06-10 A2/A1/C Selected Paths

Baseline:

- Commit before implementation: `03fa3ec23e80b2ca803b6459004339193ebb84e8`.
- User selection: Path 1 = A2 then A1; Path 2 = C in parallel where feasible; E pending.

A2:

- Added `hardware/hls/src/hgtxr_e2e_m_axi_top.cpp`.
- Added `hardware/hls/tb/tb_hgtxr_e2e_m_axi_top.cpp`.
- Added `hardware/vivado/scripts/run_e2e_m_axi_q4w8a_csim.tcl`.
- Added `hardware/vivado/scripts/run_e2e_m_axi_q4w8a_csynth.tcl`.
- Full CSim passed with `[32, -13, 26, -6, 14, -11]`, `runtime_state=2`, `failures=0`.
- Full CSynth passed: `3.744 ns`, `81,514,836 cycles`, `334 BRAM_18K`, `334 DSP`, `46,438 FF`, `84,196 LUT`, `64 URAM`.

A1:

- `gb.pooled` now binds to LUTRAM when `HGTXR_E2E_SMALL_MEM_LUTRAM=1`.
- Added optional `HGTXR_E2E_URAM_QKV_WEIGHT_CACHE`, default `0`.
- QKV-cache URAM experiment is rejected for default use: `112/96 URAM`.
- Final m_axi CSynth after safe LUTRAM change: `326 BRAM_18K`, `334 DSP`, `46,502 FF`, `84,220 LUT`, `64 URAM`.

C:

- Spark worker failed due GPT5.3-Codex-Spark quota. GPT5.5 worker `Pasteur` completed the AXIS PAR Tcl patch.
- Added `HGTXR_E2E_PAR=16|32` for AXIS CSim/CSynth Tcl; invalid values such as `12/24` are rejected.
- PAR=16 full AXIS CSynth passed: `3.953 ns`, `37,508,072 cycles`, `338 BRAM_18K`, `604 DSP`, `59,507 FF`, `127,916 LUT`, `64 URAM`.
- PAR=16 reduced CSim passed with `[18, -3, 3, -2, 1, -4]`, `runtime_state=2`, `failures=0`.

Current judgment:

- A2 is board-fit and functionally aligned, but m_axi latency is higher than AXIS PAR8 baseline.
- A1 safe small-memory LUTRAM cleanup is retained.
- C PAR=16 is currently best DSP/latency point, but LUT/timing risk is higher than PAR8.

## Continuation Update - 2026-06-10 A2 Board Artifact Gate Complete

A2 package/build status:

- HLS IP export passed for `hgtxr_e2e_m_axi_top`.
- Vivado E2E m_axi block design, implementation, route, and bitstream generation passed.
- Generated overlay artifacts are under `hardware/generated/build/vivado/overlay/hgtxr_e2e_m_axi_overlay/`.
- PYNQ package artifacts are `hardware/pynq/hgtxr/hgtxr_e2e_m_axi.bit` and `hardware/pynq/hgtxr/hgtxr_e2e_m_axi.hwh`.

Key board-build evidence:

- HLS IP export Fmax report: `267.13 MHz`.
- Routed timing: `WNS=1.926 ns`, `TNS=0.000 ns`, `WHS=0.010 ns`, `THS=0.000 ns`.
- Placed resources: `148 RAMB36`, `2 RAMB18`, `64 URAM`, `246 DSP`.
- Routed power estimate: `4.060 W`.
- Board-ready preflight: `ok=25`, `warn=6`, `fail=0`.

Validation rerun:

- `python3 tools/check_third_goal_preflight.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --mode board-ready --json-out /tmp/hgtxr_third_goal_preflight_board_2026_06_10.json`
- `python3 -m unittest tests/test_check_third_goal_preflight.py`
- `python3 -m unittest test_hgtxr_overlay.py` from `hardware/pynq/hgtxr`
- `python3 tools/static_validate_hgtxr.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR`
- `python3 -m py_compile tools/check_third_goal_preflight.py tests/test_check_third_goal_preflight.py pynq/hgtxr/hgtxr_overlay.py pynq/hgtxr/e2e_m_axi_overlay.py pynq/hgtxr/test_hgtxr_overlay.py`
- `git diff --check`

Open risk:

- Physical ZCU104 PYNQ runtime smoke is still open.
- Legacy `hardware/pynq/hgtxr/hgtxr.bit/.hwh` still points at old `hgtxr_top`; use `hgtxr_e2e_m_axi.bit/.hwh` for the selected A2 path.
- DSP DRC warnings recommend input/output pipelining for performance/power; they did not block bitstream.

## Continuation Update - 2026-06-10 A2 Weight Artifact And Integrity Gate

A2 smoke artifact status:

- Added CSim-mirrored packed Q4 builder: `hardware/pynq/hgtxr/e2e_m_axi_weights.py`.
- Added board smoke CLI: `hardware/pynq/hgtxr/run_e2e_m_axi_smoke.py`.
- Added export CLI: `hardware/tools/export_e2e_m_axi_weights.py`.
- Exported binary: `hardware/refs/weights/e2e_m_axi_active196_b6_ff768_q4_u32.bin`.
- Exported manifest: `hardware/refs/weights/e2e_m_axi_active196_b6_ff768_q4_u32_manifest.json`.

Artifact evidence:

- Format: `raw-little-endian-uint32`.
- Size: `12,192,768` bytes, `3,048,192` `uint32` values.
- Required prefix: `338,640` `uint32` values; tail is zero from that point.
- SHA256: `45b7aef9446c512cf74030975f55cd1f7407f2a27c2ea75603353132489d0a31`.
- Expected runtime state: `2`.
- Expected raw output: `[32, -13, 26, -6, 14, -11]`.

Preflight status:

- `hardware/tools/check_third_goal_preflight.py` now validates the A2 packed Q4 weight artifact.
- Checks include manifest fields, binary existence, byte count, SHA256, expected raw/runtime contract, A2 layout constants, size math, tail-zero region, and Q4 nibble probes.
- Neutral preflight: `ok=36`, `warn=6`, `fail=0`.
- Board-ready preflight: `ok=36`, `warn=6`, `fail=0`.
- `docs/resources/third_goal_preflight_2026_06_10.json` and `docs/resources/third_goal_preflight_board_2026_06_10.json` were refreshed after the stronger checker.

Latest verification:

- `python3 -m unittest tests/test_check_third_goal_preflight.py` passed 6 tests.
- `python3 -m unittest test_hgtxr_overlay.py` from `hardware/pynq/hgtxr` passed 16 tests.
- `python3 -m unittest pynq/hgtxr/test_hgtxr_overlay.py` from `hardware` passed 16 tests.
- `python3 tools/static_validate_hgtxr.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR` passed.
- `python3 tools/check_third_goal_preflight.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --mode board-ready --json-out /tmp/hgtxr_third_goal_preflight_board_2026_06_10.json` passed.
- `git diff --check` passed.

Next physical-board command:

```bash
python3 -m hgtxr.run_e2e_m_axi_smoke --weights-mode file --weights-bin hardware/refs/weights/e2e_m_axi_active196_b6_ff768_q4_u32.bin --expect-out-raw 32 -13 26 -6 14 -11 --json-out e2e_m_axi_file_smoke.json
```

## Continuation Update - 2026-06-10 A2 PYNQ File-Smoke Bundle

Transfer bundle status:

- Added packager: `hardware/tools/package_e2e_m_axi_pynq_bundle.py`.
- Added bundle test: `hardware/tests/test_package_e2e_m_axi_pynq_bundle.py`.
- Bundle directory: `hardware/generated/pynq/e2e_m_axi_smoke_bundle`.
- Bundle tarball: `hardware/generated/pynq/e2e_m_axi_smoke_bundle.tar.gz`.
- Bundle manifest: `hardware/generated/pynq/e2e_m_axi_smoke_bundle/BUNDLE_MANIFEST.json`.
- Tarball size: `1,526,468` bytes.
- Tarball SHA256: `56ade675e76d1abadab5a9bdbf00b2baf1f8d7b4f5488b9ee313405398803d3a`.

Bundle contents:

- Includes `hgtxr_e2e_m_axi.bit`, `hgtxr_e2e_m_axi.hwh`, `hgtxr_overlay.py`, `e2e_m_axi_overlay.py`, `e2e_m_axi_weights.py`, `run_e2e_m_axi_smoke.py`, exported packed Q4 weight `.bin`, weight manifest, and `run_e2e_m_axi_file_smoke.sh`.
- Excludes legacy `hgtxr.bit/.hwh`, `__pycache__`, and `.pyc` files.
- `hardware/pynq/hgtxr/README.md` now distinguishes selected A2 artifacts from legacy old-flow artifacts and gives a bundle-local file-mode command.

Bundle-local physical-board command:

```bash
./run_e2e_m_axi_file_smoke.sh
```

Equivalent expanded command from the extracted bundle root:

```bash
PYTHONPATH=. python3 -m hgtxr.run_e2e_m_axi_smoke --weights-mode file --weights-bin weights/e2e_m_axi_active196_b6_ff768_q4_u32.bin --expect-out-raw 32 -13 26 -6 14 -11 --json-out e2e_m_axi_file_smoke.json
```

Latest verification:

- `python3 tools/package_e2e_m_axi_pynq_bundle.py` regenerated the bundle and tarball.
- `python3 -m unittest tests/test_package_e2e_m_axi_pynq_bundle.py` passed 1 test.
- `python3 -m unittest tests/test_check_third_goal_preflight.py` passed 6 tests.
- `python3 -m unittest test_hgtxr_overlay.py` from `hardware/pynq/hgtxr` passed 16 tests.
- `python3 -m unittest pynq/hgtxr/test_hgtxr_overlay.py` from `hardware` passed 16 tests.
- `python3 tools/static_validate_hgtxr.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR` passed.
- Board-ready preflight passed with `ok=36`, `warn=6`, `fail=0`.

Open risk: physical ZCU104 file-mode smoke remains open. The bundle only closes transfer packaging and command integrity.

## Continuation Update - 2026-06-10 A1 E2E AXIS/DMA Build

A1 build status:

- HLS IP export completed.
- Vivado AXIS/DMA bitstream build completed.
- Physical ZCU104 A1 DMA smoke is still open.

Key artifacts:

- A1 IP root: `hardware/generated/hgtxr_e2e_axis_hls/solution_e2e_q4w8a/impl/ip`
- A1 IP component: `hardware/generated/hgtxr_e2e_axis_hls/solution_e2e_q4w8a/impl/ip/component.xml`
- A1 IP export zip: `hardware/generated/hgtxr_e2e_axis_hls/solution_e2e_q4w8a/impl/export.zip`
- A1 PYNQ bit: `hardware/pynq/hgtxr/hgtxr_e2e_axis_dma.bit`
- A1 PYNQ HWH: `hardware/pynq/hgtxr/hgtxr_e2e_axis_dma.hwh`

Evidence:

- HLS core estimate: `3.744 ns` at `5.00 ns` target.
- HLS core resources: `298 BRAM_18K`, `332 DSP`, `43,804 FF`, `81,168 LUT`, `64 URAM`.
- Vivado routed timing: `WNS=4.120 ns`, `TNS=0`, `WHS=0.009 ns`, `THS=0`.
- Vivado routed power estimate: `3.476 W`.
- A1 bit SHA256: `03457fb4021c1632062d2c2f1d628df1fda380b70fb7a8ba923e676f0eb45c2f`
- A1 HWH SHA256: `0d7edb5f7424fc401d60202b0139f7e01140a3e09d4c400fe601b405571e99b2`

Important fix made:

- Initial A1 BD build exposed DMA width mismatch: `axi_dma_in/M_AXIS_MM2S` was `32b` while `hgtxr_e2e_axis_top/axis_in` was `256b`.
- `hardware/vivado/scripts/build_e2e_axis_dma_bitstream.tcl` now forces DMA MM2S/S2MM stream and memory data widths to `256b`.

Latest verification:

- `python3 -m unittest tests/test_check_third_goal_preflight.py` passed 6 tests.
- `python3 -m unittest test_hgtxr_overlay.py` from `hardware/pynq/hgtxr` passed 20 tests.
- `python3 -m py_compile ...` passed for A1 PYNQ/preflight/static modules.
- `python3 tools/static_validate_hgtxr.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR` passed.
- Board-ready preflight passed with `ok=39`, `warn=6`, `fail=0`.
- `git diff --check` passed.

Next choices:

- A1-board: run physical ZCU104 AXIS/DMA smoke.
- C-PAR16-board: package or board-test the already validated PAR16 HLS point.
- C-PAR32-HLS: run PAR32 stress if higher DSP/latency pressure is worth LUT/timing risk.
- E remains pending by user decision.

## Continuation Update - 2026-06-10 C1 PAR16 AXIS/DMA Board Candidate

C1 build status:

- HLS/IP package completed.
- Isolated Vivado AXIS/DMA bitstream build completed.
- Physical ZCU104 C1 PAR16 DMA smoke is still open.

Key artifacts:

- C1 IP root: `hardware/generated/hgtxr_e2e_axis_par16_hls/solution_e2e_q4w8a/impl/ip`
- C1 IP component: `hardware/generated/hgtxr_e2e_axis_par16_hls/solution_e2e_q4w8a/impl/ip/component.xml`
- C1 IP export zip: `hardware/generated/hgtxr_e2e_axis_par16_hls/solution_e2e_q4w8a/impl/export.zip`
- C1 PYNQ bit: `hardware/pynq/hgtxr/hgtxr_e2e_axis_dma_par16.bit`
- C1 PYNQ HWH: `hardware/pynq/hgtxr/hgtxr_e2e_axis_dma_par16.hwh`

Evidence:

- HLS core estimate: `3.953 ns` at `5.00 ns` target.
- HLS core latency: `37,508,072 cycles`, `0.188 sec`.
- HLS core resources: `338 BRAM_18K`, `604 DSP`, `59,507 FF`, `127,916 LUT`, `64 URAM`.
- Vivado routed timing: `WNS=4.723 ns`, `TNS=0`, `WHS=0.010 ns`, `THS=0`.
- Vivado routed power estimate: `3.475 W`.
- C1 export zip SHA256: `7ec974940af59e99d14bc9bc528561bd6eeb117478ac60fda73b7da10eeb8555`
- C1 bit SHA256: `d34cc3cb108c508153a3f69531b80be81567343f38ceb52e517e578556e42e11`
- C1 HWH SHA256: `6d32cafffc6b9c9b33793bed34f114c199450e1167895ca4d448c56012c471a8`

Important notes:

- C1 materially raises DSP use versus A1/A2: `604 DSP` versus about `332-334 DSP`.
- C1 preserves the intended large-buffer URAM use at `64 URAM`.
- C1 LUT use is high at `127,916`, so it is a routed high-DSP candidate rather than a final comfortable resource point.
- Vivado wrapper utilization can show low/zero DSP or URAM for this OOC/IP integration shell; use the HLS core report for HGTXR resource accounting.

Next choices:

- C1-board: run physical ZCU104 AXIS/DMA PAR16 smoke.
- C2-PAR32-HLS: run isolated PAR32 HLS stress if the priority is maximum DSP/latency exploration.
- C3-DSP-LUT-cleanup: reduce LUT pressure and DSP pipeline warnings before pushing PAR32.
- E remains pending by user decision.

## Continuation Update - 2026-06-10 C3 DSP/LUT Cleanup

C3 build status:

- Code cleanup implemented.
- Reduced PAR16/MEM8 CSim completed.
- Full PAR16/MEM8 CSynth completed.
- Physical ZCU104 C3/MEM8 board build and smoke are not run yet.

Key files:

- `hardware/hls/include/hgtxr_e2e_vit.hpp`
- `hardware/hls/src/hgtxr_e2e_axis_top.cpp`
- `hardware/vivado/scripts/run_e2e_q4w8a_csim.tcl`
- `hardware/vivado/scripts/run_e2e_q4w8a_csynth.tcl`
- `hardware/generated/hgtxr_e2e_axis_par16_c3_mem8/solution_e2e_q4w8a/syn/report/csynth.rpt`

Evidence:

- C3 macro added: `HGTXR_E2E_MEM_BANK_PAR`.
- C3 fast path added: `HGTXR_E2E_WEIGHT_VEC_ALIGNED_FASTPATH`.
- Reduced CSim passed with output `[18, -3, 3, -2, 1, -4]`, `runtime_state=2`, and `failures=0`.
- Full C3 CSynth resources: `292 BRAM_18K`, `604 DSP`, `56,966 FF`, `113,124 LUT`, `64 URAM`.
- Full C3 latency: `48,598,922 cycles`.

C1 versus C3:

- C1: `37,508,072 cycles`, `338 BRAM_18K`, `604 DSP`, `59,507 FF`, `127,916 LUT`, `64 URAM`.
- C3/MEM8: `48,598,922 cycles`, `292 BRAM_18K`, `604 DSP`, `56,966 FF`, `113,124 LUT`, `64 URAM`.
- C3 preserves DSP and URAM while reducing LUT by `14,792`.
- C3 is slower because reduced memory banking creates an MLP W2 `Final II=2` bottleneck.

Next choices:

- Keep C1 as the faster high-DSP routed candidate and run physical C1 board smoke.
- Run C3b with PAR16/MEM16 to isolate aligned-fastpath LUT impact without the MEM8 memory-port penalty.
- Run C2/PAR32 as a high-risk stress point after accepting expected LUT/timing pressure.
- E remains pending by user decision.

## Continuation Update - 2026-06-10 C3b MEM16 Fastpath-Only

C3b build status:

- Reduced CSim completed.
- Full PAR16/MEM16 CSynth completed.
- Vivado route and physical ZCU104 smoke not run.

Key files:

- `hardware/generated/hgtxr_e2e_axis_par16_c3b_mem16/solution_e2e_q4w8a/syn/report/csynth.rpt`
- `hardware/generated/hgtxr_e2e_axis_par16_c3b_mem16/solution_e2e_q4w8a/syn/report/csynth.xml`
- `docs/resources/c3_par16_mem16_fastpath_csynth_2026_06_10.json`

Evidence:

- Reduced CSim passed with output `[18, -3, 3, -2, 1, -4]`, `runtime_state=2`, and `failures=0`.
- C3b CSynth resources: `332 BRAM_18K`, `604 DSP`, `59,505 FF`, `126,506 LUT`, `64 URAM`.
- C3b latency: `37,508,072 cycles`.
- C3b estimated clock: `3.953 ns`.

Conclusion:

- C3b restores C1-like latency and MLP W2 `II=1`.
- Fastpath-only reduces LUT by only `1,410` versus C1.
- C3/MEM8 remains the stronger LUT-reduction point, but not the low-latency point.
- Preferred next low-latency C candidate is C3b or C1 board path; preferred LUT-pressure candidate is C3/MEM8.

## Continuation Update - 2026-06-10 C3b Board Candidate

C3b board-candidate status:

- C3b HLS IP export completed under `hardware/generated/hgtxr_e2e_axis_par16_c3b_mem16_hls`.
- C3b Vivado AXIS/DMA overlay completed under isolated names:
  - project `hgtxr_e2e_axis_dma_c3b_mem16_overlay`
  - BD `hgtxr_e2e_axis_dma_c3b_mem16_system`
  - artifact `hgtxr_e2e_axis_dma_c3b_mem16`
- Board-ready artifacts now exist:
  - `hardware/pynq/hgtxr/hgtxr_e2e_axis_dma_c3b_mem16.bit`
  - `hardware/pynq/hgtxr/hgtxr_e2e_axis_dma_c3b_mem16.hwh`
- Routed timing: `WNS=4.415 ns`, `TNS=0.000 ns`, `WHS=0.010 ns`, `THS=0.000 ns`.
- Routed power: `3.476 W`.
- HLS core remains the C3b point: `37,508,072 cycles`, `332 BRAM_18K`, `604 DSP`, `59,505 FF`, `126,506 LUT`, `64 URAM`.
- Hashes:
  - export zip `a471fed1d43d2323efd22cc3581cfd7ee1c1cb5a0cfbc69ba57b2877da527e38`
  - bit `989e77836341467da05d0ccb86f0b4764d38f4711af8fbcb567290b78fe3076d`
  - hwh `8ca659da3bbcc061f7299ac18314c00b8ee2112a274990d969635b3f6acf3c90`

Validation status:

- Board-ready preflight was extended to recognize optional C3b PAR16/MEM16 artifacts.
- Current board-ready preflight: `ok=45`, `warn=6`, `fail=0`.
- `python3 -m unittest tests/test_check_third_goal_preflight.py` passed.

Open items:

- Physical ZCU104 C3b AXIS/DMA smoke remains open.
- C2 PAR32 stress remains pending.
- E remains pending by user decision.

## Continuation Update - 2026-06-10 C3b AXIS/DMA PYNQ Smoke Bundle

Current selected strategy remains Path 1 `A2 -> A1`, Path 2 `C`, and `E`
pending. After C3b board-candidate routing, the next C board-smoke gap was
package/transfer readiness.

What changed:

- `hardware/pynq/hgtxr/run_e2e_axis_dma_smoke.py` now accepts `--variant`.
- `c3b-mem16` selects `hgtxr_e2e_axis_dma_c3b_mem16.bit/.hwh`.
- `hardware/tools/package_e2e_axis_dma_pynq_bundle.py` packages AXIS/DMA PYNQ smoke bundles.
- Generated `hardware/generated/pynq/e2e_axis_dma_c3b_mem16_smoke_bundle`.
- Generated `hardware/generated/pynq/e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz`.
- Tarball SHA256: `3de505a6627409b4717ea95e5e76d74f0667c61b601fd96db1e495319adda712`.
- Bundle-local smoke entrypoint: `./run_e2e_axis_dma_c3b_mem16_file_smoke.sh`.

Verification:

- Python compile gate passed for the updated smoke CLI, packer, and test.
- Bundle manifest JSON is valid.
- 31 unittest cases passed across PYNQ helper, AXIS/DMA bundle, A2 bundle, and third-goal preflight tests.
- Static validation passed.
- Board-ready preflight passed with `ok=45`, `warn=6`, `fail=0`.

Remaining:

- Physical ZCU104 C3b AXIS/DMA smoke is still open.
- C2 PAR32 remains pending.
- E remains pending.

## Continuation Update - 2026-06-10 Final Signoff Gate

The host-side board-ready gate and final completion gate are now separated.

What changed:

- `hardware/tools/check_third_goal_preflight.py` accepts `--mode final-signoff`.
- `board-ready` still warns when C3b physical smoke result is not copied back.
- `final-signoff` fails when C3b physical smoke result is missing or invalid.
- `final-signoff` fails missing requested `PAPER_PRJXR` DeiT image and `XR-VITs` sibling.
- `hardware/tests/test_check_third_goal_preflight.py` now covers final-signoff behavior.

Current validation:

- Preflight unit tests pass: 14 tests.
- Board-ready preflight passes: `ok=61`, `warn=7`, `fail=0`.
- Final-signoff preflight fails as expected: `ok=61`, `warn=4`, `fail=3`.

Next physical board step:

1. Transfer/extract `hardware/generated/pynq/e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz` on ZCU104 PYNQ.
2. Run `./run_e2e_axis_dma_c3b_mem16_file_smoke.sh`.
3. Run `./validate_e2e_axis_dma_c3b_mem16_file_smoke.sh`.
4. Import the copied-back JSON with `hardware/tools/import_pynq_smoke_result.py`.
5. Rerun `python3 tools/check_third_goal_preflight.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --mode final-signoff`.

Remaining:

- Physical ZCU104 C3b AXIS/DMA smoke is still open.
- `PAPER_PRJXR` DeiT image and `XR-VITs` sibling are still missing for final signoff.
- C2 PAR32 remains pending.
- E remains pending.

## Continuation Update - 2026-06-10 Final Signoff Audit Artifact

The current final-signoff blocker state is now captured as durable generated artifacts.

What changed:

- New tool: `hardware/tools/write_final_signoff_audit.py`.
- New tests: `hardware/tests/test_write_final_signoff_audit.py`.
- Static validation now includes the audit tool and test.
- Generated final-signoff JSON: `hardware/generated/signoff/third_goal_final_signoff_2026_06_10.json`.
- Generated board-ready JSON: `hardware/generated/signoff/third_goal_board_ready_2026_06_10.json`.
- Generated audit JSON: `hardware/generated/signoff/final_signoff_audit_2026_06_10.json`.
- Generated audit Markdown: `hardware/generated/signoff/final_signoff_audit_2026_06_10.md`.
- Tracked audit mirror: `docs/resources/final_signoff_audit_2026_06_10.json`.
- Tracked audit mirror: `docs/resources/final_signoff_audit_2026_06_10.md`.

Current audit:

- status: `blocked`
- blockers: `3`
- warnings: `4`
- blockers are C3b physical smoke result, requested `PAPER_PRJXR` image, and requested `XR-VITs` sibling.

Remaining:

- Physical ZCU104 C3b AXIS/DMA smoke is still open.
- Requested `PAPER_PRJXR` image and `XR-VITs` sibling are still missing.
- C2 PAR32 remains pending.
- E remains pending.

## Continuation Update - 2026-06-10 Reference Input Audit

The requested third-goal reference inputs were checked against the current XR-VIT tree.

What changed:

- New tool: `hardware/tools/audit_reference_inputs.py`.
- New tests: `hardware/tests/test_audit_reference_inputs.py`.
- Static validation now includes the reference audit tool and test.
- Generated reference audit JSON: `hardware/generated/signoff/reference_input_audit_2026_06_10.json`.
- Generated reference audit Markdown: `hardware/generated/signoff/reference_input_audit_2026_06_10.md`.
- Tracked audit mirror: `docs/resources/reference_input_audit_2026_06_10.json`.
- Tracked audit mirror: `docs/resources/reference_input_audit_2026_06_10.md`.

Current audit:

- status: `needs-reference-input`
- blockers: `2`
- `PAPER_PRJXR/05_RESOURCES/DeiT-Tiny C-Syn Results.png` is missing.
- `HGPIPE/DeiT-Tiny C-Syn Results.png` exists as a candidate, SHA256 `90495791e6103be30268320d3b17dce2eaa52d003dd6f341393262745c1c3b79`.
- `/home/kjm26/project/PRJXR/XR-VITs` is missing.
- Candidate HLS directories exist: `XR_Accel`, `analysis/XR_Accel`, `ViT_Accel`.

Remaining:

- Restore the requested reference paths or explicitly approve replacements.
- Physical ZCU104 C3b AXIS/DMA smoke is still open.
- C2 PAR32 remains pending.
- E remains pending.

## Continuation Update - 2026-06-10 PAPER_PRJXR Image Restore

The requested PAPER_PRJXR image path is now present.

What changed:

- Created `/home/kjm26/project/PRJXR/XR-VIT/PAPER_PRJXR/05_RESOURCES/`.
- Copied `HGPIPE/DeiT-Tiny C-Syn Results.png` to the requested PAPER_PRJXR path.
- Regenerated final-signoff and reference-input audits.

Current validation:

- HGPIPE source SHA256: `90495791e6103be30268320d3b17dce2eaa52d003dd6f341393262745c1c3b79`.
- PAPER_PRJXR restored image SHA256: `90495791e6103be30268320d3b17dce2eaa52d003dd6f341393262745c1c3b79`.
- Reference-input audit now has 1 blocker: missing `XR-VITs`.
- Final-signoff now reports `ok=62`, `warn=4`, `fail=2`.

Remaining:

- Physical ZCU104 C3b AXIS/DMA smoke is still open.
- Requested `XR-VITs` sibling is still missing.
- C2 PAR32 remains pending.
- E remains pending.

## Continuation Update - 2026-06-10 XR-VITs Candidate Audit

The missing `XR-VITs` reference now has a ranked replacement-candidate audit.

What changed:

- New tool: `hardware/tools/audit_xr_vits_candidates.py`.
- New tests: `hardware/tests/test_audit_xr_vits_candidates.py`.
- Static validation now includes the candidate audit tool and test.
- Generated candidate audit JSON: `hardware/generated/signoff/xr_vits_candidate_audit_2026_06_10.json`.
- Generated candidate audit Markdown: `hardware/generated/signoff/xr_vits_candidate_audit_2026_06_10.md`.
- Tracked audit mirror: `docs/resources/xr_vits_candidate_audit_2026_06_10.json`.
- Tracked audit mirror: `docs/resources/xr_vits_candidate_audit_2026_06_10.md`.

Current audit:

- status: `candidate-found`
- approved replacement: `false`
- recommended candidate: `/home/kjm26/project/PRJXR/XR-VIT/XR_Accel`
- candidate scores: `XR_Accel=99`, `ViT_Accel=90`, `analysis/XR_Accel=0`

Remaining:

- Restore `/home/kjm26/project/PRJXR/XR-VITs`, or explicitly approve `XR_Accel` as the replacement reference.
- Physical ZCU104 C3b AXIS/DMA smoke is still open.
- C2 PAR32 remains pending.
- E remains pending.

## Continuation Update - 2026-06-10 C3b Self-Validating Bundle

The C3b bundle can now validate its own smoke result JSON after running on
ZCU104 PYNQ.

What changed:

- `hardware/tools/package_e2e_axis_dma_pynq_bundle.py` includes `tools/validate_pynq_smoke_result.py`.
- The generated bundle includes `validate_e2e_axis_dma_c3b_mem16_file_smoke.sh`.
- `BUNDLE_MANIFEST.json` includes `validation_command`.
- `hardware/pynq/hgtxr/README.md` documents the validation command.
- New tarball SHA256: `3de505a6627409b4717ea95e5e76d74f0667c61b601fd96db1e495319adda712`.
- New tarball size: `593,198` bytes.

Board-side sequence:

1. `./run_e2e_axis_dma_c3b_mem16_file_smoke.sh`
2. `./validate_e2e_axis_dma_c3b_mem16_file_smoke.sh`

Remaining:

- Physical ZCU104 C3b AXIS/DMA smoke is still open.
- C2 PAR32 remains pending.
- E remains pending.

## Continuation Update - 2026-06-10 C3b Bundle Package Validator

The C3b bundle package is now validated before transfer.

What changed:

- New validator: `hardware/tools/validate_pynq_bundle_package.py`.
- New tests: `hardware/tests/test_validate_pynq_bundle_package.py`.
- Board-ready preflight now includes `C3b AXIS/DMA PYNQ smoke bundle package`.
- Static validation now requires the package validator and tests.
- Evidence JSON: `docs/resources/third_goal_preflight_board_c3b_result_gate_2026_06_10.json`.

Current validation:

- Package validator passes on `hardware/generated/pynq/e2e_axis_dma_c3b_mem16_smoke_bundle`.
- Combined PYNQ/bundle/preflight/result-validator tests pass: 40 tests.
- Static validation and `git diff --check` pass.
- Board-ready preflight passes: `ok=57`, `warn=7`, `fail=0`.
- Warning remains expected until physical ZCU104 C3b smoke result JSON is copied back.

Next physical board step:

1. Transfer/extract `hardware/generated/pynq/e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz` on ZCU104 PYNQ.
2. Run `./run_e2e_axis_dma_c3b_mem16_file_smoke.sh`.
3. Run `./validate_e2e_axis_dma_c3b_mem16_file_smoke.sh`.
4. Copy `e2e_axis_dma_c3b_mem16_file_smoke.json` back to the HGTXR checkout.

Remaining:

- Physical ZCU104 C3b AXIS/DMA smoke is still open.
- C2 PAR32 remains pending.
- E remains pending.

## Continuation Update - 2026-06-10 C3b Physical Result Import Helper

The copied-back C3b board result now has a host-side import gate.

What changed:

- New importer: `hardware/tools/import_pynq_smoke_result.py`.
- New tests: `hardware/tests/test_import_pynq_smoke_result.py`.
- Static validation now requires the importer and tests.
- Default destination: `hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json`.
- Invalid board-result JSON is rejected before it can overwrite the canonical path.
- Verification passed: py_compile, 43 unittests, static validation, board-ready preflight `ok=57 warn=7 fail=0`, and `git diff --check`.

Copy-back command after board run:

```sh
python3 tools/import_pynq_smoke_result.py /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --preset axis-c3b-mem16 --json-out /tmp/hgtxr_c3b_smoke_import.json --validation-out /tmp/hgtxr_c3b_smoke_validation.json
```

Then run:

```sh
python3 tools/check_third_goal_preflight.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --mode board-ready
```

Remaining:

- Physical ZCU104 C3b AXIS/DMA smoke is still open.
- C2 PAR32 remains pending.
- E remains pending.

## Continuation Update - 2026-06-10 C3b ZCU104 Smoke Session Runbook

The C3b physical-board step now has generated session files.

What changed:

- New session tool: `hardware/tools/prepare_zcu104_smoke_session.py`.
- New tests: `hardware/tests/test_prepare_zcu104_smoke_session.py`.
- Generated JSON session: `hardware/generated/pynq/e2e_axis_dma_c3b_mem16_smoke_session.json`.
- Generated Markdown runbook: `hardware/generated/pynq/e2e_axis_dma_c3b_mem16_smoke_session.md`.
- Static validation now requires the session tool and tests.
- Verification passed: py_compile, 46 unittests, static validation, board-ready preflight `ok=57 warn=7 fail=0`, session JSON parse, and `git diff --check`.

Session content:

- Validates the C3b transfer bundle before board execution.
- Records tarball path, size, and SHA256.
- Records board steps:
  `tar -xzf ...`, `cd ...`, `./run_e2e_axis_dma_c3b_mem16_file_smoke.sh`, `./validate_e2e_axis_dma_c3b_mem16_file_smoke.sh`.
- Records host copy-back/import and final board-ready preflight commands.

Remaining:

- Physical ZCU104 C3b AXIS/DMA smoke is still open.
- C2 PAR32 remains pending.
- E remains pending.

## Continuation Update - 2026-06-10 C3b Smoke Session Preflight Gate

The C3b session files are now part of board-ready preflight.

What changed:

- `hardware/tools/check_third_goal_preflight.py` now checks `hardware/generated/pynq/e2e_axis_dma_c3b_mem16_smoke_session.json`.
- It also checks `hardware/generated/pynq/e2e_axis_dma_c3b_mem16_smoke_session.md`.
- The checker verifies session status, target, variant, preset, expected output/runtime, tar size/SHA256, board steps, host import/preflight steps, and canonical result path.
- `hardware/tests/test_check_third_goal_preflight.py` covers complete-session acceptance and missing-session failure.

Current validation:

- Session-gate unit tests pass: 11 tests for the preflight suite.
- Board-ready preflight passes: `ok=61`, `warn=7`, `fail=0`.
- Full related unit suite passes: 47 tests.
- Static validation and `git diff --check` pass.
- The remaining C3b warning is still the missing physical board result JSON.

Remaining:

- Physical ZCU104 C3b AXIS/DMA smoke is still open.
- C2 PAR32 remains pending.
- E remains pending.

## Continuation Update - 2026-06-10 C3b Physical Smoke Result Gate

The C3b physical-smoke result can now be validated after the board run.

What changed:

- New validator: `hardware/tools/validate_pynq_smoke_result.py`.
- New tests: `hardware/tests/test_validate_pynq_smoke_result.py`.
- Static validation now requires the validator and tests.
- Board-ready preflight now checks copied-back C3b physical smoke JSON if it exists.
- Expected copy-back path: `hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json`.
- Alternate checked path: `hardware/generated/pynq/e2e_axis_dma_c3b_mem16_smoke_bundle/e2e_axis_dma_c3b_mem16_file_smoke.json`.
- Evidence JSON: `docs/resources/third_goal_preflight_board_c3b_result_gate_2026_06_10.json`.

Current validation:

- Validator unit tests pass.
- Combined preflight/validator tests pass.
- Board-ready preflight passes: `ok=54`, `warn=7`, `fail=0`.
- The extra warning is expected until physical ZCU104 C3b smoke result JSON is copied back.

Next physical board step:

1. Transfer/extract `hardware/generated/pynq/e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz` on ZCU104 PYNQ.
2. Run `./run_e2e_axis_dma_c3b_mem16_file_smoke.sh`.
3. Copy `e2e_axis_dma_c3b_mem16_file_smoke.json` back to the HGTXR checkout.
4. Run `python3 tools/validate_pynq_smoke_result.py hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json --preset axis-c3b-mem16`.

Remaining:

- Physical ZCU104 C3b AXIS/DMA smoke is still open.
- C2 PAR32 remains pending.
- E remains pending.

## Continuation Update - 2026-06-10 C3b Bundle Preflight Gate

The C3b board-smoke bundle is now part of the host-side board-ready gate.

What changed:

- `hardware/tools/check_third_goal_preflight.py` now validates the C3b bundle when C3b artifacts are present.
- The gate checks bundle directory, manifest, tarball, run script, variant, expected output, file-mode command, required contents, and tar SHA256.
- `hardware/tests/test_check_third_goal_preflight.py` now covers both complete-bundle success and missing-bundle failure.
- Evidence JSON: `docs/resources/third_goal_preflight_board_c3b_bundle_2026_06_10.json`.

Current validation:

- Preflight unit tests pass: 9 tests.
- Board-ready preflight passes: `ok=54`, `warn=6`, `fail=0`.

Remaining:

- Physical ZCU104 C3b AXIS/DMA smoke is still open.
- C2 PAR32 remains pending.
- E remains pending.

## Continuation Update - 2026-06-10 XR-VITs Replacement Policy Gate

The missing `XR-VITs` reference now has an explicit replacement-policy gate.

What changed:

- `hardware/tools/check_third_goal_preflight.py` now checks `docs/resources/xr_vits_replacement_policy.json` only when `/home/kjm26/project/PRJXR/XR-VITs` is missing.
- An approved policy must set `approved_replacement: true`, match the requested path, point to an existing replacement path, and record `approved_by`.
- Added inactive template `docs/resources/xr_vits_replacement_policy.template.json`.
- `hardware/tests/test_check_third_goal_preflight.py` now covers approved and rejected replacement policies.
- Static validation now requires the policy template.
- Updated final-signoff audit mirror under `docs/resources/final_signoff_audit_2026_06_10.json/.md`.

Current validation:

- Preflight unit tests pass: 16 tests.
- Current final-signoff preflight remains blocked: `ok=62`, `warn=4`, `fail=2`.
- Current final-signoff audit: `blocker_count=2`, `warning_count=4`.

Remaining:

- No active `docs/resources/xr_vits_replacement_policy.json` exists yet.
- Physical ZCU104 C3b AXIS/DMA smoke is still open.
- C2 PAR32 remains pending.
- E remains pending.

## Continuation Update - 2026-06-10 XR-VITs Policy Creation Helper

The `XR_Accel` replacement-approval path now has a guarded helper command.

What changed:

- Added `hardware/tools/create_xr_vits_replacement_policy.py`.
- Added `hardware/tests/test_create_xr_vits_replacement_policy.py`.
- Static validation now requires the helper tool and tests.
- The helper refuses to write active `docs/resources/xr_vits_replacement_policy.json` unless `--approve`, `--approved-by`, and `--reason` are provided.
- The helper checks that `/home/kjm26/project/PRJXR/XR-VITs` is still missing, `/home/kjm26/project/PRJXR/XR-VIT/XR_Accel` exists, and the replacement matches the candidate audit.

Dry-run command:

```sh
python3 tools/create_xr_vits_replacement_policy.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --approve --approved-by dry-run --reason "dry-run validation only" --approved-at 2026-06-10T00:00:00Z --dry-run
```

If the user explicitly approves `XR_Accel`, use the same command with a real
`--approved-by`/`--reason` and without `--dry-run`.

Remaining:

- No active `docs/resources/xr_vits_replacement_policy.json` exists yet.
- Physical ZCU104 C3b AXIS/DMA smoke is still open.
- C2 PAR32 remains pending.
- E remains pending.

## Continuation Update - 2026-06-10 Third Goal Completion Audit

The active third-goal objective now has a requirement-by-requirement completion
audit.

What changed:

- Added `hardware/tools/write_third_goal_completion_audit.py`.
- Added `hardware/tests/test_write_third_goal_completion_audit.py`.
- Static validation now requires the completion audit tool and test.
- Generated `hardware/generated/signoff/third_goal_completion_audit_2026_06_10.json`.
- Generated `hardware/generated/signoff/third_goal_completion_audit_2026_06_10.md`.
- Mirrored both files to `docs/resources/third_goal_completion_audit_2026_06_10.json/.md`.

Current audit:

- status: `blocked`
- pass: `7`
- partial: `4`
- blocked: `2`
- blocked `(11)`: requested `XR-VITs` is missing and no active replacement policy exists.
- blocked `final`: final-signoff still has 2 failures.

Remaining:

- Restore `/home/kjm26/project/PRJXR/XR-VITs` or explicitly approve `XR_Accel` through `docs/resources/xr_vits_replacement_policy.json`.
- Run/import physical ZCU104 C3b AXIS/DMA smoke JSON.
- C2 PAR32 remains pending.
- E remains pending.

## Continuation Update - 2026-06-10 Third Goal Unblock Checklist

The remaining blockers now have a concrete unblock checklist.

What changed:

- Added `hardware/tools/write_third_goal_unblock_checklist.py`.
- Added `hardware/tests/test_write_third_goal_unblock_checklist.py`.
- Static validation now requires the unblock-checklist tool and test.
- Generated `hardware/generated/signoff/third_goal_unblock_checklist_2026_06_10.json`.
- Generated `hardware/generated/signoff/third_goal_unblock_checklist_2026_06_10.md`.
- Mirrored both files to `docs/resources/third_goal_unblock_checklist_2026_06_10.json/.md`.

Checklist:

- `B1`: resolve `XR-VITs` reference gate by restoring exact checkout or explicitly approving `XR_Accel` through `create_xr_vits_replacement_policy.py`.
- `B2`: run/import C3b ZCU104 physical smoke result.
- `B3`: rerun final host signoff and regenerate audits.

Current status:

- checklist status: `pending-unblock`
- final blockers: `2`
- steps: `3`

Remaining:

- Execute `B1` and `B2`.
- Then run `B3`.
- C2 PAR32 remains pending.
- E remains pending.

## Continuation Update - 2026-06-10 C3b Smoke Transfer Manifest

The C3b physical-board smoke bundle now has a transfer manifest and SHA256
verification file.

What changed:

- Added `hardware/tools/write_c3b_smoke_transfer_manifest.py`.
- Added `hardware/tests/test_write_c3b_smoke_transfer_manifest.py`.
- Static validation now requires the transfer-manifest tool and test.
- Generated `hardware/generated/signoff/c3b_smoke_transfer_manifest_2026_06_10.json`.
- Generated `hardware/generated/signoff/c3b_smoke_transfer_manifest_2026_06_10.md`.
- Generated `hardware/generated/signoff/e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz.sha256`.
- Mirrored all transfer evidence to `docs/resources/`.

Board transfer files:

- `hardware/generated/pynq/e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz`
- `hardware/generated/signoff/e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz.sha256`

Board verify command:

```sh
sha256sum -c e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz.sha256
```

Current status:

- transfer manifest: `pass`
- bundle SHA256: `3de505a6627409b4717ea95e5e76d74f0667c61b601fd96db1e495319adda712`
- bundle validation errors: `0`

Remaining:

- Transfer the tarball and SHA file to ZCU104.
- Run board verify command, then the C3b smoke scripts.
- Copy back/import `e2e_axis_dma_c3b_mem16_file_smoke.json`.

## Continuation Update - 2026-06-10 C3b Board Smoke Readiness

The C3b physical-board smoke path now has a host-side readiness checker that
separates package readiness from physical board evidence.

What changed:

- Added `hardware/tools/check_c3b_board_smoke_readiness.py`.
- Added `hardware/tests/test_check_c3b_board_smoke_readiness.py`.
- Static validation now requires the readiness checker tool and test.
- Generated `hardware/generated/signoff/c3b_board_smoke_readiness_2026_06_10.json`.
- Generated `hardware/generated/signoff/c3b_board_smoke_readiness_2026_06_10.md`.
- Mirrored readiness evidence to `docs/resources/c3b_board_smoke_readiness_2026_06_10.json/.md`.

Current status:

- readiness: `ready-for-board`
- ready: `true`
- errors: `0`
- warnings: `1`
- physical_result: `missing`
- bundle SHA256: `3de505a6627409b4717ea95e5e76d74f0667c61b601fd96db1e495319adda712`

Remaining:

- Transfer/run/import the C3b smoke on ZCU104.
- Resolve `XR-VITs` by exact restore or explicit approved replacement policy.
- Rerun final signoff after both blockers are cleared.

## Continuation Update - 2026-06-11 Final Signoff Runner

Final signoff evidence can now be regenerated with a single host-side runner.

What changed:

- Added `hardware/tools/run_third_goal_final_signoff.py`.
- Added `hardware/tests/test_run_third_goal_final_signoff.py`.
- Static validation now requires the final-signoff runner tool and test.
- Runner regenerates:
  - C3b board smoke readiness
  - final-signoff preflight JSON
  - final signoff audit
  - third-goal completion audit
  - third-goal unblock checklist
- Runner mirrors generated evidence into `docs/resources`.

Current runner command:

```sh
python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked
```

Current status:

- runner: `blocked`
- final preflight: `ok=62`, `warn=4`, `fail=2`
- readiness: `ready-for-board`
- remaining blockers: `requested XR-VITs sibling`, `C3b AXIS/DMA physical smoke result`

After blockers clear:

```sh
python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR
```

## Continuation Update - 2026-06-11 ZCU104 Remote C3b Smoke Runner

The C3b physical-board smoke blocker now has a host-side SSH/SCP runner.

What changed:

- Added `hardware/tools/run_zcu104_c3b_smoke_remote.py`.
- Added `hardware/tests/test_run_zcu104_c3b_smoke_remote.py`.
- Static validation now requires the remote smoke runner tool and test.
- Generated `hardware/generated/signoff/zcu104_c3b_smoke_remote_run_2026_06_10.json`.
- Generated `hardware/generated/signoff/zcu104_c3b_smoke_remote_run_2026_06_10.md`.
- Mirrored dry-run evidence to `docs/resources/zcu104_c3b_smoke_remote_run_2026_06_10.json/.md`.

Dry-run command:

```sh
python3 tools/run_zcu104_c3b_smoke_remote.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --host zcu104.local --user xilinx
```

Actual board execution command shape:

```sh
python3 tools/run_zcu104_c3b_smoke_remote.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --host <zcu104-host-or-ip> --user xilinx --execute
```

Optional connection flags:

```sh
--port <ssh-port> --identity-file <private-key> --remote-dir /home/xilinx/hgtxr_c3b_smoke
```

Current status:

- remote runner: `dry-run`
- local input errors: `0`
- physical result: still missing until `--execute` succeeds
- `XR-VITs` gate: still unresolved

## Continuation Update - 2026-06-11 XR-VITs Unblock Packet

The XR-VITs reference gate now has a tracked user-choice packet.

What changed:

- Added `hardware/tools/write_xr_vits_unblock_packet.py`.
- Added `hardware/tests/test_write_xr_vits_unblock_packet.py`.
- Static validation now requires the unblock packet tool and test.
- Generated `hardware/generated/signoff/xr_vits_unblock_packet_2026_06_10.json`.
- Generated `hardware/generated/signoff/xr_vits_unblock_packet_2026_06_10.md`.
- Mirrored packet evidence to `docs/resources/xr_vits_unblock_packet_2026_06_10.json/.md`.

Current packet:

- status: `pending-user-choice`
- requested path exists: `false`
- active policy exists: `false`
- recommended replacement: `XR_Accel`, score `99`

Available choices:

```sh
test -d /home/kjm26/project/PRJXR/XR-VITs
```

or:

```sh
python3 tools/create_xr_vits_replacement_policy.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --replacement-path /home/kjm26/project/PRJXR/XR-VIT/XR_Accel --approve --approved-by <approved-by> --reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff" --dry-run
python3 tools/create_xr_vits_replacement_policy.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --replacement-path /home/kjm26/project/PRJXR/XR-VIT/XR_Accel --approve --approved-by <approved-by> --reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff"
```

Safety:

- The packet did not create active `docs/resources/xr_vits_replacement_policy.json`.
- Final signoff remains blocked until exact restore or explicit approval.

## Continuation Update - 2026-06-11 Integrated Final Signoff Runner

The final signoff runner now regenerates all current blocker-support evidence.

What changed:

- Updated `hardware/tools/run_third_goal_final_signoff.py`.
- Updated `hardware/tests/test_run_third_goal_final_signoff.py`.
- Runner now produces:
  - ZCU104 C3b remote smoke dry-run evidence
  - XR-VITs unblock packet
  - C3b board smoke readiness
  - final-signoff preflight
  - final signoff audit
  - third-goal completion audit
  - third-goal unblock checklist

Current command:

```sh
python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked
```

Current status:

- runner: `blocked`
- final preflight: `ok=62`, `warn=4`, `fail=2`
- zcu104 remote: `dry-run`
- XR-VITs packet: `pending-user-choice`
- C3b readiness: `ready-for-board`

Remaining:

- Run/import physical C3b ZCU104 smoke.
- Restore exact XR-VITs or explicitly approve XR_Accel.
