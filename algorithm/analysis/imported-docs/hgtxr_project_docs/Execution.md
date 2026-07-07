# Execution

## Software Smoke

```sh
sh software/scripts/train_stage1.sh --smoke
python software/tools/eval.py
python software/tools/infer.py
```

## Hardware References

```sh
sh hardware/scripts/export_hw_refs.sh
sh hardware/scripts/run_hw_sw_compare.sh
```

## HLS

```sh
sh hardware/scripts/run_hls_csim.sh
sh hardware/scripts/run_hls_csynth.sh
```

## 2026-06-06 Continuation Execution

- Read handoff, active ZCU104 spec, cyclic architecture, experiment matrix, validation status, sweep config, HLS parameter header, cyclic Transformer block header, and `hgtxr_top.cpp`.
- Spawned two GPT-5.3-Codex-Spark sidecars for document/status analysis and `impl_repos`/image/resource scouting. The main implementation did not depend on their delayed outputs.
- Ran `python3 hardware/tools/repair_hls_to_pynq.py --apply` to patch Vivado Tcl and copy generated bit/hwh artifacts.
- Patched `hgtxr_cyclic_transformer_params.hpp` so sweep-generated macros control tiling, PE/head parallelism, bus width, bit width, buffer depth, and FIFO depth.
- Generated a 2048-run sweep manifest into `/tmp` as a smoke test.
- Ran `hardware/tools/extract_resource_metrics.py` over `../impl_repos`, producing `docs/resources/deit_tiny_csyn_resource_timing_metrics.csv`.
- Verified static artifacts, PYNQ helper smoke, and cyclic primitive smoke.

## 2026-06-06 Cyclic Top Integration Execution

- Added compile-time top gate `HGTXR_ENABLE_CYCLIC_TRANSFORMER_TOP` in `hardware/hls/src/hgtxr_top.cpp`.
- Added a tiled cyclic Transformer stage that loads active token/channel tiles, calls `transformer_block_tile`, and writes back valid tile outputs.
- Kept the public PYNQ ABI unchanged.
- Added S0 HLS define file and durable cyclic csynth Tcl.
- Updated HLS project Tcl to route generated define files into Vitis HLS builds.
- Ran default and cyclic-enabled g++ top smokes.
- Ran default HLS csynth and cyclic S0 HLS csynth.
- Fixed the cyclic MLP dataflow synthesis failure by removing invalid `DATAFLOW` from the current array-based implementation.

## 2026-06-06 Active-Token Specialization Execution

- Converted the cyclic top stage into a template over active token count.
- Instantiated separate search and track cyclic stages so HLS no longer schedules the 16-token track path and 64-token search path as if each covered the full 256-token compile-time buffer.
- Reran cyclic S0 Vitis HLS csynth with `/tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f hardware/vivado/scripts/run_cyclic_s0_csynth.tcl`.
- Updated `docs/resources/hls_csynth_summary_2026_06_06.csv` from the new report.
- Current cyclic S0 result: `267.02 MHz`, `15,023,573 cycles`, `75.118 ms`, `263 BRAM_18K`, `42 DSP`, `23397 FF`, `38752 LUT`, `0 URAM`.

## 2026-06-06 Packed Weight-Port S1 Execution

- Added `HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS` as an explicit ABI gate.
- Added packed AXI-word weight layout constants and loader helpers for norm, QKV, output projection, and MLP tile weights.
- Added `gmem_cyclic_weights` only when the weight-port gate is enabled.
- Added S1 define and csynth Tcl files.
- Ran default, cyclic, and cyclic packed-weight g++ top smokes.
- Ran `solution_cyclic_s1_weight` Vitis HLS csynth; it passed with `224.11 MHz`, `15,061,541 cycles`, and `44% BRAM` / `23% LUT` HLS-estimated ZCU104 usage.

## 2026-06-06 Host-Side Cyclic Weight Packing Execution

- Added `hardware/tools/pack_cyclic_weights.py` for fixed-point quantization and little-endian AXI word packing.
- Added `.npz`/`.json` input support and optional `.pt/.pth` support when torch is installed.
- Added manifest output that records offsets, fallback tensors, word count, and the current diagonal tile-local limitation.
- Added `tests/test_pack_cyclic_weights.py`.
- Ran packer self-test, syntax compile, and strict six-block synthetic pack validation.
- Verified the referenced DeiT-Tiny C-Syn image exists as a `955 x 429` RGBA PNG; OCR extraction remains pending.

## 2026-06-06 S2 Channel-Pair Packing Execution

- Added `s2_channel_pairs` layout mode to `hardware/tools/pack_cyclic_weights.py`.
- Generated strict synthetic S2 packed weights from `/tmp/hgtxr_synth_weights_full.npz`.
- Verified manifest shape: `216` blocks, block stride `784` words, total `169,344` words, `0` fallbacks.
- Ran packer test functions directly because `pytest` is unavailable.


## 2026-06-06 S2 HLS Projection Primitive Execution

- Added S2 HLS layout helpers for host-matching channel-pair block indexing, word-base calculation, fixed-point lane unpacking, packed element loading, and matrix tile loading.
- Added S2 projection accumulation helpers, including the thin wrapper signature that accepts `cyclic_weights`, `layer_idx`, `output_tile`, `input_tile`, `channel_tiles`, and `matrix_elem_offset`.
- Added a g++ smoke test that validates the wrapper over two packed input-tile blocks for one output tile.
- Recompiled and ran the S2 smoke test successfully with Vitis HLS 2023.2 headers.
- Reran static validation and packer self/direct tests.


## 2026-06-06 S2 WO First-Step Top Integration Execution

- Added `HGTXR_ENABLE_CYCLIC_S2_FIRST_STEP` and guarded it so the mode requires the packed weight AXI port.
- Wired the S2 channel-pair helper into `hgtxr_top.cpp` as a gated `WO` projection first-step stage.
- Added S2 first-step define and csynth Tcl artifacts.
- Ran default, S1, and S2 g++ top smokes.
- Ran standalone S2 projection smoke and packer tests.
- Ran `/tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f hardware/vivado/scripts/run_cyclic_s2_first_step_csynth.tcl`; csynth passed and generated `hardware/generated/hgtxr_hls/solution_cyclic_s2_first_step/syn/report/hgtxr_top_csynth.rpt`.


## 2026-06-06 S2 QKV First-Step Top Integration Execution

- Added an explicit `HGTXR_ENABLE_CYCLIC_S2_QKV_FIRST_STEP` mode separate from the earlier S2 `WO` first-step mode.
- Added S2 Q/K/V packed projection accumulation helper and top stage.
- Added QKV first-step define and csynth Tcl artifacts.
- Ran standalone S2 QKV projection smoke, S2 QKV top smoke, existing S2 WO top smoke, static validation, and packer tests.
- Ran `/tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f hardware/vivado/scripts/run_cyclic_s2_qkv_first_step_csynth.tcl`; csynth passed and generated `hardware/generated/hgtxr_hls/solution_cyclic_s2_qkv_first_step/syn/report/hgtxr_top_csynth.rpt`.


## 2026-06-06 S2 Attention First-Step Top Integration Execution

- Added an explicit `HGTXR_ENABLE_CYCLIC_S2_ATTN_FIRST_STEP` mode that is mutually exclusive with the S2 `WO` and S2 QKV first-step gates.
- Added an S2 first-step stage that accumulates Q/K/V channel-pair projections, feeds the resulting tiles through `attention_tile`, and applies S2 `WO` channel-pair accumulation to the attention tile.
- Added attention first-step define and csynth Tcl artifacts.
- Ran S2 attention top smoke, existing S2 QKV and S2 WO preservation smokes, static validation, and packer tests.
- Ran `/tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f hardware/vivado/scripts/run_cyclic_s2_attn_first_step_csynth.tcl`; csynth passed and generated `hardware/generated/hgtxr_hls/solution_cyclic_s2_attn_first_step/syn/report/hgtxr_top_csynth.rpt`.


## 2026-06-06 S2 MLP Hidden First-Step Top Integration Execution

- Added `HGTXR_ENABLE_CYCLIC_S2_MLP_FIRST_STEP` as a separate S2 top mode.
- Added explicit MLP hidden-dimension parameters and S2 W1/W2 hidden-pair indexing helpers.
- Added an S2 MLP first-step stage that performs W1 accumulation across input channel tiles, GELU activation per hidden tile, and W2 accumulation across hidden tiles.
- Added `s2_mlp_hidden_pairs` host packing mode, packer tests, MLP define header, and csynth Tcl artifacts.
- Ran S2 helper smoke, S2 MLP top smoke, S2 attention/QKV preservation smokes, static validation, packer tests, and `/tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f hardware/vivado/scripts/run_cyclic_s2_mlp_first_step_csynth.tcl`.
- `solution_cyclic_s2_mlp_first_step` csynth passed and generated `hardware/generated/hgtxr_hls/solution_cyclic_s2_mlp_first_step/syn/report/hgtxr_top_csynth.rpt`.


## 2026-06-06 S2 Residual-Aware Block First-Step Top Integration Execution

- Added `HGTXR_ENABLE_CYCLIC_S2_BLOCK_FIRST_STEP` as a separate fused S2 top mode.
- Added a per-layer fused packed layout that places attention channel-pair blocks before MLP hidden-pair blocks.
- Added residual copy/add helpers and a fused block stage that runs S2 attention, residual add, S2 MLP, residual add.
- Added `s2_block_first_step` host packing mode, direct packer test coverage, block define header, and csynth Tcl artifacts.
- Ran S2 block top smoke, S2 attention/MLP preservation compiles, static validation, packer tests, and `/tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f hardware/vivado/scripts/run_cyclic_s2_block_first_step_csynth.tcl`.
- `solution_cyclic_s2_block_first_step` csynth passed and generated `hardware/generated/hgtxr_hls/solution_cyclic_s2_block_first_step/syn/report/hgtxr_top_csynth.rpt`.

## 2026-06-06 S2 Pre-LayerNorm Block First-Step Execution

- Added the `HGTXR_ENABLE_CYCLIC_S2_BLOCK_PRE_LN` compile-time switch and enabled it for the ZCU104 S2 block first-step config.
- Implemented full-embedding token mean/variance pre-normalization for the fused S2 block path, with norm1 before attention and norm2 before MLP.
- Loaded norm affine parameters from the packed S2 block attention-phase records, preserving the existing fused weight layout.
- Initial Newton-style inverse-square-root synthesized but reduced estimated Fmax to `67.72 MHz`; replaced it with a bounded piecewise approximation to restore timing.
- Re-ran C++ compile, smoke, static validation, and Vitis HLS csynth.

## 2026-06-06 S2 Pre-LayerNorm Golden-Vector Execution

- Implemented `hardware/tools/validate_s2_block_preln.py` as a small host-side golden-vector validator for the S2 pre-LN contract.
- The tool builds synthetic DeiT-like packed S2 block weights, verifies the HLS norm parameter addressing rule, and quantifies exact LayerNorm versus the HLS piecewise `rsqrt` approximation.
- Initial validator run exposed high low-variance error (`~1.01`), so the low-variance piecewise bins were refined. The final validator artifact reports maximum pre-LN error below `0.004`.
- Re-ran C++ compile, C++ smoke, static validation, direct pytest-equivalent assertions, and Vitis HLS csynth.

## 2026-06-06 S2 Full-Block Golden-Vector Execution

- Implemented `hardware/tools/validate_s2_block_full.py` to validate full S2 block packed layout reconstruction and fused approximation output.
- Reused the pre-LN helper semantics and mirrored HLS approximate attention (`hgtxr_score_exp_approx`) plus hard-sigmoid GELU.
- Generated `docs/resources/s2_block_full_validation_2026_06_06.json`.
- Ran Python syntax validation, direct pytest-equivalent assertions, JSON sanity checks, and static validation.


## 2026-06-09 Software Paper-Reproduction Execution

- Copied the v3 reference package from  to .
- Copied v3 scripts to  and v3 configs to  to avoid clobbering existing HGTXR wrappers.
- Patched v3 script root detection so  resolves  as its project root and  as the package root.
- Added paper config  with submitted-version ZCU104 Option A settings and paper table targets.
- Added optional reduced-depth execution to , , , and  through  and .
- Added runtime selected-state and reliability outputs in the v3 runtime policy.
- Added reproduction manifest generation in  and run-contract artifact writing for paper configs.
- Added focused tests in .

## 2026-06-10 E2E HG-PIPE GeLUQ Opt-In Scaffold Execution

- Confirmed `spec-kit` and `specify` CLIs are unavailable in the current Ubuntu environment, so the active spec update was applied manually to `docs/Spec.md`.
- Added `HGTXR_E2E_USE_HGPIPE_INT_GELUQ` to `hardware/hls/include/hgtxr_e2e_vit.hpp`; default is `0`, preserving existing compact LUT E2E gates.
- Added explicit GeLUQ scale knobs: `HGTXR_E2E_HGPIPE_GELUQ_INPUT_SCALE` and `HGTXR_E2E_HGPIPE_GELUQ_OUTPUT_SCALE`.
- Added per-block MLP GeLUQ dispatch so block `N` uses the validated HG-PIPE `mlp{N%12}_geluq` helper.
- Extended `hardware/tools/validate_e2e_axis_vector.py` with optional `math.mlp_gelu = "hgpipe_geluq"` and contract-backed GeLUQ table loading from `hardware/refs/hgpipe_lut_math_contract.json`.
- Verified default reduced E2E still emits `[18, -3, 3, -2, 1, -4]`.
- Verified opt-in GeLUQ reduced E2E with a temporary `/tmp` spec/header emits `[19, -4, 3, -2, 1, -5]` and passes the native C++ comparator.

## 2026-06-10 E2E HG-PIPE SoftmaxQ Opt-In Scaffold Execution

- Spawned GPT5.3-Codex-Spark read-only sidecar `Euclid` for the softmaxq wiring audit. It confirmed the existing HLS helper semantics and identified the Python mirror gap.
- Added `HGTXR_E2E_USE_HGPIPE_INT_SOFTMAXQ` to `hardware/hls/include/hgtxr_e2e_vit.hpp`; default is `0`, preserving existing compact LUT E2E gates.
- Added explicit softmaxq scale knobs: `HGTXR_E2E_HGPIPE_SOFTMAX_INPUT_SCALE` and `HGTXR_E2E_HGPIPE_SOFTMAX_PROB_SCALE`.
- Added per-block softmaxq dispatch so E2E attention block `N` uses the validated HG-PIPE `attn{N%12}_softmaxq` exp, reciprocal, and uint3 requant helpers.
- Extended `hardware/tools/validate_e2e_axis_vector.py` with optional `math.attention_softmax = "hgpipe_softmaxq"` and contract-backed softmax table loading from `hardware/refs/hgpipe_lut_math_contract.json`.
- Verified default reduced E2E still emits `[18, -3, 3, -2, 1, -4]`.
- Verified opt-in softmaxq reduced E2E with a temporary `/tmp` spec/header emits `[28, -14, 12, -8, 10, -12]` and passes the native C++ comparator.

## 2026-06-10 E2E HG-PIPE Named Math Scale Execution

- Added durable `hardware/refs/e2e_axis_vector_hgpipe_math_spec.json` for the combined reduced GeLUQ plus SoftmaxQ E2E gate.
- Generated `hardware/hls/tb/e2e_axis_vector_hgpipe_math_golden.hpp`; expected raw output is `[28, -15, 12, -8, 10, -12]`.
- Updated `hardware/hls/tb/tb_hgtxr_e2e_axis_top.cpp` so `HGTXR_E2E_USE_HGPIPE_MATH_GOLDEN` selects the named generated header.
- Added `HGTXR_E2E_SCALE=hgpipe_math` to both `hardware/vivado/scripts/run_e2e_q4w8a_csim.tcl` and `hardware/vivado/scripts/run_e2e_q4w8a_csynth.tcl`.
- Registered `hgpipe_math` in `software/tests/test_e2e_axis_vector_csim.py` with GeLUQ and SoftmaxQ opt-in defines.
- Added the named spec/header to `hardware/tools/static_validate_hgtxr.py`.
- Verified reference/header sync, native C++ comparator, Vitis HLS csim, and Vitis HLS csynth for the named reduced mode.

## 2026-06-10 E2E HG-PIPE Named Math+LayerNormQ Scale Execution

- Added `hardware/refs/e2e_axis_vector_hgpipe_math_lnq_spec.json` for the combined reduced GeLUQ plus SoftmaxQ plus LayerNormQ E2E gate.
- Generated `hardware/hls/tb/e2e_axis_vector_hgpipe_math_lnq_golden.hpp`; expected raw output is `[22, -7, 6, -4, 4, -7]`.
- Updated `hardware/hls/tb/tb_hgtxr_e2e_axis_top.cpp` so `HGTXR_E2E_USE_HGPIPE_MATH_LNQ_GOLDEN` selects the named generated header and initializes LN gamma for this gate.
- Added `HGTXR_E2E_SCALE=hgpipe_math_lnq` to both `hardware/vivado/scripts/run_e2e_q4w8a_csim.tcl` and `hardware/vivado/scripts/run_e2e_q4w8a_csynth.tcl`.
- Registered `hgpipe_math_lnq` in `software/tests/test_e2e_axis_vector_csim.py` with GeLUQ, SoftmaxQ, and LayerNormQ opt-in defines.
- Verified reference/header sync, native C++ comparator, Vitis HLS csim, and Vitis HLS csynth for the named reduced mode.

## 2026-06-10 E2E HG-PIPE Math+LayerNormQ Active8 Scale Execution

- Added `hardware/refs/e2e_axis_vector_hgpipe_math_lnq_active8_spec.json` for the two-block, active8 GeLUQ plus SoftmaxQ plus LayerNormQ E2E gate.
- Generated `hardware/hls/tb/e2e_axis_vector_hgpipe_math_lnq_active8_golden.hpp`; expected raw output is `[37, -26, 22, -14, 18, -21]`.
- Added explicit `HGTXR_E2E_USE_HGPIPE_MATH_LNQ_ACTIVE8_GOLDEN` header selection in `hardware/hls/tb/tb_hgtxr_e2e_axis_top.cpp`.
- Added `HGTXR_E2E_SCALE=hgpipe_math_lnq_active8` to both `hardware/vivado/scripts/run_e2e_q4w8a_csim.tcl` and `hardware/vivado/scripts/run_e2e_q4w8a_csynth.tcl`.
- Registered `hgpipe_math_lnq_active8` in `software/tests/test_e2e_axis_vector_csim.py`.
- Verified reference/header sync, native C++ comparator, Vitis HLS csim, and Vitis HLS csynth for the active8 staged mode.

## 2026-06-10 E2E DSP/URAM Resource Rebalance Execution

- Added default-on resource steering macros in `hardware/hls/include/hgtxr_e2e_vit.hpp`: `HGTXR_E2E_FORCE_DSP_MUL` and `HGTXR_E2E_FORCE_URAM_BUFFERS`.
- Split E2E Q4 weight extraction into `hgtxr_e2e_extract_weight_bits()` plus `hgtxr_e2e_unpack_weight_word()`. For the 4-bit/256-bit ZCU104 path, lane extraction now uses constant bit ranges instead of a 256-bit variable shift.
- Added `hgtxr_e2e_dsp_mul()` and routed patch embedding, QKV projection, attention dot/product, output projection, MLP W1/W2, and MLP head MACs through it.
- Added `hgtxr_e2e_dsp_mul_acc()` for non-HGPIPE LayerNorm variance and affine multiplications identified by the GPT5.5 read-only audit.
- Changed `hardware/hls/src/hgtxr_e2e_axis_top.cpp` so large activation/global buffers use `impl=uram`; `frame` and `gb.pooled` remain BRAM.
- Added `HGTXR_E2E_RESOURCE_POLICY` to E2E CSim/CSynth Tcl with `auto_bram`, `dsp_bram`, `auto_uram`, and `dsp_uram` modes. The old `bram_lut` spelling remains an `auto_bram` compatibility alias.
- Ran native comparator, Vitis CSim, and Vitis CSynth for `HGTXR_E2E_SCALE=hgpipe_math_lnq_active16`.
- Ran same-scale active16 CSynth policy sweep and archived reports under `/tmp/hgtxr_e2e_axis_top_hgpipe_math_lnq_active16_*_csynth.rpt`.
- Wrote `docs/resources/e2e_active16_resource_policy_matrix_2026_06_10.md`; selected `dsp_uram` as the current default resource policy because it gives `39 BRAM_18K`, `55 DSP`, `27,574 FF`, `44,670 LUT`, `80 URAM`.

## 2026-06-10 E2E PAR8 DSP/LUTRAM Retune Execution

- Raised `HGTXR_PARALLELISM_FACTOR`, `HGTXR_E2E_ATTN_PAR`, and `HGTXR_E2E_DENSE_PAR` from 5 to 8 in `hardware/configs/zcu104_e2e_q4w8a_defines.h`.
- Added packed-weight vector cache controls in `hardware/hls/include/hgtxr_e2e_vit.hpp`: `HGTXR_E2E_WEIGHT_VEC_CACHE`, `HGTXR_E2E_WEIGHT_VEC_ALIGNED_FASTPATH`, `HGTXR_E2E_SMALL_MEM_LUTRAM`, and `HGTXR_E2E_LN_PARAM_CACHE`.
- Added `hgtxr_e2e_load_weight_vec()` so dense loops read one packed 256-bit word for aligned `kDensePar=8` weight groups instead of issuing lane-by-lane volatile requests.
- Routed QKV, WO, MLP W1/W2, and head dense loops through the packed vector loader.
- Added local LayerNorm gamma/beta packed-word caches; for Q4/256-bit and C=192 this is 3 gamma words plus 3 beta words per LayerNorm call.
- Bound small attention scratch arrays `score`, `prob`, and `exp_raw` to LUTRAM under `HGTXR_E2E_SMALL_MEM_LUTRAM=1`.
- Verified active16 CSim and CSynth with `HGTXR_E2E_RESOURCE_POLICY=dsp_uram`.
- Updated `docs/resources/e2e_active16_resource_policy_matrix_2026_06_10.md` with the PAR8 fast-cache comparison. Final active16 point: `42 BRAM_18K`, `128 DSP`, `19,576 FF`, `45,431 LUT`, `88 URAM`, `494,968 cycles`.

## 2026-06-10 Full Active196/B6 PAR8 Selective-URAM Execution

- Kept ZCU104 E2E `HGTXR_PARALLELISM_FACTOR`, `HGTXR_E2E_ATTN_PAR`, and `HGTXR_E2E_DENSE_PAR` at `8`.
- Replaced the all-or-nothing `HGTXR_E2E_FORCE_URAM_BUFFERS` storage binding in `hardware/hls/src/hgtxr_e2e_axis_top.cpp` with per-buffer controls for `tokens`, `gb.tokens`, `gb.norm`, `gb.q`, `gb.k`, `gb.v`, `gb.attn`, and `gb.hidden`.
- Added `HGTXR_E2E_RESOURCE_POLICY` modes `dsp_mixed_hidden`, `dsp_mixed_stream`, and `dsp_mixed` to both E2E CSim and CSynth Tcl.
- Ran full `active196_b6_ff768` PAR8 CSynth probes:
  - `dsp_uram`: fails resource fit at `160 URAM`.
  - `dsp_bram`: fits with `268 DSP` but `0 URAM`.
  - `dsp_mixed`: still over budget at `112 URAM`.
  - `dsp_mixed_stream`: fits with `210 BRAM_18K`, `268 DSP`, `39,370 FF`, `90,836 LUT`, `64 URAM`, `82,612,245 cycles`.
- Archived report copies under `/tmp/hgtxr_e2e_axis_top_active196_b6_ff768_par8_*_csynth.rpt`.

## 2026-06-10 Full Active196/B6 PAR8 QKV Cache Execution

- Spark sidecar was unavailable due usage limit; GPT5.5 sidecar `Kant` audited the QKV `II=3` report and recommended local packed Q/K/V weight caches.
- Added `HGTXR_E2E_QKV_WEIGHT_CACHE` to `hardware/hls/include/hgtxr_e2e_vit.hpp`.
- Added `kDenseModelWords` and `hgtxr_e2e_load_weight_vec_from_cache()` for packed BRAM-cache vector reads.
- Added BRAM-backed `q_weight_cache`, `k_weight_cache`, and `v_weight_cache` inside `hgtxr_e2e_project_qkv()`.
- Preloaded Q, K, and V packed weight matrices with separate `II=1` loops, then changed the QKV MAC loop to read from the three local caches instead of issuing three same-cycle AXI reads.
- Re-ran full `active196_b6_ff768`, `PAR=8`, `dsp_mixed_stream` CSynth.
- New full fit point: `306 BRAM_18K`, `332 DSP`, `43,740 FF`, `81,144 LUT`, `64 URAM`, `71,316,968 cycles`.
- QKV local report improved from `II=3`, `2,709,518 cycles`, `16 DSP` to `II=1`, `904,943 cycles`, `48 DSP`.
- Archived final report to `/tmp/hgtxr_e2e_axis_top_active196_b6_ff768_par8_dsp_mixed_stream_qkvcache_csynth.rpt`.

## 2026-06-10 E2E Board-Flow Decision Execution

- Added `docs/track/NEXT-DECISION-2026-06-10-E2E.md` to preserve the user rule that major branch points must be presented with work plan, expected result, cost, risk, and validation before execution.
- Added `docs/track/OPTION-A-PREFLIGHT-2026-06-10-E2E-BITSTREAM.md` for the board-implementation preflight.
- Confirmed the current Vivado package/build path still targets old `hgtxr_top`; `hgtxr_e2e_axis_top` needs either an AXIS/DMA path, a memory-mapped wrapper path, or an explicit in-place replacement decision.
- Attempted Spark A1 sidecar first; GPT5.3-Codex-Spark quota was exhausted until 2026-06-15 23:18.
- Spawned GPT5.5 fallback sidecar `Nash` for A1 and GPT5.5 sidecar `Epicurus` for A2. Both completed read-only and made no edits.
- Captured A1 guidance: package E2E IP separately, build an AXI DMA BD, check DMA length width for `2,097,152` input bytes, and handle one 256-bit beat per pixel.
- Captured A2 guidance: add a new `hgtxr_e2e_m_axi_top(frame, weights, out_state, runtime_state)` wrapper and rerun wrapper CSim/CSynth before any Vivado implementation.
- Updated `docs/Master-Plan.md` and `docs/Sub-Plan.md` with the current decision DAG, Task Cards, file ownership, and validation gates.

## 2026-06-10 A2/A1/C Selected-Path Execution

- User selected Path 1 = A2 then A1, Path 2 = C in parallel where feasible, and E pending.
- Baseline commit before implementation: `03fa3ec23e80b2ca803b6459004339193ebb84e8`.
- Implemented `hgtxr_e2e_m_axi_top(frame, weights, out_state, runtime_state)` plus its testbench and CSim/CSynth Tcl wrappers.
- Validated A2 full CSim with `[32, -13, 26, -6, 14, -11]`, `runtime_state=2`, `failures=0`.
- Validated A2 full CSynth, then retained the safe small-LUTRAM A1 cleanup: final `m_axi` point is `326 BRAM_18K`, `334 DSP`, `46,502 FF`, `84,220 LUT`, `64 URAM`.
- Rejected default Q/K/V local weight-cache URAM binding because the experiment reached `112/96 URAM`.
- Added `HGTXR_E2E_PAR=16|32` to the AXIS E2E Tcl flow; invalid `12/24` values are rejected.
- Validated C PAR16 full AXIS CSynth: `3.953 ns`, `37,508,072 cycles`, `338 BRAM_18K`, `604 DSP`, `59,507 FF`, `127,916 LUT`, `64 URAM`.

## 2026-06-10 Ubuntu Toolchain And Preflight Hardening

- Attempted Spark environment audit first; GPT5.3-Codex-Spark quota was exhausted until 2026-06-15 23:18, so GPT5.5 fallback completed the read-only audit.
- Added `HGTXR_XILINX_ROOT`, `HGTXR_GCC_INCLUDE`, `HGTXR_GCC_LIB`, `HGTXR_SYS_INCLUDE`, and `HGTXR_SYS_LIB` overrides to core Tcl flows, with `/tools/Xilinx` as the default Xilinx root.
- Fixed S2 vector CSim root detection so the script can run from repo root or `hardware/`.
- Extended preflight/static validation for both selected E2E tops: AXIS and A2 `m_axi`.
- Updated durable planning/spec/handover/audit docs to match the current A2/C state.

## 2026-06-10 C Path Isolated PAR Execution Prep

- Kept the selected user strategy: Path 1 A2 -> A1, Path 2 C, E pending.
- Added safe E2E HLS project naming controls to `hardware/vivado/scripts/run_e2e_q4w8a_csim.tcl` and `hardware/vivado/scripts/run_e2e_q4w8a_csynth.tcl`.
- `HGTXR_E2E_RUN_TAG=<tag>` appends the tag to the default `hgtxr_e2e_hls` generated project name.
- `HGTXR_E2E_PROJECT_NAME=<name>` forces an exact generated project name.
- Both values accept only `A-Z`, `a-z`, `0-9`, and `_`.
- Updated A1/A2 wrapper scripts to default to explicit generated project names instead of relying on fragile open-project text replacement.
- Allowed `package_e2e_axis_ip.tcl` to respect caller-provided `HGTXR_E2E_PROJECT_NAME` for isolated C/PAR package runs.
- Added `-artifact_name` to `build_e2e_axis_dma_bitstream.tcl` so C/PAR board attempts can write separate bit/HWH names.
- Saved the next C choices in `docs/track/CHOICE.md`: PAR16 board/package, PAR32 HLS stress, or DSP/LUT cleanup first.

## 2026-06-10 C1 PAR16 AXIS/DMA Board Candidate Execution

- Ran the C1 package path with `HGTXR_E2E_PROJECT_NAME=hgtxr_e2e_axis_par16_hls`, `HGTXR_E2E_PAR=16`, and `HGTXR_E2E_RESOURCE_POLICY=dsp_mixed_stream`.
- Generated C1 IP under `hardware/generated/hgtxr_e2e_axis_par16_hls/solution_e2e_q4w8a/impl/ip`.
- Ran the isolated Vivado AXIS/DMA path with project `hgtxr_e2e_axis_dma_par16_overlay`, BD `hgtxr_e2e_axis_dma_par16_system`, and artifact name `hgtxr_e2e_axis_dma_par16`.
- Copied board artifacts to `hardware/pynq/hgtxr/hgtxr_e2e_axis_dma_par16.bit/.hwh`.
- C1 HLS resource result is `604 DSP`, `64 URAM`, and `127,916 LUT`; routed timing passed with `WNS=4.723 ns`.
- Did not start C2/PAR32 in this step. Physical ZCU104 C1 DMA smoke remains open.

## 2026-06-10 C3 DSP/LUT Cleanup Execution

- Selected C3 after C1 to reduce LUT/BRAM/FF pressure before any PAR32 stress.
- Added `HGTXR_E2E_MEM_BANK_PAR` as a separate memory-bank partition factor while keeping `HGTXR_E2E_PAR=16` compute parallelism.
- Added an aligned packed-weight vector fast path guarded by `HGTXR_E2E_WEIGHT_VEC_ALIGNED_FASTPATH`.
- Updated E2E AXIS CSim/CSynth Tcl flows so `HGTXR_E2E_MEM_BANK_PAR` is validated and passed into HLS.
- Ran reduced PAR16/MEM8 CSim successfully.
- Ran full PAR16/MEM8 CSynth under `hardware/generated/hgtxr_e2e_axis_par16_c3_mem8`.
- Result: `292 BRAM_18K`, `604 DSP`, `56,966 FF`, `113,124 LUT`, `64 URAM`, `48,598,922 cycles`.
- Compared with C1, C3 reduced LUT by `14,792` while preserving `604 DSP` and `64 URAM`.
- The tradeoff is latency: reduced memory banking causes MLP W2 `Final II=2`, so C3/MEM8 is not the low-latency default.

## 2026-06-10 C3b MEM16 Fastpath-Only Execution

- Ran C3b to isolate the aligned fastpath from the C3/MEM8 memory-bank reduction.
- Used `HGTXR_E2E_PROJECT_NAME=hgtxr_e2e_axis_par16_c3b_mem16`, `HGTXR_E2E_PAR=16`, `HGTXR_E2E_MEM_BANK_PAR=16`, `HGTXR_E2E_SCALE=active196_b6_ff768`, and `HGTXR_E2E_RESOURCE_POLICY=dsp_mixed_stream`.
- Reduced CSim passed first with raw output `[18, -3, 3, -2, 1, -4]`.
- Full CSynth completed under `hardware/generated/hgtxr_e2e_axis_par16_c3b_mem16`.
- Result: `332 BRAM_18K`, `604 DSP`, `59,505 FF`, `126,506 LUT`, `64 URAM`, `37,508,072 cycles`.
- C3b restored C1-like latency and MLP W2 `II=1`.
- Conclusion: the large C3/MEM8 LUT reduction mostly comes from lower memory banking; fastpath alone gives only a small LUT drop.

## 2026-06-10 A2 m_axi Board-Flow Scaffold

- Added `hardware/vivado/scripts/package_e2e_m_axi_ip.tcl`, which packages the selected `hgtxr_e2e_m_axi_top` by reusing the validated E2E Q4W/Q8A csynth setup and adding `export_design -format ip_catalog`.
- Added `hardware/vivado/scripts/build_e2e_m_axi_bitstream.tcl`, a separate Vivado BD flow that instantiates `xilinx.com:hls:hgtxr_e2e_m_axi_top:1.0` and connects `frame`, `weights`, `out_state`, and `runtime_state` m_axi masters to PS DDR through SmartConnect.
- Added `hardware/pynq/hgtxr/e2e_m_axi_overlay.py` for board-side A2 smoke runs with frame, packed Q4 weight buffer, output state, and runtime state buffers.
- Registered the new files in static validation and package docs.
- Did not run long Vivado package/build in this step; actual component name and routed timing remain the next A2 validation gate.

## 2026-06-10 A2 E2E m_axi Board Artifact Execution

- Ran A2 HLS package export from `hardware/` with `/tmp/libtinfo.so.5` in `LD_LIBRARY_PATH`; `export_design -format ip_catalog` completed successfully.
- Confirmed exported IP interfaces: `s_axi_control`, `m_axi_gmem_frame`, `m_axi_gmem_e2e_weights`, `m_axi_gmem_out_state`, and `m_axi_gmem_runtime_state`.
- Ran the separate Vivado E2E m_axi block-design flow in `hardware/vivado/scripts/build_e2e_m_axi_bitstream.tcl`.
- `synth_1`, `opt_design`, `place_design`, `route_design`, and `write_bitstream` all completed with `0 Errors`.
- Copied generated overlay artifacts to `hardware/generated/build/vivado/overlay/hgtxr_e2e_m_axi_overlay/`.
- Copied board-ready package artifacts to `hardware/pynq/hgtxr/hgtxr_e2e_m_axi.bit` and `hardware/pynq/hgtxr/hgtxr_e2e_m_axi.hwh`.
- Updated `hardware/tools/check_third_goal_preflight.py` so board-ready mode validates the new E2E m_axi package/build path instead of treating the legacy `hgtxr_top` overlay as the selected E2E proof.
- Updated `hardware/tests/test_check_third_goal_preflight.py` for the new policy.

## 2026-06-10 C3b PAR16/MEM16 AXIS/DMA Board Candidate Execution

- Used GPT5.5 fallback evaluator `Gibbs` for the C3b board-flow risk checklist after GPT5.3-Codex-Spark quota remained exhausted.
- Ran C3b HLS package export with isolated project name `hgtxr_e2e_axis_par16_c3b_mem16_hls`.
- Ran isolated Vivado AXIS/DMA build with project `hgtxr_e2e_axis_dma_c3b_mem16_overlay`, BD `hgtxr_e2e_axis_dma_c3b_mem16_system`, and artifact name `hgtxr_e2e_axis_dma_c3b_mem16`.
- Preserved A1 and C1 artifacts by not reusing `hgtxr_e2e_axis_dma` or `hgtxr_e2e_axis_dma_par16` output names.
- The first Vivado invocation failed on `libtinfo.so.5`; reran successfully with `LD_LIBRARY_PATH=/tmp`.
- `synth_1`, `opt_design`, `place_design`, `route_design`, and `write_bitstream` completed with `0 Errors`.
- Copied artifacts to `hardware/pynq/hgtxr/hgtxr_e2e_axis_dma_c3b_mem16.bit/.hwh`.
- Extended `hardware/tools/check_third_goal_preflight.py` and `hardware/tests/test_check_third_goal_preflight.py` so board-ready mode records optional C3b artifact presence.
- Captured hashes: bit `989e77836341467da05d0ccb86f0b4764d38f4711af8fbcb567290b78fe3076d`, hwh `8ca659da3bbcc061f7299ac18314c00b8ee2112a274990d969635b3f6acf3c90`, export zip `a471fed1d43d2323efd22cc3581cfd7ee1c1cb5a0cfbc69ba57b2877da527e38`.

## 2026-06-10 C3b AXIS/DMA PYNQ Smoke Bundle Execution

- Attempted Spark sidecar for the C3b board-smoke gap; GPT5.3-Codex-Spark quota remained exhausted until 2026-06-15 23:18.
- Attempted GPT5.5 fallback spawn; native sub-agent thread limit was reached, so this step used Codex-native fallback.
- Added `--variant` selection to `hardware/pynq/hgtxr/run_e2e_axis_dma_smoke.py`.
- Added `c3b-mem16` as the variant for `hgtxr_e2e_axis_dma_c3b_mem16.bit/.hwh`.
- Added `hardware/tools/package_e2e_axis_dma_pynq_bundle.py`.
- Added `hardware/tests/test_package_e2e_axis_dma_pynq_bundle.py`.
- Generated the C3b PYNQ smoke bundle directory and tarball under `hardware/generated/pynq/`.
- Bundle tarball SHA256 is `3de505a6627409b4717ea95e5e76d74f0667c61b601fd96db1e495319adda712`.
- Updated `hardware/pynq/hgtxr/README.md` with the C3b bundle-local smoke command.
- Did not run physical ZCU104 smoke in this step.

## 2026-06-10 C3b Bundle Preflight Gate Execution

- Kept C2/PAR32 and C3c unstarted; this was branch-neutral hardening for the already selected C3b board-smoke path.
- Added `check_c3b_axis_dma_smoke_bundle()` to `hardware/tools/check_third_goal_preflight.py`.
- Board-ready mode now validates the C3b bundle when C3b bit/HWH or bundle artifacts are present.
- Added unittest coverage for complete C3b bundle acceptance and missing C3b bundle rejection.
- Wrote board-ready preflight evidence to `docs/resources/third_goal_preflight_board_c3b_bundle_2026_06_10.json`.
- Current preflight summary is `ok=54`, `warn=6`, `fail=0`.

## 2026-06-10 C3b Physical Smoke Result Gate Execution

- Attempted GPT5.3-Codex-Spark sidecar through the native multi-agent tool, but spawn failed with `agent thread limit reached`.
- Added `hardware/tools/validate_pynq_smoke_result.py`.
- Added `hardware/tests/test_validate_pynq_smoke_result.py`.
- Registered the validator and test in `hardware/tools/static_validate_hgtxr.py`.
- Extended `hardware/tools/check_third_goal_preflight.py` so board-ready mode validates copied-back C3b physical smoke JSON if present.
- If no C3b physical result JSON exists, board-ready mode now emits an explicit warning naming the expected copy-back path.
- Captured board-ready evidence at `docs/resources/third_goal_preflight_board_c3b_result_gate_2026_06_10.json`.
- Current preflight summary is `ok=54`, `warn=7`, `fail=0`.

## 2026-06-10 C3b Self-Validating Bundle Execution

- Attempted GPT5.3-Codex-Spark sidecar again; native spawn still failed with `agent thread limit reached`.
- Updated `hardware/tools/package_e2e_axis_dma_pynq_bundle.py` to include `tools/validate_pynq_smoke_result.py`.
- Added bundle-local `validate_e2e_axis_dma_c3b_mem16_file_smoke.sh`.
- Added `validation_command` to the bundle manifest.
- Updated `hardware/tests/test_package_e2e_axis_dma_pynq_bundle.py`.
- Updated C3b bundle checks in `hardware/tools/check_third_goal_preflight.py` and `hardware/tests/test_check_third_goal_preflight.py`.
- Regenerated the C3b bundle tarball.
- New tarball SHA256: `3de505a6627409b4717ea95e5e76d74f0667c61b601fd96db1e495319adda712`.
- New tarball size: `593,198` bytes.

## 2026-06-10 C3b Bundle Package Integrity Gate Execution

- Attempted GPT5.3-Codex-Spark package-QA sidecar; native spawn still failed with `agent thread limit reached`.
- Added `hardware/tools/validate_pynq_bundle_package.py`.
- Added `hardware/tests/test_validate_pynq_bundle_package.py`.
- The package validator checks C3b manifest variant/artifact prefix, run command, validation command, required files, local file SHA256 values, tar SHA256, tar contents, and tar member SHA256 values.
- Integrated package validation into `hardware/tools/check_third_goal_preflight.py` as `C3b AXIS/DMA PYNQ smoke bundle package`.
- Registered the new validator and tests in `hardware/tools/static_validate_hgtxr.py`.
- Refreshed `docs/resources/third_goal_preflight_board_c3b_result_gate_2026_06_10.json`.
- Current preflight summary is `ok=57`, `warn=7`, `fail=0`.

## 2026-06-10 C3b Physical Result Import Gate Execution

- Attempted GPT5.3-Codex-Spark result-import sidecar; native spawn failed with `agent thread limit reached`.
- Added `hardware/tools/import_pynq_smoke_result.py`.
- Added `hardware/tests/test_import_pynq_smoke_result.py`.
- The importer validates the source JSON with `validate_pynq_smoke_result.py` semantics before copying.
- Default C3b import destination is `hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json`.
- Invalid results are rejected without overwriting the canonical result.
- Registered the importer files in `hardware/tools/static_validate_hgtxr.py`.

## 2026-06-10 C3b ZCU104 Smoke Session Runbook Execution

- Attempted GPT5.3-Codex-Spark session-runbook sidecar; native spawn failed with `agent thread limit reached`.
- Added `hardware/tools/prepare_zcu104_smoke_session.py`.
- Added `hardware/tests/test_prepare_zcu104_smoke_session.py`.
- The session tool validates the C3b bundle first, then writes a JSON manifest and Markdown runbook.
- Generated `hardware/generated/pynq/e2e_axis_dma_c3b_mem16_smoke_session.json`.
- Generated `hardware/generated/pynq/e2e_axis_dma_c3b_mem16_smoke_session.md`.
- Session status is `pass`; tarball SHA256 is `3de505a6627409b4717ea95e5e76d74f0667c61b601fd96db1e495319adda712`.

## 2026-06-10 C3b Smoke Session Preflight Gate Execution

- Attempted GPT5.3-Codex-Spark session-preflight sidecar; native spawn failed with `agent thread limit reached`.
- Added `check_c3b_smoke_session()` to `hardware/tools/check_third_goal_preflight.py`.
- Board-ready preflight now validates generated C3b smoke-session JSON/Markdown when C3b artifacts or bundle/session files are present.
- Extended `hardware/tests/test_check_third_goal_preflight.py` with complete-session success and missing-session failure coverage.
- Current preflight summary is `ok=61`, `warn=7`, `fail=0`.

## 2026-06-10 Final Signoff Gate Execution

- Attempted GPT5.3-Codex-Spark final-signoff review sidecar; native spawn failed with `agent thread limit reached`.
- Added `final-signoff` mode to `hardware/tools/check_third_goal_preflight.py`.
- `board-ready` still passes with missing physical C3b smoke as a warning.
- `final-signoff` fails missing/invalid C3b physical smoke result.
- `final-signoff` also fails missing requested `PAPER_PRJXR` DeiT image and `XR-VITs` sibling.
- Extended `hardware/tests/test_check_third_goal_preflight.py` with board-ready missing-smoke warning coverage and final-signoff failure coverage.
- Current board-ready summary is `ok=61`, `warn=7`, `fail=0`.
- Current final-signoff summary is `ok=61`, `warn=4`, `fail=3`.

## 2026-06-10 Final Signoff Audit Artifact Execution

- Added `hardware/tools/write_final_signoff_audit.py`.
- Added `hardware/tests/test_write_final_signoff_audit.py`.
- Registered the audit tool/test in `hardware/tools/static_validate_hgtxr.py`.
- Captured current final-signoff and board-ready preflight JSON under `hardware/generated/signoff/`.
- Generated `hardware/generated/signoff/final_signoff_audit_2026_06_10.json`.
- Generated `hardware/generated/signoff/final_signoff_audit_2026_06_10.md`.
- The audit records 3 blockers and the C3b board/import command sequence.
