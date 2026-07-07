# HGTXR Validation Status

Date: 2026-06-05

## Status Legend

- Complete: current evidence exists and covers the requirement.
- Partial: evidence exists but does not cover the full requirement.
- Unverified: no current evidence from this turn.
- Blocked: cannot be checked until WSL shell or board access is available.

## Primary Flow

| Requirement | Status | Evidence / Gap |
|---|---|---|
| PYNQ-friendly AXI pragmas in `hgtxr_top.cpp` | Partial | Prior run summary says AXI master and AXI-Lite pragmas were added and csynth reported expected interfaces. Current turn could not re-open source through shell. |
| Vitis HLS `csynth` | Partial | Prior run summary says csynth passed after AXI bundle fixes. Current turn did not re-run. |
| HLS IP export | Partial | Prior run summary says `component.xml` and export IP were produced. Current turn did not re-open artifact. |
| Vivado block design | Partial | Prior run summary says BD validation passed and implementation reached bitgen. Current turn did not re-run. |
| Bitstream generation | Partial | Prior run summary says `write_bitstream` completed before Tcl HWH-copy failure. Current turn could not inspect `.bit`. |
| HWH generation | Partial | Prior run summary says BD handoff HWH exists. Current turn could not inspect `.hwh`. |
| PYNQ driver | Partial | Prior run summary says driver files and local helper smoke existed. Current turn did not re-run tests. |
| Board-side PYNQ execution | Unverified | Requires ZCU104 board access. |

## Secondary Goals

| Requirement | Status | Evidence / Gap |
|---|---|---|
| ZCU104 fit | Partial | A fit-oriented experiment matrix now exists. Current utilization/timing reports must still be inspected and compared. |
| Paper Transformer Block implementation | Partial | Architecture spec exists. Full module implementation remains to be done or verified. |
| Cyclic Accelerator implementation | Partial | Scheduler architecture exists. RTL/HLS implementation remains to be done or verified. |
| Tiling and parallelization | Partial | Parameter surface and sweep matrix exist. Source implementation and sweeps remain. |
| DeiT-Tiny C-Syn image reference | Blocked | `view_image` could not locate the WSL path in this runner. Must inspect when file access is restored. |
| `impl_repos` experiment reference | Blocked | Requires WSL shell or filesystem access. |
| Parameterization of tunables | Partial | Spec names required parameters. HLS headers/source still need verification or implementation. |
| Spark multi-agent use | Complete for planning | Spark sidecars produced architecture, experiment, and QA planning artifacts. Filesystem-dependent Spark tasks were blocked by runner access. |

## Required Next Evidence

Run these commands when WSL shell access is restored:

```bash
cd /home/user/project/PRJXR/HGTXR
git status --short
find . -maxdepth 4 -type f | sort | sed -n '1,200p'
ls -lh hardware/generated/build/vivado/hgtxr_overlay/hgtxr_overlay.runs/impl_1/*.bit
ls -lh hardware/generated/build/vivado/hgtxr_overlay/hgtxr_overlay.gen/sources_1/bd/hgtxr_system/hw_handoff/*.hwh
python3 -m py_compile hardware/pynq/hgtxr/hgtxr_overlay.py hardware/pynq/hgtxr/test_hgtxr_overlay.py
PYTHONPATH=hardware/pynq/hgtxr python3 hardware/pynq/hgtxr/test_hgtxr_overlay.py
```

For completion, also collect:

- HLS `csynth` XML/report.
- Vivado utilization report.
- Vivado timing report.
- PYNQ board run log.
- Paper metric comparison table.

## Continuation Verification - 2026-06-06

| Requirement | Status | Evidence / Gap |
|---|---|---|
| Vivado HWH copy Tcl patch | Complete | `hardware/tools/repair_hls_to_pynq.py --apply` patched `hardware/vivado/scripts/build_bitstream.tcl`; static validator confirmed the patch marker. |
| Overlay/PYNQ bit/hwh copies | Complete locally | `hgtxr.bit` and `hgtxr.hwh` exist under both `hardware/generated/build/vivado/overlay/hgtxr_overlay/` and `hardware/pynq/hgtxr/`. |
| Static validation | Complete | `python3 hardware/tools/static_validate_hgtxr.py --root . --require-artifacts` passed. |
| PYNQ helper local smoke | Complete locally | `python3 -m py_compile ...` and `PYTHONPATH=hardware/pynq/hgtxr python3 hardware/pynq/hgtxr/test_hgtxr_overlay.py` exited 0. |
| Cyclic primitive compile/smoke | Complete locally | Compiled with `-I/tools/Xilinx/Vitis_HLS/2023.2/include`; `/tmp/tb_cyclic_primitives` passed all checks. |
| Sweep parameterization | Partial | `hgtxr_sweep_expand.py` produced 2048 combinations and generated defines; full HLS csynth sweep not run. |
| `impl_repos` extraction | Partial | `hardware/tools/extract_resource_metrics.py` wrote 41,302 candidate rows; regex-first-pass values need csynth-focused filtering. |
| DeiT-Tiny C-Syn image extraction | Unverified | Image file exists, but `tesseract` is not installed and app image viewer could not open this WSL path. Manual/OCR extraction still required. |
| Integrated Transformer block top path | Unverified | Header-level cyclic Transformer modules exist; `hgtxr_top.cpp` still uses the current scaffold path. |
| Board-side PYNQ execution | Unverified | Requires ZCU104 access. |

## Cyclic Integration Verification - 2026-06-06

| Requirement | Status | Evidence / Gap |
|---|---|---|
| Gated cyclic Transformer top path | Complete for S0 integration | `hardware/hls/src/hgtxr_top.cpp` uses `HGTXR_ENABLE_CYCLIC_TRANSFORMER_TOP` to select cyclic Transformer tiles while preserving the public top ABI. |
| HLS project parameterization | Complete for S0 | `create_hls_project.tcl` accepts `-enable_cyclic_transformer`, `-define_file`, and `-solution`; `run_cyclic_s0_csynth.tcl` drives `solution_cyclic_s0`. |
| Default top csynth after integration | Complete | `hardware/generated/hgtxr_hls/solution/syn/report/hgtxr_top_csynth.rpt` passed; estimated `97.29 MHz`, resources `283 BRAM`, `81 DSP`, `21057 FF`, `26399 LUT`. |
| Cyclic S0 top csynth | Complete | `hardware/generated/hgtxr_hls/solution_cyclic_s0/syn/report/hgtxr_top_csynth.rpt` passed after active-token specialization; estimated `267.02 MHz`, latency `15,023,573 cycles` / `75.118 ms`, resources `263 BRAM`, `42 DSP`, `23397 FF`, `38752 LUT`. |
| ZCU104 resource fit | Partial | Cyclic S0 is under HLS resource gates (`42% BRAM`, `2% DSP`, `5% FF`, `16% LUT`), but final Vivado implementation and timing closure are not yet run. |
| Throughput/latency fit | Partial | Active-token specialization reduced cyclic S0 from the earlier `78,024,797 cycles` estimate to `15,023,573 cycles`; remaining work is packed weights, S1 parallelism sweep, and board-side throughput measurement. |
| Vitis HLS csim | Blocked by environment | `csim_design` fails because Vivado 2023.2 bundled linker cannot link Ubuntu 24.04 system libraries (`.relr.dyn` incompatibility). g++ local smoke passed, but this is not a substitute for HLS csim. |

## Active-Token Specialization Verification - 2026-06-06

| Requirement | Status | Evidence / Gap |
|---|---|---|
| Search/track token schedule specialization | Complete | `hgtxr_top.cpp` instantiates cyclic stages as `<HGTXR_SEARCH_TOKENS>` and `<HGTXR_TRACK_TOKENS>`, producing separate HLS modules for 64-token and 16-token paths. |
| Stage-level HLS evidence | Complete | `p_anonymous_namespace_hgtxr_cyclic_transformer_stage_64_s_csynth.rpt` reports `3,230,833 cycles`; `p_anonymous_namespace_hgtxr_cyclic_transformer_stage_16_s_csynth.rpt` reports `807,709 cycles`. |
| Transformer-block tile bottleneck evidence | Complete | `transformer_block_tile_16_32_32_s_csynth.rpt` reports `133,583 cycles`; largest submodules are QKV projection, MLP, and attention. |

## Packed Weight-Port S1 Verification - 2026-06-06

| Requirement | Status | Evidence / Gap |
|---|---|---|
| Default ABI preservation | Complete | Default g++ top smoke compiled and ran with the original `hgtxr_top(frame, event_pos, event_neg, prev, out, runtime_state)` signature. |
| Optional packed weight ABI | Complete for S1 HLS baseline | `HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS=1` compiles and adds `cyclic_weights` on `gmem_cyclic_weights`; `solution_cyclic_s1_weight` csynth passed. |
| S1 ZCU104 resource fit | Partial | HLS estimate is under resource gates: `44% BRAM`, `2% DSP`, `6% FF`, `23% LUT`; final Vivado implementation timing is not run. |
| Packed weight correctness | Partial | Loader and ABI compile/run smoke passed with zero-initialized test weights. Host-side trained/global weight packing and golden-vector comparison are still missing. |

## Host-Side Cyclic Weight Packer Verification - 2026-06-06

| Requirement | Status | Evidence / Gap |
|---|---|---|
| S1 host weight packing tool | Complete for current tile-local ABI | `hardware/tools/pack_cyclic_weights.py --self-test` passed; strict six-block synthetic `.npz` pack emitted `4,704` words with `0` fallbacks. |
| Packer syntax/tests | Partial | `py_compile` passed. `pytest` is not installed, so `tests/test_pack_cyclic_weights.py` could not be run through pytest in this shell. |
| Software checkpoint packing | Blocked by environment | `hardware/refs/weights/software_initial_weights.pt` exists, but active Python lacks `torch`; use `.npz` export or a torch-enabled environment. |
| DeiT-Tiny image reference | Partial | Image exists and was verified as `955 x 429` RGBA PNG. Numeric OCR extraction remains pending. |
| Full dense weight equivalence | Not met | Current S1 ABI and packer are diagonal tile-local; full `192x192` cross-channel dense projection still needs HLS accumulation/schedule work. |

## S2 Channel-Pair Packing Verification - 2026-06-06

| Requirement | Status | Evidence / Gap |
|---|---|---|
| S2 full channel-pair host schedule | Complete for host packer | `--layout-mode s2_channel_pairs` emitted `216` blocks and `169,344` words for the current 6-layer, 6-channel-tile synthetic strict case. |
| S2 packer tests | Complete without pytest | Direct execution of self-test, S1 fallback test, and S2 channel-pair test passed. `pytest` package is still unavailable. |
| HLS S2 consumption | Partial | HLS now has S2 index/load/projection accumulation helpers plus a gated top-level `WO` first-step mode. `solution_cyclic_s2_first_step` csynth passed; default and S1 paths remain preserved. Full Transformer Block S2 consumption remains. |
| Full DeiT MLP hidden expansion | Not met | Current S2 host schedule still uses `TC/TF` tile dimensions; explicit hidden-tile iteration for `mlp_ratio=4` remains required. |


## S2 HLS Projection Primitive Verification - 2026-06-06

| Requirement | Status | Evidence / Gap |
|---|---|---|
| Host/HLS S2 block index consistency | Complete locally | `tb_cyclic_s2_projection.cpp` checks `(layer * C + output_tile) * C + input_tile` and word-base stride behavior. |
| Packed lane element loading | Complete locally | The testbench packs signed fixed-point lanes into `HgtxrAxiWordT` and verifies lane0/lane1/next-word loads. |
| S2 projection partial accumulation | Complete locally for primitive | Direct tile and packed-wrapper tests pass for two input tiles accumulated into one output tile. |
| S2 top integration | Partial | `HGTXR_ENABLE_CYCLIC_S2_FIRST_STEP` now gates a `WO` channel-pair accumulation path in `hgtxr_top.cpp`, and `solution_cyclic_s2_first_step` csynth passed. Full QKV/attention/MLP S2 integration remains. |
| Full DeiT MLP hidden expansion | Not met | Hidden-tile iteration for `mlp_ratio=4` remains a future step. |


## S2 WO First-Step Top Verification - 2026-06-06

| Requirement | Status | Evidence / Gap |
|---|---|---|
| Default ABI preservation | Complete locally | Default g++ top smoke passed after S2 edits. |
| S1 weight-port preservation | Complete locally | S1 weight-port g++ top smoke passed after S2 edits. |
| S2 first-step top compile/run | Complete locally | S2 first-step g++ top smoke compiled and ran with `HGTXR_ENABLE_CYCLIC_S2_FIRST_STEP=1`. |
| S2 first-step HLS synthesis | Complete for WO first-step | `solution_cyclic_s2_first_step` csynth passed: `273.97 MHz`, `12,173,567..12,506,639 cycles`, `211 BRAM`, `22 DSP`, `10935 FF`, `17997 LUT`. |
| ZCU104 HLS resource fit | Complete for WO first-step estimate | HLS usage is `33% BRAM`, `1% DSP`, `2% FF`, `7% LUT`, `0% URAM`. Final Vivado implementation still required. |
| Full S2 Transformer Block | Not met | QKV projection first-step is now synthesized, but attention, output projection, MLP hidden expansion, LayerNorm/residual schedule, and golden-vector equivalence are still pending. |


## S2 QKV First-Step Top Verification - 2026-06-06

| Requirement | Status | Evidence / Gap |
|---|---|---|
| S2 Q/K/V packed projection helper | Complete locally | `tb_cyclic_s2_projection.cpp` checks packed WQ/WK/WV channel-pair accumulation. |
| S2 QKV first-step top compile/run | Complete locally | `HGTXR_ENABLE_CYCLIC_S2_QKV_FIRST_STEP=1` top g++ smoke compiled and ran. |
| S2 WO preservation | Complete locally | Existing S2 WO first-step top smoke still compiled and ran after QKV edits. |
| S2 QKV HLS synthesis | Complete for QKV projection first-step | `solution_cyclic_s2_qkv_first_step` csynth passed: `273.97 MHz`, `34,770,191..35,769,407 cycles`, `215 BRAM`, `22 DSP`, `11373 FF`, `19205 LUT`. |
| ZCU104 HLS resource fit | Complete for QKV projection estimate | HLS usage is `34% BRAM`, `1% DSP`, `2% FF`, `8% LUT`, `0% URAM`. Final Vivado implementation still required. |
| Attention/softmax/value path | Not met | S2 QKV output is merged for path retention; real attention scheduling and golden-vector equivalence remain. |
| Full paper Transformer Block | Not met | Output projection, residuals, LayerNorm, MLP hidden expansion, and end-to-end trained weights remain pending. |


## S2 Attention First-Step Top Verification - 2026-06-06

| Requirement | Status | Evidence / Gap |
|---|---|---|
| S2 QKV to attention tile connection | Complete locally | `HGTXR_ENABLE_CYCLIC_S2_ATTN_FIRST_STEP=1` top g++ smoke compiled and ran, retaining Q/K/V tiles long enough to call `attention_tile`. |
| S2 attention to WO projection connection | Complete locally | The attention tile is consumed by the S2 `WO` channel-pair projection helper and written back to the token buffer. |
| Existing S2 modes preserved | Complete locally | S2 QKV and S2 WO first-step top g++ smokes still compiled and ran after the attention edit. |
| S2 attention HLS synthesis | Complete for attention first-step | `solution_cyclic_s2_attn_first_step` csynth passed: `267.02 MHz`, `48,980,449..50,312,737 cycles`, `245 BRAM`, `36 DSP`, `24954 FF`, `38796 LUT`. |
| ZCU104 HLS resource fit | Complete for attention first-step estimate | HLS usage is `39% BRAM`, `2% DSP`, `5% FF`, `16% LUT`, `0% URAM`. Final Vivado implementation still required. |
| Paper-equivalent Transformer Block | Not met | Residual ordering, LayerNorm correctness, MLP hidden expansion, trained-weight golden vectors, HLS csim, final Vivado timing, and board validation remain pending. |


## S2 MLP Hidden First-Step Top Verification - 2026-06-06

| Requirement | Status | Evidence / Gap |
|---|---|---|
| Explicit `mlp_ratio=4` hidden tiling | Complete for MLP first-step | `HGTXR_CYCLIC_MLP_RATIO=4` derives `HGTXR_CYCLIC_FF_DIM=768` and `HGTXR_CYCLIC_HIDDEN_TILES=24` for the ZCU104 config. |
| S2 W1/W2 hidden-pair layout | Complete locally | `tb_cyclic_s2_projection.cpp` validates MLP layer stride, W1/W2 block indices, and W1/W2 packed wrapper accumulation. |
| Host packer MLP schedule | Complete locally | `s2_mlp_hidden_pairs` manifest records W1 and W2 phases, `hidden_dim`, and `hidden_tiles`; direct packer tests passed. |
| S2 MLP top compile/run | Complete locally | `HGTXR_ENABLE_CYCLIC_S2_MLP_FIRST_STEP=1` top g++ smoke compiled and ran. |
| Existing S2 modes preserved | Complete locally | S2 attention and S2 QKV top smokes still compiled and ran after the MLP edit. |
| S2 MLP HLS synthesis | Complete for MLP first-step | `solution_cyclic_s2_mlp_first_step` csynth passed: `273.97 MHz`, `92,490,625..95,155,201 cycles`, `243 BRAM`, `30 DSP`, `12850 FF`, `23170 LUT`. |
| ZCU104 HLS resource fit | Complete for MLP first-step estimate | HLS usage is `38% BRAM`, `1% DSP`, `2% FF`, `10% LUT`, `0% URAM`. Final Vivado implementation still required. |
| Full paper-equivalent Transformer Block | Not met | Attention and MLP are still separate first-step modes; residual ordering, LayerNorm correctness, fused block scheduling, trained-weight golden vectors, HLS csim, final Vivado timing, and board validation remain pending. |


## S2 Residual-Aware Block First-Step Top Verification - 2026-06-06

| Requirement | Status | Evidence / Gap |
|---|---|---|
| Fused attention and MLP schedule | Complete for first-step | `HGTXR_ENABLE_CYCLIC_S2_BLOCK_FIRST_STEP=1` runs S2 attention first-step followed by S2 MLP hidden first-step in one top mode. |
| Residual topology | Complete for first-step | The block stage copies token residuals before attention and before MLP, then applies two token-wise residual additions. |
| Host fused block layout | Complete locally | `s2_block_first_step` emits per-layer `attention`, `w1`, and `w2` block phases; direct packer test passed. |
| S2 block top compile/run | Complete locally | S2 block first-step top g++ smoke compiled and ran. |
| Existing S2 mode preservation | Complete locally | Existing S2 attention and MLP first-step top modes still compiled after the block edit. |
| S2 block HLS synthesis | Complete for residual-aware first-step | `solution_cyclic_s2_block_first_step` csynth passed: `267.02 MHz`, `141,204,323..145,201,187 cycles`, `319 BRAM`, `52 DSP`, `29139 FF`, `51183 LUT`. |
| ZCU104 HLS resource fit | Complete for block first-step estimate | HLS usage is `51% BRAM`, `3% DSP`, `6% FF`, `22% LUT`, `0% URAM`. Final Vivado implementation still required. |
| Paper-equivalent Transformer Block | Not met | LayerNorm numerical equivalence, trained-weight golden vectors, HLS csim, final Vivado timing, and board validation remain pending. |

## S2 Pre-LayerNorm Block First-Step Verification - 2026-06-06

| Check | Status | Evidence |
|---|---|---|
| C++ compile | Passed | `g++ ... -include hardware/configs/zcu104_cyclic_s2_block_first_step_defines.h ... -o /tmp/hgtxr_tb_cyclic_s2_block_preln` |
| C++ smoke | Passed | `/tmp/hgtxr_tb_cyclic_s2_block_preln` -> `state=1 out0=128.000000 out1=128.000000` |
| Static validation | Passed | `python3 hardware/tools/static_validate_hgtxr.py --root . --require-artifacts` |
| HLS csynth | Passed | `solution_cyclic_s2_block_first_step`, target `5.00 ns`, estimated `3.745 ns`, `267.02 MHz` |
| ZCU104 HLS resource fit | Passed at HLS estimate | `319 BRAM_18K` (`51%`), `88 DSP` (`5%`), `32699 FF` (`7%`), `58901 LUT` (`25%`), `0 URAM` |
| Paper LayerNorm equivalence | Incomplete | Pre-LN topology is present, but `rsqrt` is piecewise approximate and still needs trained-weight golden-vector validation. |

## S2 Pre-LayerNorm Golden-Vector Validation - 2026-06-06

| Check | Status | Evidence |
|---|---|---|
| Validator syntax | Passed | `python3 -m py_compile hardware/tools/validate_s2_block_preln.py hardware/tools/static_validate_hgtxr.py tests/test_pack_cyclic_weights.py` |
| Durable golden-vector artifact | Passed | `docs/resources/s2_block_preln_validation_2026_06_06.json` |
| Norm packing/addressing | Passed | `max_norm_param_error=0.0` |
| Piecewise pre-LN approximation | Passed for synthetic validation case | `max_preln_piecewise_abs_error=0.0039751529693603516`, threshold `0.01` |
| pytest | Not available | `/usr/bin/python3: No module named pytest`; direct pytest-equivalent assertions passed. |
| C++ smoke | Passed | `/tmp/hgtxr_tb_cyclic_s2_block_preln` -> `state=1 out0=128.000000 out1=128.000000` |
| HLS csynth after refinement | Passed | `3.745 ns`, `267.02 MHz`, `319 BRAM`, `88 DSP`, `32701 FF`, `59141 LUT` |
| Full trained-weight paper equivalence | Incomplete | Validation is synthetic and focused on pre-LN/norm packing only; trained checkpoints, full attention/MLP goldens, final Vivado, and board validation remain. |

## S2 Full-Block Golden-Vector Validation - 2026-06-06

| Check | Status | Evidence |
|---|---|---|
| Full-block validator syntax | Passed | `python3 -m py_compile hardware/tools/validate_s2_block_full.py ...` |
| Durable full-block artifact | Passed | `docs/resources/s2_block_full_validation_2026_06_06.json` |
| Packed QKV/WO/W1/W2 matrix reconstruction | Passed | `max_matrix_error=0.0` |
| Full approximation output | Passed | `max_output_error=0.0` for synthetic packed-vs-direct reference |
| Static validation | Passed | `python3 hardware/tools/static_validate_hgtxr.py --root . --require-artifacts` now requires the full-block validator and artifact. |
| pytest | Not available | Direct pytest-equivalent assertions passed; `/usr/bin/python3: No module named pytest` remains the environment limitation. |
| Trained checkpoint equivalence | Incomplete | This is a synthetic host reference; trained DeiT/HGTXR checkpoint vectors and HLS csim matching are still required. |



## S2 Full-Block External Harness Validation - 2026-06-06

| Check | Status | Evidence |
|---|---|---|
| Validator syntax | Passed | python3 -m py_compile hardware/tools/validate_s2_block_full.py tests/test_pack_cyclic_weights.py hardware/tools/static_validate_hgtxr.py |
| Synthetic default artifact | Passed | python3 hardware/tools/validate_s2_block_full.py --out docs/resources/s2_block_full_validation_2026_06_06.json |
| External .npz weights and tokens | Passed | Direct pytest-equivalent smoke generated external weights/tokens, wrote --out-output, then compared it as --expected-output. |
| Expected-output comparison | Passed | expected_output_compare=all_layers, max_expected_output_error=0.0 |
| Test hook | Passed directly | test_validate_s2_block_full_external_input executed directly because pytest is not installed. |
| Static validation | Passed | python3 hardware/tools/static_validate_hgtxr.py --root . --require-artifacts |
| Existing .pt candidate | Blocked by environment | hardware/refs/weights/software_initial_weights.pt requires torch; current WSL Python reports No module named torch. Export to .npz or use torch-enabled Python. |
| Quantized/HLS equivalence | Incomplete | Current check is packed float layout equivalence, not fixed-point dequantized or HLS csim equivalence. |


## S2 Software-Initial Checkpoint Validation - 2026-06-06

| Check | Status | Evidence |
|---|---|---|
| Torch-enabled runtime | Passed | HGTXR .venv Python has torch 2.12.0+cu130 and numpy 2.4.6. |
| Checkpoint key/shape contract | Passed | software_initial_weights.pt contains backbone.blocks.0..11 norm/qkv/out/MLP tensors with 192-dim and 768 hidden dim. |
| S2 full-block checkpoint validation | Passed | docs/resources/s2_block_full_validation_software_initial_2026_06_06.json |
| S2 expected-output replay | Passed | docs/resources/s2_block_full_validation_software_initial_expected_2026_06_06.json, max_expected_output_error=0.0 |
| Generated checkpoint outputs | Passed | docs/resources/s2_block_full_outputs_software_initial_2026_06_06.npz |
| S2 block packed weight ABI | Passed | hardware/refs/weights/cyclic_weights_s2_block_software_initial_manifest.json reports block_count=1944, word_count=1524096, fallback_count=0, hidden_tiles=24. |
| Static validation registration | Passed | hardware/tools/static_validate_hgtxr.py now requires the software-initial validation and packed-weight artifacts. |
| PyTorch-model exact golden | Incomplete | The current evidence is packed-vs-direct host approximation, not exact model forward equivalence. |
| HLS csim/quantized equivalence | Incomplete | Fixed-point dequantized comparison and HLS csim vector matching remain next gates. |


## Q4W/Q8A Baseline Validation - 2026-06-08

| Check | Status | Evidence |
|---|---|---|
| Q8 activation config | Passed compile smoke | hardware/configs/zcu104_cyclic_s2_block_first_step_defines.h sets HGTXR_BIT_WIDTH=8. |
| Q4 packed weight config | Passed compile smoke | HGTXR_WEIGHT_BIT_WIDTH=4 and HGTXR_WEIGHT_INT_WIDTH=2 are used for packed weight lane unpacking. |
| C++ top smoke | Passed | g++ S2 block pre-LN config produced state=1 out0=128.000000 out1=128.000000. |
| Q4 packed artifact | Passed | cyclic_weights_s2_block_software_initial_q4_manifest.json reports block_count=1944, block_word_stride=196, word_count=381024, fallback_count=0. |
| Q4 round-trip quantization | Passed | q4w8a_s2_block_quantization_2026_06_08.json reports max_abs_quant_error=0.072168730199337 <= 0.125. |
| Exact PyTorch golden export | Passed | s2_pytorch_golden_software_initial_2026_06_08.json and .npz were generated with torch-enabled HGTXR .venv. |
| Approximation-vs-exact PyTorch gap | Quantified, not closed | max_abs_error=0.34955108165740967; remaining causes include bias omission, multi-head split/scaling, and approximation tables. |
| Static validation | Passed | python3 hardware/tools/static_validate_hgtxr.py --root . --require-artifacts passed after Q4W/Q8A csynth evidence registration. |
| HLS csim for Q4W/Q8A | Blocked by environment | Vitis HLS csim fails before source validation on Ubuntu/WSL header resolution: /usr/include/features-time64.h cannot include bits/wordsize.h. The standalone g++ smoke passed, but this does not replace HLS csim. |
| HLS csynth for Q4W/Q8A | Passed | solution_cyclic_s2_block_q4w8a passed: target 5.00 ns, estimated 3.745 ns / 267.02 MHz, latency 141,665,730..145,662,594 cycles, resources 211 BRAM_18K, 76 DSP, 31674 FF, 57932 LUT, 0 URAM. |

## E2E AXI-Stream Q4W/Q8A Shell Validation - 2026-06-09

| Check | Status | Evidence |
|---|---|---|
| AXI-Stream DMA-facing top | Passed smoke/csynth | hgtxr_e2e_axis_top exposes AXIS input/output, m_axi weights, m_axi runtime_state, and AXI-Lite control. |
| Conv patch embedding | Implemented shell | hgtxr_conv_patch_embedding consumes 256x256 streamed frame and fills token buffer. |
| Global Buffer | Implemented shell | HgtxrGlobalBuffer holds token buffer and pooled vector. |
| Two ATTN units | Implemented shell | hgtxr_e2e_attn_unit<0> and <1> are instantiated by the controller. |
| Two MLP units | Implemented shell | hgtxr_e2e_mlp_unit<0> and <1> are instantiated by the controller. |
| ViT block order | Corrected | Controller now runs ATTN0 -> MLP0 -> ATTN1 -> MLP1 -> ... for six blocks. |
| C++ smoke | Passed | g++ E2E AXIS testbench produced runtime_state=1 count=6 last=1. |
| HLS csim | Blocked by environment | Same Ubuntu/WSL Vitis csim header issue: features-time64.h cannot include bits/wordsize.h. |
| HLS csynth | Passed | solution_e2e_q4w8a: 3.650 ns / 273.97 MHz, 1,610,485..1,610,491 cycles, 8.052 ms, 52 BRAM, 18 DSP, 3798 FF, 6411 LUT. |
| Weight AXI live-read | Passed | hgtxr_e2e_axis_top_csynth.rpt includes gmem_e2e_weights_m_axi_U and AR/R ports. |
| Numerical equivalence | Incomplete | This is an E2E interface/scheduler shell, not yet full HG-PIPE dense QKV/WO/W1/W2/LN/GELU/Softmax numerical equivalence. |

## Full Dense ViT E2E Validation - 2026-06-09

- Fixed Vitis HLS `csim_design` on Ubuntu 24.04/WSL by adding multiarch include
  paths, isolating the E2E project from stale testbench files, and forcing
  system linker/library paths.
- `g++` smoke passed: `runtime_state=1 count=6 last=1`.
- Vitis HLS `csim_design` passed for the reduced E2E smoke config.
- Vitis HLS `csynth_design` passed for the full `196-token, 6-block,
  C=192/H=3/FF=768` target.
- Full report:
  `hardware/generated/hgtxr_e2e_hls/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`.
