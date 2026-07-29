# HGTXR Cyclic Weight Packing Evidence

Date: 2026-06-06
Tool: `hardware/tools/pack_cyclic_weights.py`

## Purpose

This tool bridges the S1 packed-weight HLS ABI and host-side data preparation. It accepts software/global tensor maps and emits packed AXI words for `HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS=1`.

## Supported Inputs

- `.npz`: preferred in this environment; maps tensor names to numeric arrays.
- `.json`: maps tensor names to nested numeric arrays.
- `.pt` / `.pth`: supported only when `torch` is installed in the active Python environment.

The current shell does not have `torch`, so the repository checkpoint `hardware/refs/weights/software_initial_weights.pt` could not be read in this turn. Use `.npz` export or install torch in a controlled environment for checkpoint packing.

## Tensor Name Mapping

The packer recognizes the current software backbone names:

| HLS tensor | Software source |
|---|---|
| `norm1_gamma`, `norm1_beta` | `backbone.blocks.{2*layer}.norm.weight/bias` |
| `wq`, `wk`, `wv` | split from `backbone.blocks.{2*layer}.qkv.weight` |
| `wo` | `backbone.blocks.{2*layer}.out.weight` |
| `norm2_gamma`, `norm2_beta` | `backbone.blocks.{2*layer+1}.net.0.weight/bias` |
| `w1` | `backbone.blocks.{2*layer+1}.net.1.weight` |
| `w2` | `backbone.blocks.{2*layer+1}.net.4.weight` |

PyTorch `Linear.weight` tensors are transposed before packing so the HLS `mac_tile(lhs[TM][TK], rhs[TK][TN])` receives `[input][output]` layout.

## Outputs

- raw little-endian binary: suitable for loading into a 128-bit AXI word buffer.
- `.npz`: includes `words_u64` for host-side inspection/loading and `float_elems` for debugging.
- manifest JSON: records layout, offsets, word count, fallback use, and current limitation.

## Validation Run

Commands run on 2026-06-06:

```bash
python3 hardware/tools/pack_cyclic_weights.py --self-test
python3 -m py_compile hardware/tools/pack_cyclic_weights.py tests/test_pack_cyclic_weights.py
python3 hardware/tools/pack_cyclic_weights.py --input /tmp/hgtxr_synth_weights_full.npz --strict --out-bin /tmp/hgtxr_cyclic_weights_full.bin --out-npz /tmp/hgtxr_cyclic_weights_full.npz --manifest /tmp/hgtxr_cyclic_weights_full_manifest.json
```

Results:

- self-test passed with exact fixed-point round-trip for the synthetic case.
- syntax compile passed.
- strict six-block synthetic pack emitted `4,704` words with `0` fallbacks.
- pytest test file was added, but `python3 -m pytest ...` could not run because `pytest` is not installed in this shell.

## Current Limitation

The current S1 HLS block consumes one diagonal tile-local channel slice. The packer therefore packs `channel_tile=0` by default and documents this in its manifest. Full paper-equivalent dense `192x192` projection still requires HLS support for cross-channel tile accumulation or a wider per-channel-tile weight schedule.

## S2 Channel-Pair Packing

The packer now supports two layout modes:

- `s1_diagonal`: existing S1 ABI; packs one diagonal channel tile per cyclic block and is consumed by current HLS.
- `s2_channel_pairs`: future HLS schedule; packs all `(layer, output_channel_tile, input_channel_tile)` blocks for cross-channel accumulation.

Validation command run on 2026-06-06:

```bash
python3 hardware/tools/pack_cyclic_weights.py   --input /tmp/hgtxr_synth_weights_full.npz   --strict   --layout-mode s2_channel_pairs   --out-bin /tmp/hgtxr_cyclic_weights_s2_pairs.bin   --out-npz /tmp/hgtxr_cyclic_weights_s2_pairs.npz   --manifest /tmp/hgtxr_cyclic_weights_s2_pairs_manifest.json
```

Result: `216` blocks, `784` words per block, `169,344` total words, `0` fallbacks. Direct execution of the three packer test functions passed because pytest is still unavailable in this shell.


## S2 HLS Consistency Check

The host `s2_channel_pairs` block order is now mirrored by HLS helper code in `hgtxr_cyclic_weight_layout.hpp` and `hgtxr_cyclic_s2_projection.hpp`. The checked formula is:

```text
pair_idx = (layer * channel_tiles + output_tile) * channel_tiles + input_tile
word_base = pair_idx * HGTXR_CYCLIC_WEIGHT_LAYER_STRIDE_WORDS
```

`tb_cyclic_s2_projection.cpp` builds a tiny packed weight buffer with two input-tile blocks for one output tile, calls the new wrapper once with `clear=true` and once with `clear=false`, and verifies the expected accumulated partial sums.

This validates host/HLS indexing and accumulation semantics for a single matrix projection. It does not yet validate the full cyclic Transformer top schedule or DeiT `mlp_ratio=4` hidden-tile expansion.


## S2 First-Step Top Consumption

The S2 host layout is now consumed by a gated top-level first-step mode. `hardware/configs/zcu104_cyclic_s2_first_step_defines.h` enables `HGTXR_ENABLE_CYCLIC_S2_FIRST_STEP=1`, and `hgtxr_top.cpp` applies the packed channel-pair blocks to `WO` projection accumulation. This verifies that the packer manifest order and HLS word-base formula agree beyond the standalone primitive.

This does not yet require changing the packer format. The next packer-facing requirement is adding explicit metadata/tests for QKV and `mlp_ratio=4` hidden-tile traversal once those HLS paths are implemented.


## S2 QKV First-Step Top Consumption

The S2 packed layout is now consumed by two gated top modes: `WO` first-step and QKV first-step. QKV mode reads WQ/WK/WV offsets from each `(layer, output_channel_tile, input_channel_tile)` block and accumulates all input channel tiles before writing a merged debug/validation value.

The packer format is unchanged because the existing S2 block payload already contains WQ/WK/WV/WO/W1/W2 offsets. Future packer-facing work is explicit MLP hidden-tile metadata for `mlp_ratio=4` once S2 MLP is implemented.


## S2 Attention First-Step Top Consumption

The S2 packed layout is now consumed by a third gated top mode, `HGTXR_ENABLE_CYCLIC_S2_ATTN_FIRST_STEP`. This mode reads WQ/WK/WV channel-pair blocks, keeps Q/K/V tiles live through `attention_tile`, and then reads WO channel-pair blocks for output projection.

The packer format is still unchanged: the current block payload already carries WQ/WK/WV/WO offsets. The next packer-facing requirement remains explicit metadata/tests for MLP hidden-tile traversal and `mlp_ratio=4`, plus trained-weight golden-vector comparison once the full paper block schedule is present.


## S2 MLP Hidden-Pair Packing

`hardware/tools/pack_cyclic_weights.py` now supports `--layout-mode s2_mlp_hidden_pairs`. This mode emits W1 and W2 blocks for explicit hidden-tile traversal and records `hidden_dim`, `hidden_tiles`, and per-block `phase` metadata in the manifest.

The mode is separate from `s2_channel_pairs` because MLP hidden tiles are sized by `mlp_ratio`, not by the model channel-tile count. For the ZCU104 first-step config, `embed_dim=192`, `tile_ff=32`, and `mlp_ratio=4` imply `hidden_dim=768` and `hidden_tiles=24`.

Validation completed with packer self-test and direct execution of the S1, S2 channel-pair, and S2 MLP hidden-pair test functions.


## S2 Residual-Aware Block First-Step Packing

`hardware/tools/pack_cyclic_weights.py` now supports `--layout-mode s2_block_first_step`. This mode emits a fused per-layer schedule: attention channel-pair blocks first, then MLP W1 hidden-pair blocks, then MLP W2 hidden-pair blocks. Each manifest record includes a `phase` field so the host can audit whether a block is consumed by attention, W1, or W2.

For the ZCU104 first-step config, the expected fused block count is `6 * (6 * 6 + 2 * 6 * 24) = 1,944` blocks. Direct packer testing passed on a smaller synthetic case with the same ordering rule.

## S2 Pre-LayerNorm Packing Note

No new host layout mode is required for pre-LN. The `s2_block_first_step` mode already stores norm1/norm2 vectors in every attention phase pair block through `build_pair_block`; the HLS block path consumes the `input_tile=0` duplicate for each output channel tile. Manifest `phase` records remain unchanged.

## S2 Pre-LayerNorm Validation Artifact

`hardware/tools/validate_s2_block_preln.py` now exercises the S2 block packing contract directly. It uses synthetic norm1/norm2 vectors, reconstructs the HLS `output_tile/input_tile=0` addressing rule from packed blocks, and writes `docs/resources/s2_block_preln_validation_2026_06_06.json`.

## S2 Full-Block Packing Validation

`hardware/tools/validate_s2_block_full.py` now validates the full packed S2 block contract, not only phase ordering. It reconstructs HLS-layout dense matrices from attention, W1, and W2 phase blocks, runs the full approximation reference, and writes `docs/resources/s2_block_full_validation_2026_06_06.json`.



## S2 External Weight/Golden Harness

The full-block validator now keeps the synthetic default path but can also run the same packed-vs-direct fused block check from external tensors. New inputs are --input for weight maps, --tokens-input for token vectors, --expected-output for a software golden, and --out-output for generated per-layer outputs. Supported external array formats are .npy, .npz, and .json; weight maps reuse pack_cyclic_weights.py and therefore support .npz, .json, .pt, and .pth when the runtime has the needed loader dependencies.

The checked path remains the current HLS approximation contract: piecewise pre-LN rsqrt, approximate attention exponential, hard-sigmoid GELU, and two residual adds. It still compares pre-quantized packed tile floats, so fixed-point quantization and HLS csim equivalence are separate follow-up gates.

Validation completed with a synthetic external .npz weight map plus token input, generated --out-output, then re-ran with that output as --expected-output. Result: used_synthetic_payload=false, expected_output_compare=all_layers, and max_expected_output_error=0.0.

Existing candidate hardware/refs/weights/software_initial_weights.pt was probed, but this WSL Python environment does not have torch. The validator correctly stopped with: Loading .pt/.pth requires torch; export to .npz or install torch. Next trained checkpoint step is to export the candidate checkpoint to .npz or run under a torch-enabled Python environment.


## S2 Software-Initial Packed Weight Artifact

The project software_initial_weights.pt checkpoint has been packed into the S2 fused block-first-step weight ABI using the torch-enabled HGTXR .venv. The generated artifacts are:

- hardware/refs/weights/cyclic_weights_s2_block_software_initial.bin
- hardware/refs/weights/cyclic_weights_s2_block_software_initial.npz
- hardware/refs/weights/cyclic_weights_s2_block_software_initial_manifest.json

Manifest summary: layout_mode=s2_block_first_step, block_count=1944, block_word_stride=784, word_count=1524096, hidden_tiles=24, fallback_count=0. The binary size is 24385536 bytes. This is the first real checkpoint-backed packed S2 block artifact; it is still a first-step approximation path, not final HLS/PyTorch equivalence.


## Q4 Packed Weight ABI - 2026-06-08

The packed weight ABI now supports a width separate from activation/data width. HGTXR_WEIGHT_BIT_WIDTH controls packed weight lanes and HGTXR_BIT_WIDTH controls activation/data fixed-point type. For the current ZCU104 S2 block baseline, activation/data is Q8 and packed weight is Q4.

Generated Q4 software-initial artifacts:

- hardware/refs/weights/cyclic_weights_s2_block_software_initial_q4.bin
- hardware/refs/weights/cyclic_weights_s2_block_software_initial_q4.npz
- hardware/refs/weights/cyclic_weights_s2_block_software_initial_q4_manifest.json

Manifest summary: layout_mode=s2_block_first_step, block_count=1944, block_word_stride=196, word_count=381024, hidden_tiles=24, fallback_count=0. Binary size is 6096384 bytes. Q4 round-trip error is documented in docs/resources/q4w8a_s2_block_quantization_2026_06_08.json. The dedicated Q4W/Q8A HLS solution solution_cyclic_s2_block_q4w8a passed csynth at 267.02 MHz with 211 BRAM_18K, 76 DSP, 31674 FF, 57932 LUT, and 0 URAM; HLS csim remains blocked by the current Ubuntu/WSL header environment.

## 2026-06-10 Head64 Q4W/Q8A Packed Weights

Generated a head-aligned Q4 packed-weight set for the current S2 block baseline:

- Binary: hardware/refs/weights/cyclic_weights_s2_block_software_initial_q4_head64.bin.
- NPZ: hardware/refs/weights/cyclic_weights_s2_block_software_initial_q4_head64.npz.
- Manifest: hardware/refs/weights/cyclic_weights_s2_block_software_initial_q4_head64_manifest.json.
- Layout: s2_block_first_step, embed_dim=192, tile_channels=64, tile_ff=64, bus_width=128, bit_width=4, int_width=2, blocks=6, mlp_ratio=4.
- Manifest summary: block_count=486, word_count=377136, fallback_count=0.
- Companion validation: docs/resources/s2_block_full_q4w8a_head64_validation_2026_06_10.json.

