# HGTXR Cyclic Packed Weight Layout

Date: 2026-06-06
Scope: `HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS=1` S1 baseline

## Contract

The default PYNQ-facing `hgtxr_top` ABI is unchanged while `HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS=0`. When `HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS=1`, `hgtxr_top` adds one read-only AXI master pointer:

```cpp
const hgtxr::cyclic_transformer::HgtxrAxiWordT *cyclic_weights
```

The port is mapped to `gmem_cyclic_weights` and is also exposed through AXI-Lite control. The S1 config keeps `HGTXR_BUS_WIDTH=128` and `HGTXR_BIT_WIDTH=16`, so each AXI word carries `8` packed signed fixed-point lanes.

## Tile-Local Block Layout

Each cyclic layer block stores one tile-local Transformer weight set. Offsets below are in data elements, not AXI words. Word index is `element_offset / HGTXR_AXI_WEIGHT_LANES`; lane index is `element_offset % HGTXR_AXI_WEIGHT_LANES`.

| Tensor | Element offset | Shape |
|---|---:|---|
| `norm1_gamma` | `HGTXR_CYCLIC_WEIGHT_NORM1_GAMMA_OFFSET` | `[TC]` |
| `norm1_beta` | `HGTXR_CYCLIC_WEIGHT_NORM1_BETA_OFFSET` | `[TC]` |
| `norm2_gamma` | `HGTXR_CYCLIC_WEIGHT_NORM2_GAMMA_OFFSET` | `[TC]` |
| `norm2_beta` | `HGTXR_CYCLIC_WEIGHT_NORM2_BETA_OFFSET` | `[TC]` |
| `wq` | `HGTXR_CYCLIC_WEIGHT_WQ_OFFSET` | `[TC][TC]` row-major |
| `wk` | `HGTXR_CYCLIC_WEIGHT_WK_OFFSET` | `[TC][TC]` row-major |
| `wv` | `HGTXR_CYCLIC_WEIGHT_WV_OFFSET` | `[TC][TC]` row-major |
| `wo` | `HGTXR_CYCLIC_WEIGHT_WO_OFFSET` | `[TC][TC]` row-major |
| `w1` | `HGTXR_CYCLIC_WEIGHT_W1_OFFSET` | `[TC][TF]` row-major |
| `w2` | `HGTXR_CYCLIC_WEIGHT_W2_OFFSET` | `[TF][TC]` row-major |

For the S1 baseline (`TC=32`, `TF=32`), one block contains `6,272` data elements or `784` 128-bit AXI words. `HGTXR_CYCLIC_WEIGHT_BLOCKS=6` covers four search half-blocks and two track half-blocks, so the total S1 weight buffer is `4,704` AXI words.

## Current Limitation

This is a tile-local packed weight contract. It replaces synthetic/identity-like weights inside the cyclic stage, but it is not yet full global DeiT/HGTXR weight streaming across the full `192 x 192` embedding dimension. The next step is mapping host-side global weights into per-channel tile blocks and adding golden-vector checks.

## Report Evidence

`solution_cyclic_s1_weight` passed Vitis HLS csynth on 2026-06-06. Top report: `224.11 MHz`, `15,061,541 cycles` / `75.308 ms`, `279 BRAM_18K`, `42 DSP`, `30626 FF`, `54216 LUT`, `0 URAM`.

## S2 Channel-Pair Layout Extension

`hardware/tools/pack_cyclic_weights.py --layout-mode s2_channel_pairs` emits a channel-pair schedule for full cross-channel projection accumulation. The current S1 top-level kernel still consumes the diagonal S1 layout, but `hardware/hls/include/hgtxr_cyclic_weight_layout.hpp` and `hardware/hls/include/hgtxr_cyclic_s2_projection.hpp` now provide the first HLS-side S2 index/load/accumulation primitive for this contract.

Block order:

```text
for layer in [0, HGTXR_CYCLIC_WEIGHT_BLOCKS):
  for output_tile in [0, channel_tiles):
    for input_tile in [0, channel_tiles):
      block_index = (layer * channel_tiles + output_tile) * channel_tiles + input_tile
```

For `embed_dim=192` and `TC=32`, `channel_tiles=6`. With `6` cyclic blocks, S2 emits `216` blocks. Each block still has the S1 tile-local payload size of `784` 128-bit words, so the S2 packed buffer contains `169,344` words.

The S2 block records include `layer`, `input_tile`, `output_tile`, `input_channel_base`, and `output_channel_base`. A later HLS schedule should use `block_index * HGTXR_CYCLIC_WEIGHT_LAYER_STRIDE_WORDS` as the base for each pair and accumulate partial results across `input_tile` for each `output_tile`.

Current limitation: MLP hidden tiling is still represented with the same `TC/TF` tile dimensions. Full DeiT-style `mlp_ratio=4` hidden expansion requires extending the HLS schedule to iterate hidden tiles explicitly.


## S2 HLS Primitive Evidence

Added on 2026-06-06:

- `hgtxr_s2_channel_pair_block_index(layer, output_tile, input_tile, channel_tiles)` implements the host-matching block order.
- `hgtxr_s2_channel_pair_word_base(...)` maps that block index to `HGTXR_CYCLIC_WEIGHT_LAYER_STRIDE_WORDS`.
- `hgtxr_s2_project_pair_accum<TM, TI, TO>(cyclic_weights, layer, output_tile, input_tile, channel_tiles, matrix_elem_offset, x_tile, out_acc, clear)` is the thin HLS wrapper for one packed matrix projection pair.
- `HGTXR_CYCLIC_MODEL_DIM` defaults to `192` and can be overridden by generated config headers; `HGTXR_S2_DEFAULT_CHANNEL_TILES` derives from that macro and `HGTXR_TILE_MODEL_DIM`.

Verification command:

```bash
g++ -std=c++17 -I/tools/Xilinx/Vitis_HLS/2023.2/include -Ihardware/hls/include hardware/hls/tb/tb_cyclic_s2_projection.cpp -o /tmp/tb_cyclic_s2_projection
/tmp/tb_cyclic_s2_projection
```

Result: index, packed lane load, direct accumulation, and packed-wrapper `clear=true` then `clear=false` partial-sum checks all passed.

Current limitation: this is not wired into `hgtxr_top.cpp` or synthesized as a new S2 top mode yet. The next gated integration should start with one projection type, preferably `WO`, under an explicit `HGTXR_ENABLE_CYCLIC_S2_FIRST_STEP` style compile-time gate.


## S2 Top First-Step Evidence

`HGTXR_ENABLE_CYCLIC_S2_FIRST_STEP=1` now consumes the S2 channel-pair layout in `hgtxr_top.cpp` for a first-step `WO` projection path. The mode uses the same block order as the host packer and sizes `HGTXR_CYCLIC_WEIGHT_TOTAL_WORDS` for all channel-pair blocks.

For the current ZCU104 S2 config, the top expects `216` blocks and `169,344` 128-bit AXI words. Vitis HLS csynth passed in `solution_cyclic_s2_first_step` with `273.97 MHz`, `60.868..62.533 ms`, `211 BRAM_18K`, `22 DSP`, `10935 FF`, and `17997 LUT`.

Limitation: this is `WO` first-step top consumption only. The S1 full tile-local Transformer path remains separate, and full S2 QKV/attention/MLP integration is pending.


## S2 QKV Top First-Step Evidence

`HGTXR_ENABLE_CYCLIC_S2_QKV_FIRST_STEP=1` now consumes the S2 channel-pair layout for WQ/WK/WV projection accumulation in `hgtxr_top.cpp`. The same block order and word-base formula are used for all three matrix offsets.

For the current ZCU104 QKV first-step config, Vitis HLS csynth passed in `solution_cyclic_s2_qkv_first_step` with `273.97 MHz`, `0.174..0.179 sec`, `215 BRAM_18K`, `22 DSP`, `11373 FF`, and `19205 LUT`.

Limitation: Q/K/V are merged after projection only to keep the HLS paths live. Real attention, softmax, output projection, residuals, and MLP are not yet implemented in the S2 top mode.


## S2 Attention Top First-Step Evidence

`HGTXR_ENABLE_CYCLIC_S2_ATTN_FIRST_STEP=1` now consumes the same S2 channel-pair layout for a combined QKV, attention tile, and `WO` first-step path. The block order and word-base formula remain unchanged; WQ/WK/WV and WO are selected through their existing element offsets inside each block.

For the current ZCU104 attention first-step config, Vitis HLS csynth passed in `solution_cyclic_s2_attn_first_step` with `267.02 MHz`, `0.245..0.252 sec`, `245 BRAM_18K`, `36 DSP`, `24954 FF`, and `38796 LUT`.

Limitation: this validates top-level consumption of QKV, `attention_tile`, and WO over the S2 packed layout. It still does not validate residual/LayerNorm ordering, MLP hidden expansion, trained-weight numerical equivalence, final Vivado implementation, or board execution.


## S2 MLP Hidden-Pair Layout Evidence

`HGTXR_ENABLE_CYCLIC_S2_MLP_FIRST_STEP=1` uses a dedicated hidden-pair schedule instead of the QKV/WO channel-pair order. This keeps the DeiT-style `mlp_ratio=4` hidden dimension explicit.

W1 block order:

```text
for layer in [0, HGTXR_CYCLIC_WEIGHT_BLOCKS):
  for hidden_tile in [0, HGTXR_CYCLIC_HIDDEN_TILES):
    for input_tile in [0, HGTXR_CYCLIC_CHANNEL_TILES):
      block_index = layer_stride + hidden_tile * channel_tiles + input_tile
```

W2 block order:

```text
for layer in [0, HGTXR_CYCLIC_WEIGHT_BLOCKS):
  for output_tile in [0, HGTXR_CYCLIC_CHANNEL_TILES):
    for hidden_tile in [0, HGTXR_CYCLIC_HIDDEN_TILES):
      block_index = layer_stride + hidden_tiles * channel_tiles + output_tile * hidden_tiles + hidden_tile
```

For `embed_dim=192`, `TC=32`, `TF=32`, and `mlp_ratio=4`, the schedule has `6` channel tiles and `24` hidden tiles. With `6` cyclic blocks, MLP first-step expects `6 * 2 * 6 * 24 = 1,728` packed blocks.

Vitis HLS csynth passed in `solution_cyclic_s2_mlp_first_step` with `273.97 MHz`, `0.462..0.476 sec`, `243 BRAM_18K`, `30 DSP`, `12850 FF`, and `23170 LUT`.


## S2 Residual-Aware Block First-Step Layout Evidence

`HGTXR_ENABLE_CYCLIC_S2_BLOCK_FIRST_STEP=1` uses a per-layer fused packed layout:

```text
for layer in blocks:
  emit attention channel-pair blocks: output_tile, input_tile
  emit MLP W1 hidden-pair blocks: hidden_tile, input_tile
  emit MLP W2 hidden-pair blocks: output_tile, hidden_tile
```

For `embed_dim=192`, `TC=32`, `TF=32`, and `mlp_ratio=4`, each layer has `36` attention blocks plus `288` MLP blocks, or `324` packed blocks per layer. With `6` cyclic blocks, the fused mode expects `1,944` packed blocks.

Vitis HLS csynth passed in `solution_cyclic_s2_block_first_step` with `267.02 MHz`, `0.706..0.726 sec`, `319 BRAM_18K`, `52 DSP`, `29139 FF`, and `51183 LUT`.

## S2 Pre-LayerNorm Norm Parameter Use

`HGTXR_ENABLE_CYCLIC_S2_BLOCK_PRE_LN` reuses the existing `s2_block_first_step` packed layout. For each layer and channel tile, norm1/norm2 gamma and beta are read from the attention-phase channel-pair block with the matching `output_tile` and `input_tile=0`. This works because `build_pair_block` duplicates norm vectors across input tiles for a uniform block shape. The MLP-only phase blocks remain dedicated to W1/W2 payloads.

## S2 Pre-LayerNorm Golden-Vector Evidence

The validator artifact `docs/resources/s2_block_preln_validation_2026_06_06.json` confirms that norm parameters read by the HLS pre-LN path are aligned with the host `s2_block_first_step` layout. For the synthetic two-layer, two-channel-tile case, the validator reports `max_norm_param_error=0.0` and `record_count=40`.

## S2 Full-Block Golden-Vector Evidence

`docs/resources/s2_block_full_validation_2026_06_06.json` validates that the `s2_block_first_step` layout can reconstruct QKV, WO, W1, and W2 matrices for a full fused block reference. The synthetic check reports `max_matrix_error=0.0`, which confirms the block order and tile orientation for attention and MLP phases.
