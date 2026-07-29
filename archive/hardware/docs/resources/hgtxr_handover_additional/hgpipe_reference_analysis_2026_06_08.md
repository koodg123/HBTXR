# HG-PIPE Reference Analysis for ZCU104 Cyclic Baseline - 2026-06-08

## Scope

This note records the HG-PIPE artifacts inspected under ../impl_repos/HGPIPE/ICCAD24-HG-PIPE for the current HGTXR ZCU104 cyclic accelerator baseline. The current milestone is not full paper reproduction. The baseline target is an E2E-capable ViT-like cyclic block path with Q4 packed weights, Q8 activation typing, parameterized tiling/parallelism/bus/buffer/FIFO knobs, and measurable SW/HW agreement gates.

## Referenced HG-PIPE Files

- case/LAYERNORM_2X2.cpp: LayerNorm reference with TP=2, CP=2, C=192, table-based rsqrt, affine scale/bias, and integer output type.
- case/GELU.cpp: GeLU reference with T=196, C=768, CP=2, 64-entry table, and narrow integer output type.
- case/QUANT.cpp: Quantization reference with T=196, C=192, CP=2, 64-entry table, scalar/table-driven quant conversion.
- case/ATTN.cpp.template: attention stage template with explicit head count H=3, T=196, C=192, Q/K/V/O matmul parallel factors, quantization stages, and softmax table hooks.
- case/MLP.cpp.template: MLP stage template with C=192, CH=768, M1/M2 parallel factors, LayerNorm, GeLU table, and no-DSP weight options.
- case/refs: per-layer reference vectors/tables for LN, Q/K/V/O matmul, softmax, GeLU, and quant outputs.

## Baseline Decisions Reflected in HGTXR

- Keep HGTXR cyclic block as a resource-fit baseline rather than trying to clone every HG-PIPE stage at once.
- Preserve parameterized knobs: HGTXR_TILING_FACTOR, HGTXR_PARALLELISM_FACTOR, HGTXR_BUS_WIDTH, HGTXR_BIT_WIDTH, HGTXR_WEIGHT_BIT_WIDTH, HGTXR_BUFFER_SIZE, and HGTXR_FIFO_DEPTH.
- Split activation/data width from packed weight width. The ZCU104 S2 block config now sets HGTXR_BIT_WIDTH=8 for activation/data and HGTXR_WEIGHT_BIT_WIDTH=4 with HGTXR_WEIGHT_INT_WIDTH=2 for packed weights.
- Keep the current HLS LayerNorm, Softmax, and GeLU approximations as first-step equivalents, while recording exact PyTorch and quantization error separately.
- Use HG-PIPE as the next refinement source for table-driven rsqrt, GeLU, softmax reciprocal/exp, and per-stage quant tables.

## Remaining Integration Gates

- Replace the current piecewise approximations with table-driven variants compatible with HG-PIPE refs where resource/timing permits.
- Add exact SW/HLS csim vector comparison for Q4W/Q8A packed weights. The isolated head-attention fixed-point comparator exists, and the dedicated internal-stage S2 block Vitis csim vector comparator now passes; public E2E hgtxr_top token-output equivalence remains pending.
- Multi-head S2 attention tile alignment and 1/sqrt(head_dim) scaling are implemented for the Q4W/Q8A cyclic block baseline; exact paper-equivalent vector agreement remains pending.
- Run Vitis HLS csynth for the Q4W/Q8A S2 block config and compare ZCU104 resource fit against the DeiT-Tiny C-Syn reference image.

## 2026-06-10 HG-PIPE GeLUQ64 Cursor Contract

The HG-PIPE `mlp_1_geluq` slice is now captured as an exact integer cursor/table contract rather than only as a qualitative reference:

- Source harness: `/home/kjm26/project/PRJXR/XR-VIT/HGPIPE/ICCAD24-HG-PIPE/case/GELU.cpp`.
- Source implementation: `/home/kjm26/project/PRJXR/XR-VIT/HGPIPE/ICCAD24-HG-PIPE/src/gelu.h`.
- Input type: `ap_int<12>`; cursor type: `ap_int<9>`; output type: `ap_uint<3>`.
- Shape: `T=196`, `C=768`, total samples `150528`.
- Scalars: `b=94`, `s=2`, `bound=63`.
- Cursor expression: `clamp((x + b) >> s, 0, bound)`.
- Table: 64 integer entries from `case/refs/mlp_1_geluq_table_m.txt`.

HGTXR now has an isolated primitive, `hgtxr_hgpipe_geluq64_int()`, and `hardware/tools/validate_hgpipe_lut_math.py` validates all HG-PIPE `mlp_1_geluq_input.txt` / `mlp_1_geluq_output.txt` samples with `mismatches=0`.

Integration boundary: this primitive is intentionally not wired into the default E2E GeLU path yet. HG-PIPE `mlp_1_geluq` emits quantized `ap_uint<3>` table values, while the current E2E GeLU path uses activation-scale data. Wiring it into E2E requires updating the software mirror and generated golden vectors together.

## 2026-06-10 HG-PIPE Quant attn0_q Cursor Contract

The HG-PIPE `attn_0_q_q` quantization slice is now captured as an exact signed integer cursor/table contract:

- Source harness: `/home/kjm26/project/PRJXR/XR-VIT/HGPIPE/ICCAD24-HG-PIPE/case/QUANT.cpp`.
- Source implementation: `/home/kjm26/project/PRJXR/XR-VIT/HGPIPE/ICCAD24-HG-PIPE/src/quant.h`.
- Input type: `ap_int<11>`; cursor type: `ap_int<8>`; output type: `ap_int<3>`.
- Shape: `T=196`, `C=192`, total samples `37632`.
- Scalars: `b=88`, `s=2`, `bound=63`.
- Cursor expression: `clamp((x + b) >> s, 0, bound)`.
- Table: 64 signed integer entries from `case/refs/attn_0_q_q_table_m.txt`, output range `[-4, 3]`.

HGTXR now has an isolated primitive, `hgtxr_hgpipe_attn0_q_quant64_int()`, and `hardware/tools/validate_hgpipe_lut_math.py` validates all HG-PIPE `attn_0_qq_input.txt` / `attn_0_qq_output.txt` samples with `mismatches=0`.

Integration boundary: this primitive proves one attention-Q quant table and does not yet select or wire the full Q/K/V/O/MLP per-layer quant table set into default E2E.

## 2026-06-10 HG-PIPE Softmax attn0 Cursor/Recip Contract

The HG-PIPE `attn_0_softmaxq` slice is now captured as an exact rowwise integer softmax contract:

- Source harness: `/home/kjm26/project/PRJXR/XR-VIT/HGPIPE/ICCAD24-HG-PIPE/case/SOFTMAX_2X1.cpp`.
- Source implementation: `/home/kjm26/project/PRJXR/XR-VIT/HGPIPE/ICCAD24-HG-PIPE/src/softmax.h`.
- Shape: `H=3`, `T=196`, `C=196`, `TP=2`, `CP=1`, total output samples `115248`.
- Table sizes: exp `32`, reciprocal table one `64`, reciprocal table two `64`.
- Scalar order: `b1,s1,bound1,b2_one,s2_one,bound2_one,b3_one,s3_one,b2_two,s2_two,bound2_two,b3_two,s3_two,clamp_bits`.
- Scalars: `[8,4,31,-25242,14,63,32768,16,-596612,17,63,524288,20,3]`.
- Algorithm: row max, opposite-delta exp lookup, row sum, reciprocal table branch, multiply, branch-specific requant, unsigned 3-bit clamp.

HGTXR now has isolated softmaxq helper blocks for exp lookup, reciprocal branch/lookup, and uint3 requant. `hardware/tools/validate_hgpipe_lut_math.py` replays the full `attn_0_softmaxq_input.txt` / `attn_0_softmaxq_output.txt` reference with `mismatches=0`.

Integration boundary: this integer 32-entry exp plus reciprocal path is distinct from the current HGTXR compact 16-entry float-style exp approximation. Default E2E wiring requires a matching software mirror and regenerated golden vectors.

## 2026-06-10 HG-PIPE LayerNorm attn1 Contract

The HG-PIPE `attn_1_lnq` slice is now captured as an exact rowwise integer LayerNorm contract:

- Source harness: `/home/kjm26/project/PRJXR/XR-VIT/HGPIPE/ICCAD24-HG-PIPE/case/LAYERNORM_2X2.cpp`.
- Source implementation: `/home/kjm26/project/PRJXR/XR-VIT/HGPIPE/ICCAD24-HG-PIPE/src/layernorm.h`.
- Shape: `T=196`, `C=192`, `TP=2`, `CP=2`, total output samples `37632`.
- Types: input `ap_int<13>`, mean `ap_int<9>`, variance sum `ap_uint<27>`, cursor `ap_uint<7>`, rsqrt `ap_uint<12>`, affine `ap_int<38>`, output `ap_int<3>`.
- Scalars: `C_1_m=43691`, `C_1_s=23`, `b=-8585116`, `s1=20`, `bound=127`, `s2=33`, `clamp_bits=3`.
- Tables and vectors: rsqrt table `128`, `lnw` entries `192`, `lnb` entries `192`.
- Algorithm: rounded integer mean, variance-sum cursor, 128-entry rsqrt lookup, affine `(x - mean) * rsqrt * lnw + lnb`, right shift by `33`, signed 3-bit clamp.

HGTXR now has isolated helper blocks for the `attn_1_lnq` mean, rsqrt lookup, and requant path. `hardware/tools/validate_hgpipe_lut_math.py` replays the full `attn_1_lnq_input.txt` / `attn_1_lnq_output.txt` reference with `mismatches=0`.

Integration boundary: this fixed-point LayerNorm path is distinct from the current default E2E compact rsqrt/affine mirror. Default E2E wiring requires a matching software mirror and regenerated golden vectors.

## 2026-06-10 HG-PIPE Quant attn0 QKVA Contract

The HG-PIPE attention-0 quantization set is now captured as four exact signed integer cursor/table contracts:

- `attn_0_q_q`: scalars `[88, 2, 63]`, input `attn_0_qq_input.txt`, output `attn_0_qq_output.txt`.
- `attn_0_k_q`: scalars `[85, 2, 63]`, input `attn_0_kq_input.txt`, output `attn_0_kq_output.txt`.
- `attn_0_v_q`: scalars `[162, 3, 63]`, input `attn_0_vq_input.txt`, output `attn_0_vq_output.txt`.
- `attn_0_a_q`: scalars `[101, 2, 63]`, input `attn_0_aq_input.txt`, output `attn_0_aq_output.txt`.

All four use `T=196`, `C=192`, 64-entry signed `ap_int<3>` tables and the same cursor expression family: `clamp((x + b) >> s, 0, 63)`. `hardware/tools/validate_hgpipe_lut_math.py` replays all four reference groups with `mismatches=0`, covering 150,528 total samples.

Integration boundary: this proves the attention-0 Q/K/V/A quant table slices only. It does not yet select or wire all later-layer Q/K/V/A/O/MLP quant tables into default E2E.

## 2026-06-10 HG-PIPE Quant attn1 QKVA Contract

The HG-PIPE attention-1 quantization set is now captured as four exact signed integer cursor/table contracts:

- `attn_1_q_q`: scalars `[91, 2, 63]`, input range `[-271, 220]`, raw cursor range `[-45, 77]`, clamped cursor `[0, 63]`, output `[-4, 3]`.
- `attn_1_k_q`: scalars `[67, 1, 63]`, input range `[-245, 243]`, raw cursor range `[-89, 155]`, clamped cursor `[0, 63]`, output `[-4, 3]`.
- `attn_1_v_q`: scalars `[131, 2, 63]`, input range `[-318, 288]`, raw cursor range `[-47, 104]`, clamped cursor `[0, 63]`, output `[-4, 3]`.
- `attn_1_a_q`: scalars `[162, 3, 63]`, input range `[-287, 283]`, raw cursor range `[-16, 55]`, clamped cursor `[0, 55]`, output `[-4, 3]`.

All four use `T=196`, `C=192`, 64-entry signed `ap_int<3>` tables and the same cursor expression family: `clamp((x + b) >> s, 0, 63)`. Together with attention-0 Q/K/V/A, `hardware/tools/validate_hgpipe_lut_math.py` replays 301,056 attention quant samples with `mismatches=0`.

Integration boundary: this proves attention-0/1 Q/K/V/A quant table slices only. It does not yet select or wire all later-layer Q/K/V/A/O/MLP quant tables into default E2E.

## 2026-06-10 HG-PIPE GeLUQ MLP0/MLP1 Contract

The HG-PIPE quantized GeLU set now covers MLP0 and MLP1 as separate exact unsigned 3-bit cursor/table contracts:

- `mlp_0_geluq`: scalars `[76, 1, 63]`, input range `[-440, 514]`, raw cursor range `[-182, 295]`, clamped cursor `[0, 63]`, output `[0, 7]`.
- `mlp_1_geluq`: scalars `[94, 2, 63]`, input range `[-433, 403]`, raw cursor range `[-85, 124]`, clamped cursor `[0, 63]`, output `[0, 7]`.
- `mlp_10_geluq`: scalars `[-2, 1, 63]`, input range `[-357, 175]`, raw cursor range `[-180, 86]`, clamped cursor `[0, 63]`, output `[0, 7]`.
- `mlp_11_geluq`: scalars `[-5, 2, 63]`, input range `[-325, 240]`, raw cursor range `[-83, 58]`, clamped cursor `[0, 58]`, output `[0, 7]`.

All four use shape `196x768`, 64-entry `ap_uint<3>` tables, and the same cursor expression family: `clamp((x + b) >> s, 0, 63)`. `hardware/tools/validate_hgpipe_lut_math.py` replays all four reference groups with `mismatches=0`, covering 602,112 total samples.

Integration boundary: MLP0/MLP1/MLP10/MLP11 GeLUQ emit quantized unsigned 3-bit activations. Default E2E wiring requires updating the software mirror and generated golden vectors to that exact activation scale.

## 2026-06-09 Head-Aware S2 Attention Update

The cyclic S2 attention/block path now aligns channel tiles with DeiT-Tiny/HG-PIPE heads for the Q4W/Q8A block configuration:

- HGTXR_TILE_CHANNELS=64, HGTXR_CYCLIC_HEADS=3, HGTXR_CYCLIC_HEAD_DIM=64.
- HGTXR_CYCLIC_REQUIRE_HEAD_ALIGNED_ATTENTION=1 enables compile-time guards so S2 attention/block modes fail if the tile does not map to one full head.
- HGTXR_CYCLIC_ATTN_SCORE_SCALE_SHIFT=3 applies the DeiT head scale 1/sqrt(64)=1/8 through the existing attention_tile score_scale_shift path.
- The S2 attention first-step and fused block call sites now pass HGTXR_CYCLIC_ATTN_SCORE_SCALE_SHIFT instead of a hard-coded zero.

Validation evidence:

- Standalone C++ S2 Q4W/Q8A top smoke passed: state=1 out0=128.000000 out1=128.000000.
- Vitis HLS csim passed for hardware/vivado/scripts/run_cyclic_s2_block_q4w8a_csim.tcl.
- Vitis HLS csynth passed for hardware/vivado/scripts/run_cyclic_s2_block_q4w8a_csynth.tcl.
- Latest head-aligned Q4W/Q8A S2 block report: 273.67 MHz, 140,618,184..142,612,728 cycles, 0.703..0.713 sec, 231 BRAM, 80 DSP, 30,093 FF, 56,794 LUT, 0 URAM.

Remaining gate: this fixes head tiling and score scaling for the S2/cyclic baseline, but full paper-equivalent vector agreement still requires per-layer table math and direct fixed-point HW/SW comparison.

## 2026-06-09 Head Attention Vector Gate Update

A direct isolated fixed-point gate now covers the head-aligned attention tile:

- hardware/hls/tb/tb_cyclic_head_attention.cpp compiles with zcu104_cyclic_s2_block_q4w8a_defines.h and enforces TD=64, HGTXR_TILE_CHANNELS=64, and score_scale_shift=3.
- The comparator mirrors score scaling, HG-PIPE-style exp LUT lookup, softmax normalization, and value accumulation without calling attention_tile for the reference path.
- Latest evidence: head_attention_ref_max_abs_diff=0.00000000 and head_attention_scale_effect_max_abs_diff=0.68750000.

Remaining gate: lift this from host-side packed-roundtrip full-block validation into full S2 block HLS csim token-output comparison using packed Q4 weights and the pre-LN/MLP residual sequence.

## 2026-06-10 Q4W/Q8A Full-Block Host Gate Update

The full S2 host validator now follows the current head-aligned cyclic baseline more closely:

- Multi-head attention is parameterized by num_heads and score_scale_shift; the Q4W/Q8A head64 gate uses heads=3, head_dim=64, and score_scale_shift=3.
- Softmax exp and GELU use the compact HG-PIPE-style LUT path by default.
- Packed-roundtrip mode quantizes, packs, unpacks, and reconstructs matrices before running the fused S2 block reference, matching the HLS AXI weight view more closely than the prior float-only host path.
- software_initial_weights.pt has a head64 Q4 packed artifact set with word_count=377136 and fallback_count=0.
- Latest full-block host result: max_packed_roundtrip_error=0.072168730199337, max_matrix_error=0.072168730199337, max_output_error=1.5077283382415771.

This still does not prove hgtxr_top full-block csim equivalence because the top-level ABI does not expose internal token-buffer vectors. The next hardware-facing step is a debug ABI or dedicated testbench that emits the S2 block output tensor for direct comparison.

## 2026-06-10 S2 Block C++ Vector Comparator Update

A hardware-facing vector gate now exists beyond the host-only full-block validator:

- hardware/hls/tb/tb_cyclic_s2_block_vector.cpp calls the internal hgtxr_cyclic_transformer_stage<2> wrapper under the Q4W/Q8A S2 block configuration.
- The comparator consumes the actual head64 Q4 packed binary and compares all six independent layer outputs against the host golden tensor.
- Latest result: s2_block_vector_global_max_abs_diff=0.06233680, accepted under Q8 activation tolerance 0.063.

This confirms the internal S2 block wrapper tracks the head-aware packed-roundtrip host model within Q8 tolerance. The comparator has now been moved into a dedicated Vitis HLS csim_design script; the next step is a public E2E hgtxr_top token-output debug ABI or equivalent E2E vector gate.



## 2026-06-10 S2 Block Vitis CSim Gate Update

The S2 block vector comparator is now executable inside a Vitis HLS csim project:

- hardware/vivado/scripts/run_cyclic_s2_block_vector_csim.tcl creates hardware/generated/hgtxr_s2_block_vector_hls and runs csim_design for hgtxr_s2_block_vector_debug().
- hardware/hls/tb/tb_cyclic_s2_block_vector_hls_main.cpp supplies the Vitis testbench main while the original comparator still supports standalone native compilation.
- Latest Vitis evidence: CSim done with 0 errors and s2_block_vector_global_max_abs_diff=0.06233680 under tolerance 0.063.

This closes the dedicated S2 internal-stage Vitis csim vector gate. It still does not prove full public E2E top equivalence because hgtxr_top does not yet expose the complete token-buffer trajectory for comparison.
