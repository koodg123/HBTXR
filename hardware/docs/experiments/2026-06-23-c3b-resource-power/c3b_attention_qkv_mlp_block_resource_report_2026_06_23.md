# C3b Attention/QKV/MLP/Block Resource Report

Date: 2026-06-23

## 1. Scope

This report decomposes the current C3b E2E HLS synthesis result down to:

- physical C3b controller
- logical transformer block
- attention unit
- QKV projection
- attention core, including score/softmax/PV sub-pipelines
- MLP unit

Evidence source:

- `generated/hgtxr_e2e_axis_par16_c3b_mem16/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`
- `generated/hgtxr_e2e_axis_par16_c3b_mem16/solution_e2e_q4w8a/.autopilot/db/*.verbose.rpt`
- `generated/hgtxr_e2e_axis_par16_c3b_mem16/hls.app`
- `hls/include/hgtxr_e2e_vit.hpp`

Target resource denominator from HLS report:

| Resource | Available |
|---|---:|
| BRAM_18K | 624 |
| DSP | 1,728 |
| FF | 460,800 |
| LUT | 230,400 |
| URAM | 96 |

## 2. Build Fingerprint

| Item | Value |
|---|---|
| HLS project | `hgtxr_e2e_axis_par16_c3b_mem16` |
| Solution | `solution_e2e_q4w8a` |
| Vitis HLS | 2023.2 |
| Target device | `xczu7ev-ffvc1156-2-e` |
| Clock target | 5.00 ns |
| Estimated top clock | 3.953 ns |
| Active tokens | 196 |
| Patch grid | 14 x 14 |
| Blocks | 6 |
| Embed dim | 192 |
| Heads | 3 |
| Head dim | 64 |
| FF dim | 768 |
| Dense parallelism | 16 |
| Memory bank parallelism | 16 |
| QKV weight cache | enabled |
| QKV weight cache implementation | BRAM |
| Small local memory | LUTRAM |
| URAM placement | Q/K/V/attention global buffers in controller memory bucket |

The source default has `HGTXR_TOKENS = 256`, but this C3b synthesis is compiled with
`-DHGTXR_E2E_ACTIVE_TOKENS=196` and `-DHGTXR_E2E_PATCH_GRID_H=14 -DHGTXR_E2E_PATCH_GRID_W=14`.

## 3. Physical Controller Summary

The C3b controller does not instantiate six independent transformer blocks. The source loop executes
6 logical blocks and alternates two physical unit sets:

- even block: `attn_unit<0>` then `mlp_unit<0>`
- odd block: `attn_unit<1>` then `mlp_unit<1>`

This means the physical C3b controller contains two attention units and two MLP units, then reuses them
across six logical blocks.

| Scope | Latency cycles | BRAM_18K | DSP | FF | LUT | URAM |
|---|---:|---:|---:|---:|---:|---:|
| `hgtxr_e2e_axis_top` | 37,508,072 | 332 | 604 | 59,505 | 126,506 | 64 |
| `hgtxr_e2e_controller_run` | 37,296,061 | 250 | 572 | 53,901 | 114,988 | 64 |
| Controller instance subtotal, `2x attn + 2x MLP` | n/a | 90 | 572 | 53,880 | 102,864 | 0 |
| Controller memory bucket | n/a | 160 | 0 | 0 | 0 | 64 |
| Controller misc register/mux/expression | n/a | 0 | 0 | 21 | 12,124 | 0 |

Controller utilization:

| Scope | BRAM_18K | DSP | FF | LUT | URAM |
|---|---:|---:|---:|---:|---:|
| `hgtxr_e2e_controller_run` absolute | 250 | 572 | 53,901 | 114,988 | 64 |
| `hgtxr_e2e_controller_run` percentage | 40.1% | 33.1% | 11.7% | 49.9% | 66.7% |

Interpretation:

- All 572 controller DSPs are explained by two physical copies of `(attention 207 + MLP 79)`.
- The 64 URAMs are not attributed to attention/QKV/MLP function instances. They are attributed to
  controller local/global buffer memories.
- Controller memory bucket is `160 BRAM_18K + 64 URAM`, with total memory shape reported as
  `442,368 words`, `768 bits`, `96 banks`.

## 4. Logical Transformer Block

One logical block is one attention unit followed by one MLP unit. HLS controller loop reports block
iteration latency as `6,216,010` cycles. The direct module sum is `2,247,180 + 3,968,826 = 6,216,006`
cycles; the 4-cycle difference is controller-level loop overhead.

| Logical block component | Latency cycles | Latency @ 200 MHz | BRAM_18K | DSP | FF | LUT | URAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| Attention unit | 2,247,180 | 11.236 ms | 45 | 207 | 16,800 | 34,434 | 0 |
| MLP unit | 3,968,826 | 19.844 ms | 0 | 79 | 10,140 | 16,998 | 0 |
| Logical block subtotal | 6,216,006 | 31.080 ms | 45 | 286 | 26,940 | 51,432 | 0 |
| Observed controller loop iteration | 6,216,010 | 31.080 ms | n/a | n/a | n/a | n/a | n/a |

Logical block subtotal as percentage of target device:

| BRAM_18K | DSP | FF | LUT | URAM |
|---:|---:|---:|---:|---:|
| 7.2% | 16.6% | 5.8% | 22.3% | 0.0% |

## 5. Attention Unit Breakdown

Both `hgtxr_e2e_attn_unit_0_s` and `hgtxr_e2e_attn_unit_1_s` have identical synthesis results.

| Attention sub-block | Source role | Latency cycles | BRAM_18K | DSP | FF | LUT | URAM |
|---|---|---:|---:|---:|---:|---:|---:|
| LayerNorm | LN1 over active tokens | 117,619 | 0 | 15 | 2,750 | 3,160 | 0 |
| QKV projection | dense Q/K/V projection and QKV weight caches | 453,359 | 45 | 96 | 5,914 | 12,715 | 0 |
| Attention core | QK score, softmax/probability, PV accumulation | 1,186,585 | 0 | 64 | 6,086 | 7,094 | 0 |
| Output projection | WO projection and residual add | 489,609 | 0 | 32 | 1,883 | 8,070 | 0 |
| Attention unit misc overhead | unit-level expression/mux/register/mul glue | n/a | 0 | 0 | 167 | 3,395 | 0 |
| Attention unit total | HLS total | 2,247,180 | 45 | 207 | 16,800 | 34,434 | 0 |

Attention resource interpretation:

- DSP split is exactly `15 + 96 + 64 + 32 = 207`.
- QKV projection is the largest DSP consumer inside attention.
- Output projection is LUT-heavy relative to DSP: `32 DSP / 8,070 LUT`.
- QKV local caches consume all attention-local BRAM: `45 BRAM_18K`.
- No URAM is attributed inside the attention unit report.

## 6. QKV Projection Breakdown

`hgtxr_e2e_project_qkv` performs:

- Q weight cache load
- K weight cache load
- V weight cache load
- fused dense projection into Q/K/V buffers

| QKV sub-block | Latency cycles | BRAM_18K | DSP | FF | LUT | URAM |
|---|---:|---:|---:|---:|---:|---:|
| Q cache load pipeline | 579 | 0 | 0 | 291 | 83 | 0 |
| K cache load pipeline | 579 | 0 | 0 | 291 | 83 | 0 |
| V cache load pipeline | 579 | 0 | 0 | 291 | 83 | 0 |
| QKV projection pipeline | 451,590 | 0 | 96 | 4,827 | 11,719 | 0 |
| QKV expression/mux/register overhead | n/a | 0 | 0 | 214 | 747 | 0 |
| QKV local memory total | n/a | 45 | 0 | 0 | 0 | 0 |
| QKV total | 453,359 | 45 | 96 | 5,914 | 12,715 | 0 |

QKV local memory detail:

| Memory | Implementation | Words | Bits | Banks | BRAM_18K | URAM |
|---|---|---:|---:|---:|---:|---:|
| `q_weight_cache_U` | 2P BRAM | 576 | 256 | 1 | 15 | 0 |
| `k_weight_cache_U` | 2P BRAM | 576 | 256 | 1 | 15 | 0 |
| `v_weight_cache_U` | 2P BRAM | 576 | 256 | 1 | 15 | 0 |
| Total | BRAM cache | 1,728 | 768 | 3 | 45 | 0 |

Conclusion for QKV:

- Current QKV cache is explicitly BRAM, not URAM.
- If the goal is to raise URAM usage further, QKV weight cache is a candidate, but prior reports should be
  checked because moving QKV cache to URAM may trade BRAM pressure for URAM routing/latency cost.

## 7. Attention Core Breakdown

`hgtxr_e2e_attention_core` performs:

- QK score dot product
- softmax approximation/probability path
- PV/value accumulation

| Attention core sub-block | Source loop | Latency cycles | BRAM_18K | DSP | FF | LUT | URAM |
|---|---|---:|---:|---:|---:|---:|---:|
| QK score dot product | `VITIS_LOOP_949_3_VITIS_LOOP_957_5` | 790 | 0 | 32 | 1,947 | 2,629 | 0 |
| Softmax exp/sum path | `VITIS_LOOP_1011_8` | 201 | 0 | 0 | 221 | 450 | 0 |
| Softmax requant/prob path | `VITIS_LOOP_1022_9` | 230 | 0 | 0 | 2,149 | 1,597 | 0 |
| PV/value accumulation | `VITIS_LOOP_1029_10_VITIS_LOOP_1037_12` | 788 | 0 | 32 | 1,654 | 1,981 | 0 |
| Local score/prob memory | LUTRAM | n/a | 0 | 0 | 32 | 50 | 0 |
| Expression/mux/register overhead | n/a | n/a | 0 | 0 | 83 | 387 | 0 |
| Attention core total | HLS total | 1,186,585 | 0 | 64 | 6,086 | 7,094 | 0 |

Attention core loop summary:

| Loop | Latency cycles | Iteration latency | Trip count | Pipelined |
|---|---:|---:|---:|---|
| `VITIS_LOOP_943_1_VITIS_LOOP_946_2` | 1,186,584 | 2,018 | 588 | no |

Attention local memory:

| Memory | Implementation | Words | Bits | Banks | FF | LUT | BRAM_18K | URAM |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `score_U` | LUTRAM | 196 | 8 | 1 | 16 | 25 | 0 | 0 |
| `prob_U` | LUTRAM | 196 | 8 | 1 | 16 | 25 | 0 | 0 |
| Total | LUTRAM | 392 | 16 | 2 | 32 | 50 | 0 | 0 |

Conclusion for attention core:

- DSP is concentrated in QK and PV: `32 + 32 = 64`.
- Softmax/probability paths are LUT/FF based and consume no DSP.
- `score` and `prob` are successfully implemented as LUTRAM, matching the small-memory policy.

## 8. MLP Unit Breakdown

Both `hgtxr_e2e_mlp_unit_0_s` and `hgtxr_e2e_mlp_unit_1_s` have identical synthesis results.

| MLP sub-block | Source role | Latency cycles | BRAM_18K | DSP | FF | LUT | URAM |
|---|---|---:|---:|---:|---:|---:|---:|
| LayerNorm | LN2 over active tokens | 117,619 | 0 | 15 | 2,750 | 3,160 | 0 |
| W1/GELU pipeline | first FFN projection and GELU | 204 | 0 | 32 | 2,783 | 2,941 | 0 |
| W2 projection pipeline | second FFN projection and residual add | 9,230 | 0 | 32 | 4,180 | 5,476 | 0 |
| MLP LUTRAM table | GELU/local ROM table | n/a | 0 | 0 | 16 | 2 | 0 |
| MLP expression/mux/register overhead | n/a | n/a | 0 | 0 | 411 | 5,419 | 0 |
| MLP total | HLS total | 3,968,826 | 0 | 79 | 10,140 | 16,998 | 0 |

MLP outer loop summary:

The W1/GELU and W2 rows above are pipeline-module resource rows. End-to-end MLP latency is dominated by
the non-pipelined outer loops below.

| Loop | Latency cycles | Iteration latency | Trip count | Pipelined |
|---|---:|---:|---:|---|
| `VITIS_LOOP_1130_1` | 3,851,204 | 19,649 | 196 | no |
| `VITIS_LOOP_1132_2` | 10,416 | 217 | 48 | no |

MLP local memory:

| Memory | Implementation | Words | Bits | Banks | FF | LUT | BRAM_18K | URAM |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `table_1_U` | 2P LUTRAM ROM | 16 | 8 | 1 | 16 | 2 | 0 | 0 |

Conclusion for MLP:

- DSP split is `LayerNorm 15 + W1 32 + W2 32 = 79`.
- MLP latency dominates one logical block: `3,968,826 / 6,216,006 = 63.8%`.
- MLP does not consume BRAM or URAM in this report; its local table is LUTRAM.

## 9. Block-Level Bottleneck Summary

| Unit | Block latency share | DSP share in logical block | LUT share in logical block | Main bottleneck indication |
|---|---:|---:|---:|---|
| Attention unit | 36.2% | 72.4% | 67.0% | QKV + attention core DSP-heavy |
| MLP unit | 63.8% | 27.6% | 33.0% | latency-heavy despite lower DSP |

Within attention:

| Attention sub-block | Attention latency share | Attention DSP share | Attention LUT share |
|---|---:|---:|---:|
| LayerNorm | 5.2% | 7.2% | 9.2% |
| QKV projection | 20.2% | 46.4% | 36.9% |
| Attention core | 52.8% | 30.9% | 20.6% |
| Output projection | 21.8% | 15.5% | 23.4% |

## 10. Resource Mapping Assessment

Current mapping status:

| Policy target | Current C3b evidence | Status |
|---|---|---|
| Arithmetic should map to DSP, not LUT-only | Controller uses 572 DSP; QKV 96, attention core 64, MLP 79 per physical unit set | partially satisfied |
| Deep/large buffers should use URAM | Controller memory bucket uses 64 URAM | satisfied for selected global buffers |
| Small memories should use LUTRAM | attention `score/prob` and MLP table use LUTRAM | satisfied |
| QKV cache should avoid excess LUT | QKV cache uses BRAM, not LUTRAM | satisfied |
| QKV cache should raise URAM if desired | QKV cache currently BRAM, 45 BRAM_18K per attention unit | not enabled |

Important caveat:

- HLS attribution does not assign the 64 URAMs to QKV/attention/MLP function instances. The URAM appears in
  `hgtxr_e2e_controller_run` memory bucket because the large global buffers are passed through the controller
  and shared by sub-functions.

## 11. Next Optimization Reading

The report implies three concrete follow-up directions:

| Direction | Expected effect | Main risk |
|---|---|---|
| Increase MLP parallelism or pipeline the MLP outer loop | largest latency reduction opportunity because MLP is 63.8% of logical block latency | DSP/LUT increase and possible timing pressure |
| Move QKV weight caches from BRAM to URAM | increases URAM use and lowers BRAM pressure in attention | URAM port/routing pressure; may not reduce latency |
| Reduce output projection LUT overhead | may reduce attention LUT pressure without changing model math | requires operator-level rewrite or fixed-point narrowing validation |

The smallest next experiment for latency is MLP parallelism. The smallest next experiment for URAM utilization is
QKV cache URAM binding.
