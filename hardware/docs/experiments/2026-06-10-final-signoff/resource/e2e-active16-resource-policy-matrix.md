# E2E Active16 Resource Policy Matrix - 2026-06-10

Scope: `hardware/` Vitis HLS CSynth for `HGTXR_E2E_SCALE=hgpipe_math_lnq_active16`, target `xczu7ev-ffvc1156-2-e`, clock target `5.0 ns`.

Policy flags:

- `auto_bram`: `HGTXR_E2E_FORCE_DSP_MUL=0`, `HGTXR_E2E_FORCE_URAM_BUFFERS=0`
- `dsp_bram`: `HGTXR_E2E_FORCE_DSP_MUL=1`, `HGTXR_E2E_FORCE_URAM_BUFFERS=0`
- `auto_uram`: `HGTXR_E2E_FORCE_DSP_MUL=0`, `HGTXR_E2E_FORCE_URAM_BUFFERS=1`
- `dsp_uram`: `HGTXR_E2E_FORCE_DSP_MUL=1`, `HGTXR_E2E_FORCE_URAM_BUFFERS=1`

| Policy | Clock est. | Latency cycles | Latency abs. | BRAM_18K | DSP | FF | LUT | URAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `auto_bram` | 4.058 ns | 2,844,464 | 14.222 ms | 209 | 55 | 29,018 | 47,745 | 0 |
| `dsp_bram` | 4.058 ns | 2,841,904 | 14.210 ms | 209 | 55 | 27,574 | 44,670 | 0 |
| `auto_uram` | 4.058 ns | 2,844,464 | 14.222 ms | 39 | 55 | 29,018 | 47,745 | 80 |
| `dsp_uram` | 4.058 ns | 2,841,904 | 14.210 ms | 39 | 55 | 27,574 | 44,670 | 80 |

Interpretation:

- URAM steering moves large activation/global buffers from BRAM to URAM: `209 -> 39 BRAM_18K`, `0 -> 80 URAM`.
- DSP steering does not increase total DSP beyond `55`; Vitis already maps these fixed-point multiplies to DSP in auto mode. It still reduces FF/LUT and latency through the explicit DSP helper structure: `29,018 -> 27,574 FF`, `47,745 -> 44,670 LUT`, `2,844,464 -> 2,841,904 cycles`.
- Recommended resource policy for the user's current concern is `dsp_uram`: lowest LUT/FF with high URAM use.
- Remaining pressure is not a failed DSP bind. Next bottleneck is weight AXI port scheduling and URAM read-first/II warning; optimize packed-weight prefetch/cache and selective BRAM for hot scratch buffers.

Archived reports:

- `/tmp/hgtxr_e2e_axis_top_hgpipe_math_lnq_active16_auto_bram_csynth.rpt`
- `/tmp/hgtxr_e2e_axis_top_hgpipe_math_lnq_active16_dsp_bram_csynth.rpt`
- `/tmp/hgtxr_e2e_axis_top_hgpipe_math_lnq_active16_auto_uram_csynth.rpt`
- `/tmp/hgtxr_e2e_axis_top_hgpipe_math_lnq_active16_dsp_uram_csynth.rpt`

## PAR8 Packed-Weight Fast Cache Continuation

User follow-up: DSP utilization was still too low, so the active16 HG-PIPE math+LayerNormQ path was retuned from PAR5 to PAR8. Small attention scratch memories were moved to LUTRAM while large buffers stayed URAM-backed.

Implemented knobs:

- `HGTXR_PARALLELISM_FACTOR=8`
- `HGTXR_E2E_ATTN_PAR=8`
- `HGTXR_E2E_DENSE_PAR=8`
- `HGTXR_E2E_WEIGHT_VEC_CACHE=1`
- `HGTXR_E2E_WEIGHT_VEC_ALIGNED_FASTPATH=1`
- `HGTXR_E2E_SMALL_MEM_LUTRAM=1`
- `HGTXR_E2E_LN_PARAM_CACHE=1`

Same-scale comparison, `HGTXR_E2E_SCALE=hgpipe_math_lnq_active16`, `HGTXR_E2E_RESOURCE_POLICY=dsp_uram`:

| Variant | Clock est. | Latency cycles | Latency abs. | BRAM_18K | DSP | FF | LUT | URAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PAR5 `dsp_uram` | 4.058 ns | 2,841,904 | 14.210 ms | 39 | 55 | 27,574 | 44,670 | 80 |
| PAR8 vec cache interim | 4.069 ns | 816,732 | 4.084 ms | 38 | 96 | 37,909 | 65,299 | 88 |
| PAR8 aligned fast cache + LN cache | 4.069 ns | 494,968 | 2.475 ms | 42 | 128 | 19,576 | 45,431 | 88 |

Interpretation:

- DSP increased `55 -> 128` versus the PAR5 `dsp_uram` point.
- Latency improved `2,841,904 -> 494,968` cycles at the same active16 scale.
- LUT remained near the PAR5 baseline: `44,670 -> 45,431`, after the aligned packed-word fast path removed the interim LUT blow-up.
- FF decreased `27,574 -> 19,576`.
- URAM increased `80 -> 88`; this is high at 91% of the ZU7EV URAM budget, so full-scale PAR8 may need another URAM/BRAM tradeoff pass.
- `score`, `prob`, and `exp_raw` are bound as LUTRAM in the final csynth report.
- The previous packed-weight `word1` access II warning was not present in the final fast-cache report scan.

Archived reports:

- `/tmp/hgtxr_e2e_axis_top_hgpipe_math_lnq_active16_par8_vec_dsp_uram_csynth.rpt`
- `/tmp/hgtxr_e2e_axis_top_hgpipe_math_lnq_active16_par8_fastcache_dsp_uram_csynth.rpt`

## Full Active196/B6 PAR8 Selective-URAM Continuation

User follow-up: keep `PAR=8` to raise DSP use, avoid implementing arithmetic in LUTs, keep small scratch memories in LUTRAM, and use URAM for large buffers without exceeding the ZCU104 URAM budget.

New resource-policy controls:

- `dsp_mixed_hidden`: DSP MACs on, only `gb.hidden` bound to URAM.
- `dsp_mixed_stream`: DSP MACs on, `gb.q`, `gb.k`, `gb.v`, and `gb.attn` bound to URAM; `tokens`, `gb.tokens`, `gb.norm`, and `gb.hidden` stay BRAM.
- `dsp_mixed`: DSP MACs on, `gb.q`, `gb.k`, `gb.v`, `gb.attn`, and `gb.hidden` bound to URAM.

Full-scale comparison, `HGTXR_E2E_SCALE=active196_b6_ff768`, `PAR=8`:

| Policy | Fit | Clock est. | Latency cycles | Latency abs. | BRAM_18K | DSP | FF | LUT | URAM |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `dsp_uram` | no, URAM over budget | 3.744 ns | 82,612,245 | 0.413 sec | 42 | 268 | 39,370 | 90,836 | 160 |
| `dsp_bram` | yes | 3.744 ns | 82,612,245 | 0.413 sec | 338 | 268 | 39,370 | 90,836 | 0 |
| `dsp_mixed` | no, URAM over budget | 3.744 ns | 82,612,245 | 0.413 sec | 114 | 268 | 39,370 | 90,836 | 112 |
| `dsp_mixed_stream` | yes | 3.744 ns | 82,612,245 | 0.413 sec | 210 | 268 | 39,370 | 90,836 | 64 |

Interpretation:

- `dsp_mixed_stream` is the current full active196/b6 PAR8 fit point: DSP stays high at `268`, URAM rises from `0` to `64`, and LUT stays `90,836`.
- All-URAM full PAR8 is not viable on ZCU104: `160/96 URAM`.
- `dsp_mixed` still exceeds the board: `112/96 URAM`.
- Small attention scratch memories remain LUTRAM: generated RTL instantiates `score_RAM_2P_LUTRAM_1R1W`, and `prob` shares the same LUTRAM module shape.
- Remaining loop constraint failures are AXI weight-port scheduling limits, not arithmetic LUT mapping: non-HGPIPE LayerNorm final affine `II=2`, QKV packed-weight loop `II=3`.

Archived reports:

- `/tmp/hgtxr_e2e_axis_top_active196_b6_ff768_par8_dsp_uram_csynth.rpt`
- `/tmp/hgtxr_e2e_axis_top_active196_b6_ff768_par8_dsp_bram_csynth.rpt`
- `/tmp/hgtxr_e2e_axis_top_active196_b6_ff768_par8_dsp_mixed_csynth.rpt`
- `/tmp/hgtxr_e2e_axis_top_active196_b6_ff768_par8_dsp_mixed_stream_csynth.rpt`

## Full Active196/B6 PAR8 QKV Weight-Cache Continuation

User follow-up: DSP utilization was still too low after the full PAR8 selective-URAM fit, so the remaining QKV AXI weight-port bottleneck was removed by caching packed Q/K/V weight words locally.

Implemented knobs and mapping:

- `HGTXR_E2E_QKV_WEIGHT_CACHE=1`
- Q/K/V packed weight caches are local `ram_2p` BRAM arrays.
- Large streaming activation buffers remain selective-URAM under `dsp_mixed_stream`.
- Small attention scratch memories remain LUTRAM.

Full-scale comparison, `HGTXR_E2E_SCALE=active196_b6_ff768`, `PAR=8`, `HGTXR_E2E_RESOURCE_POLICY=dsp_mixed_stream`:

| Variant | Fit | Clock est. | Latency cycles | Latency abs. | BRAM_18K | DSP | FF | LUT | URAM |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| selective-URAM baseline | yes | 3.744 ns | 82,612,245 | 0.413 sec | 210 | 268 | 39,370 | 90,836 | 64 |
| LN cache continuation | yes | 3.744 ns | 82,144,418 | 0.411 sec | 210 | 268 | 42,568 | 89,756 | 64 |
| QKV BRAM weight cache | yes | 3.744 ns | 71,316,968 | 0.357 sec | 306 | 332 | 43,740 | 81,144 | 64 |

QKV local report:

| Variant | QKV latency cycles | QKV achieved II | QKV BRAM_18K | QKV DSP | QKV LUT |
|---|---:|---:|---:|---:|---:|
| before QKV cache | 2,709,518 | 3 | 0 | 16 | 12,728 |
| after QKV cache | 904,943 | 1 | 48 | 48 | 8,422 |

Interpretation:

- The QKV packed-weight hot loop moved from `II=3` to `II=1`.
- DSP increased `268 -> 332`, while LUT decreased `90,836 -> 81,144`.
- URAM stayed at `64/96`, preserving the selected ZCU104 fit point.
- BRAM increased `210 -> 306`, still within the ZU7EV budget `624`.
- The top-level timing estimate is still `3.744 ns`, with residual CSynth timing slack around `-0.09 ns` in the summary report.
- Remaining loop-constraint issue is the tiny LayerNorm gamma/beta preload loop at `II=2` with tripcount `3`; the LayerNorm affine/data loops are `II=1`.

Archived reports:

- `/tmp/hgtxr_e2e_axis_top_active196_b6_ff768_par8_dsp_mixed_stream_lncached_csynth.rpt`
- `/tmp/hgtxr_e2e_axis_top_active196_b6_ff768_par8_dsp_mixed_stream_qkvcache_csynth.rpt`
