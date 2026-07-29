# VREF-P0-01 PoT Scale Readiness Audit

- status: `pass`
- root: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR`
- checks: `14/14`
- fail_count: `0`

## Policy

- vref: `VREF-P0-01`
- source: `P2-ViT`
- scope: `SW-first PoT scale calibration readiness; no HLS source mutation`
- promotion_gate: `SW/HW exact-match, bounded LUT delta, DSP mapping preserved, C3b not overwritten`

## Scale Macros

| Macro | Value |
|---|---:|
| `HGTXR_WEIGHT_BIT_WIDTH` | `4` |
| `HGTXR_BIT_WIDTH` | `8` |
| `HGTXR_E2E_DENSE_PAR` | `8` |
| `HGTXR_PARALLELISM_FACTOR` | `8` |
| `HGTXR_E2E_ACC_SCALE` | `16` |
| `HGTXR_E2E_HGPIPE_GELUQ_INPUT_SCALE` | `16` |
| `HGTXR_E2E_HGPIPE_GELUQ_OUTPUT_SCALE` | `4` |
| `HGTXR_E2E_HGPIPE_SOFTMAX_INPUT_SCALE` | `16` |
| `HGTXR_E2E_HGPIPE_SOFTMAX_PROB_SCALE` | `4` |
| `HGTXR_E2E_HGPIPE_LAYERNORM_INPUT_SCALE` | `16` |
| `HGTXR_E2E_HGPIPE_LAYERNORM_OUTPUT_SCALE` | `4` |

## Checks

| Check | Status | Detail |
|---|---|---|
| q4_weight_contract | `pass` | 4 |
| q8_activation_contract | `pass` | 8 |
| dense_parallelism_positive | `pass` | 8 |
| pot_scale_HGTXR_E2E_ACC_SCALE | `pass` | 16 |
| pot_scale_HGTXR_E2E_HGPIPE_GELUQ_INPUT_SCALE | `pass` | 16 |
| pot_scale_HGTXR_E2E_HGPIPE_GELUQ_OUTPUT_SCALE | `pass` | 4 |
| pot_scale_HGTXR_E2E_HGPIPE_SOFTMAX_INPUT_SCALE | `pass` | 16 |
| pot_scale_HGTXR_E2E_HGPIPE_SOFTMAX_PROB_SCALE | `pass` | 4 |
| pot_scale_HGTXR_E2E_HGPIPE_LAYERNORM_INPUT_SCALE | `pass` | 16 |
| pot_scale_HGTXR_E2E_HGPIPE_LAYERNORM_OUTPUT_SCALE | `pass` | 4 |
| dsp_bind_op_present | `pass` | 2 |
| dsp_helper_used | `pass` | 15 |
| small_mem_lutram_enabled | `pass` | 1 |
| uram_buffer_policy_enabled | `pass` | 1 |

## Safety

- executes_hls: `False`
- executes_vivado: `False`
- writes_hls_source: `False`
- overwrites_c3b_artifacts: `False`
