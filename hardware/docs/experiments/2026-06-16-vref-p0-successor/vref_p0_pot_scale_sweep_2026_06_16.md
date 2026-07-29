# VREF-P0-01 PoT Scale Candidate Sweep

- status: `pass`
- root: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR`
- specs: `3`
- total candidates: `45`
- total fail_count: `0`
- next_action: `keep current PoT scales for C3b; use non-current rows only as explicit successor experiments`

## Policy

- vref: `VREF-P0-01`
- source: `P2-ViT`
- scope: `SW-only power-of-two scale candidate sweep for reduced E2E vector references`
- promotion_gate: `candidate must regenerate golden header and pass CSim before HLS macro promotion`
- c3b_protection: `no HLS/Vivado run and no C3b artifact overwrite`

## e2e_axis_vector_hgpipe_math_spec.json

- baseline_expected_raw: `[28, -15, 12, -8, 10, -12]`
- baseline_runtime_state: `2`
- recommended_candidate: `current`
- recommendation: `keep_current_scales`
- candidates: `15/15 pass`

| Candidate | Status | Raw L1 Delta | Max Raw Delta | Exact Raw Match | Scales |
|---|---|---:|---:|---|---|
| all_input_x2 | `pass` | 0 | 0 | `True` | geluq_input_scale=32, geluq_output_scale=4, softmax_input_scale=32, softmax_prob_scale=4, layernorm_input_scale=32, layernorm_output_scale=4 |
| current | `pass` | 0 | 0 | `True` | geluq_input_scale=16, geluq_output_scale=4, softmax_input_scale=16, softmax_prob_scale=4, layernorm_input_scale=16, layernorm_output_scale=4 |
| gelu_input_x0p5 | `pass` | 0 | 0 | `True` | geluq_input_scale=8, geluq_output_scale=4, softmax_input_scale=16, softmax_prob_scale=4, layernorm_input_scale=16, layernorm_output_scale=4 |
| gelu_input_x2 | `pass` | 0 | 0 | `True` | geluq_input_scale=32, geluq_output_scale=4, softmax_input_scale=16, softmax_prob_scale=4, layernorm_input_scale=16, layernorm_output_scale=4 |
| layernorm_input_x0p5 | `pass` | 0 | 0 | `True` | geluq_input_scale=16, geluq_output_scale=4, softmax_input_scale=16, softmax_prob_scale=4, layernorm_input_scale=8, layernorm_output_scale=4 |
| layernorm_input_x2 | `pass` | 0 | 0 | `True` | geluq_input_scale=16, geluq_output_scale=4, softmax_input_scale=16, softmax_prob_scale=4, layernorm_input_scale=32, layernorm_output_scale=4 |
| layernorm_output_x0p5 | `pass` | 0 | 0 | `True` | geluq_input_scale=16, geluq_output_scale=4, softmax_input_scale=16, softmax_prob_scale=4, layernorm_input_scale=16, layernorm_output_scale=2 |
| layernorm_output_x2 | `pass` | 0 | 0 | `True` | geluq_input_scale=16, geluq_output_scale=4, softmax_input_scale=16, softmax_prob_scale=4, layernorm_input_scale=16, layernorm_output_scale=8 |

## e2e_axis_vector_hgpipe_math_lnq_spec.json

- baseline_expected_raw: `[22, -7, 6, -4, 4, -7]`
- baseline_runtime_state: `2`
- recommended_candidate: `current`
- recommendation: `keep_current_scales`
- candidates: `15/15 pass`

| Candidate | Status | Raw L1 Delta | Max Raw Delta | Exact Raw Match | Scales |
|---|---|---:|---:|---|---|
| all_input_x2 | `pass` | 0 | 0 | `True` | geluq_input_scale=32, geluq_output_scale=4, softmax_input_scale=32, softmax_prob_scale=4, layernorm_input_scale=32, layernorm_output_scale=4 |
| current | `pass` | 0 | 0 | `True` | geluq_input_scale=16, geluq_output_scale=4, softmax_input_scale=16, softmax_prob_scale=4, layernorm_input_scale=16, layernorm_output_scale=4 |
| gelu_input_x0p5 | `pass` | 0 | 0 | `True` | geluq_input_scale=8, geluq_output_scale=4, softmax_input_scale=16, softmax_prob_scale=4, layernorm_input_scale=16, layernorm_output_scale=4 |
| gelu_input_x2 | `pass` | 0 | 0 | `True` | geluq_input_scale=32, geluq_output_scale=4, softmax_input_scale=16, softmax_prob_scale=4, layernorm_input_scale=16, layernorm_output_scale=4 |
| gelu_output_x2 | `pass` | 0 | 0 | `True` | geluq_input_scale=16, geluq_output_scale=8, softmax_input_scale=16, softmax_prob_scale=4, layernorm_input_scale=16, layernorm_output_scale=4 |
| layernorm_input_x0p5 | `pass` | 0 | 0 | `True` | geluq_input_scale=16, geluq_output_scale=4, softmax_input_scale=16, softmax_prob_scale=4, layernorm_input_scale=8, layernorm_output_scale=4 |
| layernorm_input_x2 | `pass` | 0 | 0 | `True` | geluq_input_scale=16, geluq_output_scale=4, softmax_input_scale=16, softmax_prob_scale=4, layernorm_input_scale=32, layernorm_output_scale=4 |
| softmax_input_x0p5 | `pass` | 0 | 0 | `True` | geluq_input_scale=16, geluq_output_scale=4, softmax_input_scale=8, softmax_prob_scale=4, layernorm_input_scale=16, layernorm_output_scale=4 |

## e2e_axis_vector_hgpipe_math_lnq_active16_spec.json

- baseline_expected_raw: `[58, -51, 42, -28, 36, -41]`
- baseline_runtime_state: `2`
- recommended_candidate: `current`
- recommendation: `keep_current_scales`
- candidates: `15/15 pass`

| Candidate | Status | Raw L1 Delta | Max Raw Delta | Exact Raw Match | Scales |
|---|---|---:|---:|---|---|
| current | `pass` | 0 | 0 | `True` | geluq_input_scale=16, geluq_output_scale=4, softmax_input_scale=16, softmax_prob_scale=4, layernorm_input_scale=16, layernorm_output_scale=4 |
| layernorm_input_x0p5 | `pass` | 0 | 0 | `True` | geluq_input_scale=16, geluq_output_scale=4, softmax_input_scale=16, softmax_prob_scale=4, layernorm_input_scale=8, layernorm_output_scale=4 |
| layernorm_input_x2 | `pass` | 0 | 0 | `True` | geluq_input_scale=16, geluq_output_scale=4, softmax_input_scale=16, softmax_prob_scale=4, layernorm_input_scale=32, layernorm_output_scale=4 |
| softmax_input_x0p5 | `pass` | 0 | 0 | `True` | geluq_input_scale=16, geluq_output_scale=4, softmax_input_scale=8, softmax_prob_scale=4, layernorm_input_scale=16, layernorm_output_scale=4 |
| softmax_input_x2 | `pass` | 0 | 0 | `True` | geluq_input_scale=16, geluq_output_scale=4, softmax_input_scale=32, softmax_prob_scale=4, layernorm_input_scale=16, layernorm_output_scale=4 |
| all_input_x2 | `pass` | 1 | 1 | `False` | geluq_input_scale=32, geluq_output_scale=4, softmax_input_scale=32, softmax_prob_scale=4, layernorm_input_scale=32, layernorm_output_scale=4 |
| gelu_input_x2 | `pass` | 1 | 1 | `False` | geluq_input_scale=32, geluq_output_scale=4, softmax_input_scale=16, softmax_prob_scale=4, layernorm_input_scale=16, layernorm_output_scale=4 |
| gelu_input_x0p5 | `pass` | 2 | 1 | `False` | geluq_input_scale=8, geluq_output_scale=4, softmax_input_scale=16, softmax_prob_scale=4, layernorm_input_scale=16, layernorm_output_scale=4 |

## Safety

- executes_hls: `False`
- executes_vivado: `False`
- writes_hls_source: `False`
- overwrites_c3b_artifacts: `False`
