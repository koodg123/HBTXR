# P2-ViT Scale Calibration Report

- status: `pass`
- decision: `keep_current_pot_scales_for_c3b`
- checks: `10/10`
- fail_count: `0`
- weight_bits: `4`
- activation_bits: `8`
- total_candidates: `45`
- next_action: `keep current PoT scales for C3b; use non-current rows only as explicit successor experiments`

## Per Spec

| Spec | Recommended | Candidates | Exact Matches |
|---|---|---:|---:|
| e2e_axis_vector_hgpipe_math_spec.json | `current` | 15/15 | 10 |
| e2e_axis_vector_hgpipe_math_lnq_spec.json | `current` | 15/15 | 9 |
| e2e_axis_vector_hgpipe_math_lnq_active16_spec.json | `current` | 15/15 | 5 |

## Checks

| Check | Status | Detail |
|---|---|---|
| req5_q4q8_pass | `pass` | pass |
| req5_precision_q4w8a | `pass` | {'activation_bits': 8, 'activation_format': 'q8/hgtxr_data_t boundary per config', 'weight_bits': 4, 'weight_format': 'signed-q4-packed-in-uint32/AXI-256'} |
| req5_csim_coverage | `pass` | pass=11 csim_logs=3 |
| pot_scale_readiness_pass | `pass` | status=pass fail=0 |
| pot_scale_readiness_checks | `pass` | 14 |
| scale_sweep_pass | `pass` | status=pass fail=0 |
| scale_sweep_candidate_coverage | `pass` | specs=3 candidates=45 |
| scale_decision_current_for_c3b | `pass` | {'next_action': 'keep current PoT scales for C3b; use non-current rows only as explicit successor experiments', 'specs_recommending_current': 3, 'total_candidate_count': 45, 'total_fail_count': 0} |
| successor_only_promotion_policy | `pass` | {'c3b_protection': 'no HLS/Vivado run and no C3b artifact overwrite', 'promotion_gate': 'candidate must regenerate golden header and pass CSim before HLS macro promotion', 'scope': 'SW-only power-of-two scale candidate sweep for reduced E2E vector references', 'source': 'P2-ViT', 'vref': 'VREF-P0-01'} |
| no_hardware_side_effects | `pass` | sweep={'executes_hls': False, 'executes_vivado': False, 'overwrites_c3b_artifacts': False, 'writes_hls_source': False} audit={'executes_hls': False, 'executes_vivado': False, 'overwrites_c3b_artifacts': False, 'writes_hls_source': False} |

## Safety

- executes_hls: `False`
- executes_vivado: `False`
- writes_hls_source: `False`
- overwrites_c3b_artifacts: `False`
- creates_board_result: `False`
- creates_xr_vits_policy: `False`
