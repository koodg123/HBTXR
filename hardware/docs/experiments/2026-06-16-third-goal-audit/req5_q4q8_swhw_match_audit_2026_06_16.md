# Req5 Q4/Q8 SW-HW Match Audit

- status: `pass`
- weight_bits: `4`
- activation_bits: `8`
- pass: `11`
- fail: `0`

## Checks

| Check | Status | Detail |
|---|---|---|
| config_q4w_q8a | `pass` | weight_bits=4 activation_bits=8 |
| packed_weight_manifest | `pass` | refs/weights/e2e_m_axi_active196_b6_ff768_q4_u32_manifest.json |
| packed_weight_binary_exists | `pass` | /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/refs/weights/e2e_m_axi_active196_b6_ff768_q4_u32.bin |
| packed_weight_sha256_matches_manifest | `pass` | 45b7aef9446c512cf74030975f55cd1f7407f2a27c2ea75603353132489d0a31 |
| packed_weight_byte_count_matches_manifest | `pass` | binary=12192768 manifest=12192768 |
| packed_weight_expected_runtime_state | `pass` | 2 |
| packed_weight_expected_c3b_output | `pass` | [32, -13, 26, -6, 14, -11] |
| testbench_strict_golden_compare | `pass` | {'axis_strict_golden': True, 'm_axi_strict_golden': True} |
| c3b_axis_csim_strict_csim | `pass` | runtime=2 failures=0 outputs=[18, -3, 3, -2, 1, -4] |
| vref_softmax_input_x2_csim_strict_csim | `pass` | runtime=2 failures=0 outputs=[58, -51, 42, -28, 36, -41] |
| qkv_uram_csim_strict_csim | `pass` | runtime=2 failures=0 outputs=[58, -51, 42, -28, 36, -41] |

## CSim Logs

| Name | Status | Outputs |
|---|---|---|
| c3b_axis_csim | `pass` | `[18, -3, 3, -2, 1, -4]` |
| vref_softmax_input_x2_csim | `pass` | `[58, -51, 42, -28, 36, -41]` |
| qkv_uram_csim | `pass` | `[58, -51, 42, -28, 36, -41]` |

## Remaining Scope
- ZCU104 physical board smoke remains separate Req4/final gate.
- This is not exhaustive over every possible arbitrary weight tensor.
