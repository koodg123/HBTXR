# HG-PIPE Operator Audit

- status: `pass`
- date_tag: `2026_06_16`
- total_ref_checks: `97`
- passed_ref_checks: `97`
- total_checked_samples: `5899008`
- property_checks: `211/211`
- vref_csim_status: `pass`

## Operators

| Operator | Status | Contracts | Samples | Properties | Missing |
|---|---|---:|---:|---:|---|
| LayerNorm | `pass` | 25 | 903360 | 2/2 | - |
| GeLU | `pass` | 12 | 1806336 | 37/37 | - |
| Softmax | `pass` | 12 | 1382976 | 26/26 | - |
| Quantization | `pass` | 48 | 1806336 | 146/146 | - |

## Property Checks

- status: `pass`, pass `211/211`, fail `0`

## Sources
- implementation: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/hls/include/hgtxr_cyclic_math.hpp`
- implementation: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/hls/include/hgtxr_e2e_vit.hpp`
- reference: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/docs/SRC_CASE_MODULE_GUIDE.md`
- reference: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/codebases/HG-PIPE/analysis.md`

## Residual Risk
- This is local sampled/reference-vector equivalence plus deterministic contract property checks, not a formal proof over every possible input.
- Final E2E hardware signoff still depends on C3b physical smoke and XR-VITs source/replacement policy.
