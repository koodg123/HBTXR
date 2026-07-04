# Master Plan

## Goal

Analyze HG-PIPE paper/code evidence and reconstruct the end-to-end quantization process in `HG-PIPE-Quantization`.

## Prompt Brief

- Inputs: HG-PIPE paper PDF, `ICCAD24-HG-PIPE` HLS sources, generated case refs, `statistics/type.npy`, `statistics/range.npy`.
- Assumptions: the original refs are golden evidence; this project should not mutate `ICCAD24-HG-PIPE`.
- Unknowns: training/QAT procedure is not present in this checkout, so this implementation reconstructs inference-time integer quantization kernels and verification, not QAT.
- Constraints: preserve provenance, keep formulas bit-exact with HLS, avoid hidden state.
- Outputs: Python package, CLI, tests, verification reports, analysis docs.
- Acceptance: all discovered quantization refs pass bit-exact verification.

## Expert Council

- Hardware Architecture Analyst: connect paper claims to LUT/table/DSP-saving mechanisms.
- Quantization Implementer: translate HLS integer formulas into Python.
- Verification Engineer: prove correctness against refs and statistics.

## Execution DAG

1. Inspect paper/code/statistics evidence.
2. Define quantization case taxonomy.
3. Implement artifact loader and HLS-equivalent kernels.
4. Implement discovery, verification, and reports.
5. Run tests and full refs verification.
6. Document evidence and remaining limits.
