# Execution

## Decisions

- Keep `../ICCAD24-HG-PIPE` read-only because its worktree already contains many modified files.
- Implement formulas directly from HLS headers rather than fitting them statistically from refs.
- Verify every discovered case against golden outputs instead of validating only selected examples.

## Evidence

- `src/quant.h`: table ReQuant formula.
- `src/gelu.h`: GeLU table formula, same as Quant.
- `src/layernorm.h`: mean/variance/rsqrt/affine/shift/clamp sequence.
- `src/softmax.h`: inverse-exp, segmented reciprocal, and final unsigned clamp.
- `statistics/print_statistics.py`: type/range `.npy` files are intended statistics sources.
