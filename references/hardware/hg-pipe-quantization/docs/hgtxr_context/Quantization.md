# Quantization

Default hardware quantization target:

- activation: int8 candidate
- weights: int8 candidate
- accumulator: 32-bit
- HLS skeleton type: `ap_fixed<16, 6>` when synthesized, `float` in host C++

Calibration and per-layer error budgets are not finalized yet.

