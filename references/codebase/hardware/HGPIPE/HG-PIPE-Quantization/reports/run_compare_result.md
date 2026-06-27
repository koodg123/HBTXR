# HG-PIPE Runner Comparison

## Summary

- Status: PASS
- Elements: 1000
- Mismatches: 0
- Max abs error: 0.0
- Mean abs error: 0.0
- Top-1 equal: True

## Outputs

| Runner | Shape | DType | Min | Max | Mean | Top-k |
|---|---:|---|---:|---:|---:|---|
| torch_int | [1000] | int64 | -40328.0 | 102624.0 | -629.392 | #1: index=0 value=102624, #2: index=758 value=80300, #3: index=391 value=69852, #4: index=389 value=56180, #5: index=395 value=37655 |
| fakequant_graph | [1000] | int64 | -40328.0 | 102624.0 | -629.392 | #1: index=0 value=102624, #2: index=758 value=80300, #3: index=391 value=69852, #4: index=389 value=56180, #5: index=395 value=37655 |

## Interpretation

The FakeQuantizer-inserted artifact graph and the torch.int artifact graph are expected to match exactly for the recovered HG-PIPE reference input. Any nonzero mismatch indicates a numerical or graph-reconstruction regression.
