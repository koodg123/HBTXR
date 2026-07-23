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
| torch_int | [1000] | int64 | -12137.0 | 25288.0 | 416.22 | #1: index=21 value=25288, #2: index=701 value=21610, #3: index=92 value=21554 |
| fakequant_graph | [1000] | int64 | -12137.0 | 25288.0 | 416.22 | #1: index=21 value=25288, #2: index=701 value=21610, #3: index=92 value=21554 |

## Interpretation

The FakeQuantizer-inserted artifact graph and the torch.int artifact graph are expected to match exactly for the recovered HG-PIPE reference input. Any nonzero mismatch indicates a numerical or graph-reconstruction regression.
