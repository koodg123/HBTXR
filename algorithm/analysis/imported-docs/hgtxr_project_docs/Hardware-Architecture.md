# HGTXR Hardware Architecture

The hardware tree follows a module-per-kernel structure. The current
implementation establishes interfaces and C simulation readiness before
applying board-specific pragmas.

## Pipeline

```text
frame/event input
  -> patch embedding
  -> attention stage
  -> mlp stage
  -> pooling
  -> search head
  -> fusion(prev_state)
  -> track head
  -> runtime_fsm
  -> selected state
```

## Next Hardware Work

- Replace placeholder attention/MLP with quantized kernels.
- Add weight loaders under `hardware/refs/weights`.
- Add per-module C testbenches with fixed reference vectors.
- Tune array partitioning and pipeline pragmas per target board.

