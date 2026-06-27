# FACET EPNet Evaluation Result

Config: `DavisEyeEllipse_EPNet_local_train_smoke.yaml`
Checkpoint: `runs/logs/EPNet_local_train_smoke/version_0/checkpoints/epoch=00-val_mean_distance=38.6962.ckpt`
Device: `cpu`
Evaluated batches: `2`

| Metric | Current | Paper Table II reference | Delta |
|---|---:|---:|---:|
| P10 | 0 | n/a | n/a |
| P5 | 0 | n/a | n/a |
| P3 | 0 | n/a | n/a |
| P1 | 0 | 0.9959 | -0.9959 |
| mean pixel error | 38.6962 | 0.203 | 38.4932 |
| IoU | 0 | n/a | n/a |
| AP | 0 | n/a | n/a |
| params M | 3.89828 | 3.92 | -0.02172 |
| FLOPs G | 3.42232 | 3.44 | -0.0176754 |
| latency ms | 20.2947 | 0.5302 | 19.7645 |

## Notes

- Paper Table II reference values available in the local analysis are P1, mean pixel error, params, FLOPs, and latency.
- P10/P5/P3/IoU/AP paper reference values were not recovered from the inspected local report text.
- Latency is measured in the current runtime and is not TensorRT latency unless the runtime is explicitly TensorRT-backed.
- This report is a metric pipeline artifact; it is only a reproduction result when run on the final full checkpoint and full validation split.
