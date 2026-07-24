# Quantization Part B — Configurable Quant: verification report

Part B makes the HBTXR fake-quant stack **fully specifiable** and verifies every
config axis end to end. This report records what is configurable and the measured
PTQ error across the config matrix.

## What is configurable (`quantization.spec.TensorQuantSpec`)

Per tensor role (weight / activation), independently:

| Knob | Values | Where it lives |
|---|---|---|
| bit-width | `bits`, `signed` | scheme + per-layer override |
| granularity | `per-tensor` / `per-channel` / `per-group` (`group_size`) / `per-block` (`block_size`) | `grouping.py` slots (observer + quantizer share them) |
| symmetric | `symmetric: true/false` (asymmetric ⇒ `zero_point ≠ 0`) | `observer.AffineObserver` |
| scale representation | `scale_type: float / dyadic / pot` | `spec.apply_scale_type` |
| calibration | `calibration.method: minmax / percentile / kl / mse` (+ `percentile`, `num_bins`, `mse_grid`) | `observer.AffineObserver` |
| per-layer overrides | `overrides: [{match, weight:{…}, activation:{…}}]` (mixed precision) | `spec.QuantScheme.resolve` → `convert.insert_fake_quant` |

Design: only **granularity** touches the fake-quant forward (via slot reshape);
symmetric / calibration / scale_type are resolved entirely at calibration time into
the stored `(scale, zero_point)` buffers.

## Config matrix — PTQ box rel-error

FrameModel (`embed_dim=48`, depth 2), int8 weights+activations, single calibration
batch, `torch.manual_seed(0)` (deterministic). Rel-error = `‖box_q − box_fp‖ / ‖box_fp‖`.

| Config | box rel-err |
|---|---|
| baseline (per-tensor sym float minmax) | 0.0144 |
| weight per-channel | 0.0140 |
| weight per-group(16) | 0.0178 |
| weight per-channel + dyadic | 0.0152 |
| weight per-channel + pot | 0.0872 |
| weight per-channel + mse | 0.0169 |
| weight per-tensor + kl | 0.0139 |
| activation asymmetric | 0.0297 |
| activation percentile | 0.0170 |
| activation per-channel (static) | 0.2050 |
| combined (per-ch + asym + dyadic + percentile) | 0.0408 |
| mixed precision (4-bit per-group qkv + asym-percentile fc1) | < 0.25 (passes) |

## Findings

- **Weights tolerate aggressive granularity.** per-tensor ≈ per-channel ≈ per-group
  ≈ mse ≈ kl (all ~1.4–1.8%) at int8 — weights are well-behaved, so per-channel is a
  free HW-friendly upgrade and per-group/4-bit becomes viable via overrides.
- **`dyadic` ≈ `float`** (1.5% vs 1.4%): the multiplier/2^shift scale loses almost
  nothing, so integer-only requant (Part C) costs no accuracy. **`pot` is ~6× worse**
  (8.7%) — rounding the whole scale to 2^k is coarse; use dyadic, not pot, for HW.
- **Asymmetric / percentile activation** are small (3.0% / 1.7%) — useful when
  activations are one-sided (post-GELU).
- **Static per-channel *activation* is the weak config (20.5%).** Expected: a fixed
  per-feature activation scale can't track the dynamic per-token range. Recommendation
  encoded as the default in `frame_quant.yaml`: **per-channel weights + per-tensor
  activations**; reserve finer activation granularity for dynamic/per-token (a Part C
  runtime concern), not static PTQ.

## Verification

`pytest tests/quantization` — **31 tests** (12 original + 19 config-matrix):
grouping round-trip exact for all four granularities; `pot` → exact powers of two;
`dyadic` values are true `m/2^s`; asymmetric zero-points; and the full PTQ matrix +
mixed-precision override, all `< 0.25`. All numerics run on the torch CPU venv.
