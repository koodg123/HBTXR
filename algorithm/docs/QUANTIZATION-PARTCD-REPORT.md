# Quantization Parts C & D — the integer tier: verification report

Part C built the I-tier kernels and Part D made them reachable, correct on real
distributions, and exportable. This report records what the integer tier converts, what
it does not, and every number that was measured rather than assumed.

Companion to `docs/QUANTIZATION-PARTB-REPORT.md`, which covers the configurable
fake-quant tier.

## What the integer tier converts

| Op | I-tier module | Datapath | Golden it is checked against |
|---|---|---|---|
| `nn.Linear` | `ILinear` | int8 weight (per-channel) × int8 act, int accumulate | `i_ops.int_matmul` |
| `nn.Conv2d` | `IConv2d` | same, strided **and padded** (pads with the activation zero-point) | `i_ops.int_conv2d` |
| `nn.GELU` | `IGeLU` | PoT-indexed integer table | `i_ops.table_quantize` |
| `nn.LayerNorm` | `ILayerNorm` | integer mean, integer variance, **two-segment** rsqrt table | `i_ops.layernorm_quantize_segmented` |
| `nn.Softmax` | `ISoftmax` | integer row max / exp / row sum / segmented reciprocal | `i_ops.softmax_quantize` |
| `Q@Kᵀ`, `attn@V` | `IMatMul` | act × act, both scales observed | `i_ops.int_matmul` |
| residual `+` | `IAdd` | two grids aligned onto one output grid | — (scale algebra) |

Every kernel is compared **element-wise `==`** against a pure-Python,
arbitrary-precision integer golden in `i_ops.py`. That idiom — not a tolerance — is what
the tier's trustworthiness rests on.

## What it does NOT do

Stated first, because a partial conversion that reads as total is the failure mode this
subsystem has already had twice.

- **Transport between modules is float32.** Each I-tier module quantizes at its input
  port and dequantizes at its output port, so every *op* is integer while the *graph* is
  not. This is what lets the converted model run through the unmodified model `forward`.
  Closing it needs an `ilayers/vit.py` that threads `QTensor` between ops — a rewrite of
  the block forwards that changes the arithmetic at every edge, so it needs its own
  pure-integer whole-block oracle first (see `QUANTIZATION-PARTD-PLAN.md` §6.4).
- **`Scale` stays float, deliberately.** The `1/√d` factor is an exact constant multiply
  — `0.125 = 2⁻³` for the shipped `head_dim=64`, a pure shift in hardware — so
  quantizing it could only add error. It is left in `left_float` rather than filtered out
  of the report: hiding an inert entry is how a report starts lying.
- **`ICat` / `IPool` are still unused.** Token pooling and the ellipse head's `torch.cat`
  are written inline in head bodies that Part D did not touch.
- **Every accuracy number below is on a randomly-initialised model.** No trained
  checkpoint exists (the legacy HGTXR checkpoints share not one state-dict key, and no
  manifest `.jsonl` exists), so these numbers characterise the *quantization*, not the
  *model*. See "Open" below.

## Conversion coverage — FrameModel (`embed_dim=48`, depth 2)

```
stages   {layernorm 5, softmax 2, gelu 6, conv 3, matmul 4, add 4, linear 14} = 38 integer modules
left_float  6 × nn.Dropout (inference no-ops) + 2 × Scale
export   {ILinear 14, IGeLU 6, ILayerNorm 5, IConv2d 3, IMatMul 4, IAddFloatIO 4, ISoftmax 2}
         num_modules 38 · all 38 replayed and "checked" · unexported []
         verify_export -> ok=True, worst max_abs_diff 0.0
artifacts  66 984 integer weights (.npy) + 50 .txt files (32 796 bytes) + manifest.json
```

## Error budget — where the error actually comes from

Box output vs the **original float model**, seed 0, one calibration batch, stages added
cumulatively:

| Conversion | box rel-err |
|---|---|
| Linear only | 0.0152 |
| + Conv2d | 0.0169 |
| + integer LayerNorm / Softmax / GeLU | 0.0171 |
| + attention matmuls and residual adds | **0.0187** |

Across seeds 0–7 the full conversion measures 0.0112 · 0.0701 · 0.0112 · 0.0125 · 0.0220
· 0.0272 · 0.0223 · **0.1119** (min 0.0112, max 0.1119).

## Findings

- **The outliers are PTQ, not the integer tier.** At the two worst seeds the *Linear-only*
  conversion already costs almost all of it: seed 1 → 0.0708 of 0.0701 (ratio 1.00×),
  seed 7 → 0.1117 of 0.1119 (ratio 1.00×). Everything Parts C and D added — integer
  conv, the three integer nonlinears, the attention matmuls and both residual joins —
  contributes ~0 there and at most 1.45× at the well-behaved seeds. **Recommendation:**
  spend effort on activation calibration (percentile / per-token for attention logits),
  not on the integer kernels.
- **The segmented rsqrt index is the one strictly-better change.** Two 64-entry segments
  (128 int16 words, *half* the previous 256-word table) beat one 256-entry segment on
  every seed, and at equal depth the win is 10.0–12.6× RMS relative rsqrt error
  (0.085–0.106 → 0.0067–0.0102) on heteroscedastic data. Homoscedastic data does not
  regress (0.00065 → 0.00032). Whole-graph effect: 0.0303 → 0.0192. **Recommendation:**
  it is the default; a hardware budget should size the table at 2×64, not 1×256.
- **Padding had to use the zero-point, not 0.** On a 3×3/pad-1 conv with `zp=17`,
  padding with the zero-point gives 1.8e-07 max abs error against the fake-quant
  reference while a naive zero-pad gives 0.24 — and the damage is *exactly the padded
  ring*, so a verification probe that never touches the border cannot see it. One
  geometry (3×3, pad 1, stride 1) was blocking all 21 refused conv sites in the repo.
- **Verification finds what review does not.** Every defect list in Parts C/D came from
  adversarially re-running claims, not from reading: a `verify_export` that returned
  `True` after every LUT was zeroed; a fabricated `max_abs_diff=0.0` for comparisons that
  never ran; a probe width that turned short-row softmax verification into a crash; a
  `padding_mode` regression introduced by the padding work itself; and several tests that
  could not fail. **Recommendation:** keep the mutation requirement — every new test is
  shown RED under the mutation it guards before it is trusted.

## Verification

```bash
cd algorithm && python -m pytest tests -q
```

**507 passed.** Breakdown: 19 config-matrix · 18 entrypoint · 56 export · 6 export-txt ·
76 conv-padding · 28 int-graph · 64 int-layernorm · 35 int-layernorm-segmented ·
50 int-softmax · 23 int-vit-graph · 29 per-channel-scales · 3 qat-to-integer ·
12 quantization · 8 seam-conversion · 33 softmax-max-tokens · 17 entrypoints-importable.

Run on the torch 2.13.0 CPU venv. Numerics are `float32`/`int64` on CPU.

## Open

- **B2 — no trained checkpoint.** Requires running the manifest build and a training run;
  there is no shortcut through the archive. Until then every accuracy figure here is on a
  randomly-initialised model.
- **Float transport between modules** (above) — the single remaining item of
  `QUANTIZATION-PLAN.md` §9.
- **`manifest["total_params"]` reads 0** on a converted model: every parameter has become
  a buffer, so the field is technically correct and practically useless. Worth either
  removing or redefining when the format next changes.
- **Hybrid conversion is exercised but not measured.** `test_entrypoint` proves the hybrid
  path calibrates and converts after the D1 fix; no accuracy number is recorded for it,
  and with `cut_point < depth` the shared early blocks carry one set of scalars for two
  modalities — a real question that only a hybrid measurement can answer.
