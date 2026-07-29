> **작성** 2026-07-27 · **갱신** 2026-07-29
> **상태** active — R4까지 반영
> **소유** algorithm/quantization

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

- ~~**Transport between modules is float32.**~~ **Closed (A1).** Each I-tier module still
  has a float I/O port — that is what lets a converted model run through the unmodified
  `forward`, and it stays — but `ilayers/` now also carries a graph that threads `QTensor`
  end to end: `IPatchEmbed` → `IViTBackbone` → `IPooledMlpHead`/`IEllipseHead`, assembled
  by `IDirectPupilDetector`. Checked element-wise against `i_block.replay_model_int`, with
  a `__torch_function__` mode proving no float tensor crosses the forward.
  **Not carried:** the mask head — because it does not run at inference. It is stage-1
  auxiliary supervision (Sec III-D.1, "Mask Head (Aux)"), and `HybridModel`, the paper's
  full deployed system, does not instantiate one. `float_io_heads` records this so the
  omission reads as scope rather than as an oversight.
- **`Scale` stays float, deliberately.** The `1/√d` factor is an exact constant multiply
  — `0.125 = 2⁻³` for the shipped `head_dim=64`, a pure shift in hardware — so
  quantizing it could only add error. It is left in `left_float` rather than filtered out
  of the report: hiding an inert entry is how a report starts lying. In the QTensor graph
  it is folded into the requant that follows the score matmul, which is where hardware
  puts it.
- ~~**`ICat` / `IPool` are still unused.**~~ **Closed (B1).** `IPool` is the token mean of
  every pooled head; `ICat` joins the ellipse head's pooled features to the host-supplied
  anchor state, which is the case it was written for — two operands genuinely not on a
  common grid.
- **Every accuracy number below is on a randomly-initialised model.** No trained
  checkpoint exists (the legacy HGTXR checkpoints share not one state-dict key, and no
  manifest `.jsonl` exists), so these numbers characterise the *quantization*, not the
  *model*. See "Open" below.

## Conversion coverage — FrameModel (`embed_dim=48`, depth 2)

**Converted ≠ deployed.** Conversion is a general pass and quantizes the auxiliary mask
head like anything else, so 3 of the 38 integer modules (two convs and a GeLU) belong to a
head no accelerator runs — stage-1 supervision, absent from `HybridModel`. `ConversionReport`
reports the split, because a single total reads as a statement about the accelerator. Which
heads are auxiliary is declared by the model (`DirectPupilDetector.AUXILIARY_HEADS`), not by
the quantizer: it is a modelling fact, and a copy of it here would be free to drift.


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
- **The float transport was never buying precision.** Removing it costs at most **2 LSB**
  against the per-op block, because the next consumer immediately re-quantized to int8
  anyway. The value of the QTensor graph is deployability, not accuracy — and saying that
  requires having measured it rather than assuming either direction.
- **An exact float kernel is still the wrong kernel.** `IConv2d` accumulated int8×int8
  products through `F.conv2d` on floats, exact only below 2^24; the shipped mask
  projection's worst case is 27,870,912. Chunking the float conv into exactly-representable
  pieces would have fixed the arithmetic and left the claim wrong: an int8×int8 MAC array
  is integer hardware, and a float kernel also punches a hole in the datapath check
  exactly where the arithmetic is hardest. The int64 im2col matmul costs single-digit
  milliseconds. **Recommendation:** on this subsystem, prefer the kernel that models the
  hardware over the kernel that merely gets the same number.
- **Two bridges in the graph are invisible by deletion, for different reasons.** The
  block-to-block requant has a ratio of exactly 1 — block *k*'s output tensor and block
  *k+1*'s input tensor are the same activation observed once. The final-norm requant has a
  ratio of 1.0042 (the same tensor, measured by two different calibrators) and is *still*
  invisible, because a LayerNorm divides by its input's own standard deviation and
  normalizes a uniform rescale away; it survives only through the PoT LUT cursor, which
  0.4% does not move. At 1.05× it moves the output by 865 LSB. **Recommendation:** test
  that a bridge is *routed through*, not that removing it changes something — two
  mutations survived here before the tests were rewritten that way.
- **Verification finds what review does not.** Every defect list in Parts C/D came from
  adversarially re-running claims, not from reading: a `verify_export` that returned
  `True` after every LUT was zeroed; a fabricated `max_abs_diff=0.0` for comparisons that
  never ran; a probe width that turned short-row softmax verification into a crash; a
  `padding_mode` regression introduced by the padding work itself; and several tests that
  could not fail. **Recommendation:** keep the mutation requirement — every new test is
  shown RED under the mutation it guards before it is trusted.

## The hybrid shared stack — what one set of scalars for two modalities costs (B2)

With `cut_point < depth` the first `c` blocks are on **both** paths, and one activation
observer sees frames and event voxels together. Measured on `depth=4, cut_point=2,
embed_dim=48`, calibration and probes drawn separately:

| Shared-block linear | frame range | event range | shared | narrower modality loses |
|---|---|---|---|---|
| `blocks.0.attn.proj` | 1.4713 | 0.6657 | 1.4713 | **2.21×** (1.14 bits, event) |
| `blocks.0.attn.qkv` | 3.4421 | 3.7168 | 3.7168 | 1.08× (frame) |
| `blocks.0.mlp.fc1` | 3.6949 | 3.2923 | 3.6949 | 1.12× (event) |
| `blocks.0.mlp.fc2` | 1.9067 | 2.5359 | 2.5359 | 1.33× (frame) |
| `blocks.1.attn.proj` | 1.4550 | 0.7835 | 1.4550 | 1.86× (event) |
| `blocks.1.attn.qkv` | 3.2367 | 3.4503 | 3.4503 | 1.07× (frame) |
| `blocks.1.mlp.fc1` | 2.8844 | 3.6125 | 3.6125 | 1.25× (frame) |
| `blocks.1.mlp.fc2` | 1.9873 | 2.5329 | 2.5329 | 1.27× (frame) |

A max-abs observer resolves the conflict by taking the **union** of the two ranges, so
the narrower modality gives up up to **1.14 bits** of resolution — and gains a range
margin, because the wider grid clips a *subset* of what either narrow grid clips. Both
effects are real, and the second is why "give each modality its own scalars" is not
automatically better.

**Net effect, RMS relative error against the float model** (12 probes × 3 weight draws,
the shared-stack linear grids overwritten with per-modality ones as the control):

| Calibration batches | search: clipped shared / own | track: clipped shared / own | cost of sharing (search / track) |
|---|---|---|---|
| 3 | 0.0336% / 0.0543% | 0.0054% / 0.0640% | +0.00447 / −0.00929 |
| 12 | 0.0163% / 0.0228% | 0.0011% / 0.0098% | −0.00466 / +0.00574 |
| 40 | 0.0033% / 0.0043% | 0.0000% / 0.0000% | −0.00165 / −0.00239 |

The clipping column is consistent in every row: the shared grid clips less. **The cost of
sharing is not.** Its sign flips with the calibration budget and its magnitude (≈0.002–
0.009) sits inside run-to-run variation on a ≈0.03 baseline. The honest reading is that
on this fixture the resolution given up and the clipping avoided cancel, and no
per-modality split is justified by measurement.

**The caveat is load-bearing.** The model is randomly initialised, and a random ViT's two
input branches produce far more similar activations than a trained one's would — the
1.5–2.2× range spread seen here is plausibly the *floor*. This measurement therefore
characterises the mechanism (union-of-ranges, and the two effects that oppose each other)
and bounds nothing about a trained network. That is the same A2 dependency as everywhere
else in this report.

## The rsqrt segment count — measured, then closed (R4)

D3 left the index as "a mitigation, not a cure", which reads as an open invitation to add
a third segment. Measured before building anything, because a third segment is not cheap:
`i_ops.layernorm_quantize_segmented` is fixed at exactly two (10 scalars), so a third
needs a new golden, a new payload and an export-format change — D3's cost again.

| LayerNorm | var dynamic range | rsqrt rel RMS | rel max | rows outside index |
|---|---|---|---|---|
| `blocks.0.norm1` | 1.72 | 0.00017 | 0.00041 | 0 |
| `blocks.0.norm2` | 1.45 | 0.00014 | 0.00026 | 0 |
| `blocks.1.norm1` | 1.43 | 0.00010 | 0.00021 | 0 |
| `blocks.1.norm2` | 1.45 | 0.00012 | 0.00027 | 1 below |
| `backbone.norm` | 1.47 | 0.00015 | 0.00029 | 0 |

The observed variance dynamic range is **under 2:1** — nowhere near the `bound+1`:1 that
D3 identified as where a linear index over a log-domain function breaks down — so two
segments are already almost exact. A simulated third segment (same index shape, same
threshold rule, greedily re-splitting whichever half carries more squared error) improves
RMS by **1.5–1.7×**, from ~2e-4 to ~1.3e-4, on a quantity four orders of magnitude below
the graph's ~2e-2 end-to-end error. **No third segment is justified.**

Ablating the integer LayerNorms entirely (original `nn.LayerNorm` restored, everything else
integer) moves RMS error by **+0.0031 on average over 16 runs**, ~10–15% of the total — so
the LayerNorm does cost something, but at 1e-4 the rsqrt index cannot be where it comes
from. Its int8 input and output ports are.

**The caveat, again:** randomly-initialised weights. A trained model's variance spread is
plausibly wider than 2:1, and that is the one thing that would reopen this. Which is why
the fit metrics now survive to the export (below) — so the question is answerable from an
artifact rather than by re-deriving the calibration.

### A defect found on the way

This report previously said of the rsqrt index that "the payload carries `metrics` so the
error is visible". **It did not.** `calibrate_int_layernorm` computed the whole fit report,
`ILayerNorm` dropped it on construction, and `export.py` builds its entry from the module —
so nothing reached the artifact. Fit quality is not recoverable from the exported tables,
and `rows_above_range` is documented as *"a warning about the fit"*: a warning nobody can
read after calibration is not a warning. Fixed, with the metric family now checked for
internal consistency (`rms ≥ max/√rows`) because "0.0" is the value a fabricated metric
reaches for — and this subsystem has shipped exactly that once, in `verify_export`.

## Verification

```bash
cd algorithm && python -m pytest tests -q
```

**594 passed.** (507 at the close of Part D; A1, B1, B2, B6 and R1–R4 added the rest.)
Historical breakdown at the Part D close: Breakdown: 19 config-matrix · 18 entrypoint · 56 export · 6 export-txt ·
76 conv-padding · 28 int-graph · 64 int-layernorm · 35 int-layernorm-segmented ·
50 int-softmax · 23 int-vit-graph · 29 per-channel-scales · 3 qat-to-integer ·
12 quantization · 8 seam-conversion · 33 softmax-max-tokens · 17 entrypoints-importable.

Run on the torch 2.13.0 CPU venv. Numerics are `float32`/`int64` on CPU.

## Open

- **A2 — no trained checkpoint.** Requires running the manifest build and a training run;
  there is no shortcut through the archive. Until then every accuracy figure here is on a
  randomly-initialised model. This is now the *only* thing blocking the remaining
  questions, including whether the hybrid shared-stack result above survives training.
- **Conv activations are calibrated symmetrically**, so no converted model has an
  asymmetric conv input grid, and `IConv2d`'s zero-point correction and zero-point padding
  — both built and measured in D2 — are never exercised by conversion. They are what makes
  an asymmetric grid legal if one is ever calibrated; the path is tested directly since
  conversion will not reach it.
