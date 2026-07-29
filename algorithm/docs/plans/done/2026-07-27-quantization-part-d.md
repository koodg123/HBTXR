> **작성** 2026-07-27 · **갱신** 2026-07-29
> **상태** frozen — D0~D6 완료
> **소유** algorithm/quantization

# HBTXR Quantization — Part D plan (closing the Part C gaps)

Parts A/B/C are committed and green (`pytest tests/quantization` = **245 passed**). This
plan covers everything `plans/active/2026-07-25-quantization-build-plan.md` §9 left open, plus three **live bugs** that
the scoping survey found and that were not on anyone's list.

Every claim below is grounded in code that was read and, where it is a claim about
behaviour, in a command that was run. Measurements are quoted with their source.

---

## 0. What changed the scope

Three findings invalidate parts of the original task list. They are listed first because
two of them are shipping bugs, not planned work.

### 0.1 The hybrid quantization entrypoint cannot run (live bug)

`quantization/entrypoint.py:47-50`:

```python
def _make_forward_fn(modality: str):
    if modality.startswith("hybrid"):
        return lambda model, batch: model.search_step(batch["frame"])
    return lambda model, batch: model(batch["image"])
```

`HybridModel.search_step` (`models/hybrid/model.py:61-69`) never touches `event_stem`,
`track_head`, `track_reliability` or `backbone.forward_track`. Of the model's 18
`QLinear`, **4 are never observed** — `track_head.regressor.{0,2}`,
`track_reliability.estimator.{0,2}` — and PTQ then dies before conversion:

```
quantization/calibrate.py:71  quant.act_fq.set_qparams(*observers[name].qparams())
quantization/observer.py:76   ValueError: AffineObserver.qparams() called before any observe()
```

This is not missing coverage; the shipped hybrid path is broken. With a both-branch
`forward_fn` the whole path works end to end (measured: 34 integer modules,
`left_float` = 6 × Dropout only, `unexported: []`, both `search_step` and `track_step`
still forward after conversion).

### 0.2 `hbtxr quantize` is a dead CLI command (live bug)

`scripts/hbtxr.py:33` maps `quantize -> quantization.entrypoint` and imports it
dynamically. `quantization/entrypoint.py:33` does `from engine.data.factory import
build_dataloader`, which reaches `dataset/hbtxr/transform.py:9` → `from PIL import Image`.
PIL is absent, so the command raises on import. The same eager import appears at
`engine/train/entrypoint.py:29`, `engine/eval/entrypoint.py:21`,
`engine/infer/entrypoint.py:26`, `engine/distill/entrypoint.py:20` — **all five
entrypoints are equally unimportable**. `quantization/__init__.py` does not import
`entrypoint`, which is why the 245 tests never noticed.

Second, independent blocker on the same chain: `dataset/preprocess/__init__.py:10`
imports `.annotation_groundedsam`, which **does not exist** (the module is at
`dataset/annotation/annotation_groundedsam.py`); the file also carries a botched merge —
lines 1-9 and 10-20 are duplicate import + `__all__` blocks.

### 0.3 The dependency wall is one package, and irrelevant anyway

Static walk of the 130-module import closure: **`pillow` is the only third-party package
that stands between the venv and an importable `entrypoint`.** `opencv-python`, `scipy`,
`scikit-image`, `hdf5plugin`, `natsort`, `numba`, `tonic`, `dv-processing`,
`albumentations`, `matplotlib`, `seaborn`, `tqdm`, `torchmetrics` are **not referenced at
all** on this path (they live under `dataset/loaders/**`, `utils/**`,
`common/losses/ellipse_loss.py`). Installing the full `requirements.txt` closure would be
60 new packages / ~286 MB — for nothing, since a real `run_quantize` test needs **zero
installs** (proof-of-concept written during the survey: 3 tests, ptq/qat/export_int, green
in 2.62 s on the current venv).

### 0.4 Smaller corrections worth knowing

- **There is no positional embedding anywhere in the model.** `models/backbones/vit.py:10-12`
  claims the caller applies one; `grep -rn "pos_embed\|positional" models/` returns zero
  hits. Nothing to quantize — but the docstring misleads a plan reader and should be fixed.
- **`HybridModel` has no `forward`**, only `search_step` / `track_step` / `run_step`
  (`models/hybrid/model.py:61,71,86`), so any driver must pass an explicit `forward_fn`.
- `models/blocks/patch_embed.py` does not exist; the stems are at
  `models/backbones/patch_embed.py:33,46`.

---

## 1. Priority order

| Phase | Content | Why here | Effort |
|---|---|---|---|
| **D0** | Live bugs: hybrid `forward_fn`, `_scalar` per-channel truncation, stale docs | Shipping defects, independent of everything else, each a small commit | S |
| **D1** | Make `entrypoint` importable + tested (all 5 entrypoints) | Removes the only untested module in the stack; unblocks D6 | S |
| **D2** | `IConv2d` padding (A2), `ISoftmax.max_tokens` guard (A3) | Small, self-contained, each closes a measured accuracy hole | S–M |
| **D3** | Segmented rsqrt index (A4) — **new golden kernel** | 9-10× accuracy *and* half the table memory; biggest win per line | M |
| **D4** | End-to-end integer graph (A1) | The headline §9 gap; largest and depends on D0's `_scalar` fix | **L** |
| **D5** | HG-PIPE `.txt` artifacts (A5), Part C/D report (A6) | Deliverables; D5's report wants D2–D4 numbers | S |
| **D6** | Verification debt: hybrid (B3), QAT→integer (B4), trained ckpt (B2) | B3/B4 are cheap once D0/D1 land; B2 needs a training run | S / S / **XL** |

D0 → D1 → D2 → D3 can proceed in that order or in parallel (disjoint files). **D4 must
come after D0's `_scalar` fix**, which it depends on.

---

## 2. D0 — live bugs (three independent commits)

### D0.1 Hybrid `forward_fn` drives only the search branch

**File**: `quantization/entrypoint.py:47-50`.

```python
def _hybrid_forward(model, batch):
    out = model.search_step(batch["frame"])
    model.track_step(batch["event"], out["state"])
    return out
```

**Verify**: a hybrid PTQ test asserting all 18 `QLinear` get calibrated (i.e. no
`AffineObserver.qparams() called before any observe()`), that `event_stem.proj` appears in
`report.stages["conv"]`, and that `track_step` still forwards after conversion. This is
also B3's test — do it once, here.

**Note for the report**: with `cut_point < depth` the shared early blocks are calibrated on
the *union* of the frame-search and event-track activation distributions, so one set of
`ILayerNorm`/`ISoftmax` scalars serves two modalities. That is a genuine accuracy question
that only a hybrid test can surface — measure it, do not assume it is fine.

### D0.2 `_scalar(v)` silently truncates a per-channel scale

`ISoftmax`, `IGeLU`, `IAdd` and `ICat` each collapse a scale tensor to element 0.
`ILinear.forward_accumulator` legitimately produces a per-channel scale, so this is
reachable. `ILayerNorm` was already fixed this way in Part C (`_rescale_ratio` handles
scalar / 1-element / `[C]` and raises otherwise) — **port that helper to the other four**.
All three D4 design proposals independently flagged this; it must land before D4.

### D0.3 Documentation that misstates the code

- `configs/experiment/frame_quant.yaml:13-17` — says the export path "LUT-calibrates
  GeLU/LayerNorm/Softmax, converts Q -> I", which describes the **Q-tier** path. It now
  runs `convert_model_to_integer`. Replace with text that says the I-tier kernels are
  installed, that the graph is integer per op but not end to end (§9), and that
  `mask_head.proj` is left float and listed in `manifest['unexported']`.
  > **Superseded.** The last two clauses were true when this plan was written and are not
  > now: D2 made padded convs convertible (`unexported` is empty for FrameModel) and A1
  > closed the end-to-end graph. The replacement text landed in R1, measured rather than
  > restated. Left here unedited because a plan is a record of what was decided, not a
  > description of the current tree.
- `quantization/entrypoint.py:12-15` — same error in the module docstring
  (`convert_to_integer`, the Linear-only converter, instead of `convert_model_to_integer`).
- `models/backbones/vit.py:10-12` — claims a positional embedding is applied by the
  caller. None exists anywhere. Say so.

---

## 3. D1 — make the entrypoints importable and tested

**Rejected — Option A (install).** `pillow` alone unblocks the import, but
`dataset/preprocess/__init__.py:10` (§0.2) fires next, so it needs a source fix regardless;
and the full closure is 60 packages / ~286 MB for no benefit.

**Chosen — Option B + C.**

**B: a lazy proxy at module scope**, following the pattern already in
`engine/train/__init__.py`:

```python
def _build_dataloader(*args, **kwargs):
    """Lazy proxy: the dataset pipeline pulls PIL and friends, which a quantization
    run does not need until it actually builds calibration batches."""
    from engine.data.factory import build_dataloader
    return build_dataloader(*args, **kwargs)
```

The shape matters. Moving the import *inside* `run_quantize` also makes the module import,
but then a test can only monkeypatch `engine.data.factory.build_dataloader` — reaching
which requires importing that module, i.e. PIL again. A module-scope proxy is patchable as
`entrypoint._build_dataloader` with no shim. **13 of the 14 other module-scope imports in
`entrypoint.py` were each verified clean**; `engine.data.factory` is the only one that is
not. The same 5-line change applies to the other four entrypoints (§0.2).

**C: a real `run_quantize` test.** Only `build_dataloader` needs faking — `load_config`,
`resolve_modality`, `make_model`, `resolve_training_entry` (never stats the *train*
manifest — `engine/runspec/run_contract.py:87-111`), `post_training_quantize`,
`prepare_qat`, `Trainer.fit`, `save_checkpoint`, `convert_model_to_integer`,
`export_integer_model` and `verify_export` all run for real. Cover the ptq path, the qat
path, and the export_int path. Also fix `dataset/preprocess/__init__.py` while here, so
`hbtxr quantize` works for a user who *does* have pillow.

---

## 4. D2 — padded conv (A2) and the softmax token guard (A3)

### D2.1 `IConv2d` padding

**Overlapping stride already works** — `i_ops.int_conv2d` is a general valid-conv loop and
was verified bit-exact against `F.conv2d` for k=3/s=1 and k=3/s=2. Only padding is missing.

The load-bearing detail: **pad with the activation zero-point, not with 0.** `IConv2d`
uses an asymmetric activation grid, and the existing uniform `- zp·Σw` correction assumes
every window has `Cin·kh·kw` real taps — zero-padded taps break that per output pixel.
Measured on a random 3×3 pad-1 conv with `zp=17`:

| route | max abs error vs float reference |
|---|---|
| pad `x_int` with `zp`, keep the uniform `- zp·Σw` correction | 3.8e-06 |
| subtract `zp` first, then zero-pad (same algebra) | 3.8e-06 |
| **naive zero-pad + uniform correction** | **2.04** |

**Files**: `i_ops.int_conv2d` (add `padding`, `pad_value`), `ilayers/conv.py`
(`__init__` stores `(ph, pw)`, `from_conv` reads `conv.padding`, `forward` convolves the
zp-corrected operand with `padding=`), `convert.py:462` `_conv_is_convertible` (drop the
`padded` refusal), and downstream `export.py` `_conv_entry` (emit `padding`),
`replay_conv:551` (same pad-with-`zp` treatment or it disagrees with the live module), and
`_probe:731` (size the probe so ≥2 output pixels survive padding).

**Payoff**: every refused conv in the entire repo is the *same* geometry — 3×3, pad 1,
stride 1. One parameter unlocks all 21 sites, including `mask_head.proj` (the only one in
the shipping detectors) and the 14 UNet convs in `models.mask.MaskModel`.

### D2.2 `ISoftmax.max_tokens` runtime guard

The row length **is** knowable in O(1) from `x_int.shape[self.dim]` — `dim` is a real
reduction axis. `max_tokens` is currently calibration-only
(`int_calibrate_softmax.py:171,195`, echoed to `payload["metrics"]` at line 258) and is
**not stored on the module** (`hasattr(sm, "max_tokens") == False`) nor emitted by
`export.py::_int_softmax_entry`.

Plumb it through `from_payload` → `__init__` → `_int_softmax_entry` → `replay_softmax`,
and **raise** in `forward_int` when `n > self.max_tokens`. Clamping is not an option: the
op's contract is a normalized row. Worst case measured on a flat 96-token row from a
16-token calibration: **row sum 4.52**.

A no-new-state fallback exists for legacy payloads —
`covered_acc_hi = ((bound2_two + 1) << s2_two) - b2_two - 1`,
`derived_max_tokens = covered_acc_hi // max(exp_table)` — but it is **loose**, admitting
rows 25-60 % longer than calibrated with gradual (not cliff) degradation. Use it only as a
fallback, never as the primary.

---

## 5. D3 — segmented rsqrt index (A4)

**This needs a new golden kernel, and that must be stated plainly.**
`i_ops.layernorm_quantize:119-153` takes exactly 7 scalars and its cursor is one
unbranched line — there is no second `(b, s1, bound)` triple, no second table argument,
no branch. A second segment **cannot** be expressed in it.

But it is *cheaper* than the softmax segmentation already in the tree, for a reason that
was verified: LayerNorm needs **no second `(b3, s3)` requant pair**. Softmax needs one per
segment because each reciprocal table has its own numerator; LayerNorm's rsqrt entries all
live at one `rsqrt_scale` and the final `>> s2` is applied per element after the affine, so
it is segment-independent. Quantizing both segments at one shared power-of-two
`rsqrt_scale` was measured to lose nothing (`2seg_int16_rms ≈ 2seg_float_rms` to 4 dp).
So the layout grows by exactly **3 scalars (7 → 10)** plus one table, plus a
`cursor_one > bound` branch copied from the softmax golden.

**Measured payoff** (heteroscedastic fixture, 4 seeds, RMS relative rsqrt error):

| entries **per segment** | 1-segment (current) | 2-segment | gain |
|---|---|---|---|
| 256 | 0.0935 / 0.0887 / 0.0903 / 0.0941 | **0.0104 / 0.0086 / 0.0089 / 0.0099** | ~9-10× |
| 128 | 0.1400 / 0.1525 / 0.1439 / 0.1371 | **0.0185 / 0.0167 / 0.0166 / 0.0166** | ~8-9× |
| 64 | 0.1868 / 0.2058 / 0.1966 / 0.1841 | **0.0280 / 0.0365 / 0.0355 / 0.0274** | ~6× |

Two 64-entry segments (**128 int16 words, half the current BRAM**) beat one 256-entry
segment (256 words) by 3×. This is the one change that is strictly better on accuracy *and*
on hardware cost, and it directly answers §9's "a spread wider than `entries`:1 cannot be
served" — a 2-segment PoT index resolves roughly `entries²`.

**Files**: `i_ops.py` (new `layernorm_quantize_segmented`, keep the 7-scalar one as the
golden for existing payloads), `ilayers/layernorm.py`, `int_calibrate.py` (two-segment
`_fit_variance_index`), `export.py` `_int_layernorm_entry` + `replay_layernorm`, and the LN
tests that assert the 7-scalar shape.

---

## 6. D4 — the end-to-end integer graph (A1)

**Decision: FX-capture lowering, with a seam refactor of the model as its first phase.**
Three designs were produced and judged; the load-bearing feasibility claims were re-run by
the judge rather than taken on trust.

### 6.1 Why FX, and the evidence

Naive `symbolic_trace(FrameModel)` fails — `models/backbones/patch_embed.py:43` does
`int(feat.shape[-2])` on a `Proxy`. With a three-module leaf set it traces cleanly:

```
FrameModel with {FramePatchEmbed, EventPatchEmbed, PupilMaskHead} as leaves
  TRACE OK. nodes: 282  Counter({call_function:125, call_module:108, call_method:47, ...})
  bitwise identical per key: head/box/state/mask/eye_box/reliability -> all True
  inventory: getitem 60, add 18, reshape 16, transpose 16, matmul 16, permute 8, mul 8
```

The **converted** model — which is what the lowering pass actually consumes — traces too,
and observers can be attached to the q/k/v edges (tensors that are the input/output of no
module) **with zero edits to `models/`**:

```
CONVERTED TRACE OK. nodes: 282   bitwise identical per key: all True
inserted observers: 24   observed graph == eager bitwise: True
```

Every node already carries `nn_module_stack`, i.e. the dotted path the calibrated payloads
are keyed on.

### 6.2 Why the seam refactor comes first

FX's weak point is the pattern matcher: it must walk a `call_function` chain
(`reshape → permute → getitem[0|1|2] → matmul/mul/matmul`) and anchor on targets ending
`.attn.qkv`. Promoting the inline ops in `models/blocks/{attention,transformer_block}.py`
to `nn.Module` attributes — exactly what was already done once when `F.softmax` became
`self.attn_softmax = nn.Softmax(dim=-1)` — collapses that matcher to a name lookup.

Measured on a real `TransformerBlock(192, 3)`:

```
state_dict keys equal: True (12 == 12)   named_parameters equal: True
bitwise identical: True   max|diff| = 0.0   strict load both ways: True True
new module paths: ['attn.qk_matmul','attn.av_matmul','attn_residual','mlp_residual']
seam modules own state: all 0 params, 0 buffers
fx trace of refactored block: 31 nodes (unchanged)
```

So the refactor is free — no float behaviour change, no `state_dict` change, no checkpoint
break — and it buys a matcher that a reshape reordering cannot confuse, a loud failure mode
(module missing) instead of a silent one, and a fallback to a pure-module design if FX ever
becomes a liability.

### 6.3 The invariant

Every edge whose producer scale differs from the consumer's `input_scale` gets an explicit
dyadic requant, so every nonlinear kernel is entered at ratio exactly 1.0. This is
mechanically checkable: monkeypatch every `ILayerNorm`/`ISoftmax`/`IGeLU` to assert
`qt.scale == module.input_scale` and that each was hit.

Note `self.scale` in attention is `head_dim**-0.5` = **0.125 = 2⁻³ exactly** for the
shipped `head_dim=64` — a pure arithmetic right shift, free in hardware for this config.

### 6.4 The biggest risk, and its mitigation

**The graph tier has no bit-exact oracle, and this repo's entire verification idiom assumes
one.** Every existing test compares element-wise `==` against an `i_ops` golden. The
lowered graph is *by design* a different function from the per-op-integer model — float
transport is strictly more precise than a dyadic requant plus an int8 re-clamp at every
edge. Concretely: per-channel dyadic requant against an exact float rescale was measured at
**max 1.0 LSB** over a `[1,64,576]` accumulator with 576 per-channel ratios (8 distinct
shifts, range 24-31). One design claimed 0.0; it is not 0.0.

The failure mode is that `allclose` gets a tolerance tuned until it passes, and a genuinely
wrong lowering — a swapped operand scale, a missed requant, a per-channel vector truncated
to element 0 — hides inside it. That risk compounds because FX's other failure modes
surface as confusing `TypeError`s deep in generated code.

**Mitigation, scheduled early rather than late**: build a pure-Python whole-block
`i_ops`-only `replay_block_int()` **before** lowering anything, and make it the reference
the lowered block is compared against with element-wise `==` and zero difference. Then the
per-op-vs-graph delta becomes a *reported number for the paper*, not the correctness gate.

### 6.5 Phases

1. **Seam refactor** (`models/blocks/{attention,transformer_block}.py`) — verified free;
   zero test churn; clean rollback.
2. **Prerequisites**: `Dropout → Identity` in the integer graph (`Dropout` raises
   `TypeError: dropout(): argument 'input' must be Tensor, not QTensor`); `QTensor.__getitem__`
   and `.contiguous()` (attention's Q/K/V split at `models/blocks/attention.py:40` has no
   handler today).
3. **Integer primitives + their goldens**: `IRequant`, `IAddInt`, `IPoolShift`, **and**
   `replay_block_int()` in pure Python.
4. **Tracer + traceability contract**: leaf set, `concrete_args={'depth_limit': None}`,
   and a test pinning the traced op inventory so a model change fails loudly.
   ⚠️ A `QTensor` survives an FX edge **only when the producer is a leaf module** —
   otherwise the tracer decomposes the dataclass into its constituent tensor ops. This is a
   hard, silent-failure-prone coupling between the leaf set and the lowering pass, and it
   is why `IGeLUFloatIO` must be a leaf (its forward branches on `isinstance(x, QTensor)`,
   which is Proxy control flow).
5. **Lower one block**, matched element-wise `==` against `replay_block_int()`.
6. **Lower the whole model**, then the hybrid (two GraphModules).
7. **Scale-provenance test**: assert every requant's `out_scale` is identical to an
   attribute of a calibrated module or a named edge scale — this catches a pass that
   silently invented a scale.
8. **Drift as a pinned property**, not just an endpoint bound: per-block drift was measured
   at 0.023 → 0.040 over 8 blocks and *saturates*, because every block ends on the next
   LayerNorm's calibrated input scale, so the residual re-anchors instead of free-running.
   Assert block-7 drift < 1.8 × block-0.

**Entry point**: a *new* function, not a flag on `convert_model_to_integer` — that is what
keeps the existing exact-set assertions (`test_int_vit_graph.py:98`, `test_export.py:266`)
green.

**Secondary risk**: q/k/v scales are per-tensor max-abs on random-weight fixtures. Trained
attention logits are heavy-tailed and K is known for outlier channels. Expose the
percentile knob from the start so this is measured on a real checkpoint rather than argued;
expect per-head scales to be a follow-up.

---

## 7. D5 — artifacts and report

### D5.1 HG-PIPE `.txt` artifacts (A5)

Reference format (`references/hardware/hg-pipe-quantization/src/lut_calibration.py:472`):
`{stem}_scalars.txt` and one `{stem}_{suffix}.txt` per table, where the generic key
`"table"` is renamed **`table_m`**. Content is a single line of comma-separated decimal
integers **with a trailing comma and no newline** — a C array initializer body.

**Put it in a new `quantization/export_txt.py` that reads `manifest.json`**, not inside
`export_integer_model`. Reasons: everything needed is already inlined in the manifest as
JSON int lists (unlike `_linear_entry`, no live module access is required); it keeps
`export_integer_model`'s one sharp invariant ("a stateful quantized module with no export
branch raises") from multiplying; and a manifest reader can be re-run against a dump
produced before the converter existed. `.txt` is lossy relative to the manifest (no dtype,
no shape, no scales), so it is a *view*, not a source of truth.

Caveat: `load_integer_manifest` materializes the seven `_INT_ARRAY_KEYS` into numpy arrays,
so `int(v)` must run over `.ravel()` — or read `manifest.json` raw. Wire an `--export-txt`
flag next to `--export-int`.

### D5.2 Part C/D report (A6)

Mirror `algorithm/docs/reports/2026-07-25-quantization-part-b.md`'s five sections, substituting §2 → "what the
integer tier converts and what it does not" and §3 → the conversion-stage error table.

**Numbers that already exist — cite, do not re-run:** end-to-end converted-graph rel-err
(`test_int_vit_graph.py:171-179`, seeds 0-7: 0.0557 0.0242 0.0094 0.0103 0.0150 0.0253
0.0237 0.1037); per-stage budget (`:206-208`, Linear-only 0.0459 → +conv 0.0517 → +3 int
nonlinears 0.0557); ILayerNorm on the real FrameModel's 5 LayerNorms
(`test_int_layernorm.py:545-551`, 1.29-1.93 %); fitted-vs-envelope rsqrt index (`:345-351`);
ISoftmax vs float (`test_int_softmax.py:228`, mean 1.02e-03 / max 4.95e-03); row-sum bands
(`:247,257`); reciprocal-bit sensitivity (`:576-581`).

**Must be measured fresh:** op-count conversion table on the real model
(`{layernorm:5, softmax:2, gelu:6, conv:2, linear:14}` = 29 integer modules; `left_float` =
6 × Dropout + `mask_head.proj`; 19 `float_composites`); artifact size / BRAM budget
(`total_int_weights`, `total_params`, per-table entry counts are in the manifest but no doc
quotes them); `verify_export` `max_abs_diff` on a real model; hybrid numbers (D6/B3); QAT
numbers (D6/B4).

Report the 0.1037 seed honestly rather than dropping it: at that seed the *Linear-only*
conversion already costs 0.1016, i.e. it is one-batch PTQ of a randomly-initialised model,
and the entire integer nonlinear tier adds ~2 % on top.

---

## 8. D6 — verification debt

- **B3 hybrid** — folded into D0.1; it is a bug fix with a test, not separate work.
- **B4 QAT → integer** — nothing blocks it. `prepare_qat` = `insert_fake_quant` +
  `calibrate` (weights *and* activations), and scales are then frozen, so the calibration
  state the converter needs is present after fine-tuning. This is a test, not a change.
- **B2 a real trained checkpoint** — **no shortcut exists.** `HBTXR/algorithm/runs/` does
  not exist; there is no `.pt` anywhere under `algorithm/`. The 63 `best.pt` under
  `backup/results/HGTXR/...` belong to the legacy HGTXR architecture: their 138 keys are
  `patch_frontend.frame_embed.proj.*` / `backbone.attn_stages.N.{q,k,v,out}_proj.*`, while
  the current `FrameModel` has 92 keys named `patch_embed.proj.*` /
  `backbone.blocks.N.{norm1,attn.qkv,attn.proj,norm2,mlp.fc1,mlp.fc2}.*` — **not one key
  matches**, and `load_checkpoint` is `strict=True` (`engine/tools/checkpoint.py:25`).
  The raw dataset is on disk (`D:\dataset\EV_Eye\raw_data`, `target_data`) and the
  preprocessing chain exists (`dataset/preprocess/build_manifests.py`), but **no manifest
  `.jsonl` exists anywhere**. B2 therefore requires (a) running the manifest build, then
  (b) a real training run. Until then every accuracy number in this stack is on a
  randomly-initialised model and must be labelled as such.

---

## 9. Verification policy for Part D

Unchanged from Parts A-C, plus one addition forced by §6.4:

1. **Bit-exact golden** — every integer kernel compared element-wise `==` against its
   `i_ops` pure-Python twin.
2. **NEW: independent integer oracle for the graph tier.** The lowered graph is compared
   against `replay_block_int()` (pure Python, `i_ops` only), *not* against the per-op
   model with a tolerance. The per-op-vs-graph delta is a reported result, not a gate.
3. **Mutation-proof tests** — every new test must be shown RED under the exact mutation it
   guards, and the mutation reverted. Part C found several tests that could not fail; that
   check is now mandatory.
4. **No self-referential tolerances** — never normalise a bound by a value the code under
   test chose.
