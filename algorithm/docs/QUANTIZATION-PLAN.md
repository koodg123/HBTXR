# HBTXR Quantization — Full Build Plan (HW-faithful · configurable · q/i tiers)

Confirmed plan for extending `algorithm/quantization/` from the current fake-quant
subsystem to a fully configurable, hardware-faithful quantization stack with a
verified pure-integer inference graph.

## 0. Status

- **Done (Q0–Q10)**: primitives (scheme / fake-quant STE / int_ops golden), Linear
  fake-quant insertion (QuantLinear), PTQ (1.7% box error), QAT, integer-linear
  equivalence (bit-exact), entrypoint/config/pytest, and the HW-friendly nonlinear
  LUTs — GeLU (1.1%), LayerNorm-rsqrt (0.12%), Softmax exp+recip (0.05%). All at the
  **fake-quant (Q) tier**; `int_ops.py` is the bit-exact integer golden reference.
- **This plan** adds: (A) a clean **q/i tier restructure**, (B) a **fully
  specifiable quant config** + observer/quantizer extensions, (C) the **HW-faithful
  integer graph (I tier)** covering **every** ViT module and functional op.
- **Progress**: Part A ✅ (`665ae48`, 12/12 regression). Part B ✅ (`d65dbcf` core +
  `1725ec8` overrides/entrypoint) — granularity / sym-asym / scale_type / calibration
  (minmax·percentile·mse·kl) + per-layer overrides; matrix report in
  `docs/QUANTIZATION-PARTB-REPORT.md`. Part C 🔵 in progress:
  - C0 ✅ (`6762e74`) QTensor + i_ops golden (requant/int_matmul/int_conv2d/dyadic_params)
    + torch int_functional, bit-exact.
  - C1 ✅ (`97a38d2`) ILinear (per-tensor/per-channel weight, asym-act zp fold) + IConv2d.
  - C2 ✅ (`8f614d6`) IMatMul (act×act, attention).
  - C3 🔵 IGeLU ✅ (`3684414`, bit-exact vs table_quantize); **ILayerNorm / ISoftmax
    pending** (fully-integer mean/var/rsqrt + exp/recip via i_ops layernorm_quantize/
    softmax_quantize goldens — need HG-PIPE calibrate_rsqrt/softmax scalars, a distinct
    calib path).
  - C4 ✅ (`3684414`) IAdd / ICat / IPool (scale alignment).
  - C5 ⏳ ilayers/vit.py integer ViT assembly + convert.py Q→I. C6 ⏳ export_int + whole-graph test.
  - **pytest 56/56** (12 regression + 19 config-matrix + 25 int-graph).

## 1. Integer representation policy (HW-faithful)

Values are stored at their real low bit-width; only accumulation / requant use wide
buffers — exactly as hardware does.

| Location | dtype | Rationale |
|---|---|---|
| value storage (weight·act) | `torch.int8` / `torch.uint8` | real bit width |
| int4 / uint4 | int8 container + clamp `[-8,7]`/`[0,15]` | torch has **no int4 compute** (storage/packing only) |
| matmul / conv accumulate | operands int8 → cast `int32`, **accumulate int32** | HW MAC (int8×int8→int32); no overflow for ViT dims (127²·192 ≈ 3M ≪ 2.1B) |
| requant multiply (`acc·mult>>shift`) | **int64** intermediate → shift → int8 | `acc(3M)·mult(~1M)=3e12 > int32` → wide multiplier (HW does the same) |
| reductions (LN mean/var, softmax sum, pool) | int32 (var → int64 if very large) | Σ int8 / Σ(x−μ)² |
| residual add / concat | align scales → int32 → requant → int8 | two int8 with different scales |
| LUT lookup (GeLU/rsqrt/exp/recip) | index int8/uint8 → value int8/uint8 | no accumulation |

`int_ops.py` (pure-Python arbitrary-precision int, no overflow) is the **golden**
that the torch int8/int32/int64 kernels are checked bit-exact against.

## 2. Complete op inventory (actual HBTXR ViT)

Every module / functional op, mapped to a Q/I module or passthrough.

| Site | Op | Kind | Q/I module |
|---|---|---|---|
| PatchEmbed | `nn.Conv2d` (strided) | weight×act | **QConv2d / IConv2d** |
| PatchEmbed | flatten / transpose | data movement | passthrough |
| Backbone/Block | `nn.LayerNorm` ×(2L+1) | norm | QLayerNorm / ILayerNorm |
| Block | `x + attn(...)`, `x + mlp(...)` | residual add | **QAdd / IAdd** (scale align) |
| MHA | `nn.Linear` qkv·proj | weight×act | QLinear / ILinear |
| MHA | `Q@Kᵀ`, `attn@V` | **act×act matmul** | **QMatMul / IMatMul** |
| MHA | `× self.scale` | scalar mul | folded into requant |
| MHA | `nn.Softmax` | norm | QSoftmax / ISoftmax |
| MHA/MLP | reshape/permute, `nn.Dropout` | data movement / inference no-op | passthrough |
| MLP | `nn.Linear` fc1·fc2 | weight×act | QLinear / ILinear |
| MLP | `nn.GELU` | nonlinear | QGeLU / IGeLU |
| Head | `pool_tokens` (mean) | reduction | **QPool / IPool** |
| Ellipse head | `torch.cat([feat, anchor])` | concat | **QCat / ICat** |
| Head | `tokens_to_grid` | data movement | passthrough |

New vs the earlier Linear-only sketch: **Conv2d · MatMul(act×act) · Add(residual) ·
Cat(concat) · Pool**.

## 3. QTensor abstraction (I tier)

Scale propagation across add/concat/passthrough needs the integer tensor to carry
its quantization metadata:

```
QTensor = (int_data: int8|uint8 tensor, scale, zero_point, dtype)
```

- passthrough (reshape/transpose): transform `int_data`, keep scale.
- IAdd: two QTensors (s1, s2) → requant each to a common `s_out` → int32 add → int8.
- ICat: requant all inputs to a common `s_out` → `torch.cat` (single scale).
- IMatMul: two activation QTensors → int32 accumulate → requant (`s_a·s_b → s_out`).

File: `ilayers/qtensor.py`. The Q tier keeps float I/O, so it needs no QTensor.

## 4. Package layout (q/i)

```
quantization/
  __init__.py
  scheme.py            # QuantDtype·qrange·clamp                (shared)
  observer.py          # observers: MinMax/Percentile/KL/MSE     (shared)
  lut_calibrate.py     # PoT/LUT table builders                  (shared)
  q_ops.py             # Q primitives: fake quantizers · STE
  i_ops.py             # I golden kernels (was int_ops.py) + requant · int-matmul · int-conv
  qlayers/
    __init__.py
    linear.py          # QLinear
    conv.py            # QConv2d
    matmul.py          # QMatMul
    nonlinear.py       # QGeLU · QLayerNorm · QSoftmax
    tensor_ops.py      # QAdd · QCat · QPool
  ilayers/
    __init__.py
    qtensor.py         # QTensor (int8 + scale)
    linear.py          # ILinear
    conv.py            # IConv2d
    matmul.py          # IMatMul
    nonlinear.py       # IGeLU · ILayerNorm · ISoftmax
    tensor_ops.py      # IAdd · ICat · IPool
    vit.py             # integer ViT assembly
  convert.py           # float→Q insert · observe→replace · Q→I convert
  calibrate.py qat.py  # PTQ / QAT flows
  entrypoint.py
```

## 5. Configurable quant spec (Part B)

Current support vs target (from the config analysis):

| Knob | Now | Target |
|---|---|---|
| bit-width | ✅ global (weight/act) | + per-layer overrides |
| granularity | ❌ per-tensor only | per-tensor/channel/group/block |
| symmetric | ✅ only mode | keep |
| asymmetric | ⚠️ structure only (zp unused) | wired observer + config |
| float scale | ✅ default | keep |
| dyadic scale | ❌ | add (`_dyadic_approx`, references/) |
| power-of-two scale | ❌ (PoT only for LUT index) | add (2^k affine) |
| calibration | ⚠️ MinMax only | MinMax/Percentile/KL/MSE (KL from references/) |

Target schema:

```yaml
quantization:
  weight:     {bits, granularity, group_size, symmetric, scale_type, calibration: {method, percentile}}
  activation: {bits, granularity, symmetric, scale_type, calibration: {method}}
  overrides:  [{match, bits, granularity, ...}]   # per-layer mixed precision
```

`granularity`: per-tensor|per-channel|per-group|per-block · `scale_type`:
float|dyadic|pot · `calibration.method`: minmax|percentile|kl|mse · `symmetric`:
true|false.

## 6. Phased plan

### Part A — q/i restructure (pure move + rename, no behavior change)
- **A1** create the layout of §4; move + rename existing Q-tier
  (`QuantLinear→QLinear`, `*LUT→QGeLU/QLayerNorm/QSoftmax`, `fake_quant.py→q_ops.py`,
  `int_ops.py→i_ops.py`, insert/calibrate → `convert.py`).
- **A2** update `__init__` / entrypoint / tests imports → **pytest 12/12 regression**.

### Part B — configurable quant (foundation, applies to ALL Q modules)
- **B0** `QuantSpec` schema + resolver (weight/act + overrides).
- **B1** granularity: quantizer scale scalar → **per-axis tensor**; observers reduce
  along the chosen axis; group/block reshape. (`quantize_group_vector` from references/)
- **B2** asymmetric: min/max → (scale, zero_point≠0) observer + config.
- **B3** scale_type: `float` / `dyadic` (`_dyadic_approx`) / `pot` (2^k).
- **B4** calibration strategy: MinMax / Percentile / KL / MSE. (`calibrate_dyadic_scale_kl`)
- **B5** per-layer overrides applied in `convert.py`.
- **B6** verify: config matrix (e.g. per-channel+asym+dyadic+percentile) PTQ-error report + pytest.

### Part C — HW-faithful integer graph (I tier, §1 policy)
- **C0** `ilayers/qtensor.py` (QTensor) + `i_ops` requant (int64 mul) / int-matmul (int32) / int-conv.
- **C1** ILinear · IConv2d (weight×act).
- **C2** IMatMul (act×act: Q@Kᵀ, attn@V).
- **C3** IGeLU · ILayerNorm · ISoftmax (scalars from B4 + HG-PIPE `calibrate_rsqrt/softmax`).
- **C4** IAdd · ICat · IPool (scale alignment).
- **C5** `ilayers/vit.py` integer ViT assembly (patch-embed → blocks with residual
  align → head) + `convert.py` Q→I.
- **C6** export (int8 weights · scalars · tables txt/json) + `entrypoint` export_int mode
  + `tests/quantization/test_int_graph.py`.

## 7. Verification (torch venv, every step)

1. **Bit-exact golden**: torch int8/int32 kernels (C) ⟺ `i_ops.py` pure-int golden.
2. **Close**: full I graph ⟺ Q graph (from A/B) within tolerance.
3. **Add/Cat align**: dedicated reconstruction-error tests after scale alignment.
4. **Config matrix**: per-knob-combo PTQ error report (B6).

## 8. Risks

- Residual/concat **scale alignment** and **dyadic requant error accumulation** in
  deep stacks → caught early by module-level golden + C5 whole-graph checks.
- **granularity × scale_type** combinatorial surface → B6 matrix verification.
- Effort: A (light) → B (medium) → C (largest, HW-codegen level). Commit + pytest each step.
