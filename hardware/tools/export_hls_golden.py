#!/usr/bin/env python3
"""S1 — emit per-stage integer golden vectors for the HLS testbenches.

The golden's authority is ``algorithm/quantization`` (SPEC §9): pure-Python
arbitrary-precision integer kernels, compared element-wise with ``==`` and never with a
tolerance. This script builds a deterministic quantized block, walks it forward once to
choose every edge's grid from what that edge actually sees, and writes each stage's
inputs, payloads and expected outputs as the comma-separated ``.txt`` form the repo
already uses (``quantization/export_txt.py``).

Stdlib only, deliberately: neither this machine nor the WSL image that runs the HLS
testbenches has numpy or torch, and ``i_ops``/``i_block`` were written dependency-free
for exactly this reason. That also rules out ``lut_calibrate`` / ``int_calibrate*`` /
``export*``, which are all numpy — so the payload builders below are pure-Python fits.

ponytail: payloads are fitted to the observed range of seeded data, NOT calibrated from
a trained model — this repo has no checkpoint. That is what V1 needs, because V1 tests
DATAPATH EQUALITY (HLS int == python int), not accuracy. When a checkpoint and an export
dump exist, replace ``build_block`` with a manifest reader; the emitted file layout is
the contract and does not change.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import types
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "algorithm"))          # the pin algorithm/conftest.py uses

# quantization/__init__.py eagerly imports the torch-backed calibrators, so importing the
# package would pull in torch to reach two modules that deliberately have no dependencies.
# Bind the namespace by hand and the submodule import below skips __init__ entirely.
_pkg = types.ModuleType("quantization")
_pkg.__path__ = [str(_ROOT / "algorithm" / "quantization")]
sys.modules.setdefault("quantization", _pkg)

from quantization.i_block import (                    # noqa: E402
    AddSpec, BlockSpec, ConvSpec, LayerNormSpec, LinearSpec, MatMulSpec, SoftmaxSpec,
    TableSpec, replay_block_int, rescale,
    # privates: the oracle's own composition helpers. Imported rather than re-written —
    # a second copy of "_linear" here is a second thing that can drift from the golden.
    _conv_tokens, _flat, _linear, _linear_acc, _requant_columns, _rows,
)
from quantization.i_ops import dyadic_params, int_matmul, requant, table_quantize  # noqa: E402
from quantization.scheme import QuantDtype, clamp_int                     # noqa: E402

EPS = 1e-5                       # LayerNorm eps, the same ceiling F.layer_norm has
NL_BITS = 16                     # SPEC §3: nonlinear LUT entries are 16-bit
EXP_SCALE = 1 << 15              # softmax exp table numerator

# SPEC §1. `img` is (channels, height, width); track carries 2 event polarities.
MODES: dict[str, dict] = {
    "search": dict(img=(1, 128, 128), patch=16, dim=192, heads=3, ff=768),
    "track":  dict(img=(2,  64,  64), patch=16, dim=192, heads=3, ff=768),
    # tiny is the development preset: same graph, minutes -> seconds.
    "tiny":   dict(img=(1,  32,  32), patch=16, dim=24,  heads=2, ff=48),
}

# SPEC §3 LUT sizes. RECIP 128 is the storage figure and the kernel has two segments,
# so each reciprocal segment gets 64 entries.
ENTRIES = dict(rsqrt=64, exp=32, recip=64, gelu=32)


# --- small numeric helpers ----------------------------------------------------

def _pot_index(lo: int, hi: int, entries: int) -> tuple[int, int, int]:
    """``(offset, shift, bound)`` for ``cursor = clamp((x + offset) >> shift, 0, bound)``.

    Numpy-free port of ``quantization/lut_calibrate.make_pot_index_params``.
    """
    bound = entries - 1
    span = max(hi - lo, 1)
    per_bin = -(-span // bound)                       # ceil
    return -lo, max(0, (per_bin - 1).bit_length()), bound


def _coords(offset: int, shift: int, bound: int) -> list[int]:
    """The lowest integer input each table entry catches."""
    return [(i << shift) - offset for i in range(bound + 1)]


def _pot_scale(max_abs: float, bits: int) -> float:
    """Smallest power-of-two scale that fits ``max_abs`` into a signed ``bits`` grid.

    Power-of-two because the two rsqrt segments must share one scale for the single
    ``>> s2`` to stay segment-independent (``i_ops`` segmented-LayerNorm docstring), and
    because a PoT scale costs a shift rather than a multiplier in hardware.
    """
    qmax = (1 << (bits - 1)) - 1
    return 2.0 ** math.ceil(math.log2(max(max_abs, 1e-30) / qmax))


def _shift_to_fit(magnitude: int, qmax: int) -> int:
    """Right shift that brings ``magnitude`` down to about ``qmax`` — fills the grid."""
    return max(0, magnitude.bit_length() - qmax.bit_length())


def _sum_grid(a: list[int], scale_a: float, b: list[int], scale_b: float,
              dtype: QuantDtype) -> float:
    """Grid for a residual join, sized to the sum it will actually carry.

    Picking this by fiat (``1.6 * stream_scale`` and the like) under-fills the wider
    grids: at 8-bit it left the whole block output spanning 37 of 255 levels.
    """
    peak = max(abs(x * scale_a + y * scale_b) for x, y in zip(a, b))
    return max(peak, 1e-12) / dtype.qmax


def _grid(values, scale: float, dtype: QuantDtype) -> float:
    """The grid a consumer should declare for integers ``values`` living on ``scale``.

    This is the whole reason the builder walks the block instead of guessing: an
    analytic sqrt(CI) estimate would be within ~2x, and 2x on a 4-bit grid is the
    difference between exercising the datapath and clamping every element flat.
    """
    peak = max((abs(int(v)) for v in values), default=1) * scale
    return max(peak, 1e-12) / dtype.qmax


# --- payload builders ---------------------------------------------------------

def _linear_spec(out_f: int, in_f: int, input_scale: float, bits: int,
                 rng: random.Random) -> LinearSpec:
    """Random weights quantized PER OUT-CHANNEL — the granularity the oracle assumes."""
    qmax = (1 << (bits - 1)) - 1
    real = [[rng.gauss(0.0, 0.05) for _ in range(in_f)] for _ in range(out_f)]
    scales = [max(max(abs(v) for v in row), 1e-8) / qmax for row in real]
    weight = [[clamp_int(round(v / s), -qmax - 1, qmax) for v in row]
              for row, s in zip(real, scales)]
    return LinearSpec(weight, scales, input_scale, [rng.gauss(0.0, 0.01) for _ in range(out_f)])


def _layernorm_payload(x_int: list[int], channels: int, input_scale: float,
                       clamp_bits: int, rng: random.Random) -> LayerNormSpec:
    """Fit the 7-scalar LayerNorm payload to the variance this input actually produces.

    One rsqrt segment (SPEC §3: RSQRT 64 entries), so this is the 7-scalar kernel; the
    10-scalar segmented one exists in ``i_ops`` for heteroscedastic real activations and
    is not what seeded data needs.
    """
    c_1_s = 16
    c_1_m = round((1 << c_1_s) / channels)
    qmax = (1 << (clamp_bits - 1)) - 1

    var_sums, diffs = [], 0
    for start in range(0, len(x_int), channels):
        row = x_int[start:start + channels]
        mean = (sum(row) * c_1_m + (1 << (c_1_s - 1))) >> c_1_s
        var_sums.append(sum((v - mean) ** 2 for v in row))
        diffs = max(diffs, max(abs(v - mean) for v in row))

    b, s1, bound = _pot_index(min(var_sums), max(var_sums), ENTRIES["rsqrt"])

    # var_real = var_sum * input_scale^2 / channels. The entry that minimaxes relative
    # error over a bin is 2/(sqrt(a)+sqrt(b)) at the bin's two ends, not 1/sqrt(a) at the
    # lower edge — the lower edge biases the whole layer high (int_calibrate.py:170).
    unit = input_scale * input_scale / channels
    rsqrt_real = []
    for coord in _coords(b, s1, bound):
        lo = max(coord, 0) * unit
        hi = lo + ((1 << s1) - 1) * unit
        rsqrt_real.append(2.0 / (math.sqrt(lo + EPS) + math.sqrt(hi + EPS)))
    rsqrt_scale = _pot_scale(max(rsqrt_real), NL_BITS)
    rsqrt_table = [max(1, round(r / rsqrt_scale)) for r in rsqrt_real]

    lnw_real = [rng.gauss(1.0, 0.1) for _ in range(channels)]
    lnb_real = [rng.gauss(0.0, 0.05) for _ in range(channels)]
    lnw_scale = _pot_scale(max(abs(v) for v in lnw_real), NL_BITS)
    lnw = [round(v / lnw_scale) for v in lnw_real]
    affine_scale = input_scale * rsqrt_scale * lnw_scale      # unit of (x-mean)*rsqrt*lnw
    lnb = [round(v / affine_scale) for v in lnb_real]

    peak = diffs * max(rsqrt_table) * max(abs(v) for v in lnw) + max(abs(v) for v in lnb)
    s2 = _shift_to_fit(peak, qmax)
    return LayerNormSpec([c_1_m, c_1_s, b, s1, bound, s2, clamp_bits],
                         lnw, lnb, rsqrt_table, None,
                         input_scale, affine_scale * (1 << s2))


def _softmax_payload(scores: list[int], tokens: int, input_scale: float,
                     out_bits: int = 8) -> SoftmaxSpec:
    """Fit the 14-scalar softmax payload: exp table + a two-segment reciprocal."""
    qmax_out = (1 << out_bits) - 1
    rows = [scores[i:i + tokens] for i in range(0, len(scores), tokens)]
    b1, s1, bound1 = _pot_index(0, max(max(r) - min(r) for r in rows), ENTRIES["exp"])

    # exp is evaluated at the bin MIDPOINT: the index floors, so the lower edge would
    # systematically over-weight every non-maximal token.
    half = ((1 << s1) - 1) / 2.0
    exp_table = [clamp_int(round(math.exp(-(max(c, 0) + half) * input_scale) * EXP_SCALE),
                           0, EXP_SCALE)
                 for c in _coords(b1, s1, bound1)]

    accs = []
    for row in rows:
        top = max(row)
        accs.append(sum(exp_table[clamp_int((top - v + b1) >> s1, 0, bound1)] for v in row))
    acc_lo, acc_hi = min(accs), max(accs)
    split = max(acc_lo + 1, int(math.sqrt(float(acc_lo) * float(acc_hi))))

    def segment(lo: int, hi: int) -> tuple[list[int], int, int, int, int, int]:
        b2, s2, bound2 = _pot_index(lo, hi, ENTRIES["recip"])
        # rel = (exp*recip + b3) >> s3 must land on qmax_out*exp/acc, so recip carries the
        # numerator qmax_out<<s3 — one per segment, which is why b3/s3 are per-segment.
        s3 = max(1, int(math.floor(math.log2(((1 << NL_BITS) - 1) * lo / float(qmax_out)))))
        table = [max(1, min((1 << NL_BITS) - 1,
                            round(qmax_out * (1 << s3) / max(c + ((1 << s2) - 1) / 2.0, 1.0))))
                 for c in _coords(b2, s2, bound2)]
        return table, b2, s2, bound2, 1 << (s3 - 1), s3

    one, b2_1, s2_1, bnd_1, b3_1, s3_1 = segment(acc_lo, split)
    two, b2_2, s2_2, bnd_2, b3_2, s3_2 = segment(split, acc_hi)
    return SoftmaxSpec([b1, s1, bound1, b2_1, s2_1, bnd_1, b3_1, s3_1,
                        b2_2, s2_2, bnd_2, b3_2, s3_2, out_bits],
                       exp_table, one, two, input_scale, 1.0 / qmax_out)


def _gelu_payload(x_int: list[int], input_scale: float, out_bits: int) -> TableSpec:
    """Pointwise GeLU LUT — ``i_ops.table_quantize``, tanh approximation."""
    b, s, bound = _pot_index(min(x_int), max(x_int), ENTRIES["gelu"])
    qmax = (1 << (out_bits - 1)) - 1
    real = []
    for coord in _coords(b, s, bound):
        v = (coord + ((1 << s) - 1) / 2.0) * input_scale
        real.append(0.5 * v * (1.0 + math.tanh(math.sqrt(2.0 / math.pi)
                                               * (v + 0.044715 * v ** 3))))
    out_scale = max(max(abs(r) for r in real), 1e-8) / qmax
    return TableSpec([b, s, bound],
                     [clamp_int(round(r / out_scale), -qmax - 1, qmax) for r in real],
                     input_scale, out_scale)


# --- the calibrating walk -----------------------------------------------------
#
# Steps through i_block.replay_block_int in its own order, choosing each consumer's
# declared scale from the accumulator it actually sees, and keeping every intermediate.
# The taps are the per-stage goldens; replay_block_int then re-runs the finished spec and
# must reproduce the walk's output, which is what proves the walk did not drift.

# Every edge is nudged off its producer's grid. An edge whose ratio is exactly 1.0 is
# short-circuited by i_block.rescale, so a golden built on equal grids would never
# exercise the requant unit at all.
_NUDGE = (0.91, 0.87, 1.13)


def build_block(x_int: list[int], tokens: int, cfg: dict, dtype: QuantDtype,
                stream_scale: float, rng: random.Random) -> tuple[BlockSpec, dict]:
    dim, heads = cfg["dim"], cfg["heads"]
    head_dim, ff = dim // heads, cfg["ff"]
    tap: dict[str, list[int]] = {"block_x": list(x_int)}

    def linear(out_f, in_f, in_scale):
        return _linear_spec(out_f, in_f, in_scale, dtype.bits, rng)

    # --- attention ------------------------------------------------------------
    ln1_in = rescale(x_int, stream_scale, stream_scale * _NUDGE[0], dtype=dtype)
    norm1 = _layernorm_payload(ln1_in, dim, stream_scale * _NUDGE[0], dtype.bits, rng)
    normed = norm1.apply(ln1_in)
    tap["ln1_x"], tap["ln1_y"] = ln1_in, normed

    qkv = linear(3 * heads * head_dim, dim, norm1.output_scale * _NUDGE[1])
    qkv_in = rescale(normed, norm1.output_scale, qkv.input_scale, dtype=dtype)
    qkv_acc = _linear_acc(qkv, _rows(qkv_in, dim))

    def part(index: int) -> tuple[float, list[list[list[int]]]]:
        span = range(index * heads * head_dim, (index + 1) * heads * head_dim)
        scale = _grid((qkv_acc[t][oc] for t in range(tokens) for oc in span),
                      max(qkv.out_scale(oc) for oc in span), dtype)
        return scale, [_requant_columns(qkv, qkv_acc, scale, dtype,
                                        columns=range(index * heads * head_dim + h * head_dim,
                                                      index * heads * head_dim + (h + 1) * head_dim))
                       for h in range(heads)]

    (s_q, query), (s_k, key), (s_v, value) = part(0), part(1), part(2)
    qk, av_b = MatMulSpec(s_q, s_k), s_v
    attn_scale = 1.0 / math.sqrt(head_dim)

    score_acc = [int_matmul(query[h], [list(c) for c in zip(*key[h])]) for h in range(heads)]
    flat_scores = [v for acc in score_acc for v in _flat(acc)]
    s_score = _grid(flat_scores, qk.out_scale * attn_scale, dtype)
    scores = rescale(flat_scores, qk.out_scale, s_score, dtype=dtype, extra=attn_scale)
    softmax = _softmax_payload(scores, tokens, s_score)
    probs = softmax.apply(scores, tokens=tokens, heads=heads)
    tap["qkv_x"], tap["qkv_y"] = qkv_in, _flat(_rows([v for h in query for r in h for v in r], head_dim))
    tap["smu_a"] = [v for h in query for r in h for v in r]
    tap["smu_b"] = [v for h in key for r in h for v in r]
    tap["smu_y"] = scores
    tap["softmax_x"], tap["softmax_y"] = scores, probs

    # softmax output is unsigned 8-bit; map its full scale onto the operand grid
    av = MatMulSpec(softmax.output_scale * ((1 << 8) - 1) / dtype.qmax, av_b)
    context_acc, ctx_flat = [], []
    for h in range(heads):
        block = probs[h * tokens * tokens:(h + 1) * tokens * tokens]
        rows = _rows(rescale(block, softmax.output_scale, av.scale_a, dtype=dtype), tokens)
        acc = int_matmul(rows, value[h])
        context_acc.append(acc)
        ctx_flat.extend(_flat(acc))
    s_ctx = _grid(ctx_flat, av.out_scale, dtype)
    heads_int = [_rows(rescale(_flat(a), av.out_scale, s_ctx, dtype=dtype), head_dim)
                 for a in context_acc]
    context = [[heads_int[h][t][c] for h in range(heads) for c in range(head_dim)]
               for t in range(tokens)]

    proj = linear(dim, dim, s_ctx)
    s_proj = _grid(_flat(_linear_acc(proj, context)),
                   max(proj.out_scale(oc) for oc in range(dim)), dtype)
    tap["rmu_x"], tap["rmu_y"] = _flat(context), _flat(_linear(proj, context, s_proj, dtype))
    attn_residual = AddSpec(stream_scale, s_proj,
                            _sum_grid(x_int, stream_scale, tap["rmu_y"], s_proj, dtype))

    # --- MLP ------------------------------------------------------------------
    from quantization.i_block import _add                       # same import rationale
    residual = _add(attn_residual, x_int, tap["rmu_y"], dtype)
    tap["mha_y"] = residual

    ln2_in = rescale(residual, attn_residual.scale_out, attn_residual.scale_out * _NUDGE[2],
                     dtype=dtype)
    norm2 = _layernorm_payload(ln2_in, dim, attn_residual.scale_out * _NUDGE[2],
                               dtype.bits, rng)
    fc1 = linear(ff, dim, norm2.output_scale * _NUDGE[0])
    fc1_in = _rows(rescale(norm2.apply(ln2_in), norm2.output_scale, fc1.input_scale,
                           dtype=dtype), dim)
    s_hidden = _grid(_flat(_linear_acc(fc1, fc1_in)),
                     max(fc1.out_scale(oc) for oc in range(ff)), dtype)
    hidden = _flat(_linear(fc1, fc1_in, s_hidden, dtype))
    # The LUT emits hbtxr_nl_t (16-bit, SPEC §3); the narrowing to the fc2 operand grid is
    # the requant on the edge below, not the table. Wiring the table to dtype.bits instead
    # collapses GeLU to 2 distinct outputs at 4-bit — check() catches it.
    gelu = _gelu_payload(hidden, s_hidden, NL_BITS)
    activated = gelu.apply(hidden)
    tap["gelu_x"], tap["gelu_y"] = hidden, activated

    fc2 = linear(dim, ff, gelu.output_scale * _NUDGE[1])
    fc2_in = _rows(rescale(activated, gelu.output_scale, fc2.input_scale, dtype=dtype), ff)
    s_mlp = _grid(_flat(_linear_acc(fc2, fc2_in)),
                  max(fc2.out_scale(oc) for oc in range(dim)), dtype)
    mlp_out = _flat(_linear(fc2, fc2_in, s_mlp, dtype))
    mlp_residual = AddSpec(attn_residual.scale_out, s_mlp,
                           _sum_grid(residual, attn_residual.scale_out, mlp_out, s_mlp, dtype))
    tap["mlp_x"] = residual
    tap["mlp_y"] = _add(mlp_residual, residual, mlp_out, dtype)

    spec = BlockSpec(heads, head_dim, norm1, qkv, qk, attn_scale, softmax,
                     MatMulSpec(av.scale_a, av.scale_b), proj, attn_residual,
                     norm2, fc1, gelu, fc2, mlp_residual, dtype)
    return spec, tap


def build_patch_embed(cfg: dict, out_scale: float, dtype: QuantDtype,
                      rng: random.Random) -> tuple[ConvSpec, list[int], list[list[int]]]:
    """Conv-F / Conv-E stem: kernel == stride == patch, 8-bit weights, asymmetric input.

    The image grid is asymmetric, so a real zero is the zero-point and NOT 0 — the
    ``- zp*sum(w)`` correction in ``_conv_tokens`` depends on it.
    """
    channels, height, width = cfg["img"]
    patch, dim = cfg["patch"], cfg["dim"]
    qmax = (1 << 7) - 1
    real = [[[[rng.gauss(0.0, 0.05) for _ in range(patch)] for _ in range(patch)]
             for _ in range(channels)] for _ in range(dim)]
    scales = [max(max(abs(v) for plane in oc for row in plane for v in row), 1e-8) / qmax
              for oc in real]
    weight = [[[[clamp_int(round(v / s), -qmax - 1, qmax) for v in row] for row in plane]
               for plane in oc] for oc, s in zip(real, scales)]
    spec = ConvSpec(weight, scales, 1.0 / 255.0, 128, patch, (0, 0),
                    [rng.gauss(0.0, 0.01) for _ in range(dim)])
    image = [rng.randrange(0, 256) for _ in range(channels * height * width)]
    tokens = _conv_tokens(spec, image, out_scale, dtype,
                          channels=channels, height=height, width=width)
    return spec, image, tokens


# --- emission -----------------------------------------------------------------

class Writer:
    """Comma-separated integer files plus an index — ``quantization/export_txt.py``'s form."""

    def __init__(self, dest: Path):
        self.dest = dest
        self.dest.mkdir(parents=True, exist_ok=True)
        for stale in self.dest.glob("*.txt"):     # a renamed field must not linger and
            stale.unlink()                        # get read by a testbench as current
        self.index: list[str] = []

    def put(self, name: str, values, shape: str, note: str) -> None:
        flat = [int(v) for v in values]
        (self.dest / f"{name}.txt").write_text(",".join(map(str, flat)) + ",", encoding="ascii")
        self.index.append(f"{name}.txt\t{len(flat)}\t{shape}\t{note}")

    def requant(self, name: str, ratios: list[float], note: str) -> None:
        """A dyadic (multiplier, shift) pair per output channel — what the HW unit holds.

        A per-tensor edge may legitimately be identity: ``rescale`` short-circuits ratio
        1.0, and ``dyadic_params(1.0)`` is ``(2, 1)``, which is bit-exact identity anyway,
        so the pair is emitted either way and the testbench needs no special case. A
        PER-CHANNEL edge cannot be: those grids are derived from the accumulator, and a
        unit ratio there means the derivation collapsed.
        """
        if len(ratios) > 1 and any(r == 1.0 for r in ratios):
            raise AssertionError(f"{name}: a per-channel requant ratio is exactly 1.0 — "
                                 "the output grid was not derived from the accumulator")
        if any(r == 1.0 for r in ratios):
            note += " [identity]"
        pairs = [dyadic_params(r) for r in ratios]
        self.put(f"{name}_mult", [m for m, _, _ in pairs], f"[{len(pairs)}]", note)
        self.put(f"{name}_shift", [s for _, s, _ in pairs], f"[{len(pairs)}]", note)

    def close(self) -> None:
        (self.dest / "index.tsv").write_text(
            "file\tcount\tshape\tnote\n" + "\n".join(self.index) + "\n", encoding="utf-8")


def _bias_int(spec: LinearSpec) -> list[int]:
    """The bias on the ACCUMULATOR grid — where the oracle adds it, and hardware too."""
    return [round(float(spec.bias[oc]) / spec.out_scale(oc)) for oc in range(len(spec.weight))]


def emit(dest: Path, mode: str, bits: int, seed: int) -> tuple[Path, dict]:
    cfg = MODES[mode]
    dim, heads = cfg["dim"], cfg["heads"]
    head_dim, ff = dim // heads, cfg["ff"]
    tokens = (cfg["img"][1] // cfg["patch"]) * (cfg["img"][2] // cfg["patch"])
    dtype = QuantDtype(bits, signed=True)
    rng = random.Random(seed)
    stream_scale = 1.0 / 64.0

    stem, image, token_rows = build_patch_embed(cfg, stream_scale, dtype, rng)
    x_int = _flat(token_rows)
    spec, tap = build_block(x_int, tokens, cfg, dtype, stream_scale, rng)
    golden = replay_block_int(spec, x_int, tokens=tokens, channels=dim)

    w = Writer(dest)
    w.put("meta", [tokens, dim, heads, head_dim, ff, bits, cfg["patch"], *cfg["img"], seed],
          "[11]", "tokens dim heads head_dim ff bits patch img_c img_h img_w seed")

    # patch embedding
    w.put("patch_image", image, f"[{cfg['img'][0]},{cfg['img'][1]},{cfg['img'][2]}]",
          "uint8 row-major, zero_point 128")
    w.put("patch_weight", [v for oc in stem.weight for p in oc for r in p for v in r],
          f"[{dim},{cfg['img'][0]},{cfg['patch']},{cfg['patch']}]", "int8 [Cout,Cin,kh,kw]")
    w.put("patch_bias_acc", [round(float(stem.bias[oc]) / stem.out_scale(oc)) for oc in range(dim)],
          f"[{dim}]", "on the accumulator grid, added before the requant")
    w.put("patch_zp_correction", [stem.zero_point * sum(v for p in stem.weight[oc] for r in p for v in r)
                                  for oc in range(dim)], f"[{dim}]", "zp * sum(w), subtracted from acc")
    w.requant("patch", [stem.out_scale(oc) / stream_scale for oc in range(dim)],
              "per out-channel, acc -> token grid")
    w.put("patch_y", x_int, f"[{tokens},{dim}]", "tokens, row-major — the block input")

    # LayerNorm (x2), softmax, GeLU: payload + vector for the S3 testbenches
    for name, ln in (("ln1", spec.norm1), ("ln2", spec.norm2)):
        w.put(f"{name}_scalars", ln.scalars, "[7]", "c_1_m c_1_s b s1 bound s2 clamp_bits")
        w.put(f"{name}_lnw", ln.lnw, f"[{dim}]", "int16")
        w.put(f"{name}_lnb", ln.lnb, f"[{dim}]", "int16, on the affine grid")
        w.put(f"{name}_rsqrt_table", ln.rsqrt_table, f"[{ENTRIES['rsqrt']}]", "int16")
    w.put("ln1_x", tap["ln1_x"], f"[{tokens},{dim}]", "on norm1.input_scale")
    w.put("ln1_y", tap["ln1_y"], f"[{tokens},{dim}]", "expected")

    w.put("softmax_scalars", spec.softmax.scalars, "[14]",
          "b1 s1 bound1 | b2/s2/bound2/b3/s3 per recip segment | clamp_bits")
    w.put("softmax_exp_table", spec.softmax.exp_table, f"[{ENTRIES['exp']}]", "int16")
    w.put("softmax_recip_table_one", spec.softmax.recip_table_one, f"[{ENTRIES['recip']}]", "int16")
    w.put("softmax_recip_table_two", spec.softmax.recip_table_two, f"[{ENTRIES['recip']}]", "int16")
    w.put("softmax_x", tap["softmax_x"], f"[{heads},{tokens},{tokens}]", "scores")
    w.put("softmax_y", tap["softmax_y"], f"[{heads},{tokens},{tokens}]", "expected, uint8")

    w.put("gelu_scalars", spec.gelu.scalars, "[3]", "b s bound")
    w.put("gelu_table", spec.gelu.table, f"[{ENTRIES['gelu']}]", "int16")
    w.put("gelu_x", tap["gelu_x"], f"[{tokens},{ff}]", "fc1 output")
    w.put("gelu_y", tap["gelu_y"], f"[{tokens},{ff}]", "expected")

    # the four resident-weight matmuls. `rmu` is the output projection, the canonical
    # single-output-grid RMU; qkv is the one whose accumulator feeds three consumers.
    qkv_out = 3 * heads * head_dim
    for name, lin, shape, note in (
            ("rmu", spec.proj, f"[{dim},{dim}]", "output projection"),
            ("qkv", spec.qkv, f"[{qkv_out},{dim}]", "out laid out (3,H,d)"),
            ("fc1", spec.fc1, f"[{ff},{dim}]", "MLP expand"),
            ("fc2", spec.fc2, f"[{dim},{ff}]", "MLP contract")):
        w.put(f"{name}_weight", [v for row in lin.weight for v in row], shape,
              f"int{bits} [out,in], {note}")
        w.put(f"{name}_bias_acc", _bias_int(lin), f"[{len(lin.weight)}]",
              "accumulator grid — added BEFORE the requant, where it is exact")
    w.put("rmu_x", tap["rmu_x"], f"[{tokens},{dim}]", "attention context")
    w.put("rmu_y", tap["rmu_y"], f"[{tokens},{dim}]", "expected")
    w.put("qkv_x", tap["qkv_x"], f"[{tokens},{dim}]", "on qkv.input_scale")

    # SMU — both operands stream. Q x K^T, with 1/sqrt(d) folded into the requant.
    w.put("smu_a", tap["smu_a"], f"[{heads},{tokens},{head_dim}]", "Q")
    w.put("smu_b", tap["smu_b"], f"[{heads},{tokens},{head_dim}]", "K, transposed by the SMU")
    w.put("smu_y", tap["smu_y"], f"[{heads},{tokens},{tokens}]", "expected")

    # Every requant in the block, in forward order. A golden a testbench cannot reproduce
    # is not a golden, so this is the whole set, not the interesting ones.
    def channels(lin: LinearSpec, target: float, span: range | None = None):
        return [lin.out_scale(oc) / target for oc in (span or range(len(lin.weight)))]

    ar, mr, sm = spec.attn_residual, spec.mlp_residual, spec.softmax
    edges: list[tuple[str, list[float], str]] = [
        ("e01_stream_to_ln1", [ar.scale_a / spec.norm1.input_scale], "per tensor"),
        ("e02_ln1_to_qkv", [spec.norm1.output_scale / spec.qkv.input_scale], "per tensor"),
        ("e03_qkv_q", channels(spec.qkv, spec.qk.scale_a, range(0, heads * head_dim)),
         "from the accumulator DIRECTLY; a shared Q/K/V grid clamps K and V"),
        ("e03_qkv_k", channels(spec.qkv, spec.qk.scale_b,
                               range(heads * head_dim, 2 * heads * head_dim)), "as above"),
        ("e03_qkv_v", channels(spec.qkv, spec.av.scale_b,
                               range(2 * heads * head_dim, qkv_out)), "as above"),
        ("e04_smu_to_softmax", [spec.qk.out_scale * spec.attn_scale / sm.input_scale],
         "per tensor; 1/sqrt(d) is folded in here, it is not a separate multiply"),
        ("e05_softmax_to_av", [sm.output_scale / spec.av.scale_a], "per tensor"),
        ("e06_av_to_proj", [spec.av.out_scale / spec.proj.input_scale], "per tensor"),
        ("e07_proj_acc", channels(spec.proj, ar.scale_b), "per out-channel"),
        ("e08_resid1_a", [ar.scale_a / ar.scale_out], "residual side of the attention join"),
        ("e08_resid1_b", [ar.scale_b / ar.scale_out], "branch side"),
        ("e09_stream_to_ln2", [ar.scale_out / spec.norm2.input_scale], "per tensor"),
        ("e10_ln2_to_fc1", [spec.norm2.output_scale / spec.fc1.input_scale], "per tensor"),
        ("e11_fc1_acc", channels(spec.fc1, spec.gelu.input_scale), "per out-channel"),
        ("e12_gelu_to_fc2", [spec.gelu.output_scale / spec.fc2.input_scale], "per tensor"),
        ("e13_fc2_acc", channels(spec.fc2, mr.scale_b), "per out-channel"),
        ("e14_resid2_a", [mr.scale_a / mr.scale_out], "residual side of the MLP join"),
        ("e14_resid2_b", [mr.scale_b / mr.scale_out], "branch side"),
    ]
    for name, ratios, note in edges:
        w.requant(name, ratios, note)

    # whole-sublayer and whole-block goldens
    w.put("mha_x", x_int, f"[{tokens},{dim}]", "block input")
    w.put("mha_y", tap["mha_y"], f"[{tokens},{dim}]", "attention sublayer incl. residual merge")
    w.put("mlp_x", tap["mlp_x"], f"[{tokens},{dim}]", "= mha_y")
    w.put("mlp_y", tap["mlp_y"], f"[{tokens},{dim}]", "MLP sublayer incl. residual merge")
    w.put("block_x", x_int, f"[{tokens},{dim}]", "block input")
    w.put("block_y", golden, f"[{tokens},{dim}]", "replay_block_int — the S8 reference")
    w.close()
    return dest, dict(spec=spec, tap=tap, golden=golden, dtype=dtype, tokens=tokens, dim=dim)


# --- the check ----------------------------------------------------------------

def _read(dest: Path, name: str) -> list[int]:
    """Parse one emitted file. The trailing comma makes a naive split yield an empty field."""
    return [int(t) for t in (dest / f"{name}.txt").read_text("ascii").split(",") if t]


def _reconstruct(dest: Path) -> None:
    """Rebuild three goldens FROM THE FILES ONLY — no spec object, no scales.

    This is the file contract: it is what the C++ testbench does, so if these three
    reproduce, a testbench reading the same files has everything it needs. Between them
    they cover per-out-channel requant with the bias on the accumulator (rmu), LUT cursor
    indexing (gelu), and the transpose with 1/sqrt(d) folded into the requant (smu).
    """
    tokens, dim, heads, head_dim, ff, bits = _read(dest, "meta")[:6]

    weight = _rows(_read(dest, "rmu_weight"), dim)
    bias, mult, shift = (_read(dest, n) for n in
                         ("rmu_bias_acc", "e07_proj_acc_mult", "e07_proj_acc_shift"))
    x = _rows(_read(dest, "rmu_x"), dim)
    acc = int_matmul(x, [list(c) for c in zip(*weight)])
    columns = [requant([row[oc] + bias[oc] for row in acc], mult[oc], shift[oc],
                       bits=bits, signed=True) for oc in range(dim)]
    if [columns[oc][t] for t in range(tokens) for oc in range(dim)] != _read(dest, "rmu_y"):
        raise AssertionError("rmu_y is not reproducible from the emitted files")

    if table_quantize(_read(dest, "gelu_x"), _read(dest, "gelu_scalars"),
                      _read(dest, "gelu_table")) != _read(dest, "gelu_y"):
        raise AssertionError("gelu_y is not reproducible from the emitted files")

    q, k = _read(dest, "smu_a"), _read(dest, "smu_b")
    (sm_mult,), (sm_shift,) = _read(dest, "e04_smu_to_softmax_mult"), \
        _read(dest, "e04_smu_to_softmax_shift")
    scores: list[int] = []
    for h in range(heads):
        base = h * tokens * head_dim
        rows_q = _rows(q[base:base + tokens * head_dim], head_dim)
        rows_k = _rows(k[base:base + tokens * head_dim], head_dim)
        acc = int_matmul(rows_q, [list(c) for c in zip(*rows_k)])
        scores.extend(requant(_flat(acc), sm_mult, sm_shift, bits=bits, signed=True))
    if scores != _read(dest, "smu_y"):
        raise AssertionError("smu_y is not reproducible from the emitted files")


def check(dest: Path, result: dict) -> None:
    """Three things, all of which have a way to be silently wrong.

    1. the walk reproduces ``replay_block_int``. The walk chooses the scales AND records
       the per-stage goldens, so if its composition drifts from the oracle every stage
       vector is wrong together and nothing else would notice;
    2. no golden is degenerate. On a 4-bit grid a mis-sized scale clamps a tensor flat,
       and an HLS block compared against a constant passes for the wrong reason;
    3. the files round-trip. The trailing comma makes a naive split yield an empty field.
    """
    tap, dtype = result["tap"], result["dtype"]
    if tap["mlp_y"] != result["golden"]:
        bad = next(i for i, (a, b) in enumerate(zip(tap["mlp_y"], result["golden"])) if a != b)
        raise AssertionError(f"walk != replay_block_int, first at {bad}: "
                             f"{tap['mlp_y'][bad]} vs {result['golden'][bad]}")

    _reconstruct(dest)

    for name in ("patch_y", "ln1_y", "softmax_y", "gelu_y", "rmu_y", "smu_y", "block_y"):
        values = _read(dest, name)
        if not values:
            raise AssertionError(f"{name}: empty")
        distinct = len(set(values))
        if distinct < 3:
            raise AssertionError(f"{name}: only {distinct} distinct values — degenerate golden")
        span = max(values) - min(values)
        if span < dtype.qmax // 2 and name not in ("softmax_y",):
            raise AssertionError(f"{name}: spans {span} on a {dtype.bits}-bit grid — "
                                 "the scale is mis-sized, the golden barely moves")

    written = {p.stem for p in dest.glob("*.txt")}
    indexed = {line.split("\t")[0][:-4]
               for line in (dest / "index.tsv").read_text("utf-8").splitlines()[1:]}
    if written != indexed:
        raise AssertionError(f"index.tsv disagrees with the directory: {written ^ indexed}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=sorted(MODES), default="search")
    ap.add_argument("--bits", type=int, default=4,
                    help="activation/weight width of the MHA and MLP matmuls (SPEC §3)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=_ROOT / "hardware" / "workspace" / "golden")
    args = ap.parse_args(argv)

    dest = args.out / f"{args.mode}-a{args.bits}"
    _, result = emit(dest, args.mode, args.bits, args.seed)
    check(dest, result)
    print(f"{dest}  {len(list(dest.glob('*.txt')))} files  "
          f"tokens={result['tokens']} dim={result['dim']} bits={args.bits}  OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
