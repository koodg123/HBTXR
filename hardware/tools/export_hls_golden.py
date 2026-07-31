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
    AddSpec, BlockSpec, ConvSpec, HeadSpec, LayerNormSpec, LinearSpec, MatMulSpec,
    ModelSpec, SoftmaxSpec, TableSpec, replay_block_int, replay_model_int, rescale,
    # privates: the oracle's own composition helpers. Imported rather than re-written —
    # a second copy of "_linear" here is a second thing that can drift from the golden.
    _conv_tokens, _flat, _linear, _linear_acc, _pool_tokens, _requant_columns, _rows,
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

# --- the whole model ----------------------------------------------------------
#
# ONE shared backbone (models/backbones/vit.py): search runs B_1:depth, track runs B_1:cut
# on the SAME blocks and the SAME final norm. Two stems and two heads differ.
#
#   search:  Conv-F(1, 128x128) -> B_1:8 -> norm -> PupilBox(192->192->5)
#   track:   Conv-E(2,  64x64)  -> B_1:4 -> norm -> PupilEllipse(197->197->5)
#                                 \_____ the same weights _____/

ANCHOR_DIM = 5      # PupilEllipseHead's (dx, dy, da, db, dtheta), supplied by the HOST
OUTPUT_FRAC = 16    # host-visible fixed point for the head accumulator

MODELS: dict[str, dict] = {
    "hbtxr": dict(dim=192, heads=3, ff=768, patch=16, depth=8, cut=4,
                  paths={"search": dict(img=(1, 128, 128), out=5, anchor=0),
                         "track":  dict(img=(2,  64,  64), out=5, anchor=ANCHOR_DIM)}),
    "tiny":  dict(dim=24, heads=2, ff=48, patch=16, depth=4, cut=2,
                  paths={"search": dict(img=(1, 64, 64), out=5, anchor=0),
                         "track":  dict(img=(2, 32, 32), out=5, anchor=ANCHOR_DIM)}),
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


def _softmax_payload(rows: list[list[int]], input_scale: float,
                     out_bits: int = 8) -> SoftmaxSpec:
    """Fit the 14-scalar softmax payload: exp table + a two-segment reciprocal.

    Takes rows rather than a flat tensor and a token count, because a SHARED block is fit
    over paths whose rows have different lengths. That matters: the reciprocal index spans
    the accumulator range and the accumulator is a sum over the row, so a payload fit on
    64-token rows alone puts every 16-token row below its first entry.
    """
    qmax_out = (1 << out_bits) - 1
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


def build_block(streams: list[tuple[list[int], int]], cfg: dict, dtype: QuantDtype,
                stream_scale: float, rng: random.Random) -> tuple[BlockSpec, list[list[int]], dict]:
    """One block, calibrated over EVERY path that runs through it.

    ``streams`` is one ``(tokens_int, token_count)`` per path. The backbone is shared
    (``vit.py``: search runs ``B_1:8``, track runs ``B_1:4`` on the SAME blocks), so a
    block before the cut point sees both, and one set of scales has to serve both. Values
    are therefore per-stream lists; anything that OBSERVES a range sees all of them at
    once, anything that transforms maps over each.

    Returns the spec, each path's output, and the first path's intermediates (the
    stage-level goldens).
    """
    dim, heads = cfg["dim"], cfg["heads"]
    head_dim, ff = dim // heads, cfg["ff"]
    xs = [s[0] for s in streams]
    ns = [s[1] for s in streams]
    tap: dict[str, list[int]] = {"block_x": list(xs[0])}

    def linear(out_f, in_f, in_scale):
        return _linear_spec(out_f, in_f, in_scale, dtype.bits, rng)

    def every(per_stream):
        """Flatten across paths — the calibration view."""
        return [v for stream in per_stream for v in stream]

    # --- attention ------------------------------------------------------------
    s_ln1 = stream_scale * _NUDGE[0]
    ln1_in = [rescale(x, stream_scale, s_ln1, dtype=dtype) for x in xs]
    norm1 = _layernorm_payload(every(ln1_in), dim, s_ln1, dtype.bits, rng)
    normed = [norm1.apply(v) for v in ln1_in]
    tap["ln1_x"], tap["ln1_y"] = ln1_in[0], normed[0]

    qkv = linear(3 * heads * head_dim, dim, norm1.output_scale * _NUDGE[1])
    qkv_in = [rescale(v, norm1.output_scale, qkv.input_scale, dtype=dtype) for v in normed]
    qkv_acc = [_linear_acc(qkv, _rows(v, dim)) for v in qkv_in]

    def part(index: int) -> tuple[float, list[list[list[list[int]]]]]:
        base = index * heads * head_dim
        span = range(base, base + heads * head_dim)
        scale = _grid((row[oc] for acc in qkv_acc for row in acc for oc in span),
                      max(qkv.out_scale(oc) for oc in span), dtype)
        return scale, [[_requant_columns(qkv, acc, scale, dtype,
                                         columns=range(base + h * head_dim,
                                                       base + (h + 1) * head_dim))
                        for h in range(heads)] for acc in qkv_acc]

    (s_q, query), (s_k, key), (s_v, value) = part(0), part(1), part(2)
    qk = MatMulSpec(s_q, s_k)
    attn_scale = 1.0 / math.sqrt(head_dim)

    score_acc = [[int_matmul(q[h], [list(c) for c in zip(*k[h])]) for h in range(heads)]
                 for q, k in zip(query, key)]
    flat_scores = [[v for acc in per for v in _flat(acc)] for per in score_acc]
    s_score = _grid(every(flat_scores), qk.out_scale * attn_scale, dtype)
    scores = [rescale(f, qk.out_scale, s_score, dtype=dtype, extra=attn_scale)
              for f in flat_scores]
    softmax = _softmax_payload([r for sc, n in zip(scores, ns) for r in _rows(sc, n)], s_score)
    probs = [softmax.apply(sc, tokens=n, heads=heads) for sc, n in zip(scores, ns)]

    tap["qkv_x"] = qkv_in[0]
    tap["smu_a"] = [v for h in query[0] for r in h for v in r]
    tap["smu_b"] = [v for h in key[0] for r in h for v in r]
    tap["smu_y"], tap["softmax_x"], tap["softmax_y"] = scores[0], scores[0], probs[0]

    # softmax output is unsigned 8-bit; map its full scale onto the operand grid
    av = MatMulSpec(softmax.output_scale * ((1 << 8) - 1) / dtype.qmax, s_v)
    context_acc = []
    for i, n in enumerate(ns):
        per_head = []
        for h in range(heads):
            block = probs[i][h * n * n:(h + 1) * n * n]
            rows = _rows(rescale(block, softmax.output_scale, av.scale_a, dtype=dtype), n)
            per_head.append(int_matmul(rows, value[i][h]))
        context_acc.append(per_head)
    s_ctx = _grid(every([_flat(a) for per in context_acc for a in per]), av.out_scale, dtype)
    context = []
    for i, n in enumerate(ns):
        hs = [_rows(rescale(_flat(a), av.out_scale, s_ctx, dtype=dtype), head_dim)
              for a in context_acc[i]]
        context.append([[hs[h][t][c] for h in range(heads) for c in range(head_dim)]
                        for t in range(n)])

    proj = linear(dim, dim, s_ctx)
    s_proj = _grid(every([_flat(_linear_acc(proj, c)) for c in context]),
                   max(proj.out_scale(oc) for oc in range(dim)), dtype)
    proj_out = [_flat(_linear(proj, c, s_proj, dtype)) for c in context]
    tap["rmu_x"], tap["rmu_y"] = _flat(context[0]), proj_out[0]
    attn_residual = AddSpec(stream_scale, s_proj,
                            _sum_grid(every(xs), stream_scale, every(proj_out), s_proj, dtype))

    # --- MLP ------------------------------------------------------------------
    from quantization.i_block import _add                       # same import rationale
    residual = [_add(attn_residual, x, p, dtype) for x, p in zip(xs, proj_out)]
    tap["mha_y"] = residual[0]

    s_ln2 = attn_residual.scale_out * _NUDGE[2]
    ln2_in = [rescale(r, attn_residual.scale_out, s_ln2, dtype=dtype) for r in residual]
    norm2 = _layernorm_payload(every(ln2_in), dim, s_ln2, dtype.bits, rng)

    fc1 = linear(ff, dim, norm2.output_scale * _NUDGE[0])
    fc1_in = [_rows(rescale(norm2.apply(v), norm2.output_scale, fc1.input_scale, dtype=dtype),
                    dim) for v in ln2_in]
    s_hidden = _grid(every([_flat(_linear_acc(fc1, v)) for v in fc1_in]),
                     max(fc1.out_scale(oc) for oc in range(ff)), dtype)
    hidden = [_flat(_linear(fc1, v, s_hidden, dtype)) for v in fc1_in]
    # The LUT emits hbtxr_nl_t (16-bit, SPEC §3); the narrowing to the fc2 operand grid is
    # the requant on the edge below, not the table. Wiring the table to dtype.bits instead
    # collapses GeLU to 2 distinct outputs at 4-bit — check() catches it.
    gelu = _gelu_payload(every(hidden), s_hidden, NL_BITS)
    activated = [gelu.apply(v) for v in hidden]
    tap["gelu_x"], tap["gelu_y"] = hidden[0], activated[0]

    fc2 = linear(dim, ff, gelu.output_scale * _NUDGE[1])
    fc2_in = [_rows(rescale(v, gelu.output_scale, fc2.input_scale, dtype=dtype), ff)
              for v in activated]
    s_mlp = _grid(every([_flat(_linear_acc(fc2, v)) for v in fc2_in]),
                  max(fc2.out_scale(oc) for oc in range(dim)), dtype)
    mlp_out = [_flat(_linear(fc2, v, s_mlp, dtype)) for v in fc2_in]
    mlp_residual = AddSpec(attn_residual.scale_out, s_mlp,
                           _sum_grid(every(residual), attn_residual.scale_out,
                                     every(mlp_out), s_mlp, dtype))
    outs = [_add(mlp_residual, r, m, dtype) for r, m in zip(residual, mlp_out)]
    tap["mlp_x"], tap["mlp_y"] = residual[0], outs[0]

    spec = BlockSpec(heads, head_dim, norm1, qkv, qk, attn_scale, softmax,
                     MatMulSpec(av.scale_a, av.scale_b), proj, attn_residual,
                     norm2, fc1, gelu, fc2, mlp_residual, dtype)
    return spec, outs, tap


def build_patch_embed(cfg: dict, out_scale: float, dtype: QuantDtype,
                      rng: random.Random) -> tuple[ConvSpec, list[int], list[list[int]]]:
    """Conv-F / Conv-E stem: kernel == stride == patch, 8-bit weights, asymmetric input.

    The image grid is asymmetric, so a real zero is the zero-point and NOT 0 — the
    ``- zp*sum(w)`` correction in ``_conv_tokens`` depends on it.
    """
    spec, image = _stem_spec(cfg["img"], cfg["patch"], cfg["dim"], rng)
    tokens = _conv_tokens(spec, image, out_scale, dtype, channels=cfg["img"][0],
                          height=cfg["img"][1], width=cfg["img"][2])
    return spec, image, tokens


# Wide enough that _conv_tokens cannot clamp — used to read an accumulator's real peak
# back out through the oracle instead of re-deriving the zero-point correction here.
_WIDE = QuantDtype(64, signed=True)
_PROBE = 1e-12


def _stem_spec(img: tuple[int, int, int], patch: int, dim: int,
               rng: random.Random) -> tuple[ConvSpec, list[int]]:
    """Weights (8-bit, per out-channel) and one uint8 image for a modality stem."""
    channels, height, width = img
    qmax = (1 << 7) - 1
    real = [[[[rng.gauss(0.0, 0.05) for _ in range(patch)] for _ in range(patch)]
             for _ in range(channels)] for _ in range(dim)]
    scales = [max(max(abs(v) for plane in oc for row in plane for v in row), 1e-8) / qmax
              for oc in real]
    weight = [[[[clamp_int(round(v / s), -qmax - 1, qmax) for v in row] for row in plane]
               for plane in oc] for oc, s in zip(real, scales)]
    spec = ConvSpec(weight, scales, 1.0 / 255.0, 128, patch, (0, 0),
                    [rng.gauss(0.0, 0.01) for _ in range(dim)])
    return spec, [rng.randrange(0, 256) for _ in range(channels * height * width)]


def _stem_peak(spec: ConvSpec, image: list[int], img: tuple[int, int, int]) -> float:
    """Real-valued peak of the stem's output, read through a grid that cannot clamp."""
    probe = _conv_tokens(spec, image, _PROBE, _WIDE, channels=img[0],
                         height=img[1], width=img[2])
    return max(abs(v) for row in probe for v in row) * _PROBE


def build_head(pooled: list[int], in_scale: float, out_dim: int, dtype: QuantDtype,
               rng: random.Random, *, anchor: list[int] | None = None,
               anchor_scale: float = 0.0) -> tuple[HeadSpec, dict]:
    """``pool -> Linear -> GeLU -> Linear``, the shape EVERY HBTXR head has.

    ``models/heads/common.py:mlp_head`` is ``Linear(in, hidden=in) -> GELU -> Linear(.., out)``
    and every head wraps it, so PupilBox and PupilEllipse differ only in ``out_dim`` and in
    whether an anchor state is concatenated.

    ``anchor`` is the track head's 5-dim state (``PupilEllipseHead.condition_on_state``).
    It is a HOST input, not a feature: it arrives on its own grid and has to be requantized
    onto the pooled feature's grid before the concat, because ``LinearSpec`` — and a real
    Linear — has ONE input grid.
    """
    tap: dict[str, object] = {}
    feat = list(pooled)
    if anchor is not None:
        moved = rescale(anchor, anchor_scale, in_scale, dtype=dtype)
        tap["anchor_x"], tap["anchor_q"] = list(anchor), moved
        tap["anchor_ratio"] = anchor_scale / in_scale
        feat = feat + moved
    hidden_in = len(feat)

    fc1 = _linear_spec(hidden_in, hidden_in, in_scale, dtype.bits, rng)
    s_hidden = _grid(_flat(_linear_acc(fc1, [feat])),
                     max(fc1.out_scale(oc) for oc in range(hidden_in)), dtype)
    hidden = _flat(_linear(fc1, [feat], s_hidden, dtype))
    gelu = _gelu_payload(hidden, s_hidden, NL_BITS)
    activated = gelu.apply(hidden)

    fc2 = _linear_spec(out_dim, hidden_in, gelu.output_scale * _NUDGE[1], dtype.bits, rng)
    fc2_in = rescale(activated, gelu.output_scale, fc2.input_scale, dtype=dtype)
    tap["head_x"], tap["head_hidden"] = feat, activated
    # The last accumulator is returned UN-requantized, exactly as replay_model_int does:
    # its consumer is the host, there is no calibrated output grid, and inventing one
    # would round the answer for nothing. The dyadic pair onto OUTPUT_UNIT is emitted
    # alongside so the accelerator can stream fixed point instead of an int32 + a float.
    tap["head_y_acc"] = _linear_acc(fc2, [fc2_in])[0]
    return HeadSpec(fc1, gelu, fc2), tap


def build_model(cfg: dict, blk_dtype: QuantDtype, seam_dtype: QuantDtype,
                head_dtype: QuantDtype, rng: random.Random) -> dict:
    """The whole deployed model in integers: two stems, ONE shared stack, two heads.

    The paper's precision is mixed — 8-bit patch/head, 4-bit MHA/MLP, 16-bit nonlinear —
    and ``ModelSpec`` carries a single dtype for every seam, so this composes the seams
    itself rather than calling ``replay_model_int``. Each block still goes through
    ``replay_block_int`` (``BlockSpec.dtype`` IS per-block), and ``crosscheck`` recovers
    the model-level oracle by running this same composition at a uniform dtype.
    """
    dim, patch, depth, cut = cfg["dim"], cfg["patch"], cfg["depth"], cfg["cut"]
    paths = cfg["paths"]
    order = list(paths)                                   # ["search", "track"]

    # --- stems: modality-specific, both 8-bit, both projecting into the same D ---
    stems, images = {}, {}
    for name in order:
        stems[name], images[name] = _stem_spec(paths[name]["img"], patch, dim, rng)
    # ONE residual-stream grid for both modalities. The blocks are shared, so their input
    # port is shared, so the two stems have to land on the same grid.
    stream_scale = max(_stem_peak(stems[n], images[n], paths[n]["img"])
                       for n in order) / seam_dtype.qmax
    tokens = {n: _conv_tokens(stems[n], images[n], stream_scale, seam_dtype,
                              channels=paths[n]["img"][0], height=paths[n]["img"][1],
                              width=paths[n]["img"][2]) for n in order}

    # --- the shared stack -----------------------------------------------------
    stream = {n: _flat(tokens[n]) for n in order}
    ntok = {n: len(tokens[n]) for n in order}
    blocks, taps, trace = [], [], {n: [] for n in order}
    scale = stream_scale
    for i in range(depth):
        active = order if i < cut else order[:1]          # track exits at the cut point
        spec, outs, tap = build_block([(stream[n], ntok[n]) for n in active],
                                      cfg, blk_dtype, scale, rng)
        for name, out in zip(active, outs):
            stream[name] = out
            trace[name].append(out)
        blocks.append(spec)
        taps.append(tap)
        # No block-to-block requant: each block's input port IS the previous block's
        # output grid, because they are calibrated in sequence. replay_model_int bridges
        # here only because it assumes independently calibrated blocks.
        scale = spec.mlp_residual.scale_out

    # --- the shared final norm ------------------------------------------------
    # Both paths land here from DIFFERENT producers — track leaves block cut-1, search
    # leaves block depth-1 — so they arrive on different grids and each needs its own
    # requant into the norm's single input port. That is a mode-dependent requant, not a
    # detail: the hardware has to switch it with the mode.
    exit_scale = {n: blocks[(cut if n == "track" else depth) - 1].mlp_residual.scale_out
                  for n in order}
    s_fn = max(exit_scale.values()) * _NUDGE[2]
    fn_in = {n: rescale(stream[n], exit_scale[n], s_fn, dtype=seam_dtype) for n in order}
    # clamp_bits is the HEAD's width: this norm's output is what the 8-bit head consumes.
    final_norm = _layernorm_payload([v for n in order for v in fn_in[n]], dim, s_fn,
                                    head_dtype.bits, rng)
    normed = {n: final_norm.apply(fn_in[n]) for n in order}
    pooled = {n: _pool_tokens(_rows(normed[n], dim), head_dtype) for n in order}

    # --- heads ----------------------------------------------------------------
    s_head = final_norm.output_scale * _NUDGE[0]
    heads, head_taps = {}, {}
    for n in order:
        anchor, anchor_scale = None, 0.0
        if paths[n]["anchor"]:
            anchor = [rng.randrange(-100, 101) for _ in range(paths[n]["anchor"])]
            anchor_scale = s_head * 1.7          # the host's grid, unrelated to the features
        feat = rescale(pooled[n], final_norm.output_scale, s_head, dtype=head_dtype)
        heads[n], head_taps[n] = build_head(feat, s_head, paths[n]["out"], head_dtype, rng,
                                            anchor=anchor, anchor_scale=anchor_scale)
    return dict(cfg=cfg, order=order, stems=stems, images=images, tokens=tokens,
                stream_scale=stream_scale, blocks=blocks, taps=taps, trace=trace,
                exit_scale=exit_scale, final_norm=final_norm, fn_in=fn_in,
                normed=normed, pooled=pooled, s_head=s_head, heads=heads,
                head_taps=head_taps, ntok=ntok,
                dtypes=(blk_dtype, seam_dtype, head_dtype))


def crosscheck(cfg: dict, dtype: QuantDtype, seed: int) -> None:
    """Confirm ``replay_model_int`` reproduces ``build_model``'s seams at a UNIFORM dtype.

    This is what buys back the oracle the mixed-precision build gives up. The seam
    composition is identical in both cases — only the clamp widths differ — so agreeing
    here proves the composition, and the mixed build then differs only in arguments this
    check has already exercised. BOTH paths run: track exits at the cut point and reaches
    the shared final norm on a different grid than search does, and that bridge is a real
    seam that only the track run touches.

    The anchor concat is removed for the check, because ``replay_model_int`` has no host
    input. It is therefore the ONE seam with no oracle — ``check_model`` rebuilds it from
    the emitted files instead.

    Two seams are unverifiable HERE no matter what, and no amount of end-to-end comparison
    will change that:

    - a scale error immediately upstream of a LayerNorm (``e01``, ``e09``) is invisible,
      because LayerNorm normalises it away. Halving the block-0 LN input changes nothing
      at the output. A testbench has to probe the LN input directly;
    - a sub-LSB error anywhere: the requant shifts are 27..31, so the accumulator LSB sits
      far below the output LSB. Halving a seam IS caught; nudging one is not.
    """
    stripped = dict(cfg, paths={n: dict(p, anchor=0) for n, p in cfg["paths"].items()})
    m = build_model(stripped, dtype, dtype, dtype, random.Random(seed))
    for name in m["order"]:
        img = stripped["paths"][name]["img"]
        used = stripped["depth"] if name == "search" else stripped["cut"]
        spec = ModelSpec(m["stems"][name], m["blocks"][:used], m["final_norm"],
                         m["heads"][name], dtype)
        want = replay_model_int(spec, m["images"][name],
                                channels=img[0], height=img[1], width=img[2])
        got = m["head_taps"][name]["head_y_acc"]
        if got != want:
            bad = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), len(want))
            raise AssertionError(
                f"{name}: seam composition != replay_model_int at {dtype.bits}-bit, "
                f"index {bad}: {got[bad:bad + 1]} vs {want[bad:bad + 1]}")


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

    def requant(self, name: str, ratios: list[float], note: str, *,
                per_channel: bool = False) -> None:
        """A dyadic (multiplier, shift) pair per output channel — what the HW unit holds.

        A per-tensor edge may legitimately be identity: ``rescale`` short-circuits ratio
        1.0, and ``dyadic_params(1.0)`` is ``(2, 1)``, which is bit-exact identity anyway,
        so the pair is emitted either way and the testbench needs no special case. A
        PER-CHANNEL edge cannot be: those grids are derived from the accumulator, and a
        unit ratio there means the derivation collapsed.

        ``per_channel`` is declared rather than inferred from the length: at model scope a
        per-TENSOR edge is concatenated across ``depth`` blocks, and inferring would then
        reject every legitimately-identity residual join in the stack.
        """
        if per_channel and any(r == 1.0 for r in ratios):
            raise AssertionError(f"{name}: a per-channel requant ratio is exactly 1.0 — "
                                 "the output grid was not derived from the accumulator")
        if any(r == 1.0 for r in ratios):
            note += " [identity]"
        pairs = [dyadic_params(r) for r in ratios]
        self.requant_raw(name, [m for m, _, _ in pairs], [s for _, s, _ in pairs], note)

    def requant_raw(self, name: str, mult: list[int], shift: list[int], note: str) -> None:
        """The same two files from (M, n) already chosen, for cases with no ratio."""
        self.put(f"{name}_mult", mult, f"[{len(mult)}]", note)
        self.put(f"{name}_shift", shift, f"[{len(shift)}]", note)

    def close(self) -> None:
        (self.dest / "index.tsv").write_text(
            "file\tcount\tshape\tnote\n" + "\n".join(self.index) + "\n", encoding="utf-8")


def _bias_int(spec: LinearSpec) -> list[int]:
    """The bias on the ACCUMULATOR grid — where the oracle adds it, and hardware too."""
    return [round(float(spec.bias[oc]) / spec.out_scale(oc)) for oc in range(len(spec.weight))]


def _block_edges(spec: BlockSpec) -> list[tuple[str, list[float], str]]:
    """Every requant in one block, in forward order.

    All of them, not the interesting ones: a golden a testbench cannot reproduce is not a
    golden, and dropping an edge means ``block_y`` cannot be rebuilt.
    """
    heads, head_dim = spec.num_heads, spec.head_dim
    qkv_out = 3 * heads * head_dim
    ar, mr, sm = spec.attn_residual, spec.mlp_residual, spec.softmax

    def ch(lin: LinearSpec, target: float, span: range | None = None) -> list[float]:
        return [lin.out_scale(oc) / target
                for oc in (span if span is not None else range(len(lin.weight)))]

    return [
        ("e01_stream_to_ln1", [ar.scale_a / spec.norm1.input_scale], "per tensor"),
        ("e02_ln1_to_qkv", [spec.norm1.output_scale / spec.qkv.input_scale], "per tensor"),
        ("e03_qkv_q", ch(spec.qkv, spec.qk.scale_a, range(0, heads * head_dim)),
         "from the accumulator DIRECTLY; a shared Q/K/V grid clamps K and V"),
        ("e03_qkv_k", ch(spec.qkv, spec.qk.scale_b,
                         range(heads * head_dim, 2 * heads * head_dim)), "as above"),
        ("e03_qkv_v", ch(spec.qkv, spec.av.scale_b,
                         range(2 * heads * head_dim, qkv_out)), "as above"),
        ("e04_smu_to_softmax", [spec.qk.out_scale * spec.attn_scale / sm.input_scale],
         "per tensor; 1/sqrt(d) is folded in here, it is not a separate multiply"),
        ("e05_softmax_to_av", [sm.output_scale / spec.av.scale_a], "per tensor"),
        ("e06_av_to_proj", [spec.av.out_scale / spec.proj.input_scale], "per tensor"),
        ("e07_proj_acc", ch(spec.proj, ar.scale_b), "per out-channel"),
        ("e08_resid1_a", [ar.scale_a / ar.scale_out], "residual side of the attention join"),
        ("e08_resid1_b", [ar.scale_b / ar.scale_out], "branch side"),
        ("e09_stream_to_ln2", [ar.scale_out / spec.norm2.input_scale], "per tensor"),
        ("e10_ln2_to_fc1", [spec.norm2.output_scale / spec.fc1.input_scale], "per tensor"),
        ("e11_fc1_acc", ch(spec.fc1, spec.gelu.input_scale), "per out-channel"),
        ("e12_gelu_to_fc2", [spec.gelu.output_scale / spec.fc2.input_scale], "per tensor"),
        ("e13_fc2_acc", ch(spec.fc2, mr.scale_b), "per out-channel"),
        ("e14_resid2_a", [mr.scale_a / mr.scale_out], "residual side of the MLP join"),
        ("e14_resid2_b", [mr.scale_b / mr.scale_out], "branch side"),
    ]


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
    spec, _outs, tap = build_block([(x_int, tokens)], cfg, dtype, stream_scale, rng)
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
              "per out-channel, acc -> token grid", per_channel=True)
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

    for name, ratios, note in _block_edges(spec):
        w.requant(name, ratios, note, per_channel=len(ratios) > 1)

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

def emit_model(dest: Path, model: str, blk_bits: int, seam_bits: int, head_bits: int,
               seed: int) -> tuple[Path, dict]:
    """The whole model at the paper's bit-widths: two stems, ONE shared stack, two heads.

    Per-block tensors carry a leading depth axis (``[L, ...]``) rather than living in 8
    directories: the weight prefetcher streams them per block from DRAM, so that is the
    layout it wants, and it keeps the file count the same as one block's.
    """
    cfg = MODELS[model]
    dim, heads, ff = cfg["dim"], cfg["heads"], cfg["ff"]
    head_dim, depth, cut = dim // heads, cfg["depth"], cfg["cut"]
    paths, order = cfg["paths"], list(cfg["paths"])
    blk_dtype = QuantDtype(blk_bits, signed=True)
    seam_dtype = QuantDtype(seam_bits, signed=True)
    head_dtype = QuantDtype(head_bits, signed=True)

    m = build_model(cfg, blk_dtype, seam_dtype, head_dtype, random.Random(seed))
    blocks, fnorm = m["blocks"], m["final_norm"]

    w = Writer(dest)
    w.put("meta", [dim, heads, head_dim, ff, cfg["patch"], depth, cut,
                   blk_bits, seam_bits, head_bits, seed], "[11]",
          "dim heads head_dim ff patch depth cut blk_bits seam_bits head_bits seed")

    # --- stems, one per modality ---------------------------------------------
    for name in order:
        stem, img = m["stems"][name], paths[name]["img"]
        n = m["ntok"][name]
        w.put(f"{name}_meta", [img[0], img[1], img[2], n, depth if name == "search" else cut,
                               len(m["heads"][name].fc1.weight), paths[name]["out"],
                               paths[name]["anchor"]], "[8]",
              "img_c img_h img_w tokens blocks_used head_in head_out anchor_dim")
        w.put(f"stem_{name}_image", m["images"][name], f"[{img[0]},{img[1]},{img[2]}]",
              "uint8 row-major, zero_point 128")
        w.put(f"stem_{name}_weight",
              [v for oc in stem.weight for p in oc for r in p for v in r],
              f"[{dim},{img[0]},{cfg['patch']},{cfg['patch']}]", "int8 [Cout,Cin,kh,kw]")
        w.put(f"stem_{name}_bias_acc",
              [round(float(stem.bias[oc]) / stem.out_scale(oc)) for oc in range(dim)],
              f"[{dim}]", "accumulator grid")
        w.put(f"stem_{name}_zp_correction",
              [stem.zero_point * sum(v for p in stem.weight[oc] for r in p for v in r)
               for oc in range(dim)], f"[{dim}]", "zp * sum(w), subtracted from the acc")
        w.requant(f"stem_{name}", [stem.out_scale(oc) / m["stream_scale"] for oc in range(dim)],
                  "per out-channel; BOTH stems land on the one shared residual grid",
                  per_channel=True)
        w.put(f"stem_{name}_y", _flat(m["tokens"][name]), f"[{n},{dim}]", "tokens")

    # --- the shared stack, depth-major ---------------------------------------
    qkv_out = 3 * heads * head_dim
    for key, shape, note in (("qkv", f"[{depth},{qkv_out},{dim}]", "out laid out (3,H,d)"),
                             ("proj", f"[{depth},{dim},{dim}]", "output projection"),
                             ("fc1", f"[{depth},{ff},{dim}]", "MLP expand"),
                             ("fc2", f"[{depth},{dim},{ff}]", "MLP contract")):
        lins = [getattr(b, key) for b in blocks]
        w.put(f"blk_{key}_weight", [v for lin in lins for row in lin.weight for v in row],
              shape, f"int{blk_bits} [out,in], {note}")
        w.put(f"blk_{key}_bias_acc", [v for lin in lins for v in _bias_int(lin)],
              f"[{depth},{len(lins[0].weight)}]", "accumulator grid")

    for tag, pick in (("ln1", lambda b: b.norm1), ("ln2", lambda b: b.norm2)):
        lns = [pick(b) for b in blocks]
        w.put(f"blk_{tag}_scalars", [v for ln in lns for v in ln.scalars], f"[{depth},7]",
              "c_1_m c_1_s b s1 bound s2 clamp_bits")
        w.put(f"blk_{tag}_lnw", [v for ln in lns for v in ln.lnw], f"[{depth},{dim}]", "int16")
        w.put(f"blk_{tag}_lnb", [v for ln in lns for v in ln.lnb], f"[{depth},{dim}]",
              "int16, on the affine grid")
        w.put(f"blk_{tag}_rsqrt_table", [v for ln in lns for v in ln.rsqrt_table],
              f"[{depth},{ENTRIES['rsqrt']}]", "int16")

    sms = [b.softmax for b in blocks]
    w.put("blk_softmax_scalars", [v for s in sms for v in s.scalars], f"[{depth},14]",
          "b1 s1 bound1 | b2/s2/bound2/b3/s3 per recip segment | clamp_bits")
    for field, entries in (("exp_table", "exp"), ("recip_table_one", "recip"),
                           ("recip_table_two", "recip")):
        w.put(f"blk_softmax_{field}", [v for s in sms for v in getattr(s, field)],
              f"[{depth},{ENTRIES[entries]}]",
              "int16; blocks 0..cut-1 are fit over BOTH token counts")
    gls = [b.gelu for b in blocks]
    w.put("blk_gelu_scalars", [v for g in gls for v in g.scalars], f"[{depth},3]", "b s bound")
    w.put("blk_gelu_table", [v for g in gls for v in g.table],
          f"[{depth},{ENTRIES['gelu']}]", "int16")

    per_block = [_block_edges(b) for b in blocks]
    for i, (name, _, note) in enumerate(per_block[0]):
        w.requant(f"blk_{name}", [r for edges in per_block for r in edges[i][1]],
                  f"[{depth} x ...] {note}", per_channel=len(per_block[0][i][1]) > 1)

    # --- the shared final norm ------------------------------------------------
    w.put("fnorm_scalars", fnorm.scalars, "[7]", "clamp_bits is the HEAD's width")
    w.put("fnorm_lnw", fnorm.lnw, f"[{dim}]", "int16")
    w.put("fnorm_lnb", fnorm.lnb, f"[{dim}]", "int16, on the affine grid")
    w.put("fnorm_rsqrt_table", fnorm.rsqrt_table, f"[{ENTRIES['rsqrt']}]", "int16")

    # --- per path: block trace, the exit bridge, the head ---------------------
    for name in order:
        used, n = (depth if name == "search" else cut), m["ntok"][name]
        head, tap = m["heads"][name], m["head_taps"][name]
        hin = len(head.fc1.weight)
        w.put(f"{name}_block_out", [v for out in m["trace"][name] for v in out],
              f"[{used},{n},{dim}]", "every block's output — locates the first mismatch")
        # search leaves block depth-1, track leaves block cut-1, so the two arrive at the
        # SHARED norm on different grids. This requant is mode-dependent.
        w.requant(f"{name}_e_exit_to_fnorm",
                  [m["exit_scale"][name] / fnorm.input_scale],
                  f"per tensor; from block {used - 1}'s output grid")
        w.put(f"{name}_fnorm_y", m["normed"][name], f"[{n},{dim}]", "expected")
        w.put(f"{name}_pooled", m["pooled"][name], f"[{dim}]",
              "integer mean over tokens, round-half-away-from-zero")
        w.requant(f"{name}_e_fnorm_to_head", [fnorm.output_scale / m["s_head"]], "per tensor")
        if paths[name]["anchor"]:
            w.put(f"{name}_anchor_x", tap["anchor_x"], f"[{paths[name]['anchor']}]",
                  "HOST input — PupilEllipseHead.condition_on_state")
            w.requant(f"{name}_e_anchor", [tap["anchor_ratio"]],
                      "the host's anchor grid -> the pooled feature's grid, before the concat")
        w.put(f"{name}_head_x", tap["head_x"], f"[{hin}]",
              "pooled feature" + (" ++ requantized anchor" if paths[name]["anchor"] else ""))
        for key, shape in (("fc1", f"[{hin},{hin}]"), ("fc2", f"[{paths[name]['out']},{hin}]")):
            lin = getattr(head, key)
            w.put(f"{name}_head_{key}_weight", [v for row in lin.weight for v in row],
                  shape, f"int{head_bits} [out,in]")
            w.put(f"{name}_head_{key}_bias_acc", _bias_int(lin), f"[{len(lin.weight)}]",
                  "accumulator grid")
        w.requant(f"{name}_head_e_fc1_acc",
                  [head.fc1.out_scale(oc) / head.gelu.input_scale for oc in range(hin)],
                  "per out-channel", per_channel=True)
        w.put(f"{name}_head_gelu_scalars", head.gelu.scalars, "[3]", "b s bound")
        w.put(f"{name}_head_gelu_table", head.gelu.table, f"[{ENTRIES['gelu']}]", "int16")
        w.requant(f"{name}_head_e_gelu_to_fc2",
                  [head.gelu.output_scale / head.fc2.input_scale], "per tensor")
        w.put(f"{name}_head_y_acc", tap["head_y_acc"], f"[{paths[name]['out']}]",
              "UN-requantized, as replay_model_int returns it — the host owns this grid")
        w.requant(f"{name}_head_out",
                  [head.fc2.out_scale(oc) * (1 << OUTPUT_FRAC)
                   for oc in range(paths[name]["out"])],
                  f"per out-channel; acc -> Q.{OUTPUT_FRAC} fixed point for the AXIS output",
                  per_channel=True)
    w.close()
    return dest, dict(model=m, dest=dest, order=order,
                      dtypes=(blk_dtype, seam_dtype, head_dtype))


def emit_requant(dest: Path, seed: int, *, acc_bits: int = 20, m_bits: int = 33,
                 n_max: int = 31, out_bits: int = 4) -> tuple[Path, dict]:
    """Direct cases for the requant primitive — the one op every matmul unit ends with.

    The block golden exercises it only with the ratios the design happens to present:
    near 1, never a tie, rarely saturating. This covers the domain instead. It matters —
    the first version of this drew multipliers uniformly over the whole range, produced
    five distinct outputs out of sixteen because almost everything saturated, and passed
    while the interesting arithmetic went untested.
    """
    acc_lo, acc_hi = -(1 << (acc_bits - 1)), (1 << (acc_bits - 1)) - 1
    m_hi = (1 << m_bits) - 1
    rng = random.Random(seed)
    accs: list[int] = []
    mults: list[int] = []
    shifts: list[int] = []
    want: list[int] = []
    skipped = 0

    def add(a: int, m: int, n: int) -> None:
        """Reject what the CONFIG cannot represent, rather than emitting it.

        REQ_M_BITS is a property of the ratios the design presents, not of dyadic_params:
        ratio 1.15 needs 33 bits at shift 31, ratio 4.5 needs 34. An oversized multiplier
        is silently truncated by ap_uint<REQ_M_BITS>, and the mismatch then reads as a
        kernel bug instead of an out-of-domain case.
        """
        nonlocal skipped
        if not (acc_lo <= a <= acc_hi and 0 < m <= m_hi and 1 <= n <= n_max):
            skipped += 1
            return
        accs.append(a)
        mults.append(m)
        shifts.append(n)
        want.append(requant([a], m, n, bits=out_bits, signed=True)[0])

    for _ in range(3000):          # in range: this is what exercises rounding and the shift
        a = rng.choice([rng.randrange(acc_lo, acc_hi), rng.randrange(-300, 300)]) or 1
        m, n, _ = dyadic_params(abs(rng.uniform(-9.0, 9.0) / a))
        add(a, m, n)
    for _ in range(600):           # saturating, both directions
        add(rng.randrange(acc_lo, acc_hi), rng.randrange(1, m_hi), rng.randrange(1, n_max + 1))
    qmax = (1 << (out_bits - 1)) - 1
    for k in range(-qmax - 1, qmax + 1):   # exact ties, both signs
        for n in (1, 2, 8, 15):
            add(k * (1 << n) + (1 << (n - 1)), 1, n)
            add(k * (1 << n) - (1 << (n - 1)), 1, n)
    for a, m, n in ((0, 1, 1), (acc_lo, m_hi, n_max), (acc_hi, 1, n_max),
                    (-1, 2, 1), (1, 2, 1), (acc_lo, 1, 1)):
        add(a, m, n)

    w = Writer(dest)
    w.put("meta", [len(accs), acc_bits, m_bits, n_max, out_bits, seed], "[6]",
          "cases acc_bits m_bits n_max out_bits seed")
    w.put("acc", accs, f"[{len(accs)}]", f"int{acc_bits}")
    w.requant_raw("edge", mults, shifts, "one (M, n) per case, not per channel")
    w.put("want", want, f"[{len(accs)}]", f"expected, int{out_bits}")
    w.close()
    distinct = len(set(want))
    span = 1 << out_bits
    if distinct < span:
        raise AssertionError(f"only {distinct}/{span} distinct outputs — the cases do not "
                             "cover the output range and the arithmetic goes untested")
    return dest, dict(cases=len(accs), skipped=skipped, distinct=distinct)


def check_model(dest: Path, result: dict) -> None:
    """Degenerate-golden guards plus a file-only rebuild of the stem.

    The stem is the piece the block-scope check never sees, and its zero-point correction
    is the easiest thing in this graph to get subtly wrong: padding is absent here, so the
    uniform ``- zp*sum(w)`` term is the ONLY place the asymmetric input grid is handled.
    """
    m = result["model"]
    dim = m["cfg"]["dim"]
    for name in result["order"]:
        img = m["cfg"]["paths"][name]["img"]
        weight = _read(dest, f"stem_{name}_weight")
        bias = _read(dest, f"stem_{name}_bias_acc")
        zp = _read(dest, f"stem_{name}_zp_correction")
        mult, shift = _read(dest, f"stem_{name}_mult"), _read(dest, f"stem_{name}_shift")
        image = _read(dest, f"stem_{name}_image")
        patch, seam_bits = m["cfg"]["patch"], result["dtypes"][1].bits
        taps = patch * patch * img[0]
        cols = []
        for oc in range(dim):
            base = oc * taps
            acc = []
            for ty in range(img[1] // patch):
                for tx in range(img[2] // patch):
                    total = 0
                    for c in range(img[0]):
                        for ky in range(patch):
                            row = (c * img[1] + ty * patch + ky) * img[2] + tx * patch
                            for kx in range(patch):
                                total += image[row + kx] * weight[base + (c * patch + ky) * patch + kx]
                    acc.append(total - zp[oc] + bias[oc])
            cols.append(requant(acc, mult[oc], shift[oc], bits=seam_bits, signed=True))
        rebuilt = [cols[oc][t] for t in range(len(cols[0])) for oc in range(dim)]
        if rebuilt != _read(dest, f"stem_{name}_y"):
            raise AssertionError(f"stem_{name}_y is not reproducible from the emitted files")

        # The anchor concat is the one seam crosscheck cannot reach (replay_model_int has
        # no host input), so it is rebuilt from the files instead.
        if m["cfg"]["paths"][name]["anchor"]:
            (am,), (ash,) = _read(dest, f"{name}_e_anchor_mult"), \
                _read(dest, f"{name}_e_anchor_shift")
            moved = requant(_read(dest, f"{name}_anchor_x"), am, ash,
                            bits=result["dtypes"][2].bits, signed=True)
            if moved != _read(dest, f"{name}_head_x")[dim:]:
                raise AssertionError(f"{name}: the anchor does not requantize onto the "
                                     "pooled feature's grid as emitted")

        for field in (f"stem_{name}_y", f"{name}_fnorm_y", f"{name}_head_y_acc"):
            values = _read(dest, field)
            if len(set(values)) < 3:
                raise AssertionError(f"{field}: {len(set(values))} distinct values — degenerate")

    written = {p.stem for p in dest.glob("*.txt")}
    indexed = {line.split("\t")[0][:-4]
               for line in (dest / "index.tsv").read_text("utf-8").splitlines()[1:]}
    if written != indexed:
        raise AssertionError(f"index.tsv disagrees with the directory: {written ^ indexed}")


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
    ap.add_argument("--scope", choices=("block", "model", "requant"), default="block",
                    help="block: one block + stage vectors (S2-S5). "
                         "model: the whole deployed model at the paper's bit-widths (S8). "
                         "requant: direct cases for the requant primitive")
    ap.add_argument("--mode", choices=sorted(MODES), default="search",
                    help="block scope: which path's token count to size the block for")
    ap.add_argument("--model", choices=sorted(MODELS), default="hbtxr",
                    help="model scope: hbtxr (L=8, c=4) or tiny")
    ap.add_argument("--bits", type=int, default=4,
                    help="W/A width of the MHA and MLP matmuls (SPEC §3)")
    ap.add_argument("--seam-bits", type=int, default=4,
                    help="model scope: stem output and the residual stream between blocks")
    ap.add_argument("--head-bits", type=int, default=8,
                    help="model scope: final norm output, pooling and the head (paper §V-B-2)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=_ROOT / "hardware" / "workspace" / "golden")
    args = ap.parse_args(argv)

    if args.scope == "requant":
        dest = args.out / "requant"
        _, result = emit_requant(dest, args.seed)
        extra = (f"{result['cases']} cases  {result['distinct']}/16 outputs  "
                 f"{result['skipped']} out-of-domain rejected")
    elif args.scope == "block":
        dest = args.out / f"{args.mode}-a{args.bits}"
        _, result = emit(dest, args.mode, args.bits, args.seed)
        check(dest, result)
        extra = f"tokens={result['tokens']} dim={result['dim']} bits={args.bits}"
    else:
        # The seam composition is size-independent, so the uniform-dtype cross-check
        # against replay_model_int runs on `tiny` and covers every model build. Running it
        # at full size would cost a second 8-block replay and prove nothing new.
        crosscheck(MODELS["tiny"], QuantDtype(args.head_bits, signed=True), args.seed)
        dest = args.out / (f"model-{args.model}-w{args.bits}"
                           f"s{args.seam_bits}h{args.head_bits}")
        _, result = emit_model(dest, args.model, args.bits, args.seam_bits,
                               args.head_bits, args.seed)
        check_model(dest, result)
        cfg = MODELS[args.model]
        extra = (f"depth={cfg['depth']} cut={cfg['cut']} dim={cfg['dim']} "
                 f"blk=a{args.bits} seam=a{args.seam_bits} head=a{args.head_bits}")
    print(f"{dest}  {len(list(dest.glob('*.txt')))} files  {extra}  OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
