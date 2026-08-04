#!/usr/bin/env python3
"""HBTXR host <-> kernel payload — the M-AXI weight blob and the AXIS streams.

    python3 hardware/deploy/hbtxr/payload.py --golden hardware/workspace/golden/model-hbtxr-w4s4h8

SPEC §8 fixes the PORTS but not what travels on them:

    void hbtxr_top(hls::stream<hbtxr_axis_t> &in_stream,     // AXIS   <- image
                   hls::stream<hbtxr_axis_t> &out_stream,    // AXIS   -> 5 results
                   const volatile hbtxr_axi_word_t *weights, // M-AXI  <- this blob
                   int mode, const hbtxr_anchor_t anchor[5], int *status);

This module is the host end of that contract and `module/include/hbtxr_payload.hpp` is
the kernel end. They are not two descriptions of one layout: **the blob carries its own
section table**, both sides read it, and neither recomputes an offset. Two independent
implementations of a binary layout is precisely the failure this project keeps finding —
an `array_reshape` that never reached the tool, a measurement taken through the datapath
it was measuring, a DSP attribution that survived two rounds of reasoning and was wrong.
A table in the file cannot drift from itself.

Stdlib only, like `tools/export_hls_golden.py`: this has to run on the board, where the
only certainty is CPython.
"""
import argparse
import io
import os
import struct
import sys

# --- blob format -------------------------------------------------------------
MAGIC = b"HBTXRPL\0"
VERSION = 1
NAME_BYTES = 24
ENTRY = struct.Struct("<%dsQQii" % NAME_BYTES)   # name, offset, bytes, count, elem_bits
HEADER = struct.Struct("<8sIIII")                # magic, version, header_bytes, n, reserved
ALIGN = 64                                       # one 512-bit M-AXI beat

# --- shape (mirrors config/design/hbtxr_config.hpp) --------------------------
D, H, HD, F = 192, 3, 64, 768
DEPTH, TRACK_DEPTH = 8, 4
N_SEARCH, N_TRACK = 64, 16
ANCHOR, OUT_DIM = 5, 5
REQ_M_BITS, REQ_S_BITS = 18, 5
AXIS_LANES = 32                                  # TP * NL_P, == stem_*_t::pix_beat_t
OUTPUT_FRAC = 16                                 # hbtxr_top.hpp:HBTXR_OUTPUT_FRAC

# Stems. `kernel == stride == patch`, so tokens = (H_IN/K) * (W_IN/K).
STEM = {
    "search": dict(cin=1, k=16, h=128, w=128, tokens=N_SEARCH),
    "track": dict(cin=2, k=16, h=64, w=64, tokens=N_TRACK),
}
HEAD_IN = {"search": D, "track": D + ANCHOR}

# The eighteen requant edges, numbered as the generator numbers them. Per-CHANNEL ones
# belong to the matmul that produced the accumulator and are packed with it; the rest are
# one (M, n) for the whole tensor.
PER_TENSOR_EDGES = [
    "e01_stream_to_ln1", "e02_ln1_to_qkv", "e04_smu_to_softmax", "e05_softmax_to_av",
    "e06_av_to_proj", "e08_resid1_a", "e08_resid1_b", "e09_stream_to_ln2",
    "e10_ln2_to_fc1", "e12_gelu_to_fc2", "e14_resid2_a", "e14_resid2_b",
]
# (member, golden weight file, CO, CI, the per-channel edges that concatenate over CO)
BLOCK_MATMULS = [
    ("qkv", "blk_qkv", 3 * D, D, ["e03_qkv_q", "e03_qkv_k", "e03_qkv_v"]),
    ("proj", "blk_proj", D, D, ["e07_proj_acc"]),
    ("fc1", "blk_fc1", F, D, ["e11_fc1_acc"]),
    ("fc2", "blk_fc2", D, F, ["e13_fc2_acc"]),
]


# --- golden reader (same contract as tb/hbtxr_golden.hpp) --------------------
def read_golden(golden_dir, name):
    """Integer CSV with a trailing comma, one file per tensor."""
    path = os.path.join(golden_dir, name + ".txt")
    with io.open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return [int(t) for t in text.replace("\n", ",").split(",") if t.strip()]


class Blob(object):
    """Sections appended in order, each 64-byte aligned, each named in the table."""

    def __init__(self):
        self.body = bytearray()
        self.entries = []

    def add(self, name, payload, count, elem_bits):
        raw = name.encode("utf-8")
        if len(raw) > NAME_BYTES:
            raise ValueError("section name too long for the table: %s" % name)
        pad = (-len(self.body)) % ALIGN
        self.body.extend(b"\0" * pad)
        self.entries.append((raw, len(self.body), len(payload), count, elem_bits))
        self.body.extend(payload)

    def render(self):
        header_bytes = HEADER.size + ENTRY.size * len(self.entries)
        header_bytes += (-header_bytes) % ALIGN
        out = bytearray(header_bytes)
        HEADER.pack_into(out, 0, MAGIC, VERSION, header_bytes, len(self.entries), 0)
        at = HEADER.size
        for name, off, nbytes, count, bits in self.entries:
            ENTRY.pack_into(out, at, name, header_bytes + off, nbytes, count, bits)
            at += ENTRY.size
        out.extend(self.body)
        return bytes(out)


# --- element encoders --------------------------------------------------------
# Widths come from the TYPES in hbtxr_config.hpp, not from the values that happen to be
# present: acc_t is 20 bits wide even when what lives in it needs 9 (SPEC §3).

def pack_int4(values):
    """Two nibbles per byte, LOW nibble first.

    The order is the half of this that cannot be inferred, and getting it wrong is silent:
    the blob is the right SIZE either way and every weight is transposed within its pair.
    """
    out = bytearray((len(values) + 1) // 2)
    for i, v in enumerate(values):
        if not -8 <= v <= 7:
            raise ValueError("weight %d does not fit int4 at index %d" % (v, i))
        if i & 1:
            out[i >> 1] |= (v & 0xF) << 4
        else:
            out[i >> 1] |= v & 0xF
    return bytes(out)


def pack_i8(values):
    return struct.pack("<%db" % len(values), *values)


def pack_i16(values):
    return struct.pack("<%dh" % len(values), *values)


def pack_u16(values):
    return struct.pack("<%dH" % len(values), *values)


def pack_i32(values):
    return struct.pack("<%di" % len(values), *values)


def pack_i64(values):
    return struct.pack("<%dq" % len(values), *values)


def pack_requant(mult, shift):
    """One u32 per channel: `mult | shift << REQ_M_BITS`.

    They travel together because they are one decision. `mult` is bounded to
    REQ_M_BITS by `i_ops.DYADIC_MULT_BITS`, and the loader REJECTS anything wider rather
    than truncating it — a silently truncated multiplier is a plausible wrong answer.
    """
    if len(mult) != len(shift):
        raise ValueError("mult/shift length mismatch: %d vs %d" % (len(mult), len(shift)))
    words = []
    for m, s in zip(mult, shift):
        if not 0 <= m < (1 << REQ_M_BITS):
            raise ValueError("multiplier %d needs more than %d bits" % (m, REQ_M_BITS))
        if not 0 <= s < (1 << REQ_S_BITS):
            raise ValueError("shift %d does not fit %d bits" % (s, REQ_S_BITS))
        words.append(m | (s << REQ_M_BITS))
    return struct.pack("<%dI" % len(words), *words)


def _slice(values, blk, span):
    lo = blk * span
    if len(values) < lo + span:
        raise ValueError("need %d elements from index %d, have %d" % (span, lo, len(values)))
    return values[lo:lo + span]


# --- the weight blob ---------------------------------------------------------
def pack_weights(golden_dir):
    """Golden directory -> the M-AXI blob the prefetcher walks.

    BLOCK-MAJOR, and that is the whole point: the prefetcher moves one TRB while the other
    core pair computes (SPEC §6), so a block's payload has to be one contiguous region.
    Interleaving by tensor would make every prefetch a scatter.
    """
    g = lambda name: read_golden(golden_dir, name)
    blob = Blob()

    # --- stems: 8-bit, and the zero-point correction folds into the bias -----
    # The input grid is asymmetric, so a real zero is the zero point and every window
    # carries a uniform `- zp * sum(w)` offset. It is per out-channel and constant, so it
    # belongs on the accumulator the RMU already adds to -- one resident number, and no
    # runtime subtract on the board (hbtxr_patch_embed.hpp).
    for mode, cfg in STEM.items():
        taps = cfg["cin"] * cfg["k"] * cfg["k"]
        w = g("stem_%s_weight" % mode)
        if len(w) != D * taps:
            raise ValueError("stem_%s_weight: %d values, want %d" % (mode, len(w), D * taps))
        bias = g("stem_%s_bias_acc" % mode)
        zp = g("stem_%s_zp_correction" % mode)
        blob.add("stem.%s.w" % mode, pack_i8(w), len(w), 8)
        blob.add("stem.%s.b" % mode, pack_i32([b + z for b, z in zip(bias, zp)]), D, 32)
        blob.add("stem.%s.rq" % mode,
                 pack_requant(g("stem_%s_mult" % mode), g("stem_%s_shift" % mode)), D, 32)

    # --- the eight TRBs, one contiguous region each --------------------------
    ln = {}
    for which in ("ln1", "ln2"):
        for field in ("scalars", "lnw", "lnb", "rsqrt_table"):
            ln[(which, field)] = g("blk_%s_%s" % (which, field))
    sm = dict((f, g("blk_softmax_" + f)) for f in
              ("scalars", "exp_table", "recip_table_one", "recip_table_two"))
    gelu = dict((f, g("blk_gelu_" + f)) for f in ("scalars", "table"))
    edges = dict((e, (g("blk_%s_mult" % e), g("blk_%s_shift" % e)))
                 for e in PER_TENSOR_EDGES)
    mm = {}
    for member, stem, co, ci, chan in BLOCK_MATMULS:
        mm[member] = (g(stem + "_weight"), g(stem + "_bias_acc"),
                      [(g("blk_%s_mult" % e), g("blk_%s_shift" % e)) for e in chan])

    for blk in range(DEPTH):
        p = "b%d." % blk
        for member, _stem, co, ci, _chan in BLOCK_MATMULS:
            w, b, chan = mm[member]
            blob.add(p + member + ".w", pack_int4(_slice(w, blk, co * ci)), co * ci, 4)
            blob.add(p + member + ".b", pack_i32(_slice(b, blk, co)), co, 32)
            # The qkv accumulator feeds three differently calibrated consumers, and they
            # concatenate in OUTPUT-CHANNEL order -- q, then k, then v. One unit, three
            # grids, expressed entirely by these per-channel pairs.
            mult, shift = [], []
            for cm, cs in chan:
                span = len(cm) // DEPTH
                mult.extend(_slice(cm, blk, span))
                shift.extend(_slice(cs, blk, span))
            if len(mult) != co:
                raise ValueError("%s%s: %d requant entries, want %d"
                                 % (p, member, len(mult), co))
            blob.add(p + member + ".rq", pack_requant(mult, shift), co, 32)

        for which in ("ln1", "ln2"):
            blob.add(p + which + ".sc", pack_i32(_slice(ln[(which, "scalars")], blk, 7)), 7, 32)
            blob.add(p + which + ".w", pack_i16(_slice(ln[(which, "lnw")], blk, D)), D, 16)
            # lnb is NOT a table entry: it lives on the affine accumulator grid, measured
            # at ~30 signed bits and typed wider. int64 rather than a tight width because
            # a bias that silently wraps is not diagnosable downstream.
            blob.add(p + which + ".b", pack_i64(_slice(ln[(which, "lnb")], blk, D)), D, 64)
            blob.add(p + which + ".rs",
                     pack_i16(_slice(ln[(which, "rsqrt_table")], blk, 64)), 64, 16)

        blob.add(p + "sm.sc", pack_i32(_slice(sm["scalars"], blk, 14)), 14, 32)
        # exp and reciprocal are UNSIGNED. exp's largest entry is its numerator,
        # 1 << 15 = 32768, which int16 cannot hold -- the sign is per operator and getting
        # it wrong is silent (hbtxr_config.hpp).
        blob.add(p + "sm.exp", pack_u16(_slice(sm["exp_table"], blk, 32)), 32, 16)
        blob.add(p + "sm.r1", pack_u16(_slice(sm["recip_table_one"], blk, 64)), 64, 16)
        blob.add(p + "sm.r2", pack_u16(_slice(sm["recip_table_two"], blk, 64)), 64, 16)
        blob.add(p + "gelu.sc", pack_i32(_slice(gelu["scalars"], blk, 3)), 3, 32)
        blob.add(p + "gelu.t", pack_i16(_slice(gelu["table"], blk, 32)), 32, 16)

        em = [edges[e][0][blk] for e in PER_TENSOR_EDGES]
        es = [edges[e][1][blk] for e in PER_TENSOR_EDGES]
        blob.add(p + "edges", pack_requant(em, es), len(em), 32)

    # --- terminal decode -----------------------------------------------------
    # ONE final norm, shared. The two paths reach it from different producers, so the
    # requant INTO it is mode-dependent and lives with the mode table below.
    blob.add("fnorm.sc", pack_i32(g("fnorm_scalars")), 7, 32)
    blob.add("fnorm.w", pack_i16(g("fnorm_lnw")), D, 16)
    blob.add("fnorm.b", pack_i64(g("fnorm_lnb")), D, 64)
    blob.add("fnorm.rs", pack_i16(g("fnorm_rsqrt_table")), 64, 16)

    for mode in ("search", "track"):
        n = HEAD_IN[mode]
        blob.add("head.%s.fc1w" % mode, pack_i8(g("%s_head_fc1_weight" % mode)), n * n, 8)
        blob.add("head.%s.fc1b" % mode, pack_i32(g("%s_head_fc1_bias_acc" % mode)), n, 32)
        blob.add("head.%s.fc1rq" % mode,
                 pack_requant(g("%s_head_e_fc1_acc_mult" % mode),
                              g("%s_head_e_fc1_acc_shift" % mode)), n, 32)
        blob.add("head.%s.gsc" % mode, pack_i32(g("%s_head_gelu_scalars" % mode)), 3, 32)
        blob.add("head.%s.gt" % mode, pack_i16(g("%s_head_gelu_table" % mode)), 32, 16)
        blob.add("head.%s.grq" % mode,
                 pack_requant(g("%s_head_e_gelu_to_fc2_mult" % mode),
                              g("%s_head_e_gelu_to_fc2_shift" % mode)), 1, 32)
        blob.add("head.%s.fc2w" % mode, pack_i8(g("%s_head_fc2_weight" % mode)),
                 OUT_DIM * n, 8)
        blob.add("head.%s.fc2b" % mode, pack_i32(g("%s_head_fc2_bias_acc" % mode)),
                 OUT_DIM, 32)

    # --- the mode table: exactly four things depend on the mode --------------
    # stem, the requant INTO the shared final norm, which head runs (+anchor), and the
    # OUTPUT requant. The middle two are the easy ones to miss, and a single output table
    # keeps passing the accumulator golden while reporting the wrong fixed-point value for
    # whichever mode ran second -- tb_top's mode-switch check is what caught that.
    for mode in ("search", "track"):
        blob.add("mode.%s.exit" % mode,
                 pack_requant(g("%s_e_exit_to_fnorm_mult" % mode),
                              g("%s_e_exit_to_fnorm_shift" % mode)), 1, 32)
        blob.add("mode.%s.feat" % mode,
                 pack_requant(g("%s_e_fnorm_to_head_mult" % mode),
                              g("%s_e_fnorm_to_head_shift" % mode)), 1, 32)
        blob.add("mode.%s.out" % mode,
                 pack_requant(g("%s_head_out_mult" % mode),
                              g("%s_head_out_shift" % mode)), OUT_DIM, 32)
    # track only: the anchor arrives on the HOST's grid and has to be requantized onto the
    # pooled feature's before the concat, because a Linear has one input grid.
    blob.add("mode.track.anchor",
             pack_requant(g("track_e_anchor_mult"), g("track_e_anchor_shift")), 1, 32)

    return blob.render()


# --- the AXIS streams --------------------------------------------------------
def pack_image(golden_dir, mode):
    """Golden image -> the AXIS beats the stem reads.

    PIXEL-INTERLEAVED `[y][x][c]`, AXIS_LANES per beat -- NOT the golden's channel-major
    `[c][y][x]`. That transpose is a DMA-descriptor choice and it is the one that makes a
    line buffer possible: with channel-major input all of channel 0 arrives before channel
    1, so a K-row window would have to buffer the whole image once Cin > 1.
    """
    cfg = STEM[mode]
    cin, h, w = cfg["cin"], cfg["h"], cfg["w"]
    px = read_golden(golden_dir, "stem_%s_image" % mode)
    if len(px) != cin * h * w:
        raise ValueError("stem_%s_image: %d pixels, want %d" % (mode, len(px), cin * h * w))
    out = bytearray()
    for y in range(h):
        for x in range(w):
            for c in range(cin):
                v = px[(c * h + y) * w + x]
                if not 0 <= v <= 255:
                    raise ValueError("pixel %d is not uint8 at (%d,%d,%d)" % (v, c, y, x))
                out.append(v)
    if len(out) % AXIS_LANES:
        raise ValueError("row length does not tile by %d lanes" % AXIS_LANES)
    return bytes(out)


def unpack_result(raw):
    """The 5 output beats -> real values. `y * 2^-OUTPUT_FRAC`.

    The kernel sends the FIXED-POINT form, not the accumulator, so the host multiplies
    nothing. The accumulator is what the golden holds and it is upstream of this.
    """
    if len(raw) != OUT_DIM * 4:
        raise ValueError("expected %d bytes, got %d" % (OUT_DIM * 4, len(raw)))
    fixed = struct.unpack("<%di" % OUT_DIM, raw)
    return [v / float(1 << OUTPUT_FRAC) for v in fixed]


def pack_anchor(values):
    """The host's 5-dim pupil state, already on the host grid. Track only."""
    if len(values) != ANCHOR:
        raise ValueError("anchor is %d-dim, want %d" % (len(values), ANCHOR))
    return struct.pack("<%di" % ANCHOR, *[int(v) for v in values])


# --- `status`, the third s_axilite word --------------------------------------
# Declared here because the host is what reads it. The kernel's copy is a comment in
# hbtxr_payload.hpp; a smoke run that prints "status 3" and nothing else is a smoke run
# nobody can act on.
STATUS_OK = 0
STATUS_BAD_BLOB = 1        # magic, version, or the section table overruns the header
STATUS_MISSING_SECTION = 2  # a section is absent, short, or the wrong element width
STATUS_TAG_SLIP = 3        # the prefetch schedule slipped: a pair ran the wrong TRB
STATUS_BAD_MODE = 4
STATUS_TEXT = {
    STATUS_OK: "ok",
    STATUS_BAD_BLOB: "the weight blob is not readable (magic/version/table)",
    STATUS_MISSING_SECTION: "a section is absent or the wrong shape",
    STATUS_TAG_SLIP: "prefetch schedule slipped -- a core pair ran the wrong block",
    STATUS_BAD_MODE: "mode is neither 0 (search) nor 1 (track)",
}


def requant(acc, mult, shift, out_bits=32):
    """`clamp((acc * M + (1 << (n-1))) >> n)` — `i_ops.py:requant`, bit for bit.

    Python's `>>` on a negative int is arithmetic and floors, which is what the hardware's
    signed shift does. Sign-extending by hand instead is where the old implementation lost
    a bit.
    """
    rnd = (1 << (shift - 1)) if shift > 0 else 0
    v = (acc * mult + rnd) >> shift
    lo, hi = -(1 << (out_bits - 1)), (1 << (out_bits - 1)) - 1
    return lo if v < lo else (hi if v > hi else v)


def expected_result(golden_dir, mode):
    """What the board should send back, computed from the golden. The smoke test's oracle.

    The golden holds the head's ACCUMULATOR — un-requantized, because the consumer is the
    host and there is no calibrated output grid. The AXIS port carries the fixed-point
    form, so the last requant happens on the kernel and this mirrors it.

    The output requant is MODE-DEPENDENT: the two heads' last linears sit on different
    accumulator grids. One table for both keeps passing the accumulator golden while
    silently reporting the wrong value for whichever mode ran second.
    """
    acc = read_golden(golden_dir, "%s_head_y_acc" % mode)
    mult = read_golden(golden_dir, "%s_head_out_mult" % mode)
    shift = read_golden(golden_dir, "%s_head_out_shift" % mode)
    fixed = [requant(a, m, s) for a, m, s in zip(acc, mult, shift)]
    return [v / float(1 << OUTPUT_FRAC) for v in fixed]


# --- section table reader (the same view the kernel gets) --------------------
def sections(raw):
    magic, version, header_bytes, n, _ = HEADER.unpack_from(raw, 0)
    if magic != MAGIC:
        raise ValueError("not an HBTXR payload blob")
    if version != VERSION:
        raise ValueError("blob version %d, this reader is %d" % (version, VERSION))
    out = {}
    at = HEADER.size
    for _ in range(n):
        name, off, nbytes, count, bits = ENTRY.unpack_from(raw, at)
        at += ENTRY.size
        out[name.rstrip(b"\0").decode("utf-8")] = (off, nbytes, count, bits)
    if at > header_bytes:
        raise ValueError("section table overruns the declared header")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--golden", required=True, help="model-scope golden directory")
    ap.add_argument("--out", default=os.path.join("hardware", "workspace", "deploy"))
    ap.add_argument("--list", action="store_true", help="print the section table and exit")
    args = ap.parse_args(argv)

    blob = pack_weights(args.golden)
    table = sections(blob)
    if args.list:
        for name in sorted(table, key=lambda k: table[k][0]):
            off, nbytes, count, bits = table[name]
            print("%-22s %10d  %9d B  %8d x %2d b" % (name, off, nbytes, count, bits))
        print("\n%d sections, %.2f MB" % (len(table), len(blob) / 1048576.0))
        return 0

    if not os.path.isdir(args.out):
        os.makedirs(args.out)
    paths = [("hbtxr_weights.bin", blob)]
    for mode in ("search", "track"):
        paths.append(("hbtxr_image_%s.bin" % mode, pack_image(args.golden, mode)))
    paths.append(("hbtxr_anchor_track.bin",
                  pack_anchor(read_golden(args.golden, "track_anchor_x"))))
    for name, data in paths:
        p = os.path.join(args.out, name)
        with io.open(p, "wb") as f:
            f.write(data)
        print("%-26s %9d B" % (name, len(data)))
    print("%d sections in the weight blob" % len(table))
    return 0


if __name__ == "__main__":
    sys.exit(main())
