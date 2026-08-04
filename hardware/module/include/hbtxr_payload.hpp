// The kernel end of the host payload contract — the M-AXI weight blob (SPEC §8).
//
// `deploy/hbtxr/payload.py` is the other end. They are NOT two descriptions of one
// layout: the blob carries a section table and both sides read it, so neither recomputes
// an offset and neither can drift from the other. Every bug this project has spent a
// stage on was one side of a pair being wrong on its own — a reshape that never reached
// the tool, a measurement taken through the datapath it was measuring, a DSP attribution
// that survived two rounds of reasoning. A table in the file cannot disagree with itself.
//
// This is the `SOURCE` the backbone's contract asks for:
//
//     src.load(i, mha, mlp)   // puts TRB i's payload into one core pair
//
// `tb/hbtxr_model.hpp` is that same interface reading the golden `.txt` directly. Both
// exist on purpose: `tb_payload` loads a core pair BOTH ways and compares every member,
// so a packing error names its own section instead of surfacing as a wrong pupil.
//
// No <string>, no allocation: this runs on the board.
#ifndef HBTXR_PAYLOAD_HPP
#define HBTXR_PAYLOAD_HPP

#include <ap_int.h>

#include "hbtxr_config.hpp"
#include "hbtxr_layernorm.hpp"
#include "hbtxr_mha_core.hpp"
#include "hbtxr_mlp_core.hpp"

namespace hbtxr {

/// One entry of the blob's section table.
struct HbtxrSection {
  unsigned long long off = 0, bytes = 0;
  int count = 0, bits = 0;
  bool ok = false;
};

/// Build "b3.fc1.rq". `b` may be null for a one-part name like "b3.edges".
inline void hbtxr_secname(char *dst, int blk, const char *a, const char *b) {
  int i = 0;
  dst[i++] = 'b';
  if (blk >= 10) dst[i++] = (char)('0' + blk / 10);
  dst[i++] = (char)('0' + blk % 10);
  dst[i++] = '.';
  for (int j = 0; a[j]; ++j) dst[i++] = a[j];
  if (b) {
    dst[i++] = '.';
    for (int j = 0; b[j]; ++j) dst[i++] = b[j];
  }
  dst[i] = 0;
}

/// A read-only view over the blob. The name lookup runs ONCE per section at load time,
/// never per element.
struct HbtxrBlobView {
  static const int NAME_BYTES = 24;
  static const int ENTRY_BYTES = NAME_BYTES + 8 + 8 + 4 + 4;   // 48
  static const int HEADER_BYTES = 8 + 4 + 4 + 4 + 4;           // 24

  const unsigned char *base = 0;
  unsigned long long size = 0;
  int n = 0;

  /// Returns false rather than aborting: the caller knows whether a bad blob is a
  /// testbench mistake or a board that was handed the wrong file.
  bool open(const unsigned char *p, unsigned long long bytes) {
    base = p;
    size = bytes;
    n = 0;
    if (!p || bytes < (unsigned long long)HEADER_BYTES) return false;
    const char magic[8] = {'H', 'B', 'T', 'X', 'R', 'P', 'L', 0};
    for (int i = 0; i < 8; ++i)
      if ((char)p[i] != magic[i]) return false;
    if (u32(8) != 1u) return false;                      // version
    const unsigned long long header_bytes = u32(12);
    const unsigned long long entries = u32(16);
    if (header_bytes > bytes) return false;
    if ((unsigned long long)HEADER_BYTES + (unsigned long long)ENTRY_BYTES * entries >
        header_bytes)
      return false;
    n = (int)entries;
    return true;
  }

  unsigned int u32(unsigned long long at) const {
    return (unsigned int)base[at] | ((unsigned int)base[at + 1] << 8) |
           ((unsigned int)base[at + 2] << 16) | ((unsigned int)base[at + 3] << 24);
  }
  unsigned long long u64(unsigned long long at) const {
    return (unsigned long long)u32(at) | ((unsigned long long)u32(at + 4) << 32);
  }

  HbtxrSection find(const char *name) const {
    HbtxrSection s;
    int len = 0;
    while (name[len] && len < NAME_BYTES) ++len;
    if (len >= NAME_BYTES) return s;          // could not have been stored
    for (int e = 0; e < n; ++e) {
      const unsigned long long at =
          (unsigned long long)HEADER_BYTES + (unsigned long long)ENTRY_BYTES * e;
      bool hit = true;
      for (int i = 0; i < len && hit; ++i)
        if ((char)base[at + i] != name[i]) hit = false;
      // The stored name has to END here too, or "b1.fc1.w" would match "b1.fc1.wx".
      if (hit && base[at + len] != 0) hit = false;
      if (!hit) continue;
      s.off = u64(at + NAME_BYTES);
      s.bytes = u64(at + NAME_BYTES + 8);
      s.count = (int)u32(at + NAME_BYTES + 16);
      s.bits = (int)u32(at + NAME_BYTES + 20);
      s.ok = s.off + s.bytes <= size;
      return s;
    }
    return s;
  }

  // --- typed element reads --------------------------------------------------
  // Each section declares its element width and the caller states what it expects;
  // `HbtxrBlobSource::expect` compares them at load time. A width mismatch that is only
  // discovered in the data is a mismatch nobody can attribute.

  /// Two nibbles per byte, LOW nibble first, sign-extended from 4 bits.
  int i4(const HbtxrSection &s, int i) const {
    const unsigned char byte = base[s.off + (unsigned long long)(i >> 1)];
    const int nib = (i & 1) ? (byte >> 4) : (byte & 0xF);
    return (nib & 0x8) ? nib - 16 : nib;
  }
  int i8(const HbtxrSection &s, int i) const {
    const unsigned char b = base[s.off + (unsigned long long)i];
    return (b & 0x80) ? (int)b - 256 : (int)b;
  }
  int i16(const HbtxrSection &s, int i) const {
    const unsigned int v = u16(s, i);
    return (v & 0x8000u) ? (int)v - 65536 : (int)v;
  }
  unsigned int u16(const HbtxrSection &s, int i) const {
    const unsigned long long at = s.off + 2 * (unsigned long long)i;
    return (unsigned int)base[at] | ((unsigned int)base[at + 1] << 8);
  }
  int i32(const HbtxrSection &s, int i) const {
    return (int)u32(s.off + 4 * (unsigned long long)i);
  }
  long long i64(const HbtxrSection &s, int i) const {
    return (long long)u64(s.off + 8 * (unsigned long long)i);
  }
  /// `mult | shift << REQ_M_BITS`. One word because they are one decision.
  void requant(const HbtxrSection &s, int i, long long &mult, int &shift) const {
    const unsigned int w = u32(s.off + 4 * (unsigned long long)i);
    mult = (long long)(w & ((1u << 18) - 1u));
    shift = (int)(w >> 18);
  }
};

/// Reads TRB payloads out of the blob. The backbone's `SOURCE`.
template <class CFG>
struct HbtxrBlobSource {
  static const int D = CFG::D, F = CFG::F;

  /// The twelve per-tensor edges, in the order `payload.py:PER_TENSOR_EDGES` packs them.
  /// The per-CHANNEL ones are not here — they belong to the matmul that produced the
  /// accumulator and are packed with it.
  enum { E01 = 0, E02, E04, E05, E06, E08A, E08B, E09, E10, E12, E14A, E14B, N_EDGES };

  HbtxrBlobView v;
  int loads = 0;
  int errors = 0;
  char missing[40];      // the first section that was absent or the wrong shape

  HbtxrBlobSource() { missing[0] = 0; }

  bool open(const unsigned char *p, unsigned long long bytes) {
    errors = 0;
    missing[0] = 0;
    return v.open(p, bytes);
  }

  /// A section that is absent, short, or the wrong element width is recorded ONCE and the
  /// load continues with zeros. Aborting would hide how many sections are wrong, and on
  /// the board there is nobody to read an abort.
  HbtxrSection expect(const char *name, int count, int bits) {
    const HbtxrSection s = v.find(name);
    if (!s.ok || s.count != count || s.bits != bits) {
      if (!missing[0]) {
        int i = 0;
        for (; name[i] && i < 39; ++i) missing[i] = name[i];
        missing[i] = 0;
      }
      ++errors;
    }
    return s;
  }

  template <int CO, int CI, class RMU>
  void put_rmu(RMU &rmu, int blk, const char *member) {
    char wn[40], bn[40], rn[40];
    hbtxr_secname(wn, blk, member, "w");
    hbtxr_secname(bn, blk, member, "b");
    hbtxr_secname(rn, blk, member, "rq");
    const HbtxrSection sw = expect(wn, CO * CI, 4);
    const HbtxrSection sb = expect(bn, CO, 32);
    const HbtxrSection sr = expect(rn, CO, 32);

    static typename CFG::w_t w[CO][CI];
    static typename CFG::acc_t b[CO];
    static ap_uint<CFG::REQ_M_BITS> m[CO];
    static ap_uint<5> s[CO];
    for (int o = 0; o < CO; ++o) {
      for (int i = 0; i < CI; ++i) w[o][i] = sw.ok ? v.i4(sw, o * CI + i) : 0;
      b[o] = sb.ok ? v.i32(sb, o) : 0;
      long long mult = 2;
      int shift = 1;                       // dyadic_params(1.0) -- bit-exact identity
      if (sr.ok) v.requant(sr, o, mult, shift);
      m[o] = (ap_uint<CFG::REQ_M_BITS>)mult;
      s[o] = (ap_uint<5>)shift;
    }
    rmu.load(w, b, m, s);
  }

  template <class LN>
  void put_ln(LN &ln, int blk, const char *which) {
    char n[4][40];
    const char *tail[4] = {"sc", "w", "b", "rs"};
    for (int i = 0; i < 4; ++i) hbtxr_secname(n[i], blk, which, tail[i]);
    load_ln(ln, n[0], n[1], n[2], n[3]);
  }

  template <class LN>
  void load_ln(LN &ln, const char *sc, const char *w, const char *b, const char *rs) {
    const HbtxrSection ss = expect(sc, 7, 32);
    const HbtxrSection sw = expect(w, D, 16);
    const HbtxrSection sb = expect(b, D, 64);
    const HbtxrSection sr = expect(rs, 64, 16);
    if (ss.ok) {
      ln.c_1_m = v.i32(ss, 0); ln.c_1_s = v.i32(ss, 1); ln.b = v.i32(ss, 2);
      ln.s1 = v.i32(ss, 3);    ln.bound = v.i32(ss, 4); ln.s2 = v.i32(ss, 5);
      ln.clamp_bits = v.i32(ss, 6);
    }
    for (int i = 0; i < D; ++i) {
      ln.lnw[i] = sw.ok ? v.i16(sw, i) : 0;
      // lnb is NOT a table entry: it lives on the affine accumulator grid, ~30 signed
      // bits, and is carried at 64 so a wrap is impossible rather than undiagnosable.
      ln.lnb[i] = sb.ok ? v.i64(sb, i) : 0;
    }
    for (int i = 0; i < 64; ++i) ln.rsqrt[i] = sr.ok ? v.i16(sr, i) : 0;
  }

  HbtxrEdge edge(const HbtxrSection &s, int i) const {
    HbtxrEdge e;
    if (s.ok) {
      long long m = 2;
      int sh = 1;
      v.requant(s, i, m, sh);
      e.mult = m;
      e.shift = sh;
    }
    return e;
  }

  void load(int blk, HbtxrMhaCore<CFG> &m, HbtxrMlpCore<CFG> &p) {
    ++loads;
    char n[40];

    put_ln(m.ln, blk, "ln1");
    put_ln(p.ln, blk, "ln2");

    hbtxr_secname(n, blk, "sm", "sc");
    const HbtxrSection ssc = expect(n, 14, 32);
    if (ssc.ok) {
      m.softmax.b1 = v.i32(ssc, 0);          m.softmax.s1 = v.i32(ssc, 1);
      m.softmax.bound1 = v.i32(ssc, 2);      m.softmax.b2_one = v.i32(ssc, 3);
      m.softmax.s2_one = v.i32(ssc, 4);      m.softmax.bound2_one = v.i32(ssc, 5);
      m.softmax.b3_one = v.i32(ssc, 6);      m.softmax.s3_one = v.i32(ssc, 7);
      m.softmax.b2_two = v.i32(ssc, 8);      m.softmax.s2_two = v.i32(ssc, 9);
      m.softmax.bound2_two = v.i32(ssc, 10); m.softmax.b3_two = v.i32(ssc, 11);
      m.softmax.s3_two = v.i32(ssc, 12);     m.softmax.clamp_bits = v.i32(ssc, 13);
    }
    hbtxr_secname(n, blk, "sm", "exp");
    const HbtxrSection sex = expect(n, 32, 16);
    hbtxr_secname(n, blk, "sm", "r1");
    const HbtxrSection sr1 = expect(n, 64, 16);
    hbtxr_secname(n, blk, "sm", "r2");
    const HbtxrSection sr2 = expect(n, 64, 16);
    // UNSIGNED: exp's largest entry is its numerator, 1<<15 = 32768, and int16 cannot
    // hold it. The sign is per operator and getting it wrong is silent.
    for (int i = 0; i < 32; ++i) m.softmax.exp_table[i] = sex.ok ? v.u16(sex, i) : 0;
    for (int i = 0; i < 64; ++i) {
      m.softmax.recip_one[i] = sr1.ok ? v.u16(sr1, i) : 0;
      m.softmax.recip_two[i] = sr2.ok ? v.u16(sr2, i) : 0;
    }

    hbtxr_secname(n, blk, "gelu", "sc");
    const HbtxrSection gsc = expect(n, 3, 32);
    hbtxr_secname(n, blk, "gelu", "t");
    const HbtxrSection gt = expect(n, 32, 16);
    if (gsc.ok) {
      p.gelu_b = v.i32(gsc, 0);
      p.gelu_s = v.i32(gsc, 1);
      p.gelu_bound = v.i32(gsc, 2);
    }
    for (int i = 0; i < 32; ++i) p.gelu_table[i] = gt.ok ? v.i16(gt, i) : 0;

    // The qkv accumulator's three consumers concatenate in OUTPUT-CHANNEL order, and the
    // packer laid them down that way -- one unit, three grids, no separate hardware.
    put_rmu<3 * D, D>(m.qkv, blk, "qkv");
    put_rmu<D, D>(m.proj, blk, "proj");
    put_rmu<F, D>(p.fc1, blk, "fc1");
    put_rmu<D, F>(p.fc2, blk, "fc2");

    hbtxr_secname(n, blk, "edges", 0);
    const HbtxrSection ed = expect(n, N_EDGES, 32);
    m.e01_stream_to_ln1 = edge(ed, E01);  m.e02_ln1_to_qkv = edge(ed, E02);
    m.e04_smu_to_softmax = edge(ed, E04); m.e05_softmax_to_av = edge(ed, E05);
    m.e06_av_to_proj = edge(ed, E06);     m.e08_resid1_a = edge(ed, E08A);
    m.e08_resid1_b = edge(ed, E08B);
    p.e09_stream_to_ln2 = edge(ed, E09);  p.e10_ln2_to_fc1 = edge(ed, E10);
    p.e12_gelu_to_fc2 = edge(ed, E12);    p.e14_resid2_a = edge(ed, E14A);
    p.e14_resid2_b = edge(ed, E14B);
  }
};

}  // namespace hbtxr

#endif  // HBTXR_PAYLOAD_HPP
