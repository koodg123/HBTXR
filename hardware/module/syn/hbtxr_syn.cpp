// S9 — concrete synthesis tops.
//
// Every unit in `module/include` is a TEMPLATE, and csynth needs a function. One file with
// N tops, `set_top` picking one: the others are still parsed and type-checked, which is
// cheaper than N files that drift apart.
//
// EVERY UNIT TAKES ITS PAYLOAD THROUGH PORTS, and that is not fidelity to the board — it
// is what makes the measurement true. The first version of this file held each unit in a
// `static` and only called its compute path. A static array that is never written is
// zero-initialised, so `weight` was provably all zeros, the whole MAC array constant-folded
// away, and csynth reported a 192x192 matmul at 90 FF and 0 DSP. It looked like a result.
// Anything that READS a resident array must have a path that WRITES it, or the report is
// measuring the absence of the design.
//
// So each top has two modes: `mode == 0` loads, anything else computes. One design, both
// paths, which is also what the RMU's third mode actually is (SPEC §4) — the reported
// resources therefore INCLUDE the load path, and the report says so.
//
// SCOPE — what is still not here:
//   - the board's M-AXI / AXIS interfaces (S10). Default ap_memory / ap_fifo ports are what
//     a datapath measurement wants; an AXI wrapper's LUTs would be reported as the unit's.
//   - `HbtxrTop` and `HbtxrBackbone`, which need a synthesizable weight SOURCE. That is an
//     interface decision, not an arithmetic one, so it belongs with S10.
//
// The tops are ordered leaf-first: if a core misses II=1 it is because one of the units
// under it did, and the leaf reports say which.
#include "hbtxr_top.hpp"   // pulls backbone, head, patch_embed and everything below

using namespace hbtxr;

typedef HbtxrCfgBase CFG;
static constexpr int D = CFG::D, F = CFG::F, N = CFG::N, HD = CFG::HD;
static constexpr int P = CFG::NL_P, LANES = CFG::TP * CFG::NL_P;
static constexpr int MB = CFG::REQ_M_BITS;

typedef CFG::act_t act_t;
typedef CFG::acc_t acc_t;
typedef CFG::w_t w_t;
typedef CFG::nl_t nl_t;
typedef CFG::nlu_t nlu_t;
typedef ap_uint<MB> mult_t;
typedef ap_uint<5> shift_t;
typedef hls::vector<act_t, LANES> beat_t;
typedef hls::vector<nl_t, LANES> nl_beat_t;

// --- payload helpers ----------------------------------------------------------
// The RMU already has `load`. LayerNorm and Softmax do not — the testbench pokes their
// members from the golden, which is file IO and does not synthesize. These are the same
// assignments with the ports in place of the files; the ORDER of the scalar array is the
// generator's, and `hbtxr_load.hpp` is the other end of that contract.

template <class LN>
static void load_ln(LN &u, const nl_t *lnw, const typename LN::affine_t *lnb,
                    const nl_t *rsq, const int *sc, int c) {
  for (int i = 0; i < c; ++i) {
#pragma HLS pipeline II = 1
    u.lnw[i] = lnw[i];
    u.lnb[i] = lnb[i];
  }
  for (int i = 0; i < 64; ++i) {
#pragma HLS pipeline II = 1
    u.rsqrt[i] = rsq[i];
  }
  u.c_1_m = sc[0]; u.c_1_s = sc[1]; u.b = sc[2];
  u.s1 = sc[3]; u.bound = sc[4]; u.s2 = sc[5]; u.clamp_bits = sc[6];
}

template <class SM>
static void load_sm(SM &u, const nlu_t *ex, const nlu_t *r1, const nlu_t *r2,
                    const int *sc) {
  for (int i = 0; i < 32; ++i) {
#pragma HLS pipeline II = 1
    u.exp_table[i] = ex[i];
  }
  for (int i = 0; i < 64; ++i) {
#pragma HLS pipeline II = 1
    u.recip_one[i] = r1[i];
    u.recip_two[i] = r2[i];
  }
  u.b1 = sc[0];     u.s1 = sc[1];      u.bound1 = sc[2];
  u.b2_one = sc[3]; u.s2_one = sc[4];  u.bound2_one = sc[5];
  u.b3_one = sc[6]; u.s3_one = sc[7];
  u.b2_two = sc[8]; u.s2_two = sc[9];  u.bound2_two = sc[10];
  u.b3_two = sc[11]; u.s3_two = sc[12]; u.clamp_bits = sc[13];
}

/// The per-tensor edges of a core, as one pair of arrays rather than n scalar ports.
static void load_edges(HbtxrEdge *dst, const mult_t *m, const shift_t *s, int n) {
  for (int i = 0; i < n; ++i) {
#pragma HLS pipeline II = 1
    dst[i].mult = (long long)m[i];
    dst[i].shift = (int)s[i];
  }
}

// --- matmul units -------------------------------------------------------------

typedef HbtxrRmu<CFG, D, D, CFG::O_CIP, CFG::O_COP> proj_t;
void syn_rmu_proj(int mode, const w_t w[D][D], const acc_t b[D], const mult_t m[D],
                  const shift_t s[D], hls::stream<proj_t::in_beat_t> &in,
                  hls::stream<proj_t::out_beat_t> &out, int tokens) {
  static proj_t u;
  if (mode == 0) u.load(w, b, m, s);
  else u.compute(in, out, tokens);
}

typedef HbtxrRmu<CFG, D, 3 * D, CFG::Q_CIP, CFG::Q_COP> qkv_t;
void syn_rmu_qkv(int mode, const w_t w[3 * D][D], const acc_t b[3 * D],
                 const mult_t m[3 * D], const shift_t s[3 * D],
                 hls::stream<qkv_t::in_beat_t> &in, hls::stream<qkv_t::out_beat_t> &out,
                 int tokens) {
  static qkv_t u;
  if (mode == 0) u.load(w, b, m, s);
  else u.compute(in, out, tokens);
}

typedef HbtxrRmu<CFG, D, F, CFG::M1_CIP, CFG::M1_COP> fc1_t;   // expand   D -> F
void syn_rmu_fc1(int mode, const w_t w[F][D], const acc_t b[F], const mult_t m[F],
                 const shift_t s[F], hls::stream<fc1_t::in_beat_t> &in,
                 hls::stream<fc1_t::out_beat_t> &out, int tokens) {
  static fc1_t u;
  if (mode == 0) u.load(w, b, m, s);
  else u.compute(in, out, tokens);
}

typedef HbtxrRmu<CFG, F, D, CFG::M2_CIP, CFG::M2_COP> fc2_t;   // contract F -> D
void syn_rmu_fc2(int mode, const w_t w[D][F], const acc_t b[D], const mult_t m[D],
                 const shift_t s[D], hls::stream<fc2_t::in_beat_t> &in,
                 hls::stream<fc2_t::out_beat_t> &out, int tokens) {
  static fc2_t u;
  if (mode == 0) u.load(w, b, m, s);
  else u.compute(in, out, tokens);
}

// The two SMUs are ONE MAC array under two load orders (SPEC §5-2). Both are synthesized so
// that claim is a measurement rather than a comment. Neither has resident payload — `bt` is
// written from the b stream — so neither needs a load mode.
typedef HbtxrSmu<CFG, HD, N, CFG::R_CIP, CFG::R_COP, true> score_t;
void syn_smu_score(hls::stream<score_t::in_beat_t> &a, hls::stream<score_t::in_beat_t> &b,
                   hls::stream<score_t::out_beat_t> &out, int rows, int cols, int kdim,
                   mult_t mult, shift_t shift) {
  static score_t u;
  u.run(a, b, out, rows, cols, kdim, mult, shift);
}

typedef HbtxrSmu<CFG, N, HD, CFG::A_CIP, CFG::A_COP, false> ctx_t;
void syn_smu_ctx(hls::stream<ctx_t::in_beat_t> &a, hls::stream<ctx_t::in_beat_t> &b,
                 hls::stream<ctx_t::out_beat_t> &out, int rows, int cols, int kdim,
                 mult_t mult, shift_t shift) {
  static ctx_t u;
  u.run(a, b, out, rows, cols, kdim, mult, shift);
}

// --- nonlinear and edge units -------------------------------------------------

typedef HbtxrLayerNorm<CFG, D, P> ln_t;
void syn_layernorm(int mode, const nl_t lnw[D], const ln_t::affine_t lnb[D],
                   const nl_t rsq[64], const int sc[7],
                   hls::stream<ln_t::in_beat_t> &in, hls::stream<ln_t::beat_t> &out,
                   int rows) {
  static ln_t u;
  if (mode == 0) load_ln(u, lnw, lnb, rsq, sc, D);
  else u.run(in, out, rows);
}

typedef HbtxrSoftmax<CFG, N, P> softmax_t;
void syn_softmax(int mode, const nlu_t ex[32], const nlu_t r1[64], const nlu_t r2[64],
                 const int sc[14], hls::stream<softmax_t::in_beat_t> &in,
                 hls::stream<softmax_t::out_beat_t> &out, int rows, int cols) {
  static softmax_t u;
  if (mode == 0) load_sm(u, ex, r1, r2, sc);
  else u.run(in, out, rows, cols);
}

// Table as a port already — it is loaded per block like any other payload, so there is no
// load mode to add.
void syn_gelu_edge(hls::stream<beat_t> &in, hls::stream<nl_beat_t> &out, int beats,
                   const nl_t table[32], int b, int s, int bound) {
  gelu_stream<CFG, 32, beat_t, nl_beat_t>(in, out, beats, LANES, table, b, s, bound);
}

// The single most-instantiated piece of logic in the design: 18 of these per block.
void syn_requant_edge(hls::stream<beat_t> &in, hls::stream<beat_t> &out, int beats,
                      mult_t mult, shift_t shift) {
  requant_stream<CFG, beat_t, beat_t, acc_t>(in, out, beats, LANES, mult, shift);
}

// --- the cores ----------------------------------------------------------------
// These are the paper's four cores: two MHA and two MLP instances of exactly this.
//
// The payload port list is long because a core's payload IS long — this is the thing the
// weight prefetcher moves every block, and its size is the reason the prefetcher exists.
// `probe` is the testbench's stage comparator and null here; the null is typed rather than
// a literal because `if (probe)` still has to compile a `check` call.

void syn_mha_core(int mode,
                  const nl_t lnw[D], const ln_t::affine_t lnb[D], const nl_t rsq[64],
                  const int ln_sc[7],
                  const nlu_t ex[32], const nlu_t r1[64], const nlu_t r2[64],
                  const int sm_sc[14],
                  const w_t qkv_w[3 * D][D], const acc_t qkv_b[3 * D],
                  const mult_t qkv_m[3 * D], const shift_t qkv_s[3 * D],
                  const w_t proj_w[D][D], const acc_t proj_b[D], const mult_t proj_m[D],
                  const shift_t proj_s[D],
                  const mult_t e_m[7], const shift_t e_s[7],
                  const act_t x[N * D], act_t y[N * D], int tokens) {
  static HbtxrMhaCore<CFG> u;
  if (mode == 0) {
    load_ln(u.ln, lnw, lnb, rsq, ln_sc, D);
    load_sm(u.softmax, ex, r1, r2, sm_sc);
    u.qkv.load(qkv_w, qkv_b, qkv_m, qkv_s);
    u.proj.load(proj_w, proj_b, proj_m, proj_s);
    // Numbered as the golden numbers them: 01, 02, 04, 05, 06, 08a, 08b.
    HbtxrEdge e[7];
    load_edges(e, e_m, e_s, 7);
    u.e01_stream_to_ln1 = e[0]; u.e02_ln1_to_qkv = e[1]; u.e04_smu_to_softmax = e[2];
    u.e05_softmax_to_av = e[3]; u.e06_av_to_proj = e[4];
    u.e08_resid1_a = e[5]; u.e08_resid1_b = e[6];
  } else {
    u.run(x, y, tokens, (HbtxrNoProbe *)0);
  }
}

void syn_mlp_core(int mode,
                  const nl_t lnw[D], const ln_t::affine_t lnb[D], const nl_t rsq[64],
                  const int ln_sc[7],
                  const w_t fc1_w[F][D], const acc_t fc1_b[F], const mult_t fc1_m[F],
                  const shift_t fc1_s[F],
                  const w_t fc2_w[D][F], const acc_t fc2_b[D], const mult_t fc2_m[D],
                  const shift_t fc2_s[D],
                  const nl_t gelu_tab[32], const int gelu_sc[3],
                  const mult_t e_m[5], const shift_t e_s[5],
                  const act_t x[N * D], act_t y[N * D], int tokens) {
  static HbtxrMlpCore<CFG> u;
  if (mode == 0) {
    load_ln(u.ln, lnw, lnb, rsq, ln_sc, D);
    u.fc1.load(fc1_w, fc1_b, fc1_m, fc1_s);
    u.fc2.load(fc2_w, fc2_b, fc2_m, fc2_s);
    for (int i = 0; i < 32; ++i) {
#pragma HLS pipeline II = 1
      u.gelu_table[i] = gelu_tab[i];
    }
    u.gelu_b = gelu_sc[0]; u.gelu_s = gelu_sc[1]; u.gelu_bound = gelu_sc[2];
    HbtxrEdge e[5];
    load_edges(e, e_m, e_s, 5);
    u.e09_stream_to_ln2 = e[0]; u.e10_ln2_to_fc1 = e[1]; u.e12_gelu_to_fc2 = e[2];
    u.e14_resid2_a = e[3]; u.e14_resid2_b = e[4];
  } else {
    u.run(x, y, tokens, (HbtxrNoProbe *)0);
  }
}

// --- stem and terminal decode -------------------------------------------------
// Two stems, as the paper has them, differing only in Cin and input size. The PE array is
// an RMU (SPEC §5-3), so the stem's payload is that RMU's.

typedef HbtxrPatchEmbed<HbtxrCfgPatch, CFG, 1, 16, 128, 128, P> stem_f_t;
void syn_patch_f(int mode, const HbtxrCfgPatch::w_t w[D][stem_f_t::TAPS],
                 const HbtxrCfgPatch::acc_t b[D], const mult_t m[D], const shift_t s[D],
                 hls::stream<stem_f_t::pix_beat_t> &in,
                 hls::stream<stem_f_t::tok_beat_t> &out) {
  static stem_f_t u;
  if (mode == 0) u.pe.load(w, b, m, s);
  else u.run(in, out);
}

typedef HbtxrPatchEmbed<HbtxrCfgPatch, CFG, 2, 16, 64, 64, P> stem_e_t;
void syn_patch_e(int mode, const HbtxrCfgPatch::w_t w[D][stem_e_t::TAPS],
                 const HbtxrCfgPatch::acc_t b[D], const mult_t m[D], const shift_t s[D],
                 hls::stream<stem_e_t::pix_beat_t> &in,
                 hls::stream<stem_e_t::tok_beat_t> &out) {
  static stem_e_t u;
  if (mode == 0) u.pe.load(w, b, m, s);
  else u.run(in, out);
}

typedef HbtxrCfgHead HCFG;
typedef HCFG::act_t head_t;
typedef HCFG::w_t hw_t;
typedef HCFG::acc_t hacc_t;

// `feat` and `acc_out` are sized ARRAYS, not pointers. A pointer port with a computed
// index is "address computation on scalar port" (214-323) and does not synthesize — the
// size is what tells the tool this is a memory rather than a single value.
template <class HEAD, int IN>
static void head_body(HEAD &u, int mode, const hw_t *fc1_w, const hacc_t *fc1_b,
                      const mult_t *fc1_m, const shift_t *fc1_s, const nl_t *gt,
                      const int *gsc, const mult_t *eg, const hw_t *fc2_w,
                      const hacc_t *fc2_b, const head_t *feat, hacc_t *acc_out) {
  if (mode == 0) {
    for (int o = 0; o < IN; ++o) {
      for (int i = 0; i < IN; ++i) {
#pragma HLS pipeline II = 1
        u.fc1_w[o][i] = fc1_w[o * IN + i];
      }
      u.fc1_b[o] = fc1_b[o];
      u.fc1_m[o] = fc1_m[o];
      u.fc1_s[o] = fc1_s[o];
    }
    for (int o = 0; o < 5; ++o) {
      for (int i = 0; i < IN; ++i) {
#pragma HLS pipeline II = 1
        u.fc2_w[o][i] = fc2_w[o * IN + i];
      }
      u.fc2_b[o] = fc2_b[o];
    }
    for (int i = 0; i < 32; ++i) {
#pragma HLS pipeline II = 1
      u.gelu_table[i] = gt[i];
    }
    u.gelu_b = gsc[0]; u.gelu_s = gsc[1]; u.gelu_bound = gsc[2];
    u.e_gelu_m = eg[0]; u.e_gelu_s = (shift_t)gsc[3];
  } else {
    u.run(feat, acc_out);
  }
}

void syn_head_box(int mode, const hw_t fc1_w[D * D], const hacc_t fc1_b[D],
                  const mult_t fc1_m[D], const shift_t fc1_s[D], const nl_t gt[32],
                  const int gsc[4], const mult_t eg[1], const hw_t fc2_w[5 * D],
                  const hacc_t fc2_b[5], const head_t feat[D], hacc_t acc_out[5]) {
  static HbtxrHead<HCFG, D, 5> u;
  head_body<HbtxrHead<HCFG, D, 5>, D>(u, mode, fc1_w, fc1_b, fc1_m, fc1_s, gt, gsc, eg,
                                      fc2_w, fc2_b, feat, acc_out);
}

// 197 = D + the host's 5-dim anchor. It tiles by nothing, which is why the head is a dense
// loop and not the RMU's PE array — this report is where that costs something, or not.
void syn_head_ellipse(int mode, const hw_t fc1_w[(D + 5) * (D + 5)],
                      const hacc_t fc1_b[D + 5], const mult_t fc1_m[D + 5],
                      const shift_t fc1_s[D + 5], const nl_t gt[32], const int gsc[4],
                      const mult_t eg[1], const hw_t fc2_w[5 * (D + 5)],
                      const hacc_t fc2_b[5], const head_t feat[D + 5], hacc_t acc_out[5]) {
  static HbtxrHead<HCFG, D + 5, 5> u;
  head_body<HbtxrHead<HCFG, D + 5, 5>, D + 5>(u, mode, fc1_w, fc1_b, fc1_m, fc1_s, gt, gsc,
                                              eg, fc2_w, fc2_b, feat, acc_out);
}
