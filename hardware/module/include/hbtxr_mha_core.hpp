// MHA Core — the nine-stage attention chain (SPEC §5-1).
//
//   residual split -> LayerNorm -> Q/K/V (RMU) -> head-wise reorder
//     -> Q x K^T (SMU) -> Softmax -> S x V (SMU) -> output proj (RMU) -> residual merge
//
// SCOPE. S4 establishes the ARITHMETIC chain and its golden. The stages are composed
// through full intermediate buffers, not through a `dataflow` region with the FIFO depths
// SPEC §5-1 specifies — those depths and the reorder-buffer split belong to S8/S9, and
// tuning a schedule for a chain that is not yet bit-exact is the wrong order. The one
// place this shows is `qkv_buf`, which holds all of Q, K and V where the target design
// keeps a per-head reorder buffer.
#ifndef HBTXR_MHA_CORE_HPP
#define HBTXR_MHA_CORE_HPP

#include <ap_int.h>
#include <hls_stream.h>
#include <hls_vector.h>

#include "hbtxr_config.hpp"
#include "hbtxr_layernorm.hpp"
#include "hbtxr_requant.hpp"
#include "hbtxr_rmu.hpp"
#include "hbtxr_smu.hpp"
#include "hbtxr_softmax.hpp"

namespace hbtxr {

/// A per-tensor requant edge: `(M, n)`. The per-channel ones live inside their matmul.
struct HbtxrEdge {
  long long mult = 2;
  int shift = 1;   // dyadic_params(1.0) -- bit-exact identity
};

template <class CFG>
struct HbtxrMhaCore {
  static constexpr int D = CFG::D, H = CFG::H, HD = CFG::HD, N = CFG::N, TP = CFG::TP;
  static constexpr int P = CFG::NL_P;           // one lane count end to end (config check)
  static constexpr int QKV_OUT = 3 * D;
  static constexpr int LANES = TP * P;

  typedef typename CFG::act_t act_t;
  typedef typename CFG::prob_t prob_t;
  typedef typename CFG::acc_t acc_t;
  typedef hls::vector<act_t, LANES> beat_t;
  typedef hls::vector<prob_t, LANES> prob_beat_t;

  typedef HbtxrLayerNorm<CFG, D, P> ln_t;
  typedef HbtxrRmu<CFG, D, QKV_OUT, CFG::Q_CIP, CFG::Q_COP> qkv_t;
  typedef HbtxrSmu<CFG, HD, N, CFG::R_CIP, CFG::R_COP, true> score_t;   // Q x K^T
  typedef HbtxrSoftmax<CFG, N, P> softmax_t;
  typedef HbtxrSmu<CFG, N, HD, CFG::A_CIP, CFG::A_COP, false> ctx_t;    // S x V
  typedef HbtxrRmu<CFG, D, D, CFG::O_CIP, CFG::O_COP> proj_t;

  ln_t ln;
  qkv_t qkv;
  score_t score;
  softmax_t softmax;
  ctx_t ctx;
  proj_t proj;

  // Per-tensor edges, numbered as the golden numbers them.
  HbtxrEdge e01_stream_to_ln1, e02_ln1_to_qkv, e04_smu_to_softmax, e05_softmax_to_av;
  HbtxrEdge e06_av_to_proj, e08_resid1_a, e08_resid1_b;

  // --- buffers ---------------------------------------------------------------
  act_t qkv_buf[N][QKV_OUT];   // reorder buffer + head-local FIFO, merged (see SCOPE)
  act_t ctx_buf[N][D];         // heads concatenated back to [tokens, D]

  // --- plumbing --------------------------------------------------------------
  // In the target design these are the FIFOs of SPEC §5-1. Here they are the boundary
  // between two units that each buffer internally anyway.
  template <class BEAT, class T>
  static void push2d(hls::stream<BEAT> &s, const T *src, int rows, int cols) {
    for (int r0 = 0; r0 < rows; r0 += TP)
      for (int c0 = 0; c0 < cols; c0 += P) {
        BEAT v;
        for (int p = 0; p < TP; ++p)
          for (int c = 0; c < P; ++c) v[p * P + c] = src[(r0 + p) * cols + c0 + c];
        s.write(v);
      }
  }
  template <class BEAT, class T>
  static void pop2d(hls::stream<BEAT> &s, T *dst, int rows, int cols) {
    for (int r0 = 0; r0 < rows; r0 += TP)
      for (int c0 = 0; c0 < cols; c0 += P) {
        const BEAT v = s.read();
        for (int p = 0; p < TP; ++p)
          for (int c = 0; c < P; ++c) dst[(r0 + p) * cols + c0 + c] = v[p * P + c];
      }
  }

  /// `x` and `y` are [tokens, D] on the residual-stream grid.
  ///
  /// `probe` is testbench-only and null in synthesis; it compares each stage against its
  /// own golden, asserts the stream drained, and REFILLS, so the first failing probe is
  /// the actual fault site rather than the first stage downstream of it.
  template <class PROBE>
  void run(const act_t *x, act_t *y, int tokens, PROBE *probe) {
    const int d_beats = (tokens / TP) * (D / P);

    // --- stage 1-2: residual split, then LayerNorm on the branch ------------
    hls::stream<beat_t> s_ln_in("ln_in"), s_ln_out("ln_out"), s_qkv_in("qkv_in");
    {
      hls::stream<beat_t> s_x("x");
      push2d(s_x, x, tokens, D);
      requant_stream<CFG, beat_t, beat_t, acc_t>(s_x, s_ln_in, d_beats, LANES,
                                                 e01_stream_to_ln1.mult,
                                                 e01_stream_to_ln1.shift);
    }
    if (probe) probe->check("ln1_x", s_ln_in, LANES);
    ln.run(s_ln_in, s_ln_out, tokens);
    if (probe) probe->check("ln1_y", s_ln_out, LANES);

    // --- stage 3: Q/K/V ------------------------------------------------------
    requant_stream<CFG, beat_t, beat_t, acc_t>(s_ln_out, s_qkv_in, d_beats, LANES,
                                               e02_ln1_to_qkv.mult, e02_ln1_to_qkv.shift);
    if (probe) probe->check("qkv_x", s_qkv_in, LANES);
    {
      // ONE unit, three output grids. Q, K and V are contiguous output-channel ranges, so
      // "requantize the accumulator directly onto each consumer's grid" is expressed
      // entirely by the per-channel (M, n) arrays — no separate hardware.
      hls::stream<beat_t> s_qkv_out("qkv_out");
      qkv.compute(s_qkv_in, s_qkv_out, tokens);
      pop2d(s_qkv_out, &qkv_buf[0][0], tokens, QKV_OUT);
    }

    // --- stage 4-7: per head ------------------------------------------------
    for (int h = 0; h < H; ++h) {
      act_t q[N * HD], k[N * HD], v[N * HD];
      for (int t = 0; t < tokens; ++t)
        for (int c = 0; c < HD; ++c) {
          q[t * HD + c] = qkv_buf[t][0 * D + h * HD + c];
          k[t * HD + c] = qkv_buf[t][1 * D + h * HD + c];
          v[t * HD + c] = qkv_buf[t][2 * D + h * HD + c];
        }

      hls::stream<beat_t> s_q("q"), s_k("k"), s_score("score"), s_av_in("av_in"), s_v("v");
      hls::stream<prob_beat_t> s_prob("prob");
      hls::stream<beat_t> s_pv("pv");

      // Q x K^T. K goes in first -- the score matrix has no row 0 until all of K has
      // arrived, which is the token-global dependency the MHA residual FIFO exists for.
      push2d(s_k, k, tokens, HD);
      push2d(s_q, q, tokens, HD);
      score.run(s_q, s_k, s_score, tokens, tokens, HD, e04_smu_to_softmax.mult,
                e04_smu_to_softmax.shift);
      if (probe) probe->check(h == 0 ? "smu_y" : "", s_score, LANES);

      softmax.run(s_score, s_prob, tokens, tokens);
      if (probe) probe->check(h == 0 ? "softmax_y" : "", s_prob, LANES);

      // S x V: no transpose, and the reduction runs over TOKENS, not the head dim.
      const int score_beats = (tokens / TP) * (tokens / P);
      requant_stream<CFG, prob_beat_t, beat_t, acc_t>(s_prob, s_av_in, score_beats, LANES,
                                                      e05_softmax_to_av.mult,
                                                      e05_softmax_to_av.shift);
      push2d(s_v, v, tokens, HD);
      ctx.run(s_av_in, s_v, s_pv, tokens, HD, tokens, e06_av_to_proj.mult,
              e06_av_to_proj.shift);

      act_t part[N * HD];
      pop2d(s_pv, part, tokens, HD);
      for (int t = 0; t < tokens; ++t)
        for (int c = 0; c < HD; ++c) ctx_buf[t][h * HD + c] = part[t * HD + c];
    }

    // --- stage 8-9: output projection, then the residual merge ---------------
    hls::stream<beat_t> s_ctx("ctx"), s_proj("proj"), s_res("res"), s_out("out");
    push2d(s_ctx, &ctx_buf[0][0], tokens, D);
    if (probe) probe->check("rmu_x", s_ctx, LANES);
    proj.compute(s_ctx, s_proj, tokens);
    if (probe) probe->check("rmu_y", s_proj, LANES);

    push2d(s_res, x, tokens, D);
    residual_merge<CFG, beat_t, acc_t>(s_res, s_proj, s_out, d_beats, LANES,
                                       e08_resid1_a.mult, e08_resid1_a.shift,
                                       e08_resid1_b.mult, e08_resid1_b.shift);
    pop2d(s_out, y, tokens, D);
  }
};

}  // namespace hbtxr

#endif  // HBTXR_MHA_CORE_HPP
