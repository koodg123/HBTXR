// MLP Core — the six-stage chain (SPEC §5-2).
//
//   residual split -> LayerNorm -> FC1 (RMU) -> GeLU -> FC2 (RMU) -> residual merge
//
// NO SMU, no score buffer, no head reorder. That is the definitional difference from the
// MHA core, not an incidental one — and it is why the residual FIFO here only has to
// cover pipeline depth while the MHA core's has to hold the whole tensor: nothing on this
// path is token-global, so FC2's row t depends on row t alone.
#ifndef HBTXR_MLP_CORE_HPP
#define HBTXR_MLP_CORE_HPP

#include <ap_int.h>
#include <hls_stream.h>
#include <hls_vector.h>

#include "hbtxr_config.hpp"
#include "hbtxr_layernorm.hpp"
#include "hbtxr_lut.hpp"
#include "hbtxr_mha_core.hpp"   // HbtxrEdge
#include "hbtxr_requant.hpp"
#include "hbtxr_rmu.hpp"

namespace hbtxr {

template <class CFG>
struct HbtxrMlpCore {
  static constexpr int D = CFG::D, F = CFG::F, N = CFG::N, TP = CFG::TP;
  static constexpr int P = CFG::NL_P;
  static constexpr int LANES = TP * P;

  typedef typename CFG::act_t act_t;
  typedef typename CFG::nl_t nl_t;
  typedef typename CFG::acc_t acc_t;
  typedef hls::vector<act_t, LANES> beat_t;
  typedef hls::vector<nl_t, LANES> nl_beat_t;

  typedef HbtxrLayerNorm<CFG, D, P> ln_t;
  typedef HbtxrRmu<CFG, D, F, CFG::M1_CIP, CFG::M1_COP> fc1_t;   // expand  D -> F
  typedef HbtxrRmu<CFG, F, D, CFG::M2_CIP, CFG::M2_COP> fc2_t;   // contract F -> D

  ln_t ln;
  fc1_t fc1;
  fc2_t fc2;

  nl_t gelu_table[32];
  int gelu_b = 0, gelu_s = 0, gelu_bound = 31;

  HbtxrEdge e09_stream_to_ln2, e10_ln2_to_fc1, e12_gelu_to_fc2;
  HbtxrEdge e14_resid2_a, e14_resid2_b;

  /// `x` and `y` are [tokens, D] on the residual-stream grid.
  template <class PROBE>
  void run(const act_t *x, act_t *y, int tokens, PROBE *probe) {
    const int d_beats = (tokens / TP) * (D / P);
    const int f_beats = (tokens / TP) * (F / P);

    // --- stage 1-2: residual split, LayerNorm on the branch -----------------
    hls::stream<beat_t> s_ln_in("ln_in"), s_ln_out("ln_out"), s_fc1_in("fc1_in");
    {
      hls::stream<beat_t> s_x("x");
      HbtxrMhaCore<CFG>::push2d(s_x, x, tokens, D);
      requant_stream<CFG, beat_t, beat_t, acc_t>(s_x, s_ln_in, d_beats, LANES,
                                                 e09_stream_to_ln2.mult,
                                                 e09_stream_to_ln2.shift);
    }
    if (probe) probe->check("ln2_x", s_ln_in, LANES);
    ln.run(s_ln_in, s_ln_out, tokens);
    if (probe) probe->check("ln2_y", s_ln_out, LANES);

    // --- stage 3-4: FC1, then GeLU ------------------------------------------
    requant_stream<CFG, beat_t, beat_t, acc_t>(s_ln_out, s_fc1_in, d_beats, LANES,
                                               e10_ln2_to_fc1.mult, e10_ln2_to_fc1.shift);
    hls::stream<beat_t> s_hidden("hidden");
    fc1.compute(s_fc1_in, s_hidden, tokens);
    if (probe) probe->check("gelu_x", s_hidden, LANES);

    hls::stream<nl_beat_t> s_act("act");
    hls::stream<beat_t> s_fc2_in("fc2_in");
    gelu_stream<CFG, 32, beat_t, nl_beat_t>(s_hidden, s_act, f_beats, LANES, gelu_table,
                                            gelu_b, gelu_s, gelu_bound);
    if (probe) probe->check("gelu_y", s_act, LANES);
    requant_stream<CFG, nl_beat_t, beat_t, acc_t>(s_act, s_fc2_in, f_beats, LANES,
                                                  e12_gelu_to_fc2.mult,
                                                  e12_gelu_to_fc2.shift);

    // --- stage 5-6: FC2, then the residual merge ----------------------------
    hls::stream<beat_t> s_fc2("fc2"), s_res("res"), s_out("out");
    fc2.compute(s_fc2_in, s_fc2, tokens);
    if (probe) probe->check("fc2_y", s_fc2, LANES);

    HbtxrMhaCore<CFG>::push2d(s_res, x, tokens, D);
    residual_merge<CFG, beat_t, acc_t>(s_res, s_fc2, s_out, d_beats, LANES,
                                       e14_resid2_a.mult, e14_resid2_a.shift,
                                       e14_resid2_b.mult, e14_resid2_b.shift);
    HbtxrMhaCore<CFG>::pop2d(s_out, y, tokens, D);
  }
};

}  // namespace hbtxr

#endif  // HBTXR_MLP_CORE_HPP
