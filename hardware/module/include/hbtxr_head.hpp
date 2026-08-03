// Terminal decode — token pooling and the two-layer regression head (SPEC §7).
//
//   pool -> [concat anchor] -> Linear -> GeLU -> Linear -> accumulator
//
// Every HBTXR head is this shape (`models/heads/common.py:mlp_head`); Pupil Box and Pupil
// Ellipse differ only in `IN` and in whether a host-supplied anchor state is concatenated.
//
// A DENSE LOOP, not the RMU. The ellipse head's input is 197 = D + 5, which tiles by
// nothing, so forcing it into the RMU's PE array would mean padding logic that has to be
// proved bit-exact for no gain: at 197x197 the head is 0.1% of a single TRB.
#ifndef HBTXR_HEAD_HPP
#define HBTXR_HEAD_HPP

#include <ap_int.h>

#include "hbtxr_config.hpp"
#include "hbtxr_lut.hpp"
#include "hbtxr_requant.hpp"

namespace hbtxr {

/// Integer mean over the token axis, keeping the input scale.
///
/// Round-half-away-from-zero in exact integer arithmetic, which is what `i_ops._pool_tokens`
/// does and why: a float divide here would put a float on the datapath, and a sum over the
/// token axis is the easiest place in this graph to pass float32's 2^24 exact-integer limit.
/// `tokens` is 64 or 16, both powers of two, so the divide is a shift.
template <class CFG, int C, class T>
void hbtxr_pool(const T *rows, typename CFG::act_t *out, int tokens) {
#pragma HLS INLINE off
  typedef ap_int<T::width + clog2(CFG::N) + 1> sum_t;
  int sh = 0;
  while ((1 << sh) < tokens) ++sh;      // tokens is a power of two; this is the divide
  const int half = tokens >> 1;
pool_ch:
  for (int c = 0; c < C; ++c) {
#pragma HLS pipeline II = 1
    sum_t total = 0;
    for (int t = 0; t < tokens; ++t) total += rows[t * C + c];
    const sum_t m = total >= 0 ? sum_t((total + half) >> sh) : sum_t(-((-total + half) >> sh));
    out[c] = m < qrange<typename CFG::act_t>::lo   ? typename CFG::act_t(qrange<typename CFG::act_t>::lo)
             : m > qrange<typename CFG::act_t>::hi ? typename CFG::act_t(qrange<typename CFG::act_t>::hi)
                                                   : typename CFG::act_t(m);
  }
}

/// `IN` is 192 (Pupil Box) or 197 (Pupil Ellipse). `OUT_DIM` is 5 for both.
template <class CFG, int IN, int OUT_DIM>
struct HbtxrHead {
  typedef typename CFG::act_t act_t;
  typedef typename CFG::w_t w_t;
  typedef typename CFG::acc_t acc_t;
  typedef typename CFG::nl_t nl_t;

  static_assert(acc_t::width >= mac_width(w_t::width, act_t::width, IN),
                "acc_t cannot hold a reduction over IN without wrapping");

  w_t fc1_w[IN][IN];
  acc_t fc1_b[IN];
  ap_uint<CFG::REQ_M_BITS> fc1_m[IN];
  ap_uint<5> fc1_s[IN];
  nl_t gelu_table[32];
  int gelu_b = 0, gelu_s = 0, gelu_bound = 31;
  ap_uint<CFG::REQ_M_BITS> e_gelu_m = 2;
  ap_uint<5> e_gelu_s = 1;
  w_t fc2_w[OUT_DIM][IN];
  acc_t fc2_b[OUT_DIM];

  /// `feat` is already on fc1's input grid, anchor included. Returns fc2's accumulator
  /// UN-requantized, exactly as `replay_model_int` does: the consumer is the host, there
  /// is no calibrated output grid, and inventing one would round the answer for nothing.
  /// `hbtxr_head_out` turns it into fixed point for the AXIS port.
  void run(const act_t *feat, acc_t *acc_out) {
#pragma HLS INLINE off
    act_t hidden[IN];
  fc1:
    for (int o = 0; o < IN; ++o) {
#pragma HLS pipeline II = 1
      acc_t a = fc1_b[o];
      for (int i = 0; i < IN; ++i) a += feat[i] * fc1_w[o][i];
      const nl_t g = gelu<CFG, 32>(requant<CFG, act_t>(a, fc1_m[o], fc1_s[o]), gelu_table,
                                   gelu_b, gelu_s, gelu_bound);
      hidden[o] = requant<CFG, act_t>(acc_t(g), e_gelu_m, e_gelu_s);
    }
  fc2:
    for (int o = 0; o < OUT_DIM; ++o) {
#pragma HLS pipeline II = 1
      acc_t a = fc2_b[o];
      for (int i = 0; i < IN; ++i) a += hidden[i] * fc2_w[o][i];
      acc_out[o] = a;
    }
  }
};

}  // namespace hbtxr

#endif  // HBTXR_HEAD_HPP
