// Integer LayerNorm — `i_ops.py:layernorm_quantize`, the 7-scalar kernel.
//
// One rsqrt segment (SPEC §3: RSQRT is 64 entries). The 10-scalar segmented kernel exists
// in i_ops for heteroscedastic real activations and is not what this payload uses.
#ifndef HBTXR_LAYERNORM_HPP
#define HBTXR_LAYERNORM_HPP

#include <ap_int.h>
#include <hls_stream.h>
#include <hls_vector.h>

#include "hbtxr_config.hpp"
#include "hbtxr_lut.hpp"
#include "hbtxr_requant.hpp"

namespace hbtxr {

/// Per row of C channels:
///   mean    = (sum * c_1_m + (1 << (c_1_s-1))) >> c_1_s
///   var_sum = sum((x - mean)^2)
///   rsqrt   = table[clamp((var_sum + b) >> s1, 0, bound)]
///   out[i]  = clamp(((x[i] - mean) * rsqrt * lnw[i] + lnb[i]) >> s2)
///
/// Beat layout `v[p * P + c]` = token `t0 + p`, channel `c0 + c` — the same convention the
/// RMU and SMU use.
/// `OUT` defaults to the activation type. The shared FINAL norm is the exception: its
/// input is the 4-bit residual stream and its `clamp_bits` is the 8-bit head grid, because
/// its consumer is the head (SPEC §3, mixed precision).
template <class CFG, int C, int P, class OUT = typename CFG::act_t>
struct HbtxrLayerNorm {
  static constexpr int TP = CFG::TP;
  typedef typename CFG::act_t act_t;
  typedef typename CFG::nl_t nl_t;

  static_assert(C % P == 0, "C must tile evenly");
  static_assert(is_pow2(P) && is_pow2(TP), "factors must be powers of two");

  // --- widths, DERIVED from this op's own operands ---------------------------
  // The reference transcribed a calibration value here and its var was 27 bits where the
  // shape needed 36. These come from the shape instead.
  //
  // EVERY multiply below is written on its NARROW operands. ap_int's operator* returns a
  // result as wide as the sum of its operands, so `wide_t(a) * wide_t(b)` asks for a
  // multiply as big as the sum of the WIDE types — the same trap -Wall caught in requant.
  typedef ap_int<20> c1m_t;                             // the integer-mean multiplier
  typedef ap_int<act_t::width + clog2(C) + 1> sum_t;    // sum over C of act_t
  typedef ap_int<sum_t::width + c1m_t::width + 1> mean_tmp_t;
  typedef ap_int<act_t::width + 1> diff_t;              // x - mean
  typedef ap_int<2 * diff_t::width> sq_t;               // d * d
  typedef ap_int<sq_t::width + clog2(C) + 1> var_t;
  typedef ap_int<diff_t::width + nl_t::width> dr_t;             // d * rsqrt
  typedef ap_int<dr_t::width + nl_t::width> drw_t;              // (d * rsqrt) * lnw
  typedef ap_int<drw_t::width + 2> affine_t;                    // + lnb, on THIS grid

  typedef hls::vector<act_t, TP * P> in_beat_t;
  typedef hls::vector<OUT, TP * P> beat_t;   // output; `beat_t` keeps the caller's name

  // Payload. lnb is affine_t, not nl_t: measured at ~30 signed bits on the golden.
  nl_t lnw[C];
  affine_t lnb[C];
  nl_t rsqrt[64];
  // Initialised: an un-loaded unit must have defined state, not whatever ap_int's
  // default constructor left behind. g++ diagnoses the static-storage case.
  c1m_t c_1_m = 0;
  int c_1_s = 1, b = 0, s1 = 0, bound = 0, s2 = 0, clamp_bits = 0;

  void run(hls::stream<in_beat_t> &in, hls::stream<beat_t> &out, int rows) {
#pragma HLS INLINE off
  row_tile:
    for (int t0 = 0; t0 < rows; t0 += TP) {
      act_t x[TP][C];
#pragma HLS array_reshape variable = x cyclic factor = P dim = 2
      sum_t sum[TP];
#pragma HLS array_partition variable = sum complete
      for (int p = 0; p < TP; ++p)
#pragma HLS unroll
        sum[p] = 0;

    // Pass 1 — buffer the row and accumulate its sum.
    fill:
      for (int c0 = 0; c0 < C; c0 += P) {
#pragma HLS pipeline II = 1
        const in_beat_t v = in.read();
        for (int p = 0; p < TP; ++p)
#pragma HLS unroll
          for (int c = 0; c < P; ++c)
#pragma HLS unroll
          {
            x[p][c0 + c] = v[p * P + c];
            sum[p] += v[p * P + c];
          }
      }

      diff_t mean[TP];
      nl_t rq[TP];
#pragma HLS array_partition variable = mean complete
#pragma HLS array_partition variable = rq complete
      for (int p = 0; p < TP; ++p) {
#pragma HLS unroll
        // Narrow multiply: sum_t x c1m_t, not mean_tmp_t x mean_tmp_t.
        // The rounding term is part of the definition, not an optimisation.
        const ap_int<sum_t::width + c1m_t::width> prod = sum[p] * c_1_m;
        const mean_tmp_t t = mean_tmp_t(prod) + (mean_tmp_t(1) << (c_1_s - 1));
        mean[p] = diff_t(t >> c_1_s);
      }

    // Pass 2 — variance. Needs the mean, so it cannot fold into pass 1.
    variance:
      for (int p = 0; p < TP; ++p) {
#pragma HLS unroll
        var_t vs = 0;
        for (int i = 0; i < C; ++i) {
#pragma HLS pipeline II = 1
          const diff_t d = diff_t(x[p][i]) - mean[p];
          const sq_t sq = d * d;          // diff_t x diff_t, not var_t x var_t
          vs += sq;
        }
        // The cursor holds the PRE-clamp value and is signed: b is negative here, so a
        // low-variance row genuinely produces a negative index that the clamp must catch.
        const ap_int<var_t::width + 2> raw = (ap_int<var_t::width + 2>(vs) + b) >> s1;
        rq[p] = rsqrt[lut_index(raw, bound)];
      }

    // Pass 3 — affine, and out.
    affine:
      for (int c0 = 0; c0 < C; c0 += P) {
#pragma HLS pipeline II = 1
        beat_t y;
        for (int p = 0; p < TP; ++p)
#pragma HLS unroll
          for (int c = 0; c < P; ++c) {
#pragma HLS unroll
            const diff_t d = diff_t(x[p][c0 + c]) - mean[p];
            const dr_t dr = d * rq[p];               // 6 x 16
            const drw_t drw = dr * lnw[c0 + c];      // 22 x 16 -- never affine_t x affine_t
            const affine_t sh = (affine_t(drw) + lnb[c0 + c]) >> s2;
            y[p * P + c] = sh < qrange<OUT>::lo   ? OUT(qrange<OUT>::lo)
                           : sh > qrange<OUT>::hi ? OUT(qrange<OUT>::hi)
                                                  : OUT(sh);
          }
        out.write(y);
      }
    }
  }
};

}  // namespace hbtxr

#endif  // HBTXR_LAYERNORM_HPP
