// Integer softmax — `i_ops.py:softmax_quantize`, 14 scalars, two reciprocal segments.
#ifndef HBTXR_SOFTMAX_HPP
#define HBTXR_SOFTMAX_HPP

#include <ap_int.h>
#include <hls_stream.h>
#include <hls_vector.h>

#include "hbtxr_config.hpp"
#include "hbtxr_lut.hpp"

namespace hbtxr {

/// Per row of `cols` scores:
///   e[j]   = exp[clamp((max - x[j] + b1) >> s1, 0, bound1)]
///   acc    = sum(e)
///   recip  = segment 1 or 2 of the reciprocal, chosen by the UNCLAMPED segment-1 cursor
///   out[j] = clamp((e[j] * recip + b3) >> s3)      unsigned
///
/// Beat layout `v[p * P + c]` = row `r0 + p`, column `j0 + c`.
template <class CFG, int COLS_MAX, int P>
struct HbtxrSoftmax {
  static constexpr int TP = CFG::TP;
  typedef typename CFG::act_t act_t;
  typedef typename CFG::nlu_t nlu_t;   // exp and reciprocal are UNSIGNED: exp's largest
  typedef typename CFG::prob_t prob_t; // entry is 1<<15 = 32768, which int16 cannot hold

  static_assert(COLS_MAX % P == 0, "COLS must tile evenly");
  static_assert(is_pow2(P) && is_pow2(TP), "factors must be powers of two");

  // --- widths, derived ------------------------------------------------------
  typedef ap_int<act_t::width + 1> delta_t;                      // max - x, non-negative
  typedef ap_uint<nlu_t::width + clog2(COLS_MAX) + 1> acc_t;     // sum of COLS exp entries
  typedef ap_uint<2 * nlu_t::width> er_t;                        // e * recip, both unsigned
  typedef ap_int<er_t::width + 4> rel_t;                         // + b3, then >> s3

  typedef hls::vector<act_t, TP * P> in_beat_t;
  typedef hls::vector<prob_t, TP * P> out_beat_t;

  nlu_t exp_table[32];
  nlu_t recip_one[64];
  nlu_t recip_two[64];
  int b1 = 0, s1 = 0, bound1 = 0;
  int b2_one = 0, s2_one = 0, bound2_one = 0, b3_one = 0, s3_one = 1;
  int b2_two = 0, s2_two = 0, bound2_two = 0, b3_two = 0, s3_two = 1;
  int clamp_bits = 0;

  void run(hls::stream<in_beat_t> &in, hls::stream<out_beat_t> &out, int rows, int cols) {
#pragma HLS INLINE off
  row_tile:
    for (int r0 = 0; r0 < rows; r0 += TP) {
      act_t x[TP][COLS_MAX];
#pragma HLS array_reshape variable = x cyclic factor = P dim = 2
      act_t mx[TP];
#pragma HLS array_partition variable = mx complete
      for (int p = 0; p < TP; ++p)
#pragma HLS unroll
        mx[p] = qrange<act_t>::lo;

    // Pass 1 — buffer and find the row max.
    fill:
      for (int j0 = 0; j0 < cols; j0 += P) {
#pragma HLS pipeline II = 1
        const in_beat_t v = in.read();
        for (int p = 0; p < TP; ++p)
#pragma HLS unroll
          for (int c = 0; c < P; ++c) {
#pragma HLS unroll
            x[p][j0 + c] = v[p * P + c];
            if (v[p * P + c] > mx[p]) mx[p] = v[p * P + c];
          }
      }

      // Pass 2 — exponent and its sum.
      acc_t acc[TP];
      nlu_t recip[TP];
      int b3[TP], s3[TP];
#pragma HLS array_partition variable = acc complete
#pragma HLS array_partition variable = recip complete
    exponent:
      for (int p = 0; p < TP; ++p) {
#pragma HLS unroll
        acc_t a = 0;
        for (int j = 0; j < cols; ++j) {
#pragma HLS pipeline II = 1
          const ap_int<delta_t::width + 18> raw =
              (ap_int<delta_t::width + 18>(delta_t(mx[p]) - delta_t(x[p][j])) + b1) >> s1;
          a += exp_table[lut_index(raw, bound1)];
        }
        acc[p] = a;

        // The segment test reads the UNCLAMPED segment-1 cursor. `cursor_one > bound2_one`
        // is exactly "this row overflowed segment one", and segment two is calibrated to
        // start at that same point, so no row lands on a clamped low entry of table two.
        typedef ap_int<acc_t::width + 2> cur_t;
        const cur_t c1 = (cur_t(acc[p]) + b2_one) >> s2_one;
        if (c1 > bound2_one) {
          const cur_t c2 = (cur_t(acc[p]) + b2_two) >> s2_two;
          recip[p] = recip_two[lut_index(c2, bound2_two)];
          b3[p] = b3_two;
          s3[p] = s3_two;
        } else {
          recip[p] = recip_one[lut_index(c1, bound2_one)];
          b3[p] = b3_one;
          s3[p] = s3_one;
        }
      }

    // Pass 3 — normalise, and out.
    normalise:
      for (int j0 = 0; j0 < cols; j0 += P) {
#pragma HLS pipeline II = 1
        out_beat_t y;
        for (int p = 0; p < TP; ++p)
#pragma HLS unroll
          for (int c = 0; c < P; ++c) {
#pragma HLS unroll
            const ap_int<delta_t::width + 18> raw =
                (ap_int<delta_t::width + 18>(delta_t(mx[p]) - delta_t(x[p][j0 + c])) + b1) >> s1;
            const nlu_t e = exp_table[lut_index(raw, bound1)];
            const er_t er = e * recip[p];      // nlu_t x nlu_t, not rel_t x rel_t
            const rel_t rel = (rel_t(er) + b3[p]) >> s3[p];
            y[p * P + c] = rel < qrange<prob_t>::lo   ? prob_t(qrange<prob_t>::lo)
                           : rel > qrange<prob_t>::hi ? prob_t(qrange<prob_t>::hi)
                                                      : prob_t(rel);
          }
        out.write(y);
      }
    }
  }
};

}  // namespace hbtxr

#endif  // HBTXR_SOFTMAX_HPP
