// SMU — Stream Matmul Unit. BOTH operands stream; nothing is resident.
//
// That is the defining difference from the RMU, not an implementation detail (SPEC §5-2).
// K changes with every token set, so it cannot be resident, and `Q x K^T` cannot emit row
// 0 of the score matrix until all of K has arrived. The buffer below is that constraint
// made explicit — it is also why the MHA core needs a full-tensor residual FIFO where the
// MLP core needs only pipeline depth.
#ifndef HBTXR_SMU_HPP
#define HBTXR_SMU_HPP

#include <ap_int.h>
#include <hls_stream.h>
#include <hls_vector.h>

#include "hbtxr_config.hpp"
#include "hbtxr_requant.hpp"

namespace hbtxr {

/// `r[i][j] = requant( sum_k a[i][k] * b[j][k] )` — B is transposed BY THE UNIT.
///
/// One `(M, n)` for the whole tensor, and `1/sqrt(d)` is already folded into it. That
/// folding happens in `algorithm/quantization`, and it is folded rather than applied as a
/// separate multiply because in hardware it is not a separate multiply (SPEC §5-1).
///
/// Beat layout:
///   a, b  `v[p * CIP + c]` = row `t0 + p`, reduction channel `k0 + c`
///   out   `v[p * COP + c]` = row `t0 + p`, column `j0 + c`
template <class CFG, int K, int ROWS_MAX, int CIP, int COP>
struct HbtxrSmu {
  static constexpr int TP = CFG::TP;
  typedef typename CFG::act_t act_t;
  typedef typename CFG::acc_t acc_t;

  // act x act, not weight x act: both operands are activations here.
  static_assert(acc_t::width >= mac_width(act_t::width, act_t::width, K),
                "acc_t cannot hold a reduction over K without wrapping");
  static_assert(K % CIP == 0 && ROWS_MAX % COP == 0, "K/ROWS must tile evenly");
  static_assert(is_pow2(CIP) && is_pow2(COP) && is_pow2(TP), "factors must be powers of two");

  typedef hls::vector<act_t, TP * CIP> in_beat_t;
  typedef hls::vector<act_t, TP * COP> out_beat_t;

  // The only buffer. ROWS_MAX x K of 4-bit = 2 KB at N=64, K=64 — one BRAM18.
  act_t bt[ROWS_MAX][K];
#pragma HLS array_reshape variable = bt cyclic factor = CIP dim = 2

  void run(hls::stream<in_beat_t> &a, hls::stream<in_beat_t> &b,
           hls::stream<out_beat_t> &out, int rows, ap_uint<CFG::REQ_M_BITS> mult,
           ap_uint<5> shift) {
#pragma HLS INLINE off
    // --- B first, in full. This is the token-global dependency. ---
  load_b:
    for (int t0 = 0; t0 < rows; t0 += TP)
      for (int k0 = 0; k0 < K; k0 += CIP) {
#pragma HLS pipeline II = 1
        const in_beat_t v = b.read();
        for (int p = 0; p < TP; ++p)
#pragma HLS unroll
          for (int c = 0; c < CIP; ++c)
#pragma HLS unroll
            bt[t0 + p][k0 + c] = v[p * CIP + c];
      }

  row_tile:
    for (int t0 = 0; t0 < rows; t0 += TP) {
      act_t at[TP][K];
#pragma HLS array_reshape variable = at cyclic factor = CIP dim = 2
    fill_a:
      for (int k0 = 0; k0 < K; k0 += CIP) {
#pragma HLS pipeline II = 1
        const in_beat_t v = a.read();
        for (int p = 0; p < TP; ++p)
#pragma HLS unroll
          for (int c = 0; c < CIP; ++c)
#pragma HLS unroll
            at[p][k0 + c] = v[p * CIP + c];
      }

    col_tile:
      for (int j0 = 0; j0 < rows; j0 += COP) {
        acc_t acc[TP][COP];
#pragma HLS array_partition variable = acc complete dim = 0
        for (int p = 0; p < TP; ++p)
#pragma HLS unroll
          for (int c = 0; c < COP; ++c)
#pragma HLS unroll
            acc[p][c] = 0;  // no bias: a score matrix has none

      reduce:
        for (int k0 = 0; k0 < K; k0 += CIP) {
#pragma HLS pipeline II = 1
          for (int p = 0; p < TP; ++p)
#pragma HLS unroll
            for (int c = 0; c < COP; ++c)
#pragma HLS unroll
              for (int k = 0; k < CIP; ++k)
#pragma HLS unroll
                acc[p][c] += at[p][k0 + k] * bt[j0 + c][k0 + k];
        }

        out_beat_t r;
        for (int p = 0; p < TP; ++p)
#pragma HLS unroll
          for (int c = 0; c < COP; ++c)
#pragma HLS unroll
            r[p * COP + c] = requant<CFG, act_t>(acc[p][c], mult, shift);
        out.write(r);
      }
    }
  }
};

}  // namespace hbtxr

#endif  // HBTXR_SMU_HPP
