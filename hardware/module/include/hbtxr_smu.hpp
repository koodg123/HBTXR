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
/// `TRANSPOSE_B` picks which of the two attention matmuls this is, at COMPILE time — a
/// runtime branch here would violate the dataflow canonical form (SPEC §3-A):
///
///   Q x K^T   TRANSPOSE_B = true   B arrives as [cols][K], reduction over the head dim
///   S x V     TRANSPOSE_B = false  B arrives as [K][cols], reduction over tokens
///
/// Only the LOAD differs. B lands in `bt[col][k]` either way, so the MAC array is one
/// piece of hardware serving both.
///
/// Beat layout:
///   a     `v[p * CIP + c]` = row `t0 + p`, reduction channel `k0 + c`
///   b     transposed: row `t0 + p`, channel `k0 + c`; plain: reduction row `k0 + p`,
///         column `j0 + c`
///   out   `v[p * COP + c]` = row `t0 + p`, column `j0 + c`
template <class CFG, int K, int ROWS_MAX, int CIP, int COP, bool TRANSPOSE_B = true>
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
           hls::stream<out_beat_t> &out, int rows, int cols, int kdim,
           ap_uint<CFG::REQ_M_BITS> mult, ap_uint<5> shift) {
#pragma HLS INLINE off
    // --- B first, in full. This is the token-global dependency. ---
  load_b:
    if (TRANSPOSE_B) {
      for (int t0 = 0; t0 < cols; t0 += TP)
        for (int k0 = 0; k0 < kdim; k0 += CIP) {
#pragma HLS pipeline II = 1
          const in_beat_t v = b.read();
          for (int p = 0; p < TP; ++p)
#pragma HLS unroll
            for (int c = 0; c < CIP; ++c)
#pragma HLS unroll
              bt[t0 + p][k0 + c] = v[p * CIP + c];
        }
    } else {
      // B is [K][cols]; the write transposes so the MAC array still sees bt[col][k].
      for (int k0 = 0; k0 < kdim; k0 += TP)
        for (int j0 = 0; j0 < cols; j0 += CIP) {
#pragma HLS pipeline II = 1
          const in_beat_t v = b.read();
          for (int p = 0; p < TP; ++p)
#pragma HLS unroll
            for (int c = 0; c < CIP; ++c)
#pragma HLS unroll
              bt[j0 + c][k0 + p] = v[p * CIP + c];
        }
    }

  row_tile:
    for (int t0 = 0; t0 < rows; t0 += TP) {
      act_t at[TP][K];
#pragma HLS array_reshape variable = at cyclic factor = CIP dim = 2
    fill_a:
      for (int k0 = 0; k0 < kdim; k0 += CIP) {
#pragma HLS pipeline II = 1
        const in_beat_t v = a.read();
        for (int p = 0; p < TP; ++p)
#pragma HLS unroll
          for (int c = 0; c < CIP; ++c)
#pragma HLS unroll
            at[p][k0 + c] = v[p * CIP + c];
      }

    col_tile:
      for (int j0 = 0; j0 < cols; j0 += COP) {
        acc_t acc[TP][COP];
#pragma HLS array_partition variable = acc complete dim = 0
        for (int p = 0; p < TP; ++p)
#pragma HLS unroll
          for (int c = 0; c < COP; ++c)
#pragma HLS unroll
            acc[p][c] = 0;  // no bias: a score matrix has none

      reduce:
        // `kdim`, NOT `K`. K is the compile-time maximum; the reduction length is a
        // runtime argument because S x V reduces over TOKENS. Running to K reads whatever
        // the buffer held from the PREVIOUS call, and that is invisible whenever
        // kdim == K — so Q x K^T (kdim = HD = K) is always right, and only a 16-token run
        // after a 64-token one sees it.
        for (int k0 = 0; k0 < kdim; k0 += CIP) {
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
