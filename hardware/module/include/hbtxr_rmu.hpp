// RMU — Resident Matmul Unit. Weights stay on chip, activations stream through.
//
// The paper's RMU holds its weights resident; ours must be RUNTIME-LOADED, which the
// ViT_Accel reference has no equivalent of (SPEC §4). The reference ROM-initialises
// weights in the RMU constructor, which is exactly why it needs twelve separate ATTN
// instances and has no weight prefetcher at all. HBTXR reuses four cores across eight
// blocks, so the weights have to arrive at runtime.
#ifndef HBTXR_RMU_HPP
#define HBTXR_RMU_HPP

#include <ap_int.h>
#include <hls_stream.h>
#include <hls_vector.h>

#include "hbtxr_config.hpp"
#include "hbtxr_requant.hpp"

namespace hbtxr {

/// `y[t][o] = requant_o( sum_i x[t][i] * W[o][i] + bias[o] )`.
///
/// Stream beat layout, which the testbench and every producer must agree on:
///   in   `v[p * CIP + c]` = token `t0 + p`, input channel  `i0 + c`
///   out  `v[p * COP + c]` = token `t0 + p`, output channel `o0 + c`
///
/// CIP/COP/TP are named template parameters, never positional — the reference took 126
/// parameters of which 76 were plain `int`, so a swapped slot compiled and c-simulated
/// clean and only surfaced in cosim.
template <class CFG, int CI, int CO, int CIP, int COP>
struct HbtxrRmu {
  static constexpr int TP = CFG::TP;
  typedef typename CFG::act_t act_t;
  typedef typename CFG::w_t w_t;
  typedef typename CFG::acc_t acc_t;

  // Derived here, where CI is known, rather than transcribed.
  static_assert(acc_t::width >= mac_width(w_t::width, act_t::width, CI),
                "acc_t cannot hold a reduction over CI without wrapping");
  static_assert(CI % CIP == 0 && CO % COP == 0, "CI/CO must tile evenly");
  static_assert(is_pow2(CIP) && is_pow2(COP) && is_pow2(TP), "factors must be powers of two");

  typedef hls::vector<act_t, TP * CIP> in_beat_t;
  typedef hls::vector<act_t, TP * COP> out_beat_t;

  // Resident state. The reshape factor MUST equal the unroll factor and the stream lane
  // count — three numbers, one value (SPEC §2-A). It is `array_reshape`, not
  // `array_partition`: the unrolled tile folds into one wide word rather than CIP banks.
  w_t weight[CO][CI];
#pragma HLS array_reshape variable = weight cyclic factor = CIP dim = 2
  acc_t bias[CO];
  ap_uint<CFG::REQ_M_BITS> mult[CO];
  ap_uint<5> shift[CO];

  /// Load the resident weights. Separate from compute because that is the whole point of
  /// the third RMU mode: the same hardware serves a different block by reloading.
  void load(const w_t w[CO][CI], const acc_t b[CO],
            const ap_uint<CFG::REQ_M_BITS> m[CO], const ap_uint<5> s[CO]) {
  load_out:
    for (int o = 0; o < CO; ++o) {
    load_in:
      for (int i = 0; i < CI; ++i) {
#pragma HLS pipeline II = 1
        weight[o][i] = w[o][i];
      }
      bias[o] = b[o];
      mult[o] = m[o];
      shift[o] = s[o];
    }
  }

  void compute(hls::stream<in_beat_t> &in, hls::stream<out_beat_t> &out, int tokens) {
#pragma HLS INLINE off
  token_tile:
    for (int t0 = 0; t0 < tokens; t0 += TP) {
      // The tile is buffered because every output tile re-reads it and a stream is
      // consumed once. TP x CI of 4-bit = 384 B at D=192.
      act_t x[TP][CI];
#pragma HLS array_reshape variable = x cyclic factor = CIP dim = 2
    fill:
      for (int i0 = 0; i0 < CI; i0 += CIP) {
#pragma HLS pipeline II = 1
        const in_beat_t v = in.read();
        for (int p = 0; p < TP; ++p)
#pragma HLS unroll
          for (int c = 0; c < CIP; ++c)
#pragma HLS unroll
            x[p][i0 + c] = v[p * CIP + c];
      }

    out_tile:
      for (int o0 = 0; o0 < CO; o0 += COP) {
        acc_t acc[TP][COP];
#pragma HLS array_partition variable = acc complete dim = 0
        for (int p = 0; p < TP; ++p)
#pragma HLS unroll
          for (int c = 0; c < COP; ++c)
#pragma HLS unroll
            // The bias is added ON THE ACCUMULATOR, before the requant. That is where it
            // is free in hardware and the only place it is exact -- after the requant it
            // would be rounded onto the coarser output grid.
            acc[p][c] = bias[o0 + c];

      reduce:
        for (int i0 = 0; i0 < CI; i0 += CIP) {
#pragma HLS pipeline II = 1
          // The reduction sits innermost by DESIGN, not by pragma: the partial sums are
          // TP*COP flip-flops, so II=1 falls out and no memory port is needed for them.
          for (int p = 0; p < TP; ++p)
#pragma HLS unroll
            for (int c = 0; c < COP; ++c)
#pragma HLS unroll
              for (int k = 0; k < CIP; ++k)
#pragma HLS unroll
                acc[p][c] += x[p][i0 + k] * weight[o0 + c][i0 + k];
        }

        out_beat_t y;
        for (int p = 0; p < TP; ++p)
#pragma HLS unroll
          for (int c = 0; c < COP; ++c)
#pragma HLS unroll
            y[p * COP + c] = requant<CFG, act_t>(acc[p][c], mult[o0 + c], shift[o0 + c]);
        out.write(y);
      }
    }
  }
};

}  // namespace hbtxr

#endif  // HBTXR_RMU_HPP
