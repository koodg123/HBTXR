// Patch-embedding stem — Conv-F (frame) and Conv-E (event), SPEC §5-3.
//
//   line buffer (K rows) -> windower -> PE array -> token stream
//
// THE PE ARRAY IS AN RMU. kernel == stride == patch, so the windows do not overlap and
// im2col is pure addressing: patch `t` reads a disjoint `Cin x K x K` tile, which makes
// the convolution a `[tokens, Cin*K*K] x [D, Cin*K*K]^T` matmul. SPEC §5-3's "Do PEs,
// each with K*K*Ui multipliers and one accumulator" is that matmul's PE array, so the
// stem reuses HbtxrRmu rather than restating it. Only the windower is stem-specific.
//
// Two things are NOT shared with the cores:
//
//   - the accumulator. 8-bit operands over Cin*K*K taps need 25 bits, not the cores' 20.
//     HbtxrCfgPatch carries that, and the width comes from the formula rather than from a
//     measurement that happens to fit (SPEC §3).
//   - the input grid is ASYMMETRIC. A real zero is the zero-point, not 0, so every window
//     carries a uniform `- zp * sum(w)` offset. It is per out-channel and constant, so it
//     folds into the bias the RMU already adds on the accumulator: one resident number,
//     no runtime subtract.
#ifndef HBTXR_PATCH_EMBED_HPP
#define HBTXR_PATCH_EMBED_HPP

#include <ap_int.h>
#include <hls_stream.h>
#include <hls_vector.h>

#include "hbtxr_config.hpp"
#include "hbtxr_rmu.hpp"

namespace hbtxr {

/// `CIN` is the modality: 1 for Conv-F, 2 for Conv-E. They are two units, as the paper
/// has them, differing only in this.
template <class CFG, class CORE_CFG, int CIN, int K, int H_IN, int W_IN, int P>
struct HbtxrPatchEmbed {
  static constexpr int TP = CFG::TP;
  static constexpr int TAPS = CIN * K * K;         // the matmul's reduction length
  static constexpr int D = CFG::D;
  static constexpr int TOK_Y = H_IN / K, TOK_X = W_IN / K;
  static constexpr int TOKENS = TOK_Y * TOK_X;
  static constexpr int LANES = TP * P;

  typedef typename CFG::act_t pix_t;               // ap_uint<8>, asymmetric grid
  typedef typename CORE_CFG::act_t tok_t;          // the residual-stream type
  typedef HbtxrRmu<CFG, TAPS, D, P, P, tok_t> pe_t;

  static_assert(TOKENS % TP == 0, "the token grid must tile evenly");
  static_assert(TAPS % P == 0, "Cin*K*K must tile evenly");

  typedef hls::vector<pix_t, LANES> pix_beat_t;
  typedef typename pe_t::in_beat_t win_beat_t;
  typedef typename pe_t::out_beat_t tok_beat_t;

  pe_t pe;

  // The line buffer. K rows of pixel-interleaved input: at K=16 that is 16*128*1 (Conv-F)
  // or 16*64*2 (Conv-E) bytes — one BRAM either way.
  pix_t lines[K][W_IN * CIN];

  /// Input is PIXEL-INTERLEAVED `[y][x][c]`, not the golden's `[c][y][x]`.
  ///
  /// That is a DMA-order choice and it is the one that makes a line buffer possible: with
  /// channel-major input, all of channel 0 arrives before channel 1, so a K-row window
  /// would have to buffer the whole image once Cin > 1. The testbench transposes, which
  /// is what the DMA descriptor does on the board.
  void run(hls::stream<pix_beat_t> &in, hls::stream<tok_beat_t> &out) {
#pragma HLS INLINE off
    hls::stream<win_beat_t> windows("windows");

  token_row:
    for (int ty = 0; ty < TOK_Y; ++ty) {
      // --- line buffer: K rows before the first patch row can be formed ---
    fill_lines:
      for (int r = 0; r < K; ++r)
        for (int i = 0; i < W_IN * CIN; i += LANES) {
#pragma HLS pipeline II = 1
          const pix_beat_t v = in.read();
          for (int l = 0; l < LANES; ++l)
#pragma HLS unroll
            lines[r][i + l] = v[l];
        }

      // --- windower: emit each patch as TAPS values in (c, ky, kx) order, which is
      //     how the weight tensor [Cout][Cin][kh][kw] flattens. Getting this order
      //     wrong is silent -- the shapes still match.
    window:
      for (int tx0 = 0; tx0 < TOK_X; tx0 += TP)
        for (int i0 = 0; i0 < TAPS; i0 += P) {
#pragma HLS pipeline II = 1
          win_beat_t w;
          for (int p = 0; p < TP; ++p)
#pragma HLS unroll
            for (int e = 0; e < P; ++e) {
#pragma HLS unroll
              const int i = i0 + e;
              const int c = i / (K * K), ky = (i / K) % K, kx = i % K;
              w[p * P + e] = lines[ky][((tx0 + p) * K + kx) * CIN + c];
            }
          windows.write(w);
        }

      // The PE array, one token row at a time.
      pe.compute(windows, out, TOK_X);
    }
  }
};

}  // namespace hbtxr

#endif  // HBTXR_PATCH_EMBED_HPP
