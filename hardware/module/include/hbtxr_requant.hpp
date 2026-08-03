// Dyadic requantization — the one primitive every matmul unit ends with.
#ifndef HBTXR_REQUANT_HPP
#define HBTXR_REQUANT_HPP

#include <type_traits>

#include <ap_int.h>
#include <hls_stream.h>

#include "hbtxr_config.hpp"

namespace hbtxr {

/// The clamp range of a quantized type. Two specializations, because the softmax path
/// clamps unsigned and everything else clamps signed.
template <class T> struct qrange;
template <int W> struct qrange<ap_int<W>> {
  static constexpr long long lo = -(1LL << (W - 1));
  static constexpr long long hi = (1LL << (W - 1)) - 1;
};
template <int W> struct qrange<ap_uint<W>> {
  static constexpr long long lo = 0;
  static constexpr long long hi = (1LL << W) - 1;
};

/// `clamp((acc * M + (1 << (n-1))) >> n)` into OUT.
///
/// This is `algorithm/quantization/i_ops.py:requant` bit for bit, and the hardware
/// follows THAT definition rather than the other way round (SPEC §3). Verified with
/// element-wise `==`, never a tolerance.
///
/// `>>` is arithmetic on a signed ap_int, which is the floor Python's `>>` also gives on
/// a negative value. Sign-extending by hand instead is where the old implementation lost
/// a bit.
template <class CFG, class OUT, class ACC>
OUT requant(ACC acc, ap_uint<CFG::REQ_M_BITS> mult, ap_uint<5> shift) {
#pragma HLS INLINE
  // Widths come from the TYPES, not from the data: acc_t is 20 bits wide even when the
  // values living in it need 9.
  //
  // The operands are multiplied AT THEIR OWN WIDTHS. Widening them first looks harmless
  // and is not: ap_int's operator* returns a result as wide as the sum of its operands,
  // so casting both to the 54-bit sum type first asks for a 54x54 -> 108-bit multiply
  // instead of the 20x33 -> 53 the datapath actually needs.
  typedef ap_int<ACC::width + CFG::REQ_M_BITS> prod_t;      // 53
  typedef ap_int<ACC::width + CFG::REQ_M_BITS + 1> wide_t;  // +1: the rounding term can carry
  static_assert(sizeof(decltype(acc * mult)) <= sizeof(prod_t),
                "the multiply widened past acc_w + REQ_M_BITS");

  const prod_t prod = acc * mult;
  const wide_t rnd = shift > 0 ? (wide_t(1) << (shift - 1)) : wide_t(0);
  const wide_t shifted = (wide_t(prod) + rnd) >> shift;

  if (shifted < qrange<OUT>::lo) return OUT(qrange<OUT>::lo);
  if (shifted > qrange<OUT>::hi) return OUT(qrange<OUT>::hi);
  return OUT(shifted);
}

/// An edge between two stages: one per-tensor requant, lane by lane.
///
/// Most of the block's 18 requants are this — the per-channel ones belong to the matmul
/// that produced the accumulator, and live inside its unit.
template <class CFG, class IN_BEAT, class OUT_BEAT, class ACC>
void requant_stream(hls::stream<IN_BEAT> &in, hls::stream<OUT_BEAT> &out, int beats,
                    int lanes, ap_uint<CFG::REQ_M_BITS> mult, ap_uint<5> shift) {
#pragma HLS INLINE off
  // hls::vector::operator[] returns a REFERENCE, so decltype alone gives `T&` and every
  // trait lookup fails on it. decay is what makes the lane's value type.
  typedef std::decay_t<decltype(std::declval<OUT_BEAT &>()[0])> out_lane_t;
edge:
  for (int i = 0; i < beats; ++i) {
#pragma HLS pipeline II = 1
    const IN_BEAT v = in.read();
    OUT_BEAT y;
    for (int l = 0; l < lanes; ++l)
#pragma HLS unroll
      y[l] = requant<CFG, out_lane_t>(ACC(v[l]), mult, shift);
    out.write(y);
  }
}

/// The residual join: align both sides onto the output grid, add, clamp.
///
/// `i_block._add` — and the clamp after the add is part of it, not an afterthought.
template <class CFG, class BEAT, class ACC>
void residual_merge(hls::stream<BEAT> &a, hls::stream<BEAT> &b, hls::stream<BEAT> &out,
                    int beats, int lanes, ap_uint<CFG::REQ_M_BITS> ma, ap_uint<5> sa,
                    ap_uint<CFG::REQ_M_BITS> mb, ap_uint<5> sb) {
#pragma HLS INLINE off
  typedef std::decay_t<decltype(std::declval<BEAT &>()[0])> lane_t;
merge:
  for (int i = 0; i < beats; ++i) {
#pragma HLS pipeline II = 1
    const BEAT va = a.read(), vb = b.read();
    BEAT y;
    for (int l = 0; l < lanes; ++l) {
#pragma HLS unroll
      const lane_t x = requant<CFG, lane_t>(ACC(va[l]), ma, sa);
      const lane_t z = requant<CFG, lane_t>(ACC(vb[l]), mb, sb);
      const ap_int<lane_t::width + 1> s = ap_int<lane_t::width + 1>(x) + z;
      y[l] = s < qrange<lane_t>::lo   ? lane_t(qrange<lane_t>::lo)
             : s > qrange<lane_t>::hi ? lane_t(qrange<lane_t>::hi)
                                      : lane_t(s);
    }
    out.write(y);
  }
}

}  // namespace hbtxr

#endif  // HBTXR_REQUANT_HPP
