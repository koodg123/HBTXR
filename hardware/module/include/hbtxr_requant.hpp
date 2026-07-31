// Dyadic requantization — the one primitive every matmul unit ends with.
#ifndef HBTXR_REQUANT_HPP
#define HBTXR_REQUANT_HPP

#include <ap_int.h>

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

}  // namespace hbtxr

#endif  // HBTXR_REQUANT_HPP
