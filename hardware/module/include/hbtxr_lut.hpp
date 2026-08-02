// Power-of-two LUT indexing, and the one pointwise LUT op (GeLU).
//
// `cursor = clamp((x + b) >> s, 0, bound)` is the same kernel every nonlinear operator
// here is built on — GeLU pointwise, LayerNorm over the variance, softmax over the
// exponent argument and again over the accumulator.
#ifndef HBTXR_LUT_HPP
#define HBTXR_LUT_HPP

#include <ap_int.h>

#include "hbtxr_config.hpp"
#include "hbtxr_requant.hpp"  // qrange

namespace hbtxr {

/// Clamp a PRE-CLAMP cursor into `[0, bound]`.
///
/// `CUR` must be signed and must hold the cursor's reachable range BEFORE the clamp.
/// Both halves of that are load-bearing and neither is diagnosed by Vitis:
///
/// - an unsigned cursor makes `clamp(., 0, bound)` a no-op on the low side, so a negative
///   index wraps to a huge positive one and reads a valid-looking entry;
/// - a too-narrow cursor aliases an out-of-range index to an in-range one instead of
///   saturating, so the clamp never sees the value it was there to catch.
///
/// `i_ops.table_quantize` is arbitrary-precision, so either one is a bit mismatch rather
/// than a small error.
template <class CUR>
inline int lut_index(CUR raw, int bound) {
#pragma HLS INLINE
  static_assert(qrange<CUR>::lo < 0, "the LUT cursor type must be SIGNED");
  if (raw < 0) return 0;
  if (raw > bound) return bound;
  return (int)raw;
}

/// GeLU — `table[clamp((x + b) >> s, 0, bound)]`, i.e. `i_ops.table_quantize`.
///
/// The table emits `nl_t` (16-bit), not the matmul width. Narrowing to the next operand's
/// grid is the requant on the edge that follows, not the table: tying the table to the
/// 4-bit activation width collapses GeLU to two distinct outputs.
template <class CFG, int ENTRIES, class X>
inline typename CFG::nl_t gelu(X x, const typename CFG::nl_t table[ENTRIES], int b, int s,
                               int bound) {
#pragma HLS INLINE
  // Derived from THIS op's own input: the offset can be as wide as the input, and the
  // shift only ever narrows, so one extra bit covers the add.
  typedef ap_int<X::width + 18> cur_t;
  const cur_t raw = (cur_t(x) + cur_t(b)) >> s;
  return table[lut_index(raw, bound)];
}

}  // namespace hbtxr

#endif  // HBTXR_LUT_HPP
