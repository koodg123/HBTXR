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

/// GeLU over a stream. Pointwise, so unlike LayerNorm and softmax it buffers nothing.
///
/// The output beat carries `nl_t`, not the activation width: narrowing to the next
/// operand's grid is the requant on the edge after this, not the table.
template <class CFG, int ENTRIES, class IN_BEAT, class OUT_BEAT>
void gelu_stream(hls::stream<IN_BEAT> &in, hls::stream<OUT_BEAT> &out, int beats, int lanes,
                 const typename CFG::nl_t table[ENTRIES], int b, int s, int bound) {
#pragma HLS INLINE off
  // A LOCAL, fully-replicated copy, because every lane indexes the table independently:
  // `lanes` is TP*P = 32 and `table` as it arrives is a 2-port memory, so reading it
  // directly costs II=16 instead of 1. csynth measured exactly that. ENTRIES is 32 and an
  // entry is 16 bits, so this is 64 bytes of registers, and copying them is `ENTRIES`
  // cycles once per stream against `beats` of 768.
  //
  // Local rather than partitioning the parameter: the caller passes a core's resident
  // member in one place and a top-level port in another, and partitioning a PORT would
  // split it into 32 ports at whatever level it happens to surface.
  typename CFG::nl_t tab[ENTRIES];
#pragma HLS array_partition variable = tab complete dim = 1
gelu_table_copy:
  for (int i = 0; i < ENTRIES; ++i) {
#pragma HLS pipeline II = 1
    tab[i] = table[i];
  }

gelu_beats:
  for (int i = 0; i < beats; ++i) {
#pragma HLS pipeline II = 1
    const IN_BEAT v = in.read();
    OUT_BEAT y;
    for (int l = 0; l < lanes; ++l)
#pragma HLS unroll
      y[l] = gelu<CFG, ENTRIES>(v[l], tab, b, s, bound);
    out.write(y);
  }
}

}  // namespace hbtxr

#endif  // HBTXR_LUT_HPP
