// HBTXR design parameters — one traits class, zero macros (SPEC §2-A).
//
// Config travels as a TYPE, not a value: the HLS front end does not accept C++20
// class-type non-type template parameters, so `template <class CFG>` is the only form
// that works. Every member is a class-scope `static constexpr` because that is what a
// pragma argument has to be.
#ifndef HBTXR_CONFIG_HPP
#define HBTXR_CONFIG_HPP

#include <ap_int.h>

namespace hbtxr {

/// ceil(log2(n)) — the width a reduction over n terms adds to its operands.
constexpr int clog2(int n) { return n <= 1 ? 0 : 1 + clog2((n + 1) / 2); }

/// Worst-case MAC accumulator width for a reduction over `ci` terms.
/// DERIVED, never transcribed: the reference copied a calibration value and its
/// LayerNorm variance was 27 bits where D=192 needs 36 (SPEC §3).
constexpr int mac_width(int w_bits, int a_bits, int ci) { return w_bits + a_bits + clog2(ci); }

constexpr bool is_pow2(int n) { return n > 0 && (n & (n - 1)) == 0; }

struct HbtxrCfgBase {
  // --- shape (SPEC §1) ------------------------------------------------------
  static constexpr int N  = 64;    // tokens, search path (128/16 squared)
  static constexpr int D  = 192;   // embedding
  static constexpr int H  = 3;     // heads
  static constexpr int HD = 64;    // D / H
  static constexpr int F  = 768;   // MLP hidden

  // --- parallelism ----------------------------------------------------------
  // Three numbers must agree per stage: the hls::vector width, the unroll factor and the
  // array_reshape cyclic factor. They are named, never positional — the reference took
  // 126 template parameters of which 76 were `int`, so a swapped slot compiled and
  // c-simulated clean and only showed up in cosim.
  static constexpr int TP    = 4;               // token parallelism, one number design-wide
  static constexpr int O_CIP = 8, O_COP = 8;    // output-projection RMU
  static constexpr int R_CIP = 8, R_COP = 8;    // relation SMU (Q x K^T)
  static constexpr int Q_CIP = 8, Q_COP = 8;    // qkv projection RMU
  static constexpr int A_CIP = 8, A_COP = 8;    // attention SMU (S x V)
  static constexpr int M1_CIP = 8, M1_COP = 8;  // MLP expand   D -> F
  static constexpr int M2_CIP = 8, M2_COP = 8;  // MLP contract F -> D
  static constexpr int NL_P  = 8;               // lanes of a pointwise / row-wise op
  // Every factor is 8, so every act_t beat in either core is TP*8 = 32 lanes and no width
  // adapter is needed between stages.

  // --- numeric system (SPEC §3) --------------------------------------------
  using act_t = ap_int<4>;     // MHA / MLP activations
  using w_t   = ap_int<4>;     // MHA / MLP weights
  using acc_t = ap_int<20>;    // every MAC accumulator

  // Nonlinear LUT entries are 16 bits, but the SIGN is per operator and getting it wrong
  // is silent. rsqrt and GeLU are signed; exp and reciprocal are not — and exp's largest
  // entry is its numerator, 1<<15 = 32768, which ap_int<16> CANNOT hold. Measured on the
  // golden: exp max 32768, recip max 46075.
  using nl_t  = ap_int<16>;    // GeLU, rsqrt      (signed)
  using nlu_t = ap_uint<16>;   // exp, reciprocal  (non-negative by construction)
  // lnb is NOT a table entry: it lives on the affine accumulator grid, ~30 signed bits.
  // The LayerNorm unit derives its width there rather than assuming a 16-bit payload.
  using prob_t = ap_uint<8>;   // softmax output, the grid the S x V operand requants from

  // --- requant (SPEC §3) ----------------------------------------------------
  // 33 is what the golden carries TODAY: dyadic_params picks shift 31 almost always, so
  // any ratio above 1 needs 33 bits and acc*M is 53. Bounding the multiplier at 18 is
  // bit-identical to the current golden — measured, see
  // algorithm/docs/reports/2026-07-31-requant-multiplier-width.md. When that change lands
  // this becomes 18 and acc*M drops to 38, which fits a DSP48E2's B port. Nothing else
  // in the design moves.
  static constexpr int REQ_M_BITS = 33;
  static constexpr int REQ_N_MAX  = 31;
};

/// Track path: the SAME backbone weights, fewer tokens (SPEC §7).
struct HbtxrCfgTrack : HbtxrCfgBase { static constexpr int N = 16; };

// --- checks that hold for any config -----------------------------------------
// Per-stage reduction widths are asserted inside the unit that knows its own CI.
template <class CFG>
struct HbtxrCfgCheck {
  static_assert(CFG::D == CFG::H * CFG::HD, "D must be H * HD");
  static_assert(CFG::acc_t::width >= mac_width(CFG::w_t::width, CFG::act_t::width, CFG::F),
                "acc_t cannot hold the widest reduction in the design (CI = F)");
  // Powers of two keep the Adapter off its LCM non-divisible path.
  static_assert(is_pow2(CFG::TP) && is_pow2(CFG::O_CIP) && is_pow2(CFG::O_COP)
                    && is_pow2(CFG::R_CIP) && is_pow2(CFG::R_COP),
                "every parallelism factor must be a power of two");
  static_assert(CFG::N % CFG::TP == 0, "N must divide into whole token tiles");
  static_assert(CFG::D % CFG::O_CIP == 0 && CFG::D % CFG::O_COP == 0, "D vs O_*P");
  static_assert(CFG::HD % CFG::R_CIP == 0, "HD vs R_CIP");
  static_assert(CFG::D % CFG::Q_CIP == 0 && (3 * CFG::D) % CFG::Q_COP == 0, "D vs Q_*P");
  static_assert(CFG::N % CFG::A_CIP == 0 && CFG::HD % CFG::A_COP == 0, "N/HD vs A_*P");
  static_assert(CFG::D % CFG::M1_CIP == 0 && CFG::F % CFG::M1_COP == 0, "D/F vs M1_*P");
  static_assert(CFG::F % CFG::M2_CIP == 0 && CFG::D % CFG::M2_COP == 0, "F/D vs M2_*P");
  static_assert(CFG::Q_CIP == CFG::O_CIP && CFG::Q_COP == CFG::O_COP
                    && CFG::R_CIP == CFG::O_CIP && CFG::A_COP == CFG::O_COP
                    && CFG::NL_P == CFG::O_CIP,
                "the MHA core assumes one lane count end to end -- see the note above");
  static_assert(CFG::REQ_N_MAX < 32, "the shift must fit ap_uint<5>");
};

}  // namespace hbtxr

#endif  // HBTXR_CONFIG_HPP
