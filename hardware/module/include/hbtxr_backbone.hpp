// Backbone — controller, global buffer, weight prefetcher, core cycling (SPEC §2, §6, §7).
//
// This is the stage where "the cores are reused by swapping weights" stops being a claim.
// The paper defines four cores (MHA0, MLP0, MHA1, MLP1) and cycles TRBs over them; there
// is exactly ONE pair of transformer cores per parity, and eight TRBs run on two pairs.
//
// The cycling IS the double buffering. While pair `i mod 2` computes TRB i, pair
// `(i+1) mod 2` is idle and its weights can be replaced — so the prefetcher needs no
// buffer of its own, and the reference's twelve separate hardware instances (which is
// what ROM-initialised weights force, SPEC §4) become two.
#ifndef HBTXR_BACKBONE_HPP
#define HBTXR_BACKBONE_HPP

#include <ap_int.h>

#include "hbtxr_config.hpp"
#include "hbtxr_mha_core.hpp"
#include "hbtxr_mlp_core.hpp"

namespace hbtxr {

/// What the mode selects (SPEC §7). The backbone only needs the depth; the stem and the
/// terminal decode are the top's business (S8).
struct HbtxrSchedule {
  int tokens;
  int depth;
};

template <class CFG>
inline HbtxrSchedule hbtxr_schedule(int mode) {
  // 0 = search: full depth. 1 = track: early exit at the cut point, on the SAME blocks.
  return mode == 0 ? HbtxrSchedule{CFG::N, CFG::SEARCH_DEPTH}
                   : HbtxrSchedule{CFG::TRACK_TOKENS, CFG::TRACK_DEPTH};
}

template <class CFG>
struct HbtxrBackbone {
  static constexpr int PAIRS = 2;              // SPEC §2: HBTXR_CORE_PAIRS
  static constexpr int D = CFG::D, N = CFG::N;
  typedef typename CFG::act_t act_t;

  HbtxrMhaCore<CFG> mha[PAIRS];
  HbtxrMlpCore<CFG> mlp[PAIRS];

  /// Which TRB's weights each pair currently holds. -1 is "nothing loaded", and running
  /// a pair whose tag is not the TRB being computed is the failure this exists to catch:
  /// an off-by-one in the prefetch schedule produces plausible numbers from the wrong
  /// block, which no value comparison downstream would attribute correctly.
  int loaded[PAIRS];

  /// The global buffer: the residual stream between TRBs. It has to exist precisely
  /// BECAUSE the cores are reused — with a core per block the stream would stay in the
  /// dataflow FIFOs and never be stored.
  act_t gbuf[N][D];

  HbtxrBackbone() {
    for (int p = 0; p < PAIRS; ++p) loaded[p] = -1;
  }

  /// `SOURCE::load(i, mha, mlp)` puts TRB `i`'s payload into one pair. On the board it
  /// reads M-AXI; in the testbench it slices the golden. Templated rather than virtual
  /// because a virtual call does not synthesize.
  ///
  /// `probe` and `trace` are both testbench-only and both null in synthesis. They are
  /// separate parameters rather than one, because a null literal cannot deduce a template
  /// pointer type — the caller has to name it.
  template <class SOURCE, class PROBE, class TRACE>
  int run(const act_t *x, act_t *y, const HbtxrSchedule &sched, SOURCE &src, PROBE *probe,
          TRACE *trace) {
    const int tokens = sched.tokens, depth = sched.depth;
    int mismatched_tag = 0;

    for (int t = 0; t < tokens; ++t)
      for (int c = 0; c < D; ++c) gbuf[t][c] = x[t * D + c];

    // Prime pair 0, then keep exactly one TRB ahead. Two pairs and a one-ahead prefetch
    // is the whole schedule: the pair being loaded is never the pair being computed.
    src.load(0, mha[0], mlp[0]);
    loaded[0] = 0;

  trb:
    for (int i = 0; i < depth; ++i) {
      const int p = i % PAIRS;
      if (i + 1 < depth) {
        const int q = (i + 1) % PAIRS;
        src.load(i + 1, mha[q], mlp[q]);
        loaded[q] = i + 1;
      }
      if (loaded[p] != i) ++mismatched_tag;   // the prefetch schedule slipped

      static act_t mid[N * D];
      mha[p].run(&gbuf[0][0], mid, tokens, probe);
      mlp[p].run(mid, &gbuf[0][0], tokens, probe);
      if (trace) trace->block(i, &gbuf[0][0], tokens);
    }

    for (int t = 0; t < tokens; ++t)
      for (int c = 0; c < D; ++c) y[t * D + c] = gbuf[t][c];
    return mismatched_tag;
  }
};

}  // namespace hbtxr

#endif  // HBTXR_BACKBONE_HPP
