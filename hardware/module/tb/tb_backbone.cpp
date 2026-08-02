// V1 — the shared backbone: eight TRBs over TWO core pairs, weights swapped between them.
//
// This is the first testbench that uses the model-scope golden, and the first that tests
// anything structural rather than arithmetic. Three claims are on trial:
//
//   1. the cores are reused. Eight TRBs run on two pairs, so the prefetcher has to put
//      the right block's payload in the right pair at the right time — an off-by-one
//      there produces plausible numbers from the wrong block.
//   2. the backbone is SHARED. Track runs blocks 0..3 with the SAME loaded weights that
//      search runs at a different token count.
//   3. the global buffer carries the residual stream across TRBs, which is only necessary
//      BECAUSE the cores are reused.
#include <cstdio>
#include <string>
#include <vector>

#include "hbtxr_backbone.hpp"
#include "hbtxr_config.hpp"
#include "hbtxr_golden.hpp"
#include "hbtxr_model.hpp"

using namespace hbtxr;
typedef HbtxrCfgBase CFG;
template struct hbtxr::HbtxrCfgCheck<CFG>;

static constexpr int D = CFG::D, N = CFG::N;
typedef HbtxrModelWeights<CFG> Weights;
static HbtxrBackbone<CFG> backbone;


/// Compares every TRB's output against the model golden's per-block trace, so a wrong
/// block reports at the TRB that produced it rather than at the end.
struct BlockTrace {
  std::vector<long long> want;
  int tokens = 0, bad = 0, checked = 0;
  void block(int i, const CFG::act_t *y, int n) {
    ++checked;
    for (int k = 0; k < n * D; ++k)
      if ((long long)y[k] != want[(size_t)i * n * D + k]) {
        std::printf("  TRB %-2d     FAIL  first at [%d]: %lld vs %lld\n", i, k,
                    (long long)y[k], want[(size_t)i * n * D + k]);
        ++bad;
        return;
      }
  }
};

static int run_path(Weights &w, const std::string &dir, const char *name, int mode) {
  const HbtxrSchedule sched = hbtxr_schedule<CFG>(mode);
  const std::vector<long long> x = read_golden(dir, std::string("stem_") + name + "_y");
  BlockTrace trace;
  trace.want = read_golden(dir, std::string(name) + "_block_out");
  trace.tokens = sched.tokens;

  static CFG::act_t xin[N * D], yout[N * D];
  for (int i = 0; i < sched.tokens * D; ++i) xin[i] = x[i];

  w.loads = 0;
  const int tag_slip = backbone.run(xin, yout, sched, w, (HbtxrNoProbe *)nullptr, &trace);

  std::printf("tb_backbone %-6s mode=%d  %d TRBs on %d core pairs, %d tokens, %d loads\n",
              name, mode, sched.depth, backbone.PAIRS, sched.tokens, w.loads);
  int bad = trace.bad;
  if (bad) std::printf("  trace      FAIL  %d of %d TRBs\n", bad, trace.checked);
  else std::printf("  trace      ok    %d TRBs, every block output matches\n", trace.checked);
  if (tag_slip) {
    std::printf("  prefetch   FAIL  %d TRBs ran on a pair holding another block\n", tag_slip);
    bad += 1;
  } else {
    std::printf("  prefetch   ok    every TRB ran on the pair holding its own weights\n");
  }
  if (w.loads != sched.depth) {
    std::printf("  loads      FAIL  %d loads for %d TRBs\n", w.loads, sched.depth);
    bad += 1;
  }
  return bad;
}

int main(int argc, char **argv) {
  const std::string dir =
      argc > 1 ? argv[1] : "hardware/workspace/golden/model-hbtxr-w4s4h8";
  const std::vector<long long> meta = read_golden(dir, "meta");
  if (meta[0] != D || meta[5] != CFG::SEARCH_DEPTH || meta[6] != CFG::TRACK_DEPTH) {
    std::printf("  meta       FAIL  golden D=%lld depth=%lld cut=%lld\n", meta[0], meta[5],
                meta[6]);
    return 1;
  }

  Weights weights(dir);
  // search, then track, then search AGAIN. State that survives a run and only matters at
  // the other token count is invisible in one order — this found exactly that: an S x V
  // reduction running to the compile-time maximum instead of the runtime length, which a
  // 64-token run covers completely and a following 16-token run reads stale.
  int bad = run_path(weights, dir, "search", 0);
  bad += run_path(weights, dir, "track", 1);
  bad += run_path(weights, dir, "search", 0);

  // The prefetch schedule has to be wrong-able, or "ok" means nothing. Load a wrong block
  // into one pair and require the trace to fail AT that TRB.
  {
    std::printf("  -- fault injection: TRB 3 given block 4's weights --\n");
    struct Slipped : Weights {
      using Weights::Weights;
      void load(int blk, HbtxrMhaCore<CFG> &m, HbtxrMlpCore<CFG> &p) {
        Weights::load(blk == 3 ? 4 : blk, m, p);
      }
    };
    Slipped bad_src(dir);
    const HbtxrSchedule sched = hbtxr_schedule<CFG>(0);
    const std::vector<long long> x = read_golden(dir, "stem_search_y");
    BlockTrace trace;
    trace.want = read_golden(dir, "search_block_out");
    trace.tokens = sched.tokens;
    static CFG::act_t xin[N * D], yout[N * D];
    for (int i = 0; i < sched.tokens * D; ++i) xin[i] = x[i];
    backbone.run(xin, yout, sched, bad_src, (HbtxrNoProbe *)nullptr, &trace);
    if (trace.bad == 0) {
      std::printf("  localize   FAIL  a wrong block's weights produced the right answer\n");
      bad += 1;
    } else {
      std::printf("  localize   ok    the trace caught it (%d TRBs affected)\n", trace.bad);
    }
  }

  std::printf("%s\n", bad ? "FAIL" : "PASS");
  return bad;
}
