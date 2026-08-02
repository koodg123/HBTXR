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
#include "hbtxr_load.hpp"
#include "hbtxr_probe.hpp"

using namespace hbtxr;
typedef HbtxrCfgBase CFG;
template struct hbtxr::HbtxrCfgCheck<CFG>;

static constexpr int D = CFG::D, F = CFG::F, N = CFG::N, DEPTH = CFG::SEARCH_DEPTH;
static HbtxrBackbone<CFG> backbone;

/// The weight image. On the board this is the M-AXI region the prefetcher walks; here it
/// is the golden's per-block arrays, read ONCE and sliced.
struct GoldenWeights {
  std::string dir;
  std::vector<long long> qkv_w, qkv_b, proj_w, proj_b, fc1_w, fc1_b, fc2_w, fc2_b;
  std::vector<long long> ln1[4], ln2[4], sm[4], gelu[2];
  std::vector<long long> em[18], es[18];
  std::vector<std::string> edge_names;
  int loads = 0;

  explicit GoldenWeights(const std::string &d) : dir(d) {
    qkv_w = read_golden(dir, "blk_qkv_weight");   qkv_b = read_golden(dir, "blk_qkv_bias_acc");
    proj_w = read_golden(dir, "blk_proj_weight"); proj_b = read_golden(dir, "blk_proj_bias_acc");
    fc1_w = read_golden(dir, "blk_fc1_weight");   fc1_b = read_golden(dir, "blk_fc1_bias_acc");
    fc2_w = read_golden(dir, "blk_fc2_weight");   fc2_b = read_golden(dir, "blk_fc2_bias_acc");
    const char *lnf[4] = {"scalars", "lnw", "lnb", "rsqrt_table"};
    for (int i = 0; i < 4; ++i) {
      ln1[i] = read_golden(dir, std::string("blk_ln1_") + lnf[i]);
      ln2[i] = read_golden(dir, std::string("blk_ln2_") + lnf[i]);
    }
    const char *smf[4] = {"scalars", "exp_table", "recip_table_one", "recip_table_two"};
    for (int i = 0; i < 4; ++i) sm[i] = read_golden(dir, std::string("blk_softmax_") + smf[i]);
    gelu[0] = read_golden(dir, "blk_gelu_scalars");
    gelu[1] = read_golden(dir, "blk_gelu_table");
    edge_names = {"e01_stream_to_ln1", "e02_ln1_to_qkv", "e03_qkv_q", "e03_qkv_k",
                  "e03_qkv_v", "e04_smu_to_softmax", "e05_softmax_to_av", "e06_av_to_proj",
                  "e07_proj_acc", "e08_resid1_a", "e08_resid1_b", "e09_stream_to_ln2",
                  "e10_ln2_to_fc1", "e11_fc1_acc", "e12_gelu_to_fc2", "e13_fc2_acc",
                  "e14_resid2_a", "e14_resid2_b"};
    for (size_t i = 0; i < edge_names.size(); ++i) {
      em[i] = read_golden(dir, "blk_" + edge_names[i] + "_mult");
      es[i] = read_golden(dir, "blk_" + edge_names[i] + "_shift");
    }
  }

  HbtxrEdge edge(int e, int blk) const {
    HbtxrEdge r;
    r.mult = em[e][blk];
    r.shift = (int)es[e][blk];
    return r;
  }

  template <class LN>
  void put_ln(LN &ln, const std::vector<long long> *src, int blk) const {
    ln.c_1_m = (int)src[0][blk * 7 + 0]; ln.c_1_s = (int)src[0][blk * 7 + 1];
    ln.b = (int)src[0][blk * 7 + 2];     ln.s1 = (int)src[0][blk * 7 + 3];
    ln.bound = (int)src[0][blk * 7 + 4]; ln.s2 = (int)src[0][blk * 7 + 5];
    ln.clamp_bits = (int)src[0][blk * 7 + 6];
    for (int i = 0; i < D; ++i) {
      ln.lnw[i] = src[1][blk * D + i];
      ln.lnb[i] = src[2][blk * D + i];
    }
    for (int i = 0; i < 64; ++i) ln.rsqrt[i] = src[3][blk * 64 + i];
  }

  template <int CO, int CI, class RMU>
  void put_rmu(RMU &rmu, const std::vector<long long> &w, const std::vector<long long> &b,
               int blk, const std::vector<int> &edges) const {
    static typename CFG::w_t wt[CO][CI];
    static typename CFG::acc_t bt[CO];
    static ap_uint<CFG::REQ_M_BITS> mt[CO];
    static ap_uint<5> st[CO];
    const long long wo = (long long)blk * CO * CI, bo = (long long)blk * CO;
    for (int o = 0; o < CO; ++o) {
      for (int i = 0; i < CI; ++i) wt[o][i] = w[wo + (long long)o * CI + i];
      bt[o] = b[bo + o];
    }
    // Per-channel edges are [depth * CO]; the qkv accumulator's three consumers
    // concatenate in output-channel order.
    int at = 0;
    for (int e : edges) {
      const int span = (int)(em[e].size() / DEPTH);
      for (int i = 0; i < span; ++i, ++at) {
        mt[at] = em[e][(long long)blk * span + i];
        st[at] = (int)es[e][(long long)blk * span + i];
      }
    }
    rmu.load(wt, bt, mt, st);
  }

  void load(int blk, HbtxrMhaCore<CFG> &m, HbtxrMlpCore<CFG> &p) {
    ++loads;
    put_ln(m.ln, ln1, blk);
    put_ln(p.ln, ln2, blk);
    for (int i = 0; i < 32; ++i) m.softmax.exp_table[i] = sm[1][blk * 32 + i];
    for (int i = 0; i < 64; ++i) {
      m.softmax.recip_one[i] = sm[2][blk * 64 + i];
      m.softmax.recip_two[i] = sm[3][blk * 64 + i];
    }
    const long long s = (long long)blk * 14;
    m.softmax.b1 = sm[0][s + 0];  m.softmax.s1 = sm[0][s + 1];  m.softmax.bound1 = sm[0][s + 2];
    m.softmax.b2_one = sm[0][s + 3]; m.softmax.s2_one = sm[0][s + 4];
    m.softmax.bound2_one = sm[0][s + 5]; m.softmax.b3_one = sm[0][s + 6];
    m.softmax.s3_one = sm[0][s + 7];
    m.softmax.b2_two = sm[0][s + 8]; m.softmax.s2_two = sm[0][s + 9];
    m.softmax.bound2_two = sm[0][s + 10]; m.softmax.b3_two = sm[0][s + 11];
    m.softmax.s3_two = sm[0][s + 12]; m.softmax.clamp_bits = sm[0][s + 13];
    p.gelu_b = (int)gelu[0][blk * 3 + 0];
    p.gelu_s = (int)gelu[0][blk * 3 + 1];
    p.gelu_bound = (int)gelu[0][blk * 3 + 2];
    for (int i = 0; i < 32; ++i) p.gelu_table[i] = gelu[1][blk * 32 + i];

    put_rmu<3 * D, D>(m.qkv, qkv_w, qkv_b, blk, {2, 3, 4});
    put_rmu<D, D>(m.proj, proj_w, proj_b, blk, {8});
    put_rmu<F, D>(p.fc1, fc1_w, fc1_b, blk, {13});
    put_rmu<D, F>(p.fc2, fc2_w, fc2_b, blk, {15});

    m.e01_stream_to_ln1 = edge(0, blk);  m.e02_ln1_to_qkv = edge(1, blk);
    m.e04_smu_to_softmax = edge(5, blk); m.e05_softmax_to_av = edge(6, blk);
    m.e06_av_to_proj = edge(7, blk);     m.e08_resid1_a = edge(9, blk);
    m.e08_resid1_b = edge(10, blk);
    p.e09_stream_to_ln2 = edge(11, blk); p.e10_ln2_to_fc1 = edge(12, blk);
    p.e12_gelu_to_fc2 = edge(14, blk);   p.e14_resid2_a = edge(16, blk);
    p.e14_resid2_b = edge(17, blk);
  }
};

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

static int run_path(GoldenWeights &w, const std::string &dir, const char *name, int mode) {
  const HbtxrSchedule sched = hbtxr_schedule<CFG>(mode);
  const std::vector<long long> x = read_golden(dir, std::string("stem_") + name + "_y");
  BlockTrace trace;
  trace.want = read_golden(dir, std::string(name) + "_block_out");
  trace.tokens = sched.tokens;

  static CFG::act_t xin[N * D], yout[N * D];
  for (int i = 0; i < sched.tokens * D; ++i) xin[i] = x[i];

  w.loads = 0;
  const int tag_slip = backbone.run(xin, yout, sched, w, (HbtxrProbe *)nullptr, &trace);

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
  if (meta[0] != D || meta[5] != DEPTH || meta[6] != CFG::TRACK_DEPTH) {
    std::printf("  meta       FAIL  golden D=%lld depth=%lld cut=%lld\n", meta[0], meta[5],
                meta[6]);
    return 1;
  }

  GoldenWeights weights(dir);
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
    struct Slipped : GoldenWeights {
      using GoldenWeights::GoldenWeights;
      void load(int blk, HbtxrMhaCore<CFG> &m, HbtxrMlpCore<CFG> &p) {
        GoldenWeights::load(blk == 3 ? 4 : blk, m, p);
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
    backbone.run(xin, yout, sched, bad_src, (HbtxrProbe *)nullptr, &trace);
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
