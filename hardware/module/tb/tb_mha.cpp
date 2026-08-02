// V1 — the whole MHA core: mha_x in, mha_y out, with a probe at every stage boundary.
#include <cstdio>
#include <string>
#include <vector>

#include "hbtxr_config.hpp"
#include "hbtxr_golden.hpp"
#include "hbtxr_mha_core.hpp"
#include "hbtxr_probe.hpp"

using namespace hbtxr;
typedef HbtxrCfgBase CFG;
template struct hbtxr::HbtxrCfgCheck<CFG>;

static constexpr int D = CFG::D, H = CFG::H, HD = CFG::HD, N = CFG::N;
typedef HbtxrMhaCore<CFG> Core;
static Core core;

static HbtxrEdge edge(const std::string &dir, const std::string &name) {
  std::vector<long long> m, s;
  read_requant<CFG>(dir, name, m, s);
  if (m.size() != 1) {
    std::printf("  %s: %zu multipliers, expected a per-tensor edge\n", name.c_str(), m.size());
    std::exit(2);
  }
  HbtxrEdge e;
  e.mult = m[0];
  e.shift = (int)s[0];
  return e;
}

int main(int argc, char **argv) {
  const std::string dir = argc > 1 ? argv[1] : "hardware/workspace/golden/search-a4";
  const std::vector<long long> meta = read_golden(dir, "meta");
  const int tokens = (int)meta[0];
  if (meta[1] != D || meta[2] != H || meta[3] != HD) {
    std::printf("  meta       FAIL  golden D=%lld H=%lld HD=%lld\n", meta[1], meta[2], meta[3]);
    return 1;
  }

  // --- payloads ------------------------------------------------------------
  {
    const std::vector<long long> sc = read_golden(dir, "ln1_scalars");
    const std::vector<long long> lnw = read_golden(dir, "ln1_lnw");
    const std::vector<long long> lnb = read_golden(dir, "ln1_lnb");
    const std::vector<long long> rs = read_golden(dir, "ln1_rsqrt_table");
    core.ln.c_1_m = (int)sc[0]; core.ln.c_1_s = (int)sc[1]; core.ln.b = (int)sc[2];
    core.ln.s1 = (int)sc[3]; core.ln.bound = (int)sc[4]; core.ln.s2 = (int)sc[5];
    core.ln.clamp_bits = (int)sc[6];
    for (int i = 0; i < D; ++i) { core.ln.lnw[i] = lnw[i]; core.ln.lnb[i] = lnb[i]; }
    for (int i = 0; i < 64; ++i) core.ln.rsqrt[i] = rs[i];
  }
  {
    const std::vector<long long> sc = read_golden(dir, "softmax_scalars");
    const std::vector<long long> et = read_golden(dir, "softmax_exp_table");
    const std::vector<long long> r1 = read_golden(dir, "softmax_recip_table_one");
    const std::vector<long long> r2 = read_golden(dir, "softmax_recip_table_two");
    for (int i = 0; i < 32; ++i) core.softmax.exp_table[i] = et[i];
    for (int i = 0; i < 64; ++i) { core.softmax.recip_one[i] = r1[i]; core.softmax.recip_two[i] = r2[i]; }
    core.softmax.b1 = sc[0]; core.softmax.s1 = sc[1]; core.softmax.bound1 = sc[2];
    core.softmax.b2_one = sc[3]; core.softmax.s2_one = sc[4]; core.softmax.bound2_one = sc[5];
    core.softmax.b3_one = sc[6]; core.softmax.s3_one = sc[7];
    core.softmax.b2_two = sc[8]; core.softmax.s2_two = sc[9]; core.softmax.bound2_two = sc[10];
    core.softmax.b3_two = sc[11]; core.softmax.s3_two = sc[12]; core.softmax.clamp_bits = sc[13];
  }
  // qkv: one unit, three output grids. The three per-channel (M, n) arrays concatenate in
  // output-channel order, because Q, K and V ARE contiguous output-channel ranges.
  {
    const std::vector<long long> w = read_golden(dir, "qkv_weight");
    const std::vector<long long> b = read_golden(dir, "qkv_bias_acc");
    static CFG::w_t wt[3 * D][D];
    static CFG::acc_t bt[3 * D];
    static ap_uint<CFG::REQ_M_BITS> mt[3 * D];
    static ap_uint<5> st[3 * D];
    for (int o = 0; o < 3 * D; ++o) {
      for (int i = 0; i < D; ++i) wt[o][i] = w[(long long)o * D + i];
      bt[o] = b[o];
    }
    int at = 0;
    for (const char *part : {"e03_qkv_q", "e03_qkv_k", "e03_qkv_v"}) {
      std::vector<long long> m, s;
      read_requant<CFG>(dir, part, m, s);
      if ((int)m.size() != D) { std::printf("  %s: %zu entries, want %d\n", part, m.size(), D); return 1; }
      for (int i = 0; i < D; ++i, ++at) { mt[at] = m[i]; st[at] = (int)s[i]; }
    }
    core.qkv.load(wt, bt, mt, st);
  }
  {
    const std::vector<long long> w = read_golden(dir, "rmu_weight");
    const std::vector<long long> b = read_golden(dir, "rmu_bias_acc");
    std::vector<long long> m, s;
    read_requant<CFG>(dir, "e07_proj_acc", m, s);
    static CFG::w_t wt[D][D];
    static CFG::acc_t bt[D];
    static ap_uint<CFG::REQ_M_BITS> mt[D];
    static ap_uint<5> st[D];
    for (int o = 0; o < D; ++o) {
      for (int i = 0; i < D; ++i) wt[o][i] = w[o * D + i];
      bt[o] = b[o]; mt[o] = m[o]; st[o] = (int)s[o];
    }
    core.proj.load(wt, bt, mt, st);
  }
  core.e01_stream_to_ln1 = edge(dir, "e01_stream_to_ln1");
  core.e02_ln1_to_qkv = edge(dir, "e02_ln1_to_qkv");
  core.e04_smu_to_softmax = edge(dir, "e04_smu_to_softmax");
  core.e05_softmax_to_av = edge(dir, "e05_softmax_to_av");
  core.e06_av_to_proj = edge(dir, "e06_av_to_proj");
  core.e08_resid1_a = edge(dir, "e08_resid1_a");
  core.e08_resid1_b = edge(dir, "e08_resid1_b");

  // --- probes: every stage that has its own golden -------------------------
  // The goldens are row-major; a stream carries TP x P beats. expect2d permutes.
  HbtxrProbe probe;
  const int TP = CFG::TP, P = Core::P;
  for (const char *stage : {"ln1_x", "ln1_y", "qkv_x", "rmu_x", "rmu_y"})
    probe.expect2d(stage, read_golden(dir, stage), tokens, D, TP, P);
  // smu_y and softmax_y are [H][tokens][tokens]; the core probes head 0 only.
  for (const char *stage : {"smu_y", "softmax_y"}) {
    const std::vector<long long> all = read_golden(dir, stage);
    probe.expect2d(stage, std::vector<long long>(all.begin(), all.begin() + tokens * tokens),
                   tokens, tokens, TP, P);
  }

  const std::vector<long long> x = read_golden(dir, "mha_x");
  const std::vector<long long> want = read_golden(dir, "mha_y");
  static CFG::act_t xin[N * D], yout[N * D];
  for (int i = 0; i < tokens * D; ++i) xin[i] = x[i];

  std::printf("tb_mha      %d tokens, D=%d H=%d HD=%d, lanes=%d\n", tokens, D, H, HD,
              Core::LANES);
  core.run(xin, yout, tokens, &probe);

  std::vector<long long> got(tokens * D);
  for (int i = 0; i < tokens * D; ++i) got[i] = (long long)yout[i];
  int bad = compare("mha_y", got, want);
  if (probe.failures) {
    std::printf("  probes     FAIL  %d of %d stages\n", probe.failures, probe.checked);
    bad += 1;
  } else {
    std::printf("  probes     ok    %d stages\n", probe.checked);
  }
  bad += expect_reject("negctl", got, want);

  // --- negative control for the PROBE itself -------------------------------
  // A probe that never fires is decoration. Break one edge and require two things: the
  // probes localize (the first failure is the stage right after the broken edge), and the
  // refill contains it (downstream probes still pass because they were handed the golden).
  {
    std::printf("  -- fault injection: e02_ln1_to_qkv multiplier x2 --\n");
    HbtxrProbe p2 = probe;
    p2.failures = p2.checked = 0;
    core.e02_ln1_to_qkv.mult *= 2;
    core.run(xin, yout, tokens, &p2);
    core.e02_ln1_to_qkv.mult /= 2;
    if (p2.failures == 0) {
      std::printf("  localize   FAIL  a broken edge produced no probe failure\n");
      bad += 1;
    } else if (p2.failures > 2) {
      std::printf("  localize   FAIL  %d probes fired; the refill should contain it\n",
                  p2.failures);
      bad += 1;
    } else {
      std::printf("  localize   ok    %d of %d probes fired, downstream contained\n",
                  p2.failures, p2.checked);
    }
  }

  std::printf("%s\n", bad ? "FAIL" : "PASS");
  return bad;
}
