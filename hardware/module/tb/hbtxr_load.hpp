// Payload loading, shared by every core testbench.
//
// Testbench-only. It lives here rather than in each tb because three of them load the
// same block and a second copy of "which golden file feeds which member" is a second
// thing that can drift from the generator's layout.
#ifndef HBTXR_LOAD_HPP
#define HBTXR_LOAD_HPP

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "hbtxr_golden.hpp"
#include "hbtxr_mha_core.hpp"
#include "hbtxr_mlp_core.hpp"

namespace hbtxr {

template <class CFG>
HbtxrEdge load_edge(const std::string &dir, const std::string &name) {
  std::vector<long long> m, s;
  read_requant<CFG>(dir, name, m, s);
  if (m.size() != 1) {
    std::fprintf(stderr, "%s: %zu multipliers, expected a per-tensor edge\n", name.c_str(),
                 m.size());
    std::exit(2);
  }
  HbtxrEdge e;
  e.mult = m[0];
  e.shift = (int)s[0];
  return e;
}

template <class CFG, class LN>
void load_layernorm(LN &ln, const std::string &dir, const std::string &prefix) {
  const std::vector<long long> sc = read_golden(dir, prefix + "_scalars");
  const std::vector<long long> lnw = read_golden(dir, prefix + "_lnw");
  const std::vector<long long> lnb = read_golden(dir, prefix + "_lnb");
  const std::vector<long long> rs = read_golden(dir, prefix + "_rsqrt_table");
  if (sc.size() != 7 || rs.size() != 64) {
    std::fprintf(stderr, "%s: %zu scalars, %zu rsqrt entries\n", prefix.c_str(), sc.size(),
                 rs.size());
    std::exit(2);
  }
  ln.c_1_m = (int)sc[0];
  ln.c_1_s = (int)sc[1];
  ln.b = (int)sc[2];
  ln.s1 = (int)sc[3];
  ln.bound = (int)sc[4];
  ln.s2 = (int)sc[5];
  ln.clamp_bits = (int)sc[6];
  for (size_t i = 0; i < lnw.size(); ++i) {
    ln.lnw[i] = lnw[i];
    ln.lnb[i] = lnb[i];   // affine grid, ~30 signed bits, NOT a 16-bit table entry
    if ((long long)ln.lnb[i] != lnb[i]) {
      std::fprintf(stderr, "%s: lnb[%zu]=%lld does not fit affine_t\n", prefix.c_str(), i,
                   lnb[i]);
      std::exit(2);
    }
  }
  for (int i = 0; i < 64; ++i) ln.rsqrt[i] = rs[i];
}

/// A resident-weight matmul: weights, bias on the accumulator grid, per-channel requant.
/// `edges` may name more than one range — the qkv projection's accumulator feeds three
/// differently calibrated consumers, and they concatenate in output-channel order.
template <class CFG, int CO, int CI, class RMU>
void load_rmu(RMU &rmu, const std::string &dir, const std::string &wname,
              const std::string &bname, const std::vector<std::string> &edges) {
  const std::vector<long long> w = read_golden(dir, wname);
  const std::vector<long long> b = read_golden(dir, bname);
  static typename CFG::w_t wt[CO][CI];
  static typename CFG::acc_t bt[CO];
  static ap_uint<CFG::REQ_M_BITS> mt[CO];
  static ap_uint<5> st[CO];
  if ((long long)w.size() != (long long)CO * CI) {
    std::fprintf(stderr, "%s: %zu weights, want %lld\n", wname.c_str(), w.size(),
                 (long long)CO * CI);
    std::exit(2);
  }
  for (int o = 0; o < CO; ++o) {
    for (int i = 0; i < CI; ++i) wt[o][i] = w[(long long)o * CI + i];
    bt[o] = b[o];
  }
  int at = 0;
  for (const std::string &e : edges) {
    std::vector<long long> m, s;
    read_requant<CFG>(dir, e, m, s);
    for (size_t i = 0; i < m.size(); ++i, ++at) {
      if (at >= CO) { std::fprintf(stderr, "%s: too many requant entries\n", e.c_str()); std::exit(2); }
      mt[at] = m[i];
      st[at] = (int)s[i];
    }
  }
  if (at != CO) {
    std::fprintf(stderr, "%s: %d requant entries, want %d\n", wname.c_str(), at, CO);
    std::exit(2);
  }
  rmu.load(wt, bt, mt, st);
}

template <class CFG, class SM>
void load_softmax(SM &sm, const std::string &dir) {
  const std::vector<long long> sc = read_golden(dir, "softmax_scalars");
  const std::vector<long long> et = read_golden(dir, "softmax_exp_table");
  const std::vector<long long> r1 = read_golden(dir, "softmax_recip_table_one");
  const std::vector<long long> r2 = read_golden(dir, "softmax_recip_table_two");
  // Unsigned: exp's largest entry is its numerator, 1<<15, which int16 cannot hold.
  for (int i = 0; i < 32; ++i) {
    sm.exp_table[i] = et[i];
    if ((long long)sm.exp_table[i] != et[i]) {
      std::fprintf(stderr, "exp_table[%d]=%lld does not fit nlu_t\n", i, et[i]);
      std::exit(2);
    }
  }
  for (int i = 0; i < 64; ++i) { sm.recip_one[i] = r1[i]; sm.recip_two[i] = r2[i]; }
  sm.b1 = sc[0]; sm.s1 = sc[1]; sm.bound1 = sc[2];
  sm.b2_one = sc[3]; sm.s2_one = sc[4]; sm.bound2_one = sc[5];
  sm.b3_one = sc[6]; sm.s3_one = sc[7];
  sm.b2_two = sc[8]; sm.s2_two = sc[9]; sm.bound2_two = sc[10];
  sm.b3_two = sc[11]; sm.s3_two = sc[12]; sm.clamp_bits = sc[13];
}

template <class CFG>
void load_mha(HbtxrMhaCore<CFG> &c, const std::string &dir) {
  constexpr int D = CFG::D;
  load_layernorm<CFG>(c.ln, dir, "ln1");
  load_softmax<CFG>(c.softmax, dir);
  load_rmu<CFG, 3 * D, D>(c.qkv, dir, "qkv_weight", "qkv_bias_acc",
                          {"e03_qkv_q", "e03_qkv_k", "e03_qkv_v"});
  load_rmu<CFG, D, D>(c.proj, dir, "rmu_weight", "rmu_bias_acc", {"e07_proj_acc"});
  c.e01_stream_to_ln1 = load_edge<CFG>(dir, "e01_stream_to_ln1");
  c.e02_ln1_to_qkv = load_edge<CFG>(dir, "e02_ln1_to_qkv");
  c.e04_smu_to_softmax = load_edge<CFG>(dir, "e04_smu_to_softmax");
  c.e05_softmax_to_av = load_edge<CFG>(dir, "e05_softmax_to_av");
  c.e06_av_to_proj = load_edge<CFG>(dir, "e06_av_to_proj");
  c.e08_resid1_a = load_edge<CFG>(dir, "e08_resid1_a");
  c.e08_resid1_b = load_edge<CFG>(dir, "e08_resid1_b");
}

template <class CFG>
void load_mlp(HbtxrMlpCore<CFG> &c, const std::string &dir) {
  constexpr int D = CFG::D, F = CFG::F;
  load_layernorm<CFG>(c.ln, dir, "ln2");
  load_rmu<CFG, F, D>(c.fc1, dir, "fc1_weight", "fc1_bias_acc", {"e11_fc1_acc"});
  load_rmu<CFG, D, F>(c.fc2, dir, "fc2_weight", "fc2_bias_acc", {"e13_fc2_acc"});
  const std::vector<long long> sc = read_golden(dir, "gelu_scalars");
  const std::vector<long long> tb = read_golden(dir, "gelu_table");
  c.gelu_b = (int)sc[0];
  c.gelu_s = (int)sc[1];
  c.gelu_bound = (int)sc[2];
  for (int i = 0; i < 32; ++i) c.gelu_table[i] = tb[i];
  c.e09_stream_to_ln2 = load_edge<CFG>(dir, "e09_stream_to_ln2");
  c.e10_ln2_to_fc1 = load_edge<CFG>(dir, "e10_ln2_to_fc1");
  c.e12_gelu_to_fc2 = load_edge<CFG>(dir, "e12_gelu_to_fc2");
  c.e14_resid2_a = load_edge<CFG>(dir, "e14_resid2_a");
  c.e14_resid2_b = load_edge<CFG>(dir, "e14_resid2_b");
}

}  // namespace hbtxr

#endif  // HBTXR_LOAD_HPP
