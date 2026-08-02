// Model-scope golden loading — the per-block weight image the prefetcher walks.
//
// Testbench-only, and a header rather than a copy in each tb because tb_backbone and
// tb_top read the SAME image: a second copy of "which file feeds which member" is a
// second thing that can drift from the generator's layout.
#ifndef HBTXR_MODEL_HPP
#define HBTXR_MODEL_HPP

#include <cstdio>
#include <string>
#include <vector>

#include "hbtxr_golden.hpp"
#include "hbtxr_load.hpp"

namespace hbtxr {

/// The weight image. On the board this is the M-AXI region the prefetcher walks; here it
/// is the golden's per-block arrays, read ONCE and sliced.
template <class CFG>
struct HbtxrModelWeights {
  std::string dir;
  std::vector<long long> qkv_w, qkv_b, proj_w, proj_b, fc1_w, fc1_b, fc2_w, fc2_b;
  std::vector<long long> ln1[4], ln2[4], sm[4], gelu[2];
  std::vector<long long> em[18], es[18];
  std::vector<std::string> edge_names;
  int loads = 0;

  explicit HbtxrModelWeights(const std::string &d) : dir(d) {
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
    for (int i = 0; i < CFG::D; ++i) {
      ln.lnw[i] = src[1][blk * CFG::D + i];
      ln.lnb[i] = src[2][blk * CFG::D + i];
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
      const int span = (int)(em[e].size() / CFG::SEARCH_DEPTH);
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

    put_rmu<3 * CFG::D, CFG::D>(m.qkv, qkv_w, qkv_b, blk, {2, 3, 4});
    put_rmu<CFG::D, CFG::D>(m.proj, proj_w, proj_b, blk, {8});
    put_rmu<CFG::F, CFG::D>(p.fc1, fc1_w, fc1_b, blk, {13});
    put_rmu<CFG::D, CFG::F>(p.fc2, fc2_w, fc2_b, blk, {15});

    m.e01_stream_to_ln1 = edge(0, blk);  m.e02_ln1_to_qkv = edge(1, blk);
    m.e04_smu_to_softmax = edge(5, blk); m.e05_softmax_to_av = edge(6, blk);
    m.e06_av_to_proj = edge(7, blk);     m.e08_resid1_a = edge(9, blk);
    m.e08_resid1_b = edge(10, blk);
    p.e09_stream_to_ln2 = edge(11, blk); p.e10_ln2_to_fc1 = edge(12, blk);
    p.e12_gelu_to_fc2 = edge(14, blk);   p.e14_resid2_a = edge(16, blk);
    p.e14_resid2_b = edge(17, blk);
  }
};

}  // namespace hbtxr

#endif  // HBTXR_MODEL_HPP
