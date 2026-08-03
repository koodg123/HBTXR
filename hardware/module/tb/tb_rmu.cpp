// V1 — the RMU against the block golden's output projection.
//
// `rmu_*` in the golden IS the output projection: [D,D] weights, one output grid, a
// per-out-channel requant, and the bias on the accumulator. That makes it the canonical
// single-grid RMU. The qkv projection, whose accumulator feeds three differently
// calibrated consumers, arrives with S4.
#include <cstdio>
#include <string>
#include <vector>

#include "hbtxr_config.hpp"
#include "hbtxr_golden.hpp"
#include "hbtxr_rmu.hpp"

using namespace hbtxr;
typedef HbtxrCfgBase CFG;
template struct hbtxr::HbtxrCfgCheck<CFG>;

static constexpr int D = CFG::D, CIP = CFG::O_CIP, COP = CFG::O_COP, TP = CFG::TP;
typedef HbtxrRmu<CFG, D, D, CIP, COP> Rmu;

static Rmu rmu;  // resident: too big for the stack at D=192

int main(int argc, char **argv) {
  const std::string dir = argc > 1 ? argv[1] : "hardware/workspace/golden/search-a4";
  const std::vector<long long> meta = read_golden(dir, "meta");
  const int tokens = (int)meta[0];
  if (meta[1] != D || meta[5] != CFG::act_t::width) {
    std::printf("  meta       FAIL  golden is D=%lld a%lld, config is D=%d a%d\n", meta[1],
                meta[5], D, (int)CFG::act_t::width);
    return 1;
  }

  const std::vector<long long> w = read_golden(dir, "rmu_weight");
  const std::vector<long long> b = read_golden(dir, "rmu_bias_acc");
  const std::vector<long long> x = read_golden(dir, "rmu_x");
  const std::vector<long long> want = read_golden(dir, "rmu_y");
  std::vector<long long> mult, shift;
  read_requant<CFG>(dir, "e07_proj_acc", mult, shift);

  // --- load the resident weights ------------------------------------------
  {
    static CFG::w_t wt[D][D];
    static CFG::acc_t bt[D];
    static ap_uint<CFG::REQ_M_BITS> mt[D];
    static ap_uint<5> st[D];
    for (int o = 0; o < D; ++o) {
      for (int i = 0; i < D; ++i) wt[o][i] = w[o * D + i];
      bt[o] = b[o];
      mt[o] = mult[o];
      st[o] = shift[o];
    }
    rmu.load(wt, bt, mt, st);
  }

  // --- stream the activations in the beat layout the unit declares ---------
  hls::stream<Rmu::in_beat_t> in("in");
  hls::stream<Rmu::out_beat_t> out("out");
  for (int t0 = 0; t0 < tokens; t0 += TP)
    for (int i0 = 0; i0 < D; i0 += CIP) {
      Rmu::in_beat_t v;
      for (int p = 0; p < TP; ++p)
        for (int c = 0; c < CIP; ++c) v[p * CIP + c] = x[(t0 + p) * D + i0 + c];
      in.write(v);
    }

  rmu.compute(in, out, tokens);

  // --- unpack back to [tokens, D] row-major --------------------------------
  std::vector<long long> got(tokens * D, 0);
  for (int t0 = 0; t0 < tokens; t0 += TP)
    for (int o0 = 0; o0 < D; o0 += COP) {
      const Rmu::out_beat_t v = out.read();
      for (int p = 0; p < TP; ++p)
        for (int c = 0; c < COP; ++c) got[(t0 + p) * D + o0 + c] = (long long)v[p * COP + c];
    }

  std::printf("tb_rmu      %d tokens, %dx%d, TP=%d CIP=%d COP=%d\n", tokens, D, D, TP, CIP,
              COP);
  int bad = compare("rmu_y", got, want);
  // Over-production is invisible to a value comparison, so assert the stream drained.
  if (!in.empty() || !out.empty()) {
    std::printf("  drain      FAIL  in=%zu out=%zu beats left\n", in.size(), out.size());
    bad += 1;
  } else {
    std::printf("  drain      ok    both streams empty\n");
  }
  bad += expect_reject("negctl", got, want);
  std::printf("%s\n", bad ? "FAIL" : "PASS");
  return bad;
}
