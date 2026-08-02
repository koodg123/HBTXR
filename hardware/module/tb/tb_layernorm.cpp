// V1 — integer LayerNorm against i_ops.py:layernorm_quantize (the 7-scalar kernel).
#include <cstdio>
#include <string>
#include <vector>

#include "hbtxr_config.hpp"
#include "hbtxr_golden.hpp"
#include "hbtxr_layernorm.hpp"

using namespace hbtxr;
typedef HbtxrCfgBase CFG;
template struct hbtxr::HbtxrCfgCheck<CFG>;

static constexpr int D = CFG::D, P = CFG::NL_P, TP = CFG::TP;
typedef HbtxrLayerNorm<CFG, D, P> Ln;
static Ln ln;

int main(int argc, char **argv) {
  const std::string dir = argc > 1 ? argv[1] : "hardware/workspace/golden/search-a4";
  const std::vector<long long> meta = read_golden(dir, "meta");
  const int tokens = (int)meta[0];
  const std::vector<long long> sc = read_golden(dir, "ln1_scalars");
  const std::vector<long long> lnw = read_golden(dir, "ln1_lnw");
  const std::vector<long long> lnb = read_golden(dir, "ln1_lnb");
  const std::vector<long long> rs = read_golden(dir, "ln1_rsqrt_table");
  const std::vector<long long> x = read_golden(dir, "ln1_x");
  const std::vector<long long> want = read_golden(dir, "ln1_y");
  if (sc.size() != 7 || rs.size() != 64) {
    std::printf("  payload    FAIL  %zu scalars, %zu rsqrt entries (want 7, 64)\n",
                sc.size(), rs.size());
    return 1;
  }
  if (sc[6] != CFG::act_t::width) {
    std::printf("  payload    FAIL  clamp_bits=%lld, act_t is %d\n", sc[6],
                (int)CFG::act_t::width);
    return 1;
  }

  ln.c_1_m = (int)sc[0];
  ln.c_1_s = (int)sc[1];
  ln.b = (int)sc[2];
  ln.s1 = (int)sc[3];
  ln.bound = (int)sc[4];
  ln.s2 = (int)sc[5];
  ln.clamp_bits = (int)sc[6];
  for (int i = 0; i < D; ++i) {
    ln.lnw[i] = lnw[i];
    ln.lnb[i] = lnb[i];   // affine grid, ~30 signed bits -- NOT a 16-bit table entry
    if ((long long)ln.lnb[i] != lnb[i]) {
      std::printf("  payload    FAIL  lnb[%d]=%lld does not fit affine_t (%d bits)\n", i,
                  lnb[i], (int)Ln::affine_t::width);
      return 1;
    }
  }
  for (int i = 0; i < 64; ++i) ln.rsqrt[i] = rs[i];

  hls::stream<Ln::beat_t> in("in"), out("out");
  for (int t0 = 0; t0 < tokens; t0 += TP)
    for (int c0 = 0; c0 < D; c0 += P) {
      Ln::beat_t v;
      for (int p = 0; p < TP; ++p)
        for (int c = 0; c < P; ++c) v[p * P + c] = x[(t0 + p) * D + c0 + c];
      in.write(v);
    }

  ln.run(in, out, tokens);

  std::vector<long long> got(tokens * D, 0);
  for (int t0 = 0; t0 < tokens; t0 += TP)
    for (int c0 = 0; c0 < D; c0 += P) {
      const Ln::beat_t v = out.read();
      for (int p = 0; p < TP; ++p)
        for (int c = 0; c < P; ++c) got[(t0 + p) * D + c0 + c] = (long long)v[p * P + c];
    }

  std::printf("tb_layernorm %d rows x %d ch, P=%d  widths: sum%d var%d affine%d\n", tokens, D,
              P, (int)Ln::sum_t::width, (int)Ln::var_t::width, (int)Ln::affine_t::width);
  int bad = compare("ln1_y", got, want);
  if (!in.empty() || !out.empty()) {
    std::printf("  drain      FAIL  in=%zu out=%zu\n", in.size(), out.size());
    bad += 1;
  } else {
    std::printf("  drain      ok    both streams empty\n");
  }
  bad += expect_reject("negctl", got, want);
  std::printf("%s\n", bad ? "FAIL" : "PASS");
  return bad;
}
