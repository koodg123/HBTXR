// V1 — the SMU against the block golden's Q x K^T.
//
// Three things this covers that the RMU testbench cannot: the transpose, a per-TENSOR
// requant, and the 1/sqrt(d) factor arriving folded into (M, n) rather than as a multiply.
#include <cstdio>
#include <string>
#include <vector>

#include "hbtxr_config.hpp"
#include "hbtxr_golden.hpp"
#include "hbtxr_smu.hpp"

using namespace hbtxr;
typedef HbtxrCfgBase CFG;
template struct hbtxr::HbtxrCfgCheck<CFG>;

static constexpr int HD = CFG::HD, N = CFG::N, CIP = CFG::R_CIP, COP = CFG::R_COP;
static constexpr int TP = CFG::TP;
typedef HbtxrSmu<CFG, HD, N, CIP, COP> Smu;

static Smu smu;

int main(int argc, char **argv) {
  const std::string dir = argc > 1 ? argv[1] : "hardware/workspace/golden/search-a4";
  const std::vector<long long> meta = read_golden(dir, "meta");
  const int tokens = (int)meta[0], heads = (int)meta[2];
  if (meta[3] != HD || tokens > N) {
    std::printf("  meta       FAIL  golden HD=%lld tokens=%d, config HD=%d N=%d\n", meta[3],
                tokens, HD, N);
    return 1;
  }

  const std::vector<long long> q = read_golden(dir, "smu_a");
  const std::vector<long long> k = read_golden(dir, "smu_b");
  const std::vector<long long> want = read_golden(dir, "smu_y");
  std::vector<long long> mult, shift;
  read_requant<CFG>(dir, "e04_smu_to_softmax", mult, shift);
  if (mult.size() != 1) {
    std::printf("  requant    FAIL  %zu multipliers, the score requant is per-tensor\n",
                mult.size());
    return 1;
  }

  std::vector<long long> got(heads * tokens * tokens, 0);
  hls::stream<Smu::in_beat_t> a("a"), b("b");
  hls::stream<Smu::out_beat_t> out("out");

  for (int h = 0; h < heads; ++h) {
    const int base = h * tokens * HD;
    // B first: the unit consumes all of K before it can emit a single score row.
    for (int t0 = 0; t0 < tokens; t0 += TP)
      for (int c0 = 0; c0 < HD; c0 += CIP) {
        Smu::in_beat_t v;
        for (int p = 0; p < TP; ++p)
          for (int c = 0; c < CIP; ++c) v[p * CIP + c] = k[base + (t0 + p) * HD + c0 + c];
        b.write(v);
      }
    for (int t0 = 0; t0 < tokens; t0 += TP)
      for (int c0 = 0; c0 < HD; c0 += CIP) {
        Smu::in_beat_t v;
        for (int p = 0; p < TP; ++p)
          for (int c = 0; c < CIP; ++c) v[p * CIP + c] = q[base + (t0 + p) * HD + c0 + c];
        a.write(v);
      }

    smu.run(a, b, out, tokens, tokens, HD, mult[0], shift[0]);

    for (int t0 = 0; t0 < tokens; t0 += TP)
      for (int j0 = 0; j0 < tokens; j0 += COP) {
        const Smu::out_beat_t r = out.read();
        for (int p = 0; p < TP; ++p)
          for (int c = 0; c < COP; ++c)
            got[base / HD * tokens + (t0 + p) * tokens + j0 + c] = (long long)r[p * COP + c];
      }
  }

  std::printf("tb_smu      %d heads, %dx%d over K=%d, TP=%d CIP=%d COP=%d\n", heads, tokens,
              tokens, HD, TP, CIP, COP);
  int bad = compare("smu_y", got, want);
  if (!a.empty() || !b.empty() || !out.empty()) {
    std::printf("  drain      FAIL  a=%zu b=%zu out=%zu beats left\n", a.size(), b.size(),
                out.size());
    bad += 1;
  } else {
    std::printf("  drain      ok    all streams empty\n");
  }
  bad += expect_reject("negctl", got, want);
  std::printf("%s\n", bad ? "FAIL" : "PASS");
  return bad;
}
