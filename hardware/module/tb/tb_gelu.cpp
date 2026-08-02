// V1 — the pointwise GeLU LUT against i_ops.py:table_quantize.
#include <cstdio>
#include <string>
#include <vector>

#include "hbtxr_config.hpp"
#include "hbtxr_golden.hpp"
#include "hbtxr_lut.hpp"

using namespace hbtxr;
typedef HbtxrCfgBase CFG;
template struct hbtxr::HbtxrCfgCheck<CFG>;

int main(int argc, char **argv) {
  const std::string dir = argc > 1 ? argv[1] : "hardware/workspace/golden/search-a4";
  const std::vector<long long> sc = read_golden(dir, "gelu_scalars");
  const std::vector<long long> tb = read_golden(dir, "gelu_table");
  const std::vector<long long> x = read_golden(dir, "gelu_x");
  const std::vector<long long> want = read_golden(dir, "gelu_y");
  if (sc.size() != 3 || tb.size() != 32) {
    std::printf("  payload    FAIL  %zu scalars, %zu entries (want 3, 32)\n", sc.size(),
                tb.size());
    return 1;
  }
  const int b = (int)sc[0], s = (int)sc[1], bound = (int)sc[2];

  static CFG::nl_t table[32];
  for (int i = 0; i < 32; ++i) table[i] = tb[i];

  std::vector<long long> got;
  got.reserve(x.size());
  for (size_t i = 0; i < x.size(); ++i)
    got.push_back((long long)gelu<CFG, 32>(CFG::act_t(x[i]), table, b, s, bound));

  // The cursor never leaves [0, bound] on this payload, so the clamp is untested here.
  // Say so rather than let the pass imply otherwise.
  int outside = 0;
  for (size_t i = 0; i < x.size(); ++i) {
    const long long raw = ((long long)x[i] + b) >> s;
    outside += (raw < 0 || raw > bound);
  }
  std::printf("tb_gelu     %zu values, table[32], cursor b=%d s=%d bound=%d\n", x.size(), b,
              s, bound);
  std::printf("  clamp      %s\n",
              outside ? "exercised" : "NOT exercised by this payload (0 cursors out of range)");
  int bad = compare("gelu_y", got, want);
  bad += expect_reject("negctl", got, want);

  // So test it directly: an out-of-range cursor must saturate, not alias.
  const long long lo = (long long)gelu<CFG, 32>(CFG::act_t(0), table, -(1 << 20), 0, bound);
  const long long hi = (long long)gelu<CFG, 32>(CFG::act_t(0), table, (1 << 20), 0, bound);
  if (lo != (long long)table[0] || hi != (long long)table[bound]) {
    std::printf("  saturate   FAIL  out-of-range cursors gave %lld/%lld, want %lld/%lld\n", lo,
                hi, (long long)table[0], (long long)table[bound]);
    bad += 1;
  } else {
    std::printf("  saturate   ok    out-of-range cursors saturate to table[0]/table[bound]\n");
  }
  std::printf("%s\n", bad ? "FAIL" : "PASS");
  return bad;
}
