// V1 — integer softmax against i_ops.py:softmax_quantize (14 scalars, two recip segments).
#include <cstdio>
#include <string>
#include <vector>

#include "hbtxr_config.hpp"
#include "hbtxr_golden.hpp"
#include "hbtxr_softmax.hpp"

using namespace hbtxr;
typedef HbtxrCfgBase CFG;
template struct hbtxr::HbtxrCfgCheck<CFG>;

static constexpr int N = CFG::N, P = CFG::NL_P, TP = CFG::TP;
typedef HbtxrSoftmax<CFG, N, P> Sm;
static Sm sm;

int main(int argc, char **argv) {
  const std::string dir = argc > 1 ? argv[1] : "hardware/workspace/golden/search-a4";
  const std::vector<long long> meta = read_golden(dir, "meta");
  const int tokens = (int)meta[0], heads = (int)meta[2];
  const std::vector<long long> sc = read_golden(dir, "softmax_scalars");
  const std::vector<long long> et = read_golden(dir, "softmax_exp_table");
  const std::vector<long long> r1 = read_golden(dir, "softmax_recip_table_one");
  const std::vector<long long> r2 = read_golden(dir, "softmax_recip_table_two");
  const std::vector<long long> x = read_golden(dir, "softmax_x");
  const std::vector<long long> want = read_golden(dir, "softmax_y");
  if (sc.size() != 14 || et.size() != 32 || r1.size() != 64 || r2.size() != 64) {
    std::printf("  payload    FAIL  %zu scalars, tables %zu/%zu/%zu\n", sc.size(), et.size(),
                r1.size(), r2.size());
    return 1;
  }

  // The tables are UNSIGNED. exp's largest entry is its numerator, 1<<15 = 32768, and
  // ap_int<16> tops out at 32767 -- loading it signed would wrap to -32768 silently.
  for (int i = 0; i < 32; ++i) {
    sm.exp_table[i] = et[i];
    if ((long long)sm.exp_table[i] != et[i]) {
      std::printf("  payload    FAIL  exp_table[%d]=%lld does not fit nlu_t\n", i, et[i]);
      return 1;
    }
  }
  for (int i = 0; i < 64; ++i) {
    sm.recip_one[i] = r1[i];
    sm.recip_two[i] = r2[i];
  }
  sm.b1 = sc[0];   sm.s1 = sc[1];   sm.bound1 = sc[2];
  sm.b2_one = sc[3]; sm.s2_one = sc[4]; sm.bound2_one = sc[5];
  sm.b3_one = sc[6]; sm.s3_one = sc[7];
  sm.b2_two = sc[8]; sm.s2_two = sc[9]; sm.bound2_two = sc[10];
  sm.b3_two = sc[11]; sm.s3_two = sc[12]; sm.clamp_bits = sc[13];

  const int rows = heads * tokens, cols = tokens;
  hls::stream<Sm::in_beat_t> in("in");
  hls::stream<Sm::out_beat_t> out("out");
  for (int r0 = 0; r0 < rows; r0 += TP)
    for (int j0 = 0; j0 < cols; j0 += P) {
      Sm::in_beat_t v;
      for (int p = 0; p < TP; ++p)
        for (int c = 0; c < P; ++c) v[p * P + c] = x[(r0 + p) * cols + j0 + c];
      in.write(v);
    }

  sm.run(in, out, rows, cols);

  std::vector<long long> got(rows * cols, 0);
  for (int r0 = 0; r0 < rows; r0 += TP)
    for (int j0 = 0; j0 < cols; j0 += P) {
      const Sm::out_beat_t v = out.read();
      for (int p = 0; p < TP; ++p)
        for (int c = 0; c < P; ++c) got[(r0 + p) * cols + j0 + c] = (long long)v[p * P + c];
    }

  // Both reciprocal segments must actually be taken, or half the kernel is untested.
  int seg2 = 0;
  for (int r = 0; r < rows; ++r) {
    long long mx = x[(long long)r * cols], a = 0;
    for (int j = 1; j < cols; ++j) mx = x[(long long)r * cols + j] > mx ? x[(long long)r * cols + j] : mx;
    for (int j = 0; j < cols; ++j) {
      long long raw = ((mx - x[(long long)r * cols + j]) + sc[0]) >> sc[1];
      a += et[raw < 0 ? 0 : (raw > sc[2] ? sc[2] : raw)];
    }
    seg2 += (((a + sc[3]) >> sc[4]) > sc[5]);
  }

  std::printf("tb_softmax  %d rows x %d cols, P=%d  acc%db rel%db\n", rows, cols, P,
              (int)Sm::acc_t::width, (int)Sm::rel_t::width);
  std::printf("  segments   %s  (%d/%d rows take segment 2)\n",
              (seg2 > 0 && seg2 < rows) ? "both taken" : "ONLY ONE TAKEN", seg2, rows);
  int bad = compare("softmax_y", got, want);
  if (seg2 == 0 || seg2 == rows) {
    std::printf("  segments   FAIL  one reciprocal segment is never exercised\n");
    bad += 1;
  }
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
