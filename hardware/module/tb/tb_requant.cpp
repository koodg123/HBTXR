// V1 — the requant primitive against algorithm/quantization/i_ops.py:requant.
//
// Every matmul unit ends in this op, so it gets a testbench of its own rather than being
// covered incidentally by the RMU golden. That golden only presents ratios near 1, never
// a tie, and rarely saturates.
#include <cstdio>
#include <string>
#include <vector>

#include "hbtxr_golden.hpp"
#include "hbtxr_requant.hpp"

using namespace hbtxr;
typedef HbtxrCfgBase CFG;
template struct hbtxr::HbtxrCfgCheck<CFG>;

int main(int argc, char **argv) {
  const std::string dir = argc > 1 ? argv[1] : "hardware/workspace/golden/requant";
  const std::vector<long long> meta = read_golden(dir, "meta");
  const std::vector<long long> acc = read_golden(dir, "acc");
  const std::vector<long long> want = read_golden(dir, "want");
  std::vector<long long> mult, shift;
  read_requant<CFG>(dir, "edge", mult, shift);

  if (meta[1] != CFG::acc_t::width || meta[2] != CFG::REQ_M_BITS || meta[3] != CFG::REQ_N_MAX) {
    std::printf("  meta       FAIL  golden is acc%lld/M%lld/n%lld, config is acc%d/M%d/n%d\n",
                meta[1], meta[2], meta[3], (int)CFG::acc_t::width, CFG::REQ_M_BITS,
                CFG::REQ_N_MAX);
    return 1;
  }

  std::vector<long long> got;
  got.reserve(acc.size());
  for (size_t i = 0; i < acc.size(); ++i)
    got.push_back((long long)requant<CFG, CFG::act_t>(CFG::acc_t(acc[i]), mult[i], shift[i]));

  std::printf("tb_requant  %zu cases\n", acc.size());
  int bad = compare("requant", got, want);
  bad += expect_reject("negctl", got, want);
  std::printf("%s\n", bad ? "FAIL" : "PASS");
  return bad;
}
