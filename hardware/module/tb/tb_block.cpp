// V1 — one whole transformer block: MHA core then MLP core, against block_y.
//
// This is the first thing that tests the SEAM between the two cores, and the answer is
// that there isn't one: the MLP core's input port IS the attention residual's output
// grid, so `mlp_x == mha_y` with no requant between them. A bridge here would be a bug,
// not an optimisation, and the golden is what says so.
#include <cstdio>
#include <string>
#include <vector>

#include "hbtxr_config.hpp"
#include "hbtxr_golden.hpp"
#include "hbtxr_load.hpp"
#include "hbtxr_probe.hpp"

using namespace hbtxr;
typedef HbtxrCfgBase CFG;
template struct hbtxr::HbtxrCfgCheck<CFG>;

static constexpr int D = CFG::D, N = CFG::N;
static HbtxrMhaCore<CFG> mha;
static HbtxrMlpCore<CFG> mlp;

int main(int argc, char **argv) {
  const std::string dir = argc > 1 ? argv[1] : "hardware/workspace/golden/search-a4";
  const int tokens = (int)read_golden(dir, "meta")[0];
  load_mha<CFG>(mha, dir);
  load_mlp<CFG>(mlp, dir);

  HbtxrProbe probe;
  const int TP = CFG::TP, P = HbtxrMhaCore<CFG>::P;
  for (const char *stage : {"ln1_x", "ln1_y", "qkv_x", "rmu_x", "rmu_y", "ln2_x", "ln2_y",
                            "fc2_y"})
    probe.expect2d(stage, read_golden(dir, stage), tokens, D, TP, P);
  for (const char *stage : {"gelu_x", "gelu_y"})
    probe.expect2d(stage, read_golden(dir, stage), tokens, CFG::F, TP, P);
  for (const char *stage : {"smu_y", "softmax_y"}) {
    const std::vector<long long> all = read_golden(dir, stage);
    probe.expect2d(stage, std::vector<long long>(all.begin(), all.begin() + tokens * tokens),
                   tokens, tokens, TP, P);
  }

  const std::vector<long long> x = read_golden(dir, "block_x");
  const std::vector<long long> mid_want = read_golden(dir, "mha_y");
  const std::vector<long long> want = read_golden(dir, "block_y");
  static CFG::act_t xin[N * D], mid[N * D], yout[N * D];
  for (int i = 0; i < tokens * D; ++i) xin[i] = x[i];

  std::printf("tb_block    %d tokens, D=%d F=%d — MHA then MLP\n", tokens, D, CFG::F);
  mha.run(xin, mid, tokens, &probe);
  // No requant between the cores: the MLP's input port is the attention residual's
  // output grid. Assert that rather than assume it.
  std::vector<long long> got_mid(tokens * D);
  for (int i = 0; i < tokens * D; ++i) got_mid[i] = (long long)mid[i];
  int bad = compare("mha_y", got_mid, mid_want);

  mlp.run(mid, yout, tokens, &probe);
  std::vector<long long> got(tokens * D);
  for (int i = 0; i < tokens * D; ++i) got[i] = (long long)yout[i];
  bad += compare("block_y", got, want);

  if (probe.failures) {
    std::printf("  probes     FAIL  %d of %d stages\n", probe.failures, probe.checked);
    bad += 1;
  } else {
    std::printf("  probes     ok    %d stages across both cores\n", probe.checked);
  }
  bad += expect_reject("negctl", got, want);
  std::printf("%s\n", bad ? "FAIL" : "PASS");
  return bad;
}
