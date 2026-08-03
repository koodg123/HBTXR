// V1 — the whole MHA core: mha_x in, mha_y out, with a probe at every stage boundary.
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

static constexpr int D = CFG::D, H = CFG::H, HD = CFG::HD, N = CFG::N;
typedef HbtxrMhaCore<CFG> Core;
static Core core;

int main(int argc, char **argv) {
  const std::string dir = argc > 1 ? argv[1] : "hardware/workspace/golden/search-a4";
  const std::vector<long long> meta = read_golden(dir, "meta");
  const int tokens = (int)meta[0];
  if (meta[1] != D || meta[2] != H || meta[3] != HD) {
    std::printf("  meta       FAIL  golden D=%lld H=%lld HD=%lld\n", meta[1], meta[2], meta[3]);
    return 1;
  }

  load_mha<CFG>(core, dir);

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
