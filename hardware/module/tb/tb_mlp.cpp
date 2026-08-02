// V1 — the whole MLP core: mlp_x in, mlp_y out, with a probe at every seam.
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

static constexpr int D = CFG::D, F = CFG::F, N = CFG::N;
typedef HbtxrMlpCore<CFG> Core;
static Core core;

int main(int argc, char **argv) {
  const std::string dir = argc > 1 ? argv[1] : "hardware/workspace/golden/search-a4";
  const std::vector<long long> meta = read_golden(dir, "meta");
  const int tokens = (int)meta[0];
  if (meta[1] != D || meta[4] != F) {
    std::printf("  meta       FAIL  golden D=%lld F=%lld, config D=%d F=%d\n", meta[1],
                meta[4], D, F);
    return 1;
  }

  load_mlp<CFG>(core, dir);

  HbtxrProbe probe;
  const int TP = CFG::TP, P = Core::P;
  for (const char *stage : {"ln2_x", "ln2_y", "fc2_y"})
    probe.expect2d(stage, read_golden(dir, stage), tokens, D, TP, P);
  for (const char *stage : {"gelu_x", "gelu_y"})
    probe.expect2d(stage, read_golden(dir, stage), tokens, F, TP, P);

  const std::vector<long long> x = read_golden(dir, "mlp_x");
  const std::vector<long long> want = read_golden(dir, "mlp_y");
  static CFG::act_t xin[N * D], yout[N * D];
  for (int i = 0; i < tokens * D; ++i) xin[i] = x[i];

  std::printf("tb_mlp      %d tokens, D=%d F=%d, lanes=%d\n", tokens, D, F, Core::LANES);
  core.run(xin, yout, tokens, &probe);

  std::vector<long long> got(tokens * D);
  for (int i = 0; i < tokens * D; ++i) got[i] = (long long)yout[i];
  int bad = compare("mlp_y", got, want);
  if (probe.failures) {
    std::printf("  probes     FAIL  %d of %d stages\n", probe.failures, probe.checked);
    bad += 1;
  } else {
    std::printf("  probes     ok    %d stages\n", probe.checked);
  }
  bad += expect_reject("negctl", got, want);

  // The probe must localize, or it is decoration. Break FC1's output grid and require
  // exactly one probe to fire — the refill contains the rest.
  {
    std::printf("  -- fault injection: e10_ln2_to_fc1 multiplier x2 --\n");
    HbtxrProbe p2 = probe;
    p2.failures = p2.checked = 0;
    core.e10_ln2_to_fc1.mult *= 2;
    core.run(xin, yout, tokens, &p2);
    core.e10_ln2_to_fc1.mult /= 2;
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
