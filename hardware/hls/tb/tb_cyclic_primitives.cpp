#include "hgtxr_cyclic_attention.hpp"
#include "hgtxr_cyclic_mac.hpp"
#include "hgtxr_cyclic_mlp.hpp"
#include "hgtxr_cyclic_scheduler.hpp"

#include <cstdio>

using namespace hgtxr::cyclic_transformer;

static int check(bool cond, const char *name) {
  if (!cond) {
    std::printf("FAIL: %s\n", name);
    return 1;
  }
  std::printf("PASS: %s\n", name);
  return 0;
}

int main() {
  int failures = 0;

  HgtxrDataT lhs[2][3] = {
      {1, 2, 3},
      {4, 5, 6},
  };
  HgtxrDataT rhs[3][2] = {
      {1, 0},
      {0, 1},
      {1, 1},
  };
  HgtxrAccumT gemm_out[2][2];
  mac_tile<2, 2, 3>(lhs, rhs, gemm_out, false);
  failures += check(gemm_out[0][0] == 4, "mac 0,0");
  failures += check(gemm_out[0][1] == 5, "mac 0,1");
  failures += check(gemm_out[1][0] == 10, "mac 1,0");
  failures += check(gemm_out[1][1] == 11, "mac 1,1");

  HgtxrSchedulerLimits lim;
  lim.tiles_per_layer = 1;
  lim.kv_tiles_per_attention = 1;
  lim.heads_per_layer = 1;
  lim.num_layers = 1;

  HgtxrSchedulerCounters ctr;
  ctr.tile_idx = 0;
  ctr.kv_tile_idx = 0;
  ctr.head_idx = 0;
  ctr.layer_idx = 0;

  HgtxrSchedulerState state = HGTXR_SCHED_IDLE;
  state = hgtxr_scheduler_next_state(state, ctr, lim);
  failures += check(state == HGTXR_SCHED_LOAD_X_TILE, "scheduler starts load");

  HgtxrDataT q[2][2] = {
      {1, 0},
      {0, 1},
  };
  HgtxrDataT k[2][2] = {
      {1, 0},
      {0, 1},
  };
  HgtxrDataT v[2][2] = {
      {1, 2},
      {3, 4},
  };
  HgtxrDataT attn_out[2][2];
  attention_tile<2, 2, 2>(q, k, v, attn_out, 0, 0);
  failures += check(attn_out[0][0] >= 1, "attention output finite 0,0");
  failures += check(attn_out[1][1] >= 2, "attention output finite 1,1");

  HgtxrDataT x[2][2] = {
      {1, -1},
      {0, 1},
  };
  HgtxrDataT w1[2][2] = {
      {1, 0},
      {0, 1},
  };
  HgtxrDataT w2[2][2] = {
      {1, 0},
      {0, 1},
  };
  HgtxrDataT mlp_out[2][2];
  mlp_tile<2, 2, 2>(x, w1, w2, x, mlp_out, 0, 0, true);
  failures += check(mlp_out[0][0] >= x[0][0], "mlp residual 0,0");
  failures += check(mlp_out[1][1] >= x[1][1], "mlp residual 1,1");

  if (failures != 0) {
    std::printf("cyclic primitive smoke failed: %d\n", failures);
    return 1;
  }
  std::printf("cyclic primitive smoke passed\n");
  return 0;
}

