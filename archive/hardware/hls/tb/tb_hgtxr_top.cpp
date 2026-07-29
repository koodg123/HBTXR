#include "../include/common.h"
#include <cstdio>

int main() {
  static hgtxr_data_t frame[HGTXR_HEIGHT][HGTXR_WIDTH];
  static hgtxr_data_t event_pos[HGTXR_HEIGHT][HGTXR_WIDTH];
  static hgtxr_data_t event_neg[HGTXR_HEIGHT][HGTXR_WIDTH];
  hgtxr_data_t prev[HGTXR_STATE] = {128, 128, 40, 24, 1, 0};
  hgtxr_data_t out[HGTXR_STATE];
#if HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS
  static hgtxr::cyclic_transformer::HgtxrAxiWordT cyclic_weights[
      hgtxr::cyclic_transformer::HGTXR_CYCLIC_WEIGHT_TOTAL_WORDS];
#endif
  int state = -1;
#if HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS
  hgtxr_top(frame, event_pos, event_neg, prev, cyclic_weights, out, &state);
#else
  hgtxr_top(frame, event_pos, event_neg, prev, out, &state);
#endif
  std::printf("state=%d out0=%f out1=%f\n", state, (float)out[0], (float)out[1]);
  return 0;
}

