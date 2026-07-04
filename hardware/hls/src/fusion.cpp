#include "../include/common.h"

void pool_tokens(const token_buffer_t tokens, pooled_buffer_t pooled) {
  for (int c = 0; c < HGTXR_EMBED; ++c) {
    hgtxr_acc_t acc = 0;
    for (int t = 0; t < HGTXR_TOKENS; ++t) acc += tokens[t][c];
    pooled[c] = (hgtxr_data_t)(acc / HGTXR_TOKENS);
  }
}

void fusion(const pooled_buffer_t frame, const pooled_buffer_t event, const hgtxr_data_t prev_state[HGTXR_STATE], hgtxr_data_t fused[HGTXR_EMBED * 2]) {
  hgtxr_acc_t state_mean = 0;
  for (int i = 0; i < HGTXR_STATE; ++i) state_mean += prev_state[i];
  state_mean /= HGTXR_STATE;
  for (int c = 0; c < HGTXR_EMBED; ++c) {
    fused[c] = clamp_data(frame[c] + (hgtxr_data_t)0.1 * state_mean);
    fused[HGTXR_EMBED + c] = clamp_data(event[c] + (hgtxr_data_t)0.1 * state_mean);
  }
}

