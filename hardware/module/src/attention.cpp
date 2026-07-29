#include "../include/common.h"

void attention_stage(token_buffer_t tokens) {
  token_buffer_t residual;
#pragma HLS bind_storage variable=residual type=ram_2p impl=bram

  for (int t = 0; t < HGTXR_TOKENS; ++t) {
    for (int c = 0; c < HGTXR_EMBED; ++c) {
#pragma HLS pipeline II=1
      residual[t][c] = tokens[t][c];
    }
  }

  rmu_projection_stage(tokens);   // Q/K/V projection surrogate
  smu_relation_stage(tokens);     // QK softmax and AV relation surrogate
  rmu_projection_stage(tokens);   // output projection surrogate

  for (int t = 0; t < HGTXR_TOKENS; ++t) {
    for (int c = 0; c < HGTXR_EMBED; ++c) {
#pragma HLS pipeline II=1
      tokens[t][c] = clamp_data(residual[t][c] + tokens[t][c] * (hgtxr_data_t)0.5);
    }
  }
}
