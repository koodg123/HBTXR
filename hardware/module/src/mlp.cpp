#include "../include/common.h"

void mlp_stage(token_buffer_t tokens) {
  token_buffer_t residual;
#pragma HLS bind_storage variable=residual type=ram_2p impl=bram

  for (int t = 0; t < HGTXR_TOKENS; ++t) {
    for (int c = 0; c < HGTXR_EMBED; ++c) {
#pragma HLS pipeline II=1
      residual[t][c] = tokens[t][c];
    }
  }

  layernorm_stage(tokens);
  rmu_projection_stage(tokens);   // FC1 / expansion surrogate
  gelu_stage(tokens);
  rmu_projection_stage(tokens);   // FC2 / projection surrogate

  for (int t = 0; t < HGTXR_TOKENS; ++t) {
    for (int c = 0; c < HGTXR_EMBED; ++c) {
#pragma HLS pipeline II=1
      tokens[t][c] = clamp_data(residual[t][c] + tokens[t][c] * (hgtxr_data_t)0.5);
    }
  }
}
