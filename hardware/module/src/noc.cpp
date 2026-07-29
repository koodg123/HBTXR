#include "../include/common.h"

void noc_route_tokens(const token_buffer_t in, token_buffer_t out, int mode) {
  const int active = controller_active_tokens(mode);
  for (int t = 0; t < HGTXR_TOKENS; ++t) {
    for (int c = 0; c < HGTXR_EMBED; ++c) {
      out[t][c] = (t < active) ? in[t][c] : (hgtxr_data_t)0;
    }
  }
}
