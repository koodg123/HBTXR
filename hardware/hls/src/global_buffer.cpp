#include "../include/common.h"

void global_buffer_write_search(const token_buffer_t in, token_buffer_t out) {
  for (int t = 0; t < HGTXR_TOKENS; ++t)
    for (int c = 0; c < HGTXR_EMBED; ++c) out[t][c] = in[t][c];
}

void global_buffer_write_track(const token_buffer_t in, token_buffer_t out) {
  for (int t = 0; t < HGTXR_TOKENS; ++t)
    for (int c = 0; c < HGTXR_EMBED; ++c) out[t][c] = (t < HGTXR_TRACK_TOKENS) ? in[t][c] : (hgtxr_data_t)0;
}
