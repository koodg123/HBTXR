#include "../include/common.h"

void event_patch_embed(const hgtxr_data_t event_pos[HGTXR_HEIGHT][HGTXR_WIDTH], const hgtxr_data_t event_neg[HGTXR_HEIGHT][HGTXR_WIDTH], token_buffer_t tokens) {
  for (int gy = 0; gy < HGTXR_GRID_H; ++gy) {
    for (int gx = 0; gx < HGTXR_GRID_W; ++gx) {
      hgtxr_acc_t sum = 0;
      for (int py = 0; py < HGTXR_PATCH; ++py) {
        for (int px = 0; px < HGTXR_PATCH; ++px) {
          int y = gy * HGTXR_PATCH + py;
          int x = gx * HGTXR_PATCH + px;
          sum += event_pos[y][x] - event_neg[y][x];
        }
      }
      int token = gy * HGTXR_GRID_W + gx;
      hgtxr_data_t mean = (hgtxr_data_t)(sum / (HGTXR_PATCH * HGTXR_PATCH));
      for (int c = 0; c < HGTXR_EMBED; ++c) tokens[token][c] = mean;
    }
  }
}

