#include "../include/common.h"

static hgtxr_data_t clamp_range(hgtxr_data_t x, hgtxr_data_t lo, hgtxr_data_t hi) {
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

void track_head(const hgtxr_data_t fused[HGTXR_EMBED * 2], const hgtxr_data_t prev_state[HGTXR_STATE], hgtxr_data_t out[HGTXR_STATE]) {
  hgtxr_acc_t frame_mean = 0;
  hgtxr_acc_t event_mean = 0;
  hgtxr_acc_t frame_energy = 0;
  hgtxr_acc_t event_energy = 0;
  hgtxr_acc_t cross = 0;

  for (int c = 0; c < HGTXR_EMBED; ++c) {
#pragma HLS pipeline II=1
    hgtxr_data_t f = fused[c];
    hgtxr_data_t e = fused[HGTXR_EMBED + c];
    frame_mean += f;
    event_mean += e;
    frame_energy += f * f;
    event_energy += e * e;
    cross += f * e;
  }
  frame_mean /= HGTXR_EMBED;
  event_mean /= HGTXR_EMBED;
  frame_energy /= HGTXR_EMBED;
  event_energy /= HGTXR_EMBED;

  hgtxr_data_t residual_x = clamp_range((event_mean - frame_mean) * (hgtxr_data_t)2.0, (hgtxr_data_t)-12, (hgtxr_data_t)12);
  hgtxr_data_t residual_y = clamp_range((event_energy - frame_energy) * (hgtxr_data_t)0.0625, (hgtxr_data_t)-12, (hgtxr_data_t)12);
  hgtxr_data_t residual_s = clamp_range((event_energy + frame_energy) * (hgtxr_data_t)0.001953125, (hgtxr_data_t)-4, (hgtxr_data_t)4);
  hgtxr_data_t similarity = clamp_range(cross * (hgtxr_data_t)0.000244140625, (hgtxr_data_t)0, (hgtxr_data_t)1);

  out[0] = clamp_range(prev_state[0] + residual_x, (hgtxr_data_t)0, (hgtxr_data_t)(HGTXR_WIDTH - 1));
  out[1] = clamp_range(prev_state[1] + residual_y, (hgtxr_data_t)0, (hgtxr_data_t)(HGTXR_HEIGHT - 1));
  out[2] = clamp_range(prev_state[2] + residual_s, (hgtxr_data_t)8, (hgtxr_data_t)96);
  out[3] = clamp_range(prev_state[3] + residual_s * (hgtxr_data_t)0.625, (hgtxr_data_t)8, (hgtxr_data_t)72);
  out[4] = clamp_range((hgtxr_data_t)0.45 + similarity, (hgtxr_data_t)0, (hgtxr_data_t)1);
  out[5] = clamp_range(prev_state[5] + (event_mean - frame_mean) * (hgtxr_data_t)0.03125, (hgtxr_data_t)-1, (hgtxr_data_t)1);
}
