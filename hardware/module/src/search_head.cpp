#include "../include/common.h"

static hgtxr_data_t positive_clamp(hgtxr_data_t x, hgtxr_data_t lo, hgtxr_data_t hi) {
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

void search_head(const pooled_buffer_t pooled, hgtxr_data_t out[HGTXR_SEARCH_LOGITS]) {
  hgtxr_acc_t mean = 0;
  hgtxr_acc_t spread = 0;
  hgtxr_acc_t left = 0;
  hgtxr_acc_t right = 0;
  hgtxr_acc_t angle_acc = 0;

  for (int c = 0; c < HGTXR_EMBED; ++c) {
#pragma HLS pipeline II=1
    hgtxr_data_t v = pooled[c];
    mean += v;
    spread += v * v;
    if (c < HGTXR_EMBED / 2) left += v; else right += v;
    angle_acc += v * (hgtxr_data_t)((c & 15) - 8);
  }
  mean /= HGTXR_EMBED;
  spread /= HGTXR_EMBED;
  hgtxr_data_t dx = positive_clamp((hgtxr_data_t)((right - left) / HGTXR_EMBED) * (hgtxr_data_t)8, (hgtxr_data_t)-32, (hgtxr_data_t)32);
  hgtxr_data_t dy = positive_clamp(mean * (hgtxr_data_t)4, (hgtxr_data_t)-32, (hgtxr_data_t)32);
  hgtxr_data_t size = positive_clamp((hgtxr_data_t)24 + spread * (hgtxr_data_t)0.125, (hgtxr_data_t)12, (hgtxr_data_t)72);
  hgtxr_data_t conf = positive_clamp((hgtxr_data_t)0.5 + spread * (hgtxr_data_t)0.015625, (hgtxr_data_t)0, (hgtxr_data_t)1);

  out[0] = (hgtxr_data_t)(HGTXR_WIDTH / 2) + dx;
  out[1] = (hgtxr_data_t)(HGTXR_HEIGHT / 2) + dy;
  out[2] = size;
  out[3] = positive_clamp(size * (hgtxr_data_t)0.625, (hgtxr_data_t)8, (hgtxr_data_t)56);
  out[4] = conf;
  out[5] = positive_clamp((hgtxr_data_t)(angle_acc / (HGTXR_EMBED * 64)), (hgtxr_data_t)-1, (hgtxr_data_t)1);
  out[6] = conf;
}
