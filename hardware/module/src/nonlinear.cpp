#include "../include/common.h"

static hgtxr_data_t hgtxr_rsqrt_approx(hgtxr_data_t x) {
  if (x <= (hgtxr_data_t)0.0001) return (hgtxr_data_t)100.0;
  hgtxr_data_t y = (hgtxr_data_t)1.0;
  for (int i = 0; i < 5; ++i) {
#pragma HLS pipeline II=1
    y = (hgtxr_data_t)0.5 * y * ((hgtxr_data_t)3.0 - x * y * y);
  }
  return y;
}

static hgtxr_data_t hgtxr_exp_neg_approx(hgtxr_data_t x) {
  if (x < (hgtxr_data_t)0) x = (hgtxr_data_t)0;
  if (x > (hgtxr_data_t)8) return (hgtxr_data_t)0;
  hgtxr_data_t x2 = x * x;
  hgtxr_data_t denom = (hgtxr_data_t)1 + x + (hgtxr_data_t)0.5 * x2 + (hgtxr_data_t)0.1666667 * x2 * x;
  hgtxr_data_t recip = (hgtxr_data_t)0;
  if (denom > (hgtxr_data_t)0) {
    recip = (hgtxr_data_t)((hgtxr_data_t)1 / denom);
  }
  return recip;
}

static hgtxr_data_t hgtxr_tanh_approx(hgtxr_data_t x) {
  if (x > (hgtxr_data_t)3) return (hgtxr_data_t)1;
  if (x < (hgtxr_data_t)-3) return (hgtxr_data_t)-1;
  hgtxr_data_t x2 = x * x;
  return x * ((hgtxr_data_t)27 + x2) / ((hgtxr_data_t)27 + (hgtxr_data_t)9 * x2);
}

void layernorm_stage(token_buffer_t tokens) {
  for (int t = 0; t < HGTXR_TOKENS; ++t) {
    hgtxr_acc_t mean = 0;
    hgtxr_acc_t var = 0;
    for (int c = 0; c < HGTXR_EMBED; ++c) {
#pragma HLS pipeline II=1
      mean += tokens[t][c];
    }
    mean /= HGTXR_EMBED;
    for (int c = 0; c < HGTXR_EMBED; ++c) {
#pragma HLS pipeline II=1
      hgtxr_data_t diff = tokens[t][c] - mean;
      var += diff * diff;
    }
    hgtxr_data_t inv_std = hgtxr_rsqrt_approx((hgtxr_data_t)(var / HGTXR_EMBED) + (hgtxr_data_t)0.0001);
    for (int c = 0; c < HGTXR_EMBED; ++c) {
#pragma HLS pipeline II=1
      tokens[t][c] = clamp_data((tokens[t][c] - mean) * inv_std);
    }
  }
}

void softmax_stage(token_buffer_t tokens) {
  // Three-pass LUT-friendly softmax approximation over the token axis.
  for (int c = 0; c < HGTXR_EMBED; ++c) {
    hgtxr_data_t max_val = tokens[0][c];
    for (int t = 1; t < HGTXR_TOKENS; ++t) {
#pragma HLS pipeline II=1
      if (tokens[t][c] > max_val) max_val = tokens[t][c];
    }
    hgtxr_acc_t sum = 0;
    for (int t = 0; t < HGTXR_TOKENS; ++t) {
#pragma HLS pipeline II=1
      hgtxr_data_t e = hgtxr_exp_neg_approx(max_val - tokens[t][c]);
      tokens[t][c] = e;
      sum += e;
    }
    hgtxr_data_t recip = (hgtxr_data_t)0;
    if (sum > (hgtxr_acc_t)0) {
      recip = (hgtxr_data_t)((hgtxr_data_t)1 / (hgtxr_data_t)sum);
    }
    for (int t = 0; t < HGTXR_TOKENS; ++t) {
#pragma HLS pipeline II=1
      tokens[t][c] = clamp_data(tokens[t][c] * recip);
    }
  }
}

void gelu_stage(token_buffer_t tokens) {
  for (int t = 0; t < HGTXR_TOKENS; ++t) {
    for (int c = 0; c < HGTXR_EMBED; ++c) {
#pragma HLS pipeline II=1
      hgtxr_data_t x = tokens[t][c];
      hgtxr_data_t inner = (hgtxr_data_t)0.79788456 * (x + (hgtxr_data_t)0.044715 * x * x * x);
      tokens[t][c] = clamp_data((hgtxr_data_t)0.5 * x * ((hgtxr_data_t)1 + hgtxr_tanh_approx(inner)));
    }
  }
}
