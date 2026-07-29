#include "../include/common.h"

static hgtxr_acc_t rmu_smu_dsp_mul(hgtxr_data_t lhs, hgtxr_data_t rhs) {
#pragma HLS inline
  hgtxr_acc_t product;
#pragma HLS bind_op variable=product op=mul impl=dsp
  product = (hgtxr_acc_t)lhs * (hgtxr_acc_t)rhs;
  return product;
}

static hgtxr_acc_t rmu_smu_dsp_mul_acc(hgtxr_acc_t lhs, hgtxr_data_t rhs) {
#pragma HLS inline
  hgtxr_acc_t product;
#pragma HLS bind_op variable=product op=mul impl=dsp
  product = lhs * (hgtxr_acc_t)rhs;
  return product;
}

static hgtxr_data_t channel_gain(int c) {
  int bucket = c & 7;
  return (hgtxr_data_t)0.875 + (hgtxr_data_t)0.03125 * bucket;
}

static hgtxr_data_t token_gain(int t) {
  int bucket = t & 15;
  return (hgtxr_data_t)0.9375 + (hgtxr_data_t)0.0078125 * bucket;
}

void rmu_projection_stage(token_buffer_t tokens) {
  // RMU-like projection: LN followed by deterministic low-rank channel mixing.
  // Final trained weights are expected to replace these generated coefficients.
  layernorm_stage(tokens);
  for (int t = 0; t < HGTXR_TOKENS; ++t) {
    hgtxr_acc_t token_mean = 0;
    for (int c = 0; c < HGTXR_EMBED; ++c) {
#pragma HLS pipeline II=1
      token_mean += tokens[t][c];
    }
    token_mean /= HGTXR_EMBED;
    hgtxr_data_t prev = tokens[t][0];
    for (int c = 0; c < HGTXR_EMBED; ++c) {
#pragma HLS pipeline II=1
      hgtxr_data_t cur = tokens[t][c];
      hgtxr_data_t next = tokens[t][(c + 1) == HGTXR_EMBED ? 0 : c + 1];
      hgtxr_acc_t neighbor = (hgtxr_acc_t)prev + (hgtxr_acc_t)next;
      hgtxr_acc_t mixed =
          rmu_smu_dsp_mul(cur, channel_gain(c)) +
          rmu_smu_dsp_mul_acc(neighbor, (hgtxr_data_t)0.03125) +
          rmu_smu_dsp_mul_acc(token_mean, (hgtxr_data_t)0.0625);
      tokens[t][c] = clamp_data(mixed);
      prev = cur;
    }
  }
}

void smu_relation_stage(token_buffer_t tokens) {
  // SMU-like token relation stage. It forms a single attention distribution
  // from token saliency, accumulates context, and injects it into all tokens.
  hgtxr_data_t score[HGTXR_TOKENS];
  hgtxr_data_t prob[HGTXR_TOKENS];
#pragma HLS bind_storage variable=score type=ram_1p impl=lutram
#pragma HLS bind_storage variable=prob type=ram_1p impl=lutram

  hgtxr_data_t max_score = (hgtxr_data_t)-32768;
  for (int t = 0; t < HGTXR_TOKENS; ++t) {
    hgtxr_acc_t acc = 0;
    for (int c = 0; c < HGTXR_EMBED; c += 8) {
#pragma HLS pipeline II=1
      acc += rmu_smu_dsp_mul(tokens[t][c], channel_gain(c));
    }
    score[t] = (hgtxr_data_t)(acc / (HGTXR_EMBED / 8));
    if (score[t] > max_score) max_score = score[t];
  }

  hgtxr_acc_t sum_score = 0;
  for (int t = 0; t < HGTXR_TOKENS; ++t) {
#pragma HLS pipeline II=1
    hgtxr_data_t centered = score[t] - max_score;
    hgtxr_data_t relu_exp = (hgtxr_data_t)0.00390625;
    if (centered > (hgtxr_data_t)-1) {
      relu_exp = (hgtxr_data_t)((hgtxr_data_t)1 + centered +
                                rmu_smu_dsp_mul(centered, centered) *
                                    (hgtxr_data_t)0.5);
    }
    prob[t] = relu_exp;
    sum_score += relu_exp;
  }
  hgtxr_data_t recip = (hgtxr_data_t)0;
  if (sum_score > (hgtxr_acc_t)0) {
    recip = (hgtxr_data_t)((hgtxr_data_t)1 / (hgtxr_data_t)sum_score);
  }

  for (int c = 0; c < HGTXR_EMBED; ++c) {
    hgtxr_acc_t context = 0;
    for (int t = 0; t < HGTXR_TOKENS; ++t) {
#pragma HLS pipeline II=1
      context += rmu_smu_dsp_mul_acc(rmu_smu_dsp_mul(tokens[t][c], prob[t]), recip);
    }
    for (int t = 0; t < HGTXR_TOKENS; ++t) {
#pragma HLS pipeline II=1
      hgtxr_acc_t out =
          (hgtxr_acc_t)tokens[t][c] +
          rmu_smu_dsp_mul_acc(context, (hgtxr_data_t)0.25) +
          rmu_smu_dsp_mul_acc(rmu_smu_dsp_mul(score[t], token_gain(t)),
                              (hgtxr_data_t)0.03125);
      tokens[t][c] = clamp_data(out);
    }
  }
}
