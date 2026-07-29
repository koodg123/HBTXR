#include "hgtxr_cyclic_attention.hpp"

#include <cmath>
#include <cstdio>

using namespace hgtxr::cyclic_transformer;

namespace {

static constexpr int TQ = 3;
static constexpr int TK = 4;
static constexpr int TD = HGTXR_CYCLIC_HEAD_DIM;

static int clamp_int(int value, int low, int high) {
  if (value < low) {
    return low;
  }
  if (value > high) {
    return high;
  }
  return value;
}

static HgtxrScoreT ref_exp_lut(HgtxrScoreT shifted_score) {
  static const HgtxrScoreT table[16] = {
      static_cast<HgtxrScoreT>(1.000000), static_cast<HgtxrScoreT>(0.586646),
      static_cast<HgtxrScoreT>(0.344154), static_cast<HgtxrScoreT>(0.201897),
      static_cast<HgtxrScoreT>(0.118442), static_cast<HgtxrScoreT>(0.069483),
      static_cast<HgtxrScoreT>(0.040762), static_cast<HgtxrScoreT>(0.023916),
      static_cast<HgtxrScoreT>(0.014025), static_cast<HgtxrScoreT>(0.008230),
      static_cast<HgtxrScoreT>(0.004827), static_cast<HgtxrScoreT>(0.002831),
      static_cast<HgtxrScoreT>(0.001661), static_cast<HgtxrScoreT>(0.000974),
      static_cast<HgtxrScoreT>(0.000572), static_cast<HgtxrScoreT>(0.000335)};
  HgtxrScoreT opposite_delta = 0;
  if (shifted_score < 0) {
    opposite_delta = static_cast<HgtxrScoreT>(-shifted_score);
  }
  HgtxrScoreT scaled = opposite_delta * static_cast<HgtxrScoreT>(1.875);
  int cursor = clamp_int(static_cast<int>(scaled), 0, 15);
  return table[cursor];
}

template <int Shift>
static void reference_attention(const HgtxrDataT q[TQ][TD],
                                const HgtxrDataT k[TK][TD],
                                const HgtxrDataT v[TK][TD],
                                HgtxrDataT out[TQ][TD]) {
  HgtxrScoreT scores[TQ][TK];
  HgtxrScoreT row_max[TQ];
  HgtxrScoreT probs_score[TQ][TK];
  HgtxrAccumT row_sum[TQ];
  HgtxrDataT probs[TQ][TK];

  for (int i = 0; i < TQ; ++i) {
    for (int j = 0; j < TK; ++j) {
      HgtxrAccumT acc = 0;
      for (int d = 0; d < TD; ++d) {
        acc += static_cast<HgtxrAccumT>(q[i][d]) *
               static_cast<HgtxrAccumT>(k[j][d]);
      }
      if (Shift > 0) {
        acc = acc >> Shift;
      } else if (Shift < 0) {
        acc = acc << (-Shift);
      }
      scores[i][j] = static_cast<HgtxrScoreT>(acc);
    }
  }

  for (int i = 0; i < TQ; ++i) {
    row_max[i] = scores[i][0];
    for (int j = 1; j < TK; ++j) {
      if (scores[i][j] > row_max[i]) {
        row_max[i] = scores[i][j];
      }
    }
  }

  for (int i = 0; i < TQ; ++i) {
    HgtxrAccumT sum = 0;
    for (int j = 0; j < TK; ++j) {
      probs_score[i][j] = ref_exp_lut(scores[i][j] - row_max[i]);
      sum += static_cast<HgtxrAccumT>(probs_score[i][j]);
    }
    row_sum[i] = sum;
  }

  for (int i = 0; i < TQ; ++i) {
    HgtxrAccumT denom =
        row_sum[i] == 0 ? static_cast<HgtxrAccumT>(1) : row_sum[i];
    for (int j = 0; j < TK; ++j) {
      probs[i][j] = static_cast<HgtxrDataT>(
          static_cast<HgtxrAccumT>(probs_score[i][j]) / denom);
    }
  }

  for (int i = 0; i < TQ; ++i) {
    for (int d = 0; d < TD; ++d) {
      HgtxrAccumT acc = 0;
      for (int j = 0; j < TK; ++j) {
        acc += static_cast<HgtxrAccumT>(probs[i][j]) *
               static_cast<HgtxrAccumT>(v[j][d]);
      }
      out[i][d] = static_cast<HgtxrDataT>(acc);
    }
  }
}

static void fill_inputs(HgtxrDataT q[TQ][TD],
                        HgtxrDataT k[TK][TD],
                        HgtxrDataT v[TK][TD]) {
  for (int i = 0; i < TQ; ++i) {
    for (int d = 0; d < TD; ++d) {
      int raw = ((i + 2) * (d % 9) + (d / 7)) % 11;
      q[i][d] = static_cast<HgtxrDataT>((raw - 5) / 8.0);
    }
  }
  for (int j = 0; j < TK; ++j) {
    for (int d = 0; d < TD; ++d) {
      int raw = ((j + 3) * ((d + 2) % 13) + d / 5) % 15;
      k[j][d] = static_cast<HgtxrDataT>((raw - 7) / 8.0);
      int vraw = ((j + 1) * (d % 7) + (d / 3)) % 9;
      v[j][d] = static_cast<HgtxrDataT>((vraw - 4) / 4.0);
    }
  }
}

static double abs_diff(HgtxrDataT lhs, HgtxrDataT rhs) {
  return std::fabs(static_cast<double>(lhs) - static_cast<double>(rhs));
}

}  // namespace

int main() {
  static_assert(HGTXR_CYCLIC_HEAD_DIM == 64,
                "head attention vector test expects DeiT/HG-PIPE head_dim=64");
  static_assert(HGTXR_TILE_CHANNELS == HGTXR_CYCLIC_HEAD_DIM,
                "head attention vector test expects one tile per head");
  static_assert(HGTXR_CYCLIC_ATTN_SCORE_SCALE_SHIFT == 3,
                "head attention vector test expects 1/sqrt(64) score shift");

  HgtxrDataT q[TQ][TD];
  HgtxrDataT k[TK][TD];
  HgtxrDataT v[TK][TD];
  HgtxrDataT hw[TQ][TD];
  HgtxrDataT ref[TQ][TD];
  HgtxrDataT no_scale[TQ][TD];
  fill_inputs(q, k, v);

  attention_tile<TQ, TK, TD>(q, k, v, hw, HGTXR_CYCLIC_ATTN_SCORE_SCALE_SHIFT, 0);
  reference_attention<HGTXR_CYCLIC_ATTN_SCORE_SCALE_SHIFT>(q, k, v, ref);
  attention_tile<TQ, TK, TD>(q, k, v, no_scale, 0, 0);

  double max_ref_diff = 0.0;
  double max_scale_effect = 0.0;
  for (int i = 0; i < TQ; ++i) {
    for (int d = 0; d < TD; ++d) {
      double ref_diff = abs_diff(hw[i][d], ref[i][d]);
      if (ref_diff > max_ref_diff) {
        max_ref_diff = ref_diff;
      }
      double scale_effect = abs_diff(hw[i][d], no_scale[i][d]);
      if (scale_effect > max_scale_effect) {
        max_scale_effect = scale_effect;
      }
    }
  }

  std::printf("head_attention_ref_max_abs_diff=%.8f\n", max_ref_diff);
  std::printf("head_attention_scale_effect_max_abs_diff=%.8f\n",
              max_scale_effect);
  std::printf("head_attention_score_shift=%d\n",
              HGTXR_CYCLIC_ATTN_SCORE_SCALE_SHIFT);

  if (max_ref_diff > 0.0001) {
    std::printf("FAIL: head attention output differs from reference\n");
    return 1;
  }
  if (max_scale_effect <= 0.01) {
    std::printf("FAIL: score scale shift is not observable in output\n");
    return 1;
  }
  std::printf("head attention fixed-point vector comparison passed\n");
  return 0;
}
