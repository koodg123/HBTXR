#ifndef HGTXR_CYCLIC_ATTENTION_HPP
#define HGTXR_CYCLIC_ATTENTION_HPP

#include "hgtxr_cyclic_mac.hpp"
#include "hgtxr_cyclic_math.hpp"

namespace hgtxr {
namespace cyclic_transformer {

template <int TQ, int TK, int TD>
void attention_score_tile(const HgtxrDataT q[TQ][TD],
                          const HgtxrDataT k[TK][TD],
                          HgtxrScoreT scores[TQ][TK],
                          ap_int<8> score_scale_shift) {
#pragma HLS INLINE off
  for (int i = 0; i < TQ; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_SEQ
    for (int j = 0; j < TK; ++j) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_SEQ
      HgtxrAccumT acc = 0;
      for (int d = 0; d < TD; ++d) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_HEAD_DIM
#pragma HLS PIPELINE II=1
        acc += static_cast<HgtxrAccumT>(q[i][d]) *
               static_cast<HgtxrAccumT>(k[j][d]);
      }
      if (score_scale_shift > 0) {
        acc = acc >> score_scale_shift;
      } else if (score_scale_shift < 0) {
        acc = acc << (-score_scale_shift);
      }
      scores[i][j] = static_cast<HgtxrScoreT>(acc);
    }
  }
}

template <int TQ, int TK, int TD>
void attention_value_tile(const HgtxrDataT probs[TQ][TK],
                          const HgtxrDataT v[TK][TD],
                          HgtxrDataT out[TQ][TD],
                          ap_int<8> value_scale_shift) {
#pragma HLS INLINE off
  HgtxrAccumT acc[TQ][TD];
#pragma HLS ARRAY_PARTITION variable=acc cyclic factor=HGTXR_PAR_ATTN dim=2

  for (int i = 0; i < TQ; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_SEQ
    for (int d = 0; d < TD; ++d) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_HEAD_DIM
#pragma HLS PIPELINE II=1
      acc[i][d] = 0;
    }
  }

  for (int j = 0; j < TK; ++j) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_SEQ
    for (int i = 0; i < TQ; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_SEQ
      for (int d = 0; d < TD; ++d) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_HEAD_DIM
#pragma HLS PIPELINE II=1
        acc[i][d] += static_cast<HgtxrAccumT>(probs[i][j]) *
                     static_cast<HgtxrAccumT>(v[j][d]);
      }
    }
  }

  for (int i = 0; i < TQ; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_SEQ
    for (int d = 0; d < TD; ++d) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_HEAD_DIM
#pragma HLS PIPELINE II=1
      HgtxrAccumT value = acc[i][d];
      if (value_scale_shift > 0) {
        value = value >> value_scale_shift;
      } else if (value_scale_shift < 0) {
        value = value << (-value_scale_shift);
      }
      out[i][d] = static_cast<HgtxrDataT>(value);
    }
  }
}

template <int TQ, int TK, int TD>
void attention_tile(const HgtxrDataT q[TQ][TD],
                    const HgtxrDataT k[TK][TD],
                    const HgtxrDataT v[TK][TD],
                    HgtxrDataT out[TQ][TD],
                    ap_int<8> score_scale_shift,
                    ap_int<8> value_scale_shift) {
#pragma HLS INLINE off

  HgtxrScoreT scores[TQ][TK];
  HgtxrScoreT row_max[TQ];
  HgtxrScoreT probs_score[TQ][TK];
  HgtxrAccumT row_sum[TQ];
  HgtxrDataT probs[TQ][TK];

#pragma HLS ARRAY_PARTITION variable=scores cyclic factor=HGTXR_PAR_ATTN dim=2
#pragma HLS ARRAY_PARTITION variable=probs_score cyclic factor=HGTXR_PAR_ATTN dim=2
#pragma HLS ARRAY_PARTITION variable=probs cyclic factor=HGTXR_PAR_ATTN dim=2

  attention_score_tile<TQ, TK, TD>(q, k, scores, score_scale_shift);
  row_max_tile<TQ, TK>(scores, row_max);
  softmax_approx_tile<TQ, TK>(scores, row_max, probs_score, row_sum);
  softmax_normalize_tile<TQ, TK>(probs_score, row_sum, probs);
  attention_value_tile<TQ, TK, TD>(probs, v, out, value_scale_shift);
}

}  // namespace cyclic_transformer
}  // namespace hgtxr

#endif  // HGTXR_CYCLIC_ATTENTION_HPP

