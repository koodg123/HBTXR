#ifndef HGTXR_CYCLIC_NORM_HPP
#define HGTXR_CYCLIC_NORM_HPP

#include "hgtxr_cyclic_math.hpp"

namespace hgtxr {
namespace cyclic_transformer {

template <int TM, int TC>
void identity_norm_tile(const HgtxrDataT in[TM][TC], HgtxrDataT out[TM][TC]) {
#pragma HLS INLINE off
  for (int i = 0; i < TM; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_SEQ
    for (int j = 0; j < TC; ++j) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_MODEL_DIM
#pragma HLS PIPELINE II=1
      out[i][j] = in[i][j];
    }
  }
}

template <int TM, int TC>
void mean_center_tile(const HgtxrDataT in[TM][TC], HgtxrDataT out[TM][TC]) {
#pragma HLS INLINE off
  HgtxrAccumT mean[TM];
#pragma HLS ARRAY_PARTITION variable=mean cyclic factor=HGTXR_PAR_ATTN dim=1

  for (int i = 0; i < TM; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_SEQ
    HgtxrAccumT sum = 0;
    for (int j = 0; j < TC; ++j) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_MODEL_DIM
#pragma HLS PIPELINE II=1
      sum += static_cast<HgtxrAccumT>(in[i][j]);
    }
    mean[i] = sum / static_cast<HgtxrAccumT>(TC);
  }

  for (int i = 0; i < TM; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_SEQ
    for (int j = 0; j < TC; ++j) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_MODEL_DIM
#pragma HLS PIPELINE II=1
      out[i][j] = static_cast<HgtxrDataT>(static_cast<HgtxrAccumT>(in[i][j]) - mean[i]);
    }
  }
}

template <int TM, int TC>
void affine_tile(const HgtxrDataT in[TM][TC],
                 const HgtxrDataT gamma[TC],
                 const HgtxrDataT beta[TC],
                 HgtxrDataT out[TM][TC]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=gamma cyclic factor=HGTXR_PAR_FFN dim=1
#pragma HLS ARRAY_PARTITION variable=beta cyclic factor=HGTXR_PAR_FFN dim=1

  for (int i = 0; i < TM; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_SEQ
    for (int j = 0; j < TC; ++j) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_MODEL_DIM
#pragma HLS PIPELINE II=1
      HgtxrAccumT y = static_cast<HgtxrAccumT>(in[i][j]) *
                      static_cast<HgtxrAccumT>(gamma[j]);
      y += static_cast<HgtxrAccumT>(beta[j]);
      out[i][j] = static_cast<HgtxrDataT>(y);
    }
  }
}

template <int TM, int TC>
void norm_tile(const HgtxrDataT in[TM][TC],
               const HgtxrDataT gamma[TC],
               const HgtxrDataT beta[TC],
               HgtxrDataT out[TM][TC],
               bool enable_mean_center,
               bool enable_affine) {
#pragma HLS INLINE off

  HgtxrDataT centered[TM][TC];
#pragma HLS ARRAY_PARTITION variable=centered cyclic factor=HGTXR_PAR_FFN dim=2

  if (enable_mean_center) {
    mean_center_tile<TM, TC>(in, centered);
  } else {
    identity_norm_tile<TM, TC>(in, centered);
  }

  if (enable_affine) {
    affine_tile<TM, TC>(centered, gamma, beta, out);
  } else {
    identity_norm_tile<TM, TC>(centered, out);
  }
}

}  // namespace cyclic_transformer
}  // namespace hgtxr

#endif  // HGTXR_CYCLIC_NORM_HPP

