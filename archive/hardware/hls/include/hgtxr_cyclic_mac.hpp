#ifndef HGTXR_CYCLIC_MAC_HPP
#define HGTXR_CYCLIC_MAC_HPP

#include "hgtxr_cyclic_transformer_params.hpp"

namespace hgtxr {
namespace cyclic_transformer {

template <int TM, int TN, int TK>
void zero_tile(HgtxrAccumT out[TM][TN]) {
#pragma HLS INLINE
  for (int i = 0; i < TM; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_SEQ
    for (int j = 0; j < TN; ++j) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_MODEL_DIM
#pragma HLS PIPELINE II=1
      out[i][j] = 0;
    }
  }
}

template <int TM, int TN, int TK>
void mac_tile(const HgtxrDataT lhs[TM][TK],
              const HgtxrDataT rhs[TK][TN],
              HgtxrAccumT out[TM][TN],
              bool accumulate) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=lhs cyclic factor=HGTXR_PAR_QK dim=2
#pragma HLS ARRAY_PARTITION variable=rhs cyclic factor=HGTXR_PAR_QK dim=1
#pragma HLS ARRAY_PARTITION variable=out cyclic factor=HGTXR_PAR_FFN dim=2

  if (!accumulate) {
    zero_tile<TM, TN, TK>(out);
  }

  for (int k = 0; k < TK; ++k) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_HEAD_DIM
    for (int i = 0; i < TM; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_SEQ
      for (int j = 0; j < TN; ++j) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_MODEL_DIM
#pragma HLS PIPELINE II=1
        HgtxrAccumT product = static_cast<HgtxrAccumT>(lhs[i][k]) *
                              static_cast<HgtxrAccumT>(rhs[k][j]);
        out[i][j] = out[i][j] + product;
      }
    }
  }
}

template <int TM, int TN>
void accum_to_data_tile(const HgtxrAccumT in[TM][TN],
                        HgtxrDataT out[TM][TN],
                        ap_int<8> scale_shift) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=in cyclic factor=HGTXR_PAR_FFN dim=2
#pragma HLS ARRAY_PARTITION variable=out cyclic factor=HGTXR_PAR_FFN dim=2

  for (int i = 0; i < TM; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_SEQ
    for (int j = 0; j < TN; ++j) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_MODEL_DIM
#pragma HLS PIPELINE II=1
      HgtxrAccumT value = in[i][j];
      if (scale_shift > 0) {
        value = value >> scale_shift;
      } else if (scale_shift < 0) {
        value = value << (-scale_shift);
      }
      out[i][j] = static_cast<HgtxrDataT>(value);
    }
  }
}

template <int TM, int TN>
void residual_add_tile(const HgtxrDataT lhs[TM][TN],
                       const HgtxrDataT rhs[TM][TN],
                       HgtxrDataT out[TM][TN]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=lhs cyclic factor=HGTXR_PAR_FFN dim=2
#pragma HLS ARRAY_PARTITION variable=rhs cyclic factor=HGTXR_PAR_FFN dim=2
#pragma HLS ARRAY_PARTITION variable=out cyclic factor=HGTXR_PAR_FFN dim=2

  for (int i = 0; i < TM; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_SEQ
    for (int j = 0; j < TN; ++j) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_MODEL_DIM
#pragma HLS PIPELINE II=1
      out[i][j] = lhs[i][j] + rhs[i][j];
    }
  }
}

}  // namespace cyclic_transformer
}  // namespace hgtxr

#endif  // HGTXR_CYCLIC_MAC_HPP

