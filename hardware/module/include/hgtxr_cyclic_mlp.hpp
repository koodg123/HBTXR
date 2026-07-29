#ifndef HGTXR_CYCLIC_MLP_HPP
#define HGTXR_CYCLIC_MLP_HPP

#include "hgtxr_cyclic_mac.hpp"
#include "hgtxr_cyclic_math.hpp"

namespace hgtxr {
namespace cyclic_transformer {

template <int TM, int TC, int TF>
void mlp_tile(const HgtxrDataT x[TM][TC],
              const HgtxrDataT w1[TC][TF],
              const HgtxrDataT w2[TF][TC],
              const HgtxrDataT residual[TM][TC],
              HgtxrDataT out[TM][TC],
              ap_int<8> acc1_scale_shift,
              ap_int<8> acc2_scale_shift,
              bool enable_residual) {
#pragma HLS INLINE off

  HgtxrAccumT hidden_acc[TM][TF];
  HgtxrDataT hidden[TM][TF];
  HgtxrAccumT out_acc[TM][TC];
  HgtxrDataT projected[TM][TC];

#pragma HLS ARRAY_PARTITION variable=hidden_acc cyclic factor=HGTXR_PAR_FFN dim=2
#pragma HLS ARRAY_PARTITION variable=hidden cyclic factor=HGTXR_PAR_FFN dim=2
#pragma HLS ARRAY_PARTITION variable=out_acc cyclic factor=HGTXR_PAR_FFN dim=2
#pragma HLS ARRAY_PARTITION variable=projected cyclic factor=HGTXR_PAR_FFN dim=2

  mac_tile<TM, TF, TC>(x, w1, hidden_acc, false);
  accum_to_data_tile<TM, TF>(hidden_acc, hidden, acc1_scale_shift);

  for (int i = 0; i < TM; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_SEQ
    for (int j = 0; j < TF; ++j) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_FF_DIM
#pragma HLS PIPELINE II=1
      hidden[i][j] = hgtxr_gelu_approx(hidden[i][j]);
    }
  }

  mac_tile<TM, TC, TF>(hidden, w2, out_acc, false);
  accum_to_data_tile<TM, TC>(out_acc, projected, acc2_scale_shift);

  if (enable_residual) {
    residual_add_tile<TM, TC>(projected, residual, out);
  } else {
    for (int i = 0; i < TM; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_SEQ
      for (int j = 0; j < TC; ++j) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_MODEL_DIM
#pragma HLS PIPELINE II=1
        out[i][j] = projected[i][j];
      }
    }
  }
}

}  // namespace cyclic_transformer
}  // namespace hgtxr

#endif  // HGTXR_CYCLIC_MLP_HPP

