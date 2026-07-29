#ifndef HGTXR_CYCLIC_S2_PROJECTION_HPP
#define HGTXR_CYCLIC_S2_PROJECTION_HPP

#include "hgtxr_cyclic_weight_layout.hpp"

namespace hgtxr {
namespace cyclic_transformer {

template <int TM, int TI, int TO>
void hgtxr_s2_project_pair_accum(const HgtxrDataT x_tile[TM][TI],
                                 const HgtxrDataT weight_tile[TI][TO],
                                 HgtxrAccumT out_acc[TM][TO],
                                 bool clear_accumulator) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=x_tile cyclic factor=HGTXR_PAR_QK dim=2
#pragma HLS ARRAY_PARTITION variable=weight_tile cyclic factor=HGTXR_PAR_QK dim=1
#pragma HLS ARRAY_PARTITION variable=out_acc cyclic factor=HGTXR_PAR_FFN dim=2

  if (clear_accumulator) {
    for (int tm = 0; tm < TM; ++tm) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_SEQ
      for (int to = 0; to < TO; ++to) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_MODEL_DIM
#pragma HLS PIPELINE II=1
        out_acc[tm][to] = 0;
      }
    }
  }

  for (int ti = 0; ti < TI; ++ti) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_MODEL_DIM
    for (int tm = 0; tm < TM; ++tm) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_SEQ
      for (int to = 0; to < TO; ++to) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_MODEL_DIM
#pragma HLS PIPELINE II=1
        out_acc[tm][to] = out_acc[tm][to] +
                          static_cast<HgtxrAccumT>(x_tile[tm][ti]) *
                              static_cast<HgtxrAccumT>(weight_tile[ti][to]);
      }
    }
  }
}

template <int TM, int TI, int TO>
void hgtxr_s2_load_project_pair_accum(const HgtxrAxiWordT *weights,
                                      int block_word_base,
                                      int matrix_elem_offset,
                                      const HgtxrDataT x_tile[TM][TI],
                                      HgtxrAccumT out_acc[TM][TO],
                                      bool clear_accumulator) {
#pragma HLS INLINE off
  HgtxrDataT weight_tile[TI][TO];
#pragma HLS bind_storage variable=weight_tile type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=weight_tile cyclic factor=HGTXR_PAR_QK dim=1

  hgtxr_load_weight_matrix_tile<TI, TO>(weights + block_word_base,
                                        matrix_elem_offset, weight_tile);
  hgtxr_s2_project_pair_accum<TM, TI, TO>(x_tile, weight_tile, out_acc,
                                          clear_accumulator);
}

// Thin S2 wrapper matching the host-side s2_channel_pairs packer order.
// It preserves S1/top behavior unless an explicit caller wires this path in.
template <int TM, int TI, int TO>
void hgtxr_s2_project_pair_accum(const HgtxrAxiWordT *cyclic_weights,
                                 int layer_idx,
                                 int output_tile,
                                 int input_tile,
                                 int channel_tiles,
                                 int matrix_elem_offset,
                                 const HgtxrDataT x_tile[TM][TI],
                                 HgtxrAccumT out_acc[TM][TO],
                                 bool clear_accumulator) {
#pragma HLS INLINE off
  const int block_word_base = hgtxr_s2_channel_pair_word_base(
      layer_idx, output_tile, input_tile, channel_tiles);
  hgtxr_s2_load_project_pair_accum<TM, TI, TO>(
      cyclic_weights, block_word_base, matrix_elem_offset, x_tile, out_acc,
      clear_accumulator);
}

// QKV first-step wrapper: consumes one channel-pair block and accumulates all
// three Transformer input projections so HLS cannot optimize unused paths away.
template <int TM, int TI, int TO>
void hgtxr_s2_qkv_project_pair_accum(const HgtxrAxiWordT *cyclic_weights,
                                     int layer_idx,
                                     int output_tile,
                                     int input_tile,
                                     int channel_tiles,
                                     const HgtxrDataT x_tile[TM][TI],
                                     HgtxrAccumT q_acc[TM][TO],
                                     HgtxrAccumT k_acc[TM][TO],
                                     HgtxrAccumT v_acc[TM][TO],
                                     bool clear_accumulator) {
#pragma HLS INLINE off
  hgtxr_s2_project_pair_accum<TM, TI, TO>(
      cyclic_weights, layer_idx, output_tile, input_tile, channel_tiles,
      HGTXR_CYCLIC_WEIGHT_WQ_OFFSET, x_tile, q_acc, clear_accumulator);
  hgtxr_s2_project_pair_accum<TM, TI, TO>(
      cyclic_weights, layer_idx, output_tile, input_tile, channel_tiles,
      HGTXR_CYCLIC_WEIGHT_WK_OFFSET, x_tile, k_acc, clear_accumulator);
  hgtxr_s2_project_pair_accum<TM, TI, TO>(
      cyclic_weights, layer_idx, output_tile, input_tile, channel_tiles,
      HGTXR_CYCLIC_WEIGHT_WV_OFFSET, x_tile, v_acc, clear_accumulator);
}


template <int TM, int TI, int TH>
void hgtxr_s2_mlp_w1_pair_accum(const HgtxrAxiWordT *cyclic_weights,
                                int layer_idx,
                                int hidden_tile,
                                int input_tile,
                                int channel_tiles,
                                int hidden_tiles,
                                const HgtxrDataT x_tile[TM][TI],
                                HgtxrAccumT hidden_acc[TM][TH],
                                bool clear_accumulator) {
#pragma HLS INLINE off
  const int block_word_base = hgtxr_s2_mlp_w1_word_base(
      layer_idx, hidden_tile, input_tile, channel_tiles, hidden_tiles);
  hgtxr_s2_load_project_pair_accum<TM, TI, TH>(
      cyclic_weights, block_word_base, HGTXR_CYCLIC_WEIGHT_W1_OFFSET, x_tile,
      hidden_acc, clear_accumulator);
}

template <int TM, int TH, int TO>
void hgtxr_s2_mlp_w2_pair_accum(const HgtxrAxiWordT *cyclic_weights,
                                int layer_idx,
                                int output_tile,
                                int hidden_tile,
                                int channel_tiles,
                                int hidden_tiles,
                                const HgtxrDataT hidden_tile_data[TM][TH],
                                HgtxrAccumT out_acc[TM][TO],
                                bool clear_accumulator) {
#pragma HLS INLINE off
  const int block_word_base = hgtxr_s2_mlp_w2_word_base(
      layer_idx, output_tile, hidden_tile, channel_tiles, hidden_tiles);
  hgtxr_s2_load_project_pair_accum<TM, TH, TO>(
      cyclic_weights, block_word_base, HGTXR_CYCLIC_WEIGHT_W2_OFFSET,
      hidden_tile_data, out_acc, clear_accumulator);
}


}  // namespace cyclic_transformer
}  // namespace hgtxr

#endif  // HGTXR_CYCLIC_S2_PROJECTION_HPP
