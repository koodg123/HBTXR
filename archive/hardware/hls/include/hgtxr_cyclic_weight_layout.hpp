#ifndef HGTXR_CYCLIC_WEIGHT_LAYOUT_HPP
#define HGTXR_CYCLIC_WEIGHT_LAYOUT_HPP

#include "hgtxr_cyclic_transformer_params.hpp"

namespace hgtxr {
namespace cyclic_transformer {

static constexpr int HGTXR_S2_DEFAULT_CHANNEL_TILES = HGTXR_CYCLIC_CHANNEL_TILES;

inline int hgtxr_s2_channel_pair_block_index(int layer,
                                             int output_channel_tile,
                                             int input_channel_tile,
                                             int channel_tiles) {
#pragma HLS INLINE
  return (layer * channel_tiles + output_channel_tile) * channel_tiles +
         input_channel_tile;
}

inline int hgtxr_s2_channel_pair_word_base(int layer,
                                           int output_channel_tile,
                                           int input_channel_tile,
                                           int channel_tiles) {
#pragma HLS INLINE
  return hgtxr_s2_channel_pair_block_index(layer, output_channel_tile,
                                           input_channel_tile, channel_tiles) *
         HGTXR_CYCLIC_WEIGHT_LAYER_STRIDE_WORDS;
}

inline int hgtxr_s2_mlp_layer_block_stride(int channel_tiles,
                                           int hidden_tiles) {
#pragma HLS INLINE
  return 2 * channel_tiles * hidden_tiles;
}

inline int hgtxr_s2_mlp_w1_block_index(int layer,
                                       int hidden_tile,
                                       int input_channel_tile,
                                       int channel_tiles,
                                       int hidden_tiles) {
#pragma HLS INLINE
  return layer * hgtxr_s2_mlp_layer_block_stride(channel_tiles, hidden_tiles) +
         hidden_tile * channel_tiles + input_channel_tile;
}

inline int hgtxr_s2_mlp_w2_block_index(int layer,
                                       int output_channel_tile,
                                       int hidden_tile,
                                       int channel_tiles,
                                       int hidden_tiles) {
#pragma HLS INLINE
  return layer * hgtxr_s2_mlp_layer_block_stride(channel_tiles, hidden_tiles) +
         hidden_tiles * channel_tiles + output_channel_tile * hidden_tiles +
         hidden_tile;
}

inline int hgtxr_s2_mlp_w1_word_base(int layer,
                                     int hidden_tile,
                                     int input_channel_tile,
                                     int channel_tiles,
                                     int hidden_tiles) {
#pragma HLS INLINE
  return hgtxr_s2_mlp_w1_block_index(layer, hidden_tile, input_channel_tile,
                                     channel_tiles, hidden_tiles) *
         HGTXR_CYCLIC_WEIGHT_LAYER_STRIDE_WORDS;
}

inline int hgtxr_s2_mlp_w2_word_base(int layer,
                                     int output_channel_tile,
                                     int hidden_tile,
                                     int channel_tiles,
                                     int hidden_tiles) {
#pragma HLS INLINE
  return hgtxr_s2_mlp_w2_block_index(layer, output_channel_tile, hidden_tile,
                                     channel_tiles, hidden_tiles) *
         HGTXR_CYCLIC_WEIGHT_LAYER_STRIDE_WORDS;
}

inline HgtxrDataT hgtxr_unpack_weight_word(HgtxrAxiWordT word, int lane) {
#pragma HLS INLINE
  HgtxrAxiWordT shifted = word >> (lane * HGTXR_WEIGHT_WIDTH);
  HgtxrWeightIntT bits = shifted.range(HGTXR_WEIGHT_WIDTH - 1, 0);
  HgtxrWeightT value = 0;
  value.range(HGTXR_WEIGHT_WIDTH - 1, 0) = bits.range(HGTXR_WEIGHT_WIDTH - 1, 0);
  return (HgtxrDataT)value;
}

inline HgtxrDataT hgtxr_load_weight_elem(const HgtxrAxiWordT *weights,
                                         int elem_offset) {
#pragma HLS INLINE
  int word_idx = elem_offset / HGTXR_AXI_WEIGHT_LANES;
  int lane_idx = elem_offset % HGTXR_AXI_WEIGHT_LANES;
  return hgtxr_unpack_weight_word(weights[word_idx], lane_idx);
}

template <int TI, int TO>
void hgtxr_load_weight_matrix_tile(const HgtxrAxiWordT *weights,
                                   int elem_offset,
                                   HgtxrDataT matrix[TI][TO]) {
#pragma HLS INLINE off
  for (int i = 0; i < TI; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_MODEL_DIM
    for (int j = 0; j < TO; ++j) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_MODEL_DIM
#pragma HLS PIPELINE II=1
      matrix[i][j] = hgtxr_load_weight_elem(weights, elem_offset + i * TO + j);
    }
  }
}

}  // namespace cyclic_transformer
}  // namespace hgtxr

#endif  // HGTXR_CYCLIC_WEIGHT_LAYOUT_HPP
