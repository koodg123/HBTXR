#ifndef HGTXR_CYCLIC_MATH_HPP
#define HGTXR_CYCLIC_MATH_HPP

#include "hgtxr_cyclic_transformer_params.hpp"

#ifndef HGTXR_USE_HGPIPE_LUT_MATH
#define HGTXR_USE_HGPIPE_LUT_MATH 1
#endif
#ifndef HGTXR_HGPIPE_QUANT_CLAMP_BITS
#define HGTXR_HGPIPE_QUANT_CLAMP_BITS HGTXR_BIT_WIDTH
#endif

namespace hgtxr {
namespace cyclic_transformer {


static inline int hgtxr_clamp_int(int value, int low, int high) {
#pragma HLS INLINE
  if (value < low) {
    return low;
  }
  if (value > high) {
    return high;
  }
  return value;
}

static inline int hgtxr_hgpipe_cursor_lut_index(int value,
                                                int bias,
                                                int shift,
                                                int bound) {
#pragma HLS INLINE
  int cursor = (value + bias) >> shift;
  return hgtxr_clamp_int(cursor, 0, bound);
}

template <int ENTRIES>
static inline int hgtxr_hgpipe_int_table_lookup(int value,
                                                int bias,
                                                int shift,
                                                int bound,
                                                const int table[ENTRIES]) {
#pragma HLS INLINE
  int cursor = hgtxr_hgpipe_cursor_lut_index(value, bias, shift, bound);
  return table[cursor];
}

static inline int hgtxr_hgpipe_mlp0_geluq64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
      1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3,
      3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7};
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 76, 1, 63, table);
}

static inline int hgtxr_hgpipe_mlp1_geluq64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3,
      4, 4, 5, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7,
      7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7};
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 94, 2, 63, table);
}

static inline int hgtxr_hgpipe_mlp2_geluq64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
      2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6,
      6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 114, 2, 63, table);
}

static inline int hgtxr_hgpipe_mlp3_geluq64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
      1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6,
      6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 116, 2, 63, table);
}

static inline int hgtxr_hgpipe_mlp4_geluq64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
      2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6,
      7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 112, 2, 63, table);
}

static inline int hgtxr_hgpipe_mlp5_geluq64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2,
      2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7,
      7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 108, 2, 63, table);
}

static inline int hgtxr_hgpipe_mlp6_geluq64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2,
      2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7,
      7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 107, 2, 63, table);
}

static inline int hgtxr_hgpipe_mlp7_geluq64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2,
      2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7,
      7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 105, 2, 63, table);
}

static inline int hgtxr_hgpipe_mlp8_geluq64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
      2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 7,
      7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 109, 2, 63, table);
}

static inline int hgtxr_hgpipe_mlp9_geluq64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2,
      2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 7, 7, 7, 7,
      7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 110, 2, 63, table);
}
static inline int hgtxr_hgpipe_mlp10_geluq64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2,
      2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4,
      4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6,
      6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7};
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, -2, 1, 63, table);
}

static inline int hgtxr_hgpipe_mlp11_geluq64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3,
      4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6,
      7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
      7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7};
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, -5, 2, 63, table);
}

static inline int hgtxr_hgpipe_geluq64_int(int value) {
#pragma HLS INLINE
  return hgtxr_hgpipe_mlp1_geluq64_int(value);
}

static inline int hgtxr_hgpipe_attn0_q_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2, -1, -1, -1,
      -1, -1, -1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2,
      2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
      3, 3, 3, 3};
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 88, 2, 63, table);
}

static inline int hgtxr_hgpipe_attn0_k_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2, -1, -1, -1,
      -1, -1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2,
      2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 85, 2, 63, table);
}

static inline int hgtxr_hgpipe_attn0_v_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2, -1, -1, -1, -1,
      -1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2,
      2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 162, 3, 63, table);
}

static inline int hgtxr_hgpipe_attn0_a_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2, -2, -1,
      -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
      1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 101, 2, 63, table);
}

static inline int hgtxr_hgpipe_attn1_q_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2, -1, -1, -1,
      -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
      2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 91, 2, 63, table);
}

static inline int hgtxr_hgpipe_attn1_k_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2,
      -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
      2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3};
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 67, 1, 63, table);
}

static inline int hgtxr_hgpipe_attn1_v_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2,
      -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
      2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3};
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 131, 2, 63, table);
}

static inline int hgtxr_hgpipe_attn1_a_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -1, -1, -1, -1,
      -1, -1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2,
      2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 162, 3, 63, table);
}

static inline int hgtxr_hgpipe_attn2_q_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2, -1, -1,
      -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
      1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 48, 1, 63, table);
}

static inline int hgtxr_hgpipe_attn2_k_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2,
      -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
      2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3};
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 32, 0, 63, table);
}

static inline int hgtxr_hgpipe_attn2_v_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2, -1, -1,
      -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
      1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 95, 2, 63, table);
}

static inline int hgtxr_hgpipe_attn2_a_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2, -2, -1,
      -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
      1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 25, 0, 63, table);
}

static inline int hgtxr_hgpipe_attn3_q_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2, -1, -1, -1, -1,
      -1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 77, 2, 63, table);
}

static inline int hgtxr_hgpipe_attn3_k_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2, -2, -1, -1,
      -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
      1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 48, 1, 63, table);
}

static inline int hgtxr_hgpipe_attn3_v_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2, -2,
      -2, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
      2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 117, 2, 63, table);
}

static inline int hgtxr_hgpipe_attn3_a_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2, -1, -1, -1,
      -1, -1, -1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2,
      2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 43, 1, 63, table);
}

static inline int hgtxr_hgpipe_attn4_q_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2, -2,
      -2, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2,
      2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 116, 2, 63, table);
}

static inline int hgtxr_hgpipe_attn4_k_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2,
      -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 73, 1, 63, table);
}

static inline int hgtxr_hgpipe_attn4_v_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2,
      -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
      2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 132, 2, 63, table);
}

static inline int hgtxr_hgpipe_attn4_a_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2,
      -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
      2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 132, 2, 63, table);
}

static inline int hgtxr_hgpipe_attn5_q_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2, -2, -1,
      -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
      1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 101, 2, 63, table);
}

static inline int hgtxr_hgpipe_attn5_k_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2,
      -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2,
      2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 64, 1, 63, table);
}

static inline int hgtxr_hgpipe_attn5_v_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2,
      -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
      2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 121, 2, 63, table);
}

static inline int hgtxr_hgpipe_attn5_a_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2, -2, -1, -1,
      -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
      1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 93, 2, 63, table);
}

static inline int hgtxr_hgpipe_attn6_q_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2, -2,
      -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 111, 2, 63, table);
}

static inline int hgtxr_hgpipe_attn6_k_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2, -1, -1, -1, -1,
      -1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2,
      2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 81, 2, 63, table);
}

static inline int hgtxr_hgpipe_attn6_v_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2,
      -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
      2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 120, 2, 63, table);
}

static inline int hgtxr_hgpipe_attn6_a_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2, -2, -1, -1,
      -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
      1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 98, 2, 63, table);
}

static inline int hgtxr_hgpipe_attn7_q_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2,
      -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
      2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 129, 2, 63, table);
}

static inline int hgtxr_hgpipe_attn7_k_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2, -2, -1, -1,
      -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
      2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 92, 2, 63, table);
}

static inline int hgtxr_hgpipe_attn7_v_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2, -2,
      -2, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 113, 2, 63, table);
}

static inline int hgtxr_hgpipe_attn7_a_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -1, -1, -1, -1,
      -1, -1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2,
      2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 162, 3, 63, table);
}

static inline int hgtxr_hgpipe_attn8_q_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2, -2, -2,
      -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
      1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 109, 2, 63, table);
}

static inline int hgtxr_hgpipe_attn8_k_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -1, -1, -1, -1,
      -1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2,
      2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 80, 2, 63, table);
}

static inline int hgtxr_hgpipe_attn8_v_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2, -2,
      -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 111, 2, 63, table);
}

static inline int hgtxr_hgpipe_attn8_a_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2, -2, -2,
      -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 1, 1,
      1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 107, 2, 63, table);
}

static inline int hgtxr_hgpipe_attn9_q_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1,
      0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 152, 3, 63, table);
}

static inline int hgtxr_hgpipe_attn9_k_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2, -1, -1,
      -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
      1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 97, 2, 63, table);
}

static inline int hgtxr_hgpipe_attn9_v_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2, -2, -2,
      -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 1, 1,
      1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 107, 2, 63, table);
}

static inline int hgtxr_hgpipe_attn9_a_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2, -1, -1, -1,
      -1, -1, -1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2,
      2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 174, 3, 63, table);
}

static inline int hgtxr_hgpipe_attn10_q_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2, -1, -1, -1,
      -1, -1, -1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2,
      2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 173, 3, 63, table);
}

static inline int hgtxr_hgpipe_attn10_k_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2, -2,
      -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 112, 2, 63, table);
}

static inline int hgtxr_hgpipe_attn10_v_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2, -2,
      -2, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 114, 2, 63, table);
}

static inline int hgtxr_hgpipe_attn10_a_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2,
      -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 135, 2, 63, table);
}

static inline int hgtxr_hgpipe_attn11_q_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2, -1, -1,
      -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
      1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 189, 3, 63, table);
}

static inline int hgtxr_hgpipe_attn11_k_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1,
      -1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 77, 2, 63, table);
}

static inline int hgtxr_hgpipe_attn11_v_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2, -2, -2,
      -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 1,
      1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 110, 2, 63, table);
}

static inline int hgtxr_hgpipe_attn11_a_quant64_int(int value) {
#pragma HLS INLINE
  static const int table[64] = {
      -4, -3, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2, -2, -1,
      -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
      1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<64>(value, 102, 2, 63, table);
}

// Isolated HG-PIPE Softmax integer helpers generated from case/refs/*_softmaxq_*.

static inline int hgtxr_hgpipe_attn0_softmax_exp32_int(int opposite_delta) {
#pragma HLS INLINE
  static const int table[32] = {
      32768, 21631, 14280, 9426, 6223, 4108, 2711, 1790,
      1181, 780, 515, 340, 224, 148, 97, 64,
      42, 28, 18, 12, 8, 5, 3, 2,
      1, 1, 0, 0, 0, 0, 0, 0
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<32>(opposite_delta, 8, 4, 31, table);
}

static inline bool hgtxr_hgpipe_attn0_softmax_recip_uses_table_two(int acc) {
#pragma HLS INLINE
  return ((acc + -25242) >> 14) > 63;
}

static inline int hgtxr_hgpipe_attn0_softmax_recip_int(int acc) {
#pragma HLS INLINE
  static const int table_one[64] = {
      244, 164, 123, 99, 82, 70, 62, 55,
      49, 45, 41, 38, 35, 33, 31, 29,
      27, 26, 24, 23, 22, 21, 20, 19,
      19, 18, 17, 17, 16, 16, 15, 15,
      14, 14, 13, 13, 13, 12, 12, 12,
      11, 11, 11, 11, 10, 10, 10, 10,
      9, 9, 9, 9, 9, 9, 8, 8,
      8, 8, 8, 8, 8, 7, 7, 7
  };
  static const int table_two[64] = {
      197, 165, 141, 124, 110, 99, 90, 82,
      76, 71, 66, 62, 58, 55, 52, 49,
      47, 45, 43, 41, 39, 38, 36, 35,
      34, 33, 32, 31, 30, 29, 28, 27,
      26, 26, 25, 24, 24, 23, 23, 22,
      22, 21, 21, 20, 20, 19, 19, 19,
      18, 18, 18, 17, 17, 17, 16, 16,
      16, 16, 15, 15, 15, 15, 14, 14
  };
#pragma HLS bind_storage variable=table_one type=rom_2p impl=lutram
#pragma HLS bind_storage variable=table_two type=rom_2p impl=lutram
  if (hgtxr_hgpipe_attn0_softmax_recip_uses_table_two(acc)) {
    return hgtxr_hgpipe_int_table_lookup<64>(acc, -596612, 17, 63, table_two);
  }
  return hgtxr_hgpipe_int_table_lookup<64>(acc, -25242, 14, 63, table_one);
}

static inline int hgtxr_hgpipe_attn0_softmax_requant_uint3_int(int exp_score,
                                                               int recip,
                                                               bool in_table_two) {
#pragma HLS INLINE
  int rel_b = in_table_two ? 524288 : 32768;
  int rel_s = in_table_two ? 20 : 16;
  int rel = ((exp_score * recip) + rel_b) >> rel_s;
  return hgtxr_clamp_int(rel, 0, 7);
}

static inline int hgtxr_hgpipe_attn1_softmax_exp32_int(int opposite_delta) {
#pragma HLS INLINE
  static const int table[32] = {
      32768, 22149, 14971, 10120, 6840, 4623, 3125, 2112,
      1428, 965, 652, 441, 298, 201, 136, 92,
      62, 42, 28, 19, 12, 8, 5, 4,
      2, 1, 1, 0, 0, 0, 0, 0
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<32>(opposite_delta, 8, 4, 31, table);
}

static inline bool hgtxr_hgpipe_attn1_softmax_recip_uses_table_two(int acc) {
#pragma HLS INLINE
  return ((acc + -28690) >> 13) > 63;
}

static inline int hgtxr_hgpipe_attn1_softmax_recip_int(int acc) {
#pragma HLS INLINE
  static const int table_one[64] = {
      147, 118, 98, 84, 73, 65, 59, 53,
      49, 45, 42, 39, 36, 34, 32, 31,
      29, 28, 26, 25, 24, 23, 22, 21,
      21, 20, 19, 19, 18, 17, 17, 16,
      16, 15, 15, 15, 14, 14, 14, 13,
      13, 13, 12, 12, 12, 12, 11, 11,
      11, 11, 10, 10, 10, 10, 10, 10,
      9, 9, 9, 9, 9, 9, 8, 8
  };
  static const int table_two[64] = {
      143, 127, 115, 105, 96, 89, 83, 77,
      72, 68, 64, 61, 58, 55, 53, 50,
      48, 46, 45, 43, 41, 40, 39, 37,
      36, 35, 34, 33, 32, 31, 30, 30,
      29, 28, 28, 27, 26, 26, 25, 25,
      24, 24, 23, 23, 22, 22, 21, 21,
      21, 20, 20, 19, 19, 19, 19, 18,
      18, 18, 17, 17, 17, 17, 16, 16
  };
#pragma HLS bind_storage variable=table_one type=rom_2p impl=lutram
#pragma HLS bind_storage variable=table_two type=rom_2p impl=lutram
  if (hgtxr_hgpipe_attn1_softmax_recip_uses_table_two(acc)) {
    return hgtxr_hgpipe_int_table_lookup<64>(acc, -507831, 16, 63, table_two);
  }
  return hgtxr_hgpipe_int_table_lookup<64>(acc, -28690, 13, 63, table_one);
}

static inline int hgtxr_hgpipe_attn1_softmax_requant_uint3_int(int exp_score,
                                                               int recip,
                                                               bool in_table_two) {
#pragma HLS INLINE
  int rel_b = in_table_two ? 262144 : 16384;
  int rel_s = in_table_two ? 19 : 15;
  int rel = ((exp_score * recip) + rel_b) >> rel_s;
  return hgtxr_clamp_int(rel, 0, 7);
}

static inline int hgtxr_hgpipe_attn2_softmax_exp32_int(int opposite_delta) {
#pragma HLS INLINE
  static const int table[32] = {
      32768, 19952, 12148, 7397, 4504, 2742, 1669, 1016,
      619, 376, 229, 139, 85, 51, 31, 19,
      11, 7, 4, 2, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<32>(opposite_delta, 2, 2, 31, table);
}

static inline bool hgtxr_hgpipe_attn2_softmax_recip_uses_table_two(int acc) {
#pragma HLS INLINE
  return ((acc + -24576) >> 14) > 63;
}

static inline int hgtxr_hgpipe_attn2_softmax_recip_int(int acc) {
#pragma HLS INLINE
  static const int table_one[64] = {
      148, 99, 74, 59, 49, 42, 37, 33,
      29, 27, 24, 22, 21, 19, 18, 17,
      16, 15, 14, 14, 13, 12, 12, 11,
      11, 11, 10, 10, 9, 9, 9, 9,
      8, 8, 8, 8, 7, 7, 7, 7,
      7, 6, 6, 6, 6, 6, 6, 6,
      5, 5, 5, 5, 5, 5, 5, 5,
      5, 5, 4, 4, 4, 4, 4, 4
  };
  static const int table_two[64] = {
      198, 170, 148, 132, 119, 108, 99, 91,
      85, 79, 74, 70, 66, 62, 59, 56,
      54, 51, 49, 47, 45, 44, 42, 41,
      39, 38, 37, 36, 34, 33, 33, 32,
      31, 30, 29, 29, 28, 27, 27, 26,
      25, 25, 24, 24, 23, 23, 22, 22,
      22, 21, 21, 20, 20, 20, 19, 19,
      19, 18, 18, 18, 18, 17, 17, 17
  };
#pragma HLS bind_storage variable=table_one type=rom_2p impl=lutram
#pragma HLS bind_storage variable=table_two type=rom_2p impl=lutram
  if (hgtxr_hgpipe_attn2_softmax_recip_uses_table_two(acc)) {
    return hgtxr_hgpipe_int_table_lookup<64>(acc, -720252, 17, 63, table_two);
  }
  return hgtxr_hgpipe_int_table_lookup<64>(acc, -24576, 14, 63, table_one);
}

static inline int hgtxr_hgpipe_attn2_softmax_requant_uint3_int(int exp_score,
                                                               int recip,
                                                               bool in_table_two) {
#pragma HLS INLINE
  int rel_b = in_table_two ? 1048576 : 32768;
  int rel_s = in_table_two ? 21 : 16;
  int rel = ((exp_score * recip) + rel_b) >> rel_s;
  return hgtxr_clamp_int(rel, 0, 7);
}

static inline int hgtxr_hgpipe_attn3_softmax_exp32_int(int opposite_delta) {
#pragma HLS INLINE
  static const int table[32] = {
      32768, 19220, 11273, 6612, 3878, 2275, 1334, 782,
      459, 269, 157, 92, 54, 31, 18, 10,
      6, 3, 2, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<32>(opposite_delta, 8, 4, 31, table);
}

static inline bool hgtxr_hgpipe_attn3_softmax_recip_uses_table_two(int acc) {
#pragma HLS INLINE
  return ((acc + -28672) >> 13) > 63;
}

static inline int hgtxr_hgpipe_attn3_softmax_recip_int(int acc) {
#pragma HLS INLINE
  static const int table_one[64] = {
      130, 104, 86, 74, 65, 57, 52, 47,
      43, 40, 37, 34, 32, 30, 28, 27,
      26, 24, 23, 22, 21, 20, 20, 19,
      18, 17, 17, 16, 16, 15, 15, 14,
      14, 14, 13, 13, 13, 12, 12, 12,
      11, 11, 11, 11, 10, 10, 10, 10,
      10, 9, 9, 9, 9, 9, 8, 8,
      8, 8, 8, 8, 8, 8, 7, 7
  };
  static const int table_two[64] = {
      146, 128, 114, 103, 93, 85, 79, 73,
      68, 64, 60, 57, 54, 51, 49, 47,
      45, 43, 41, 39, 38, 37, 35, 34,
      33, 32, 31, 30, 29, 28, 28, 27,
      26, 25, 25, 24, 24, 23, 23, 22,
      22, 21, 21, 20, 20, 19, 19, 19,
      18, 18, 18, 17, 17, 17, 17, 16,
      16, 16, 15, 15, 15, 15, 15, 14
  };
#pragma HLS bind_storage variable=table_one type=rom_2p impl=lutram
#pragma HLS bind_storage variable=table_two type=rom_2p impl=lutram
  if (hgtxr_hgpipe_attn3_softmax_recip_uses_table_two(acc)) {
    return hgtxr_hgpipe_int_table_lookup<64>(acc, -432345, 16, 63, table_two);
  }
  return hgtxr_hgpipe_int_table_lookup<64>(acc, -28672, 13, 63, table_one);
}

static inline int hgtxr_hgpipe_attn3_softmax_requant_uint3_int(int exp_score,
                                                               int recip,
                                                               bool in_table_two) {
#pragma HLS INLINE
  int rel_b = in_table_two ? 1048576 : 65536;
  int rel_s = in_table_two ? 21 : 17;
  int rel = ((exp_score * recip) + rel_b) >> rel_s;
  return hgtxr_clamp_int(rel, 0, 7);
}

static inline int hgtxr_hgpipe_attn4_softmax_exp32_int(int opposite_delta) {
#pragma HLS INLINE
  static const int table[32] = {
      32768, 18593, 10550, 5986, 3397, 1927, 1093, 620,
      352, 199, 113, 64, 36, 20, 11, 6,
      3, 2, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<32>(opposite_delta, 8, 4, 31, table);
}

static inline bool hgtxr_hgpipe_attn4_softmax_recip_uses_table_two(int acc) {
#pragma HLS INLINE
  return ((acc + -28707) >> 13) > 63;
}

static inline int hgtxr_hgpipe_attn4_softmax_recip_int(int acc) {
#pragma HLS INLINE
  static const int table_one[64] = {
      253, 203, 169, 145, 126, 112, 101, 92,
      84, 78, 72, 67, 63, 59, 56, 53,
      50, 48, 46, 44, 42, 40, 39, 37,
      36, 35, 33, 32, 31, 30, 29, 29,
      28, 27, 26, 26, 25, 24, 24, 23,
      23, 22, 22, 21, 21, 20, 20, 19,
      19, 19, 18, 18, 18, 17, 17, 17,
      16, 16, 16, 16, 15, 15, 15, 15
  };
  static const int table_two[64] = {
      152, 132, 117, 105, 95, 87, 80, 74,
      69, 64, 61, 57, 54, 51, 49, 46,
      44, 42, 41, 39, 38, 36, 35, 34,
      33, 32, 31, 30, 29, 28, 27, 26,
      26, 25, 24, 24, 23, 23, 22, 22,
      21, 21, 20, 20, 20, 19, 19, 18,
      18, 18, 17, 17, 17, 17, 16, 16,
      16, 15, 15, 15, 15, 15, 14, 14
  };
#pragma HLS bind_storage variable=table_one type=rom_2p impl=lutram
#pragma HLS bind_storage variable=table_two type=rom_2p impl=lutram
  if (hgtxr_hgpipe_attn4_softmax_recip_uses_table_two(acc)) {
    return hgtxr_hgpipe_int_table_lookup<64>(acc, -403413, 16, 63, table_two);
  }
  return hgtxr_hgpipe_int_table_lookup<64>(acc, -28707, 13, 63, table_one);
}

static inline int hgtxr_hgpipe_attn4_softmax_requant_uint3_int(int exp_score,
                                                               int recip,
                                                               bool in_table_two) {
#pragma HLS INLINE
  int rel_b = in_table_two ? 262144 : 32768;
  int rel_s = in_table_two ? 19 : 16;
  int rel = ((exp_score * recip) + rel_b) >> rel_s;
  return hgtxr_clamp_int(rel, 0, 7);
}

static inline int hgtxr_hgpipe_attn5_softmax_exp32_int(int opposite_delta) {
#pragma HLS INLINE
  static const int table[32] = {
      32768, 18271, 10187, 5680, 3167, 1766, 984, 549,
      306, 170, 95, 53, 29, 16, 9, 5,
      2, 1, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<32>(opposite_delta, 8, 4, 31, table);
}

static inline bool hgtxr_hgpipe_attn5_softmax_recip_uses_table_two(int acc) {
#pragma HLS INLINE
  return ((acc + -28732) >> 13) > 63;
}

static inline int hgtxr_hgpipe_attn5_softmax_recip_int(int acc) {
#pragma HLS INLINE
  static const int table_one[64] = {
      172, 138, 115, 98, 86, 76, 69, 62,
      57, 53, 49, 46, 43, 40, 38, 36,
      34, 32, 31, 30, 28, 27, 26, 25,
      24, 23, 23, 22, 21, 20, 20, 19,
      19, 18, 18, 17, 17, 16, 16, 16,
      15, 15, 15, 14, 14, 14, 13, 13,
      13, 13, 12, 12, 12, 12, 11, 11,
      11, 11, 11, 10, 10, 10, 10, 10
  };
  static const int table_two[64] = {
      186, 164, 147, 132, 121, 111, 103, 96,
      89, 84, 79, 75, 71, 67, 64, 61,
      59, 56, 54, 52, 50, 48, 47, 45,
      44, 42, 41, 40, 39, 38, 37, 36,
      35, 34, 33, 32, 31, 31, 30, 29,
      29, 28, 28, 27, 26, 26, 25, 25,
      24, 24, 24, 23, 23, 22, 22, 22,
      21, 21, 21, 20, 20, 20, 19, 19
  };
#pragma HLS bind_storage variable=table_one type=rom_2p impl=lutram
#pragma HLS bind_storage variable=table_two type=rom_2p impl=lutram
  if (hgtxr_hgpipe_attn5_softmax_recip_uses_table_two(acc)) {
    return hgtxr_hgpipe_int_table_lookup<64>(acc, -452800, 16, 63, table_two);
  }
  return hgtxr_hgpipe_int_table_lookup<64>(acc, -28732, 13, 63, table_one);
}

static inline int hgtxr_hgpipe_attn5_softmax_requant_uint3_int(int exp_score,
                                                               int recip,
                                                               bool in_table_two) {
#pragma HLS INLINE
  int rel_b = in_table_two ? 524288 : 32768;
  int rel_s = in_table_two ? 20 : 16;
  int rel = ((exp_score * recip) + rel_b) >> rel_s;
  return hgtxr_clamp_int(rel, 0, 7);
}

static inline int hgtxr_hgpipe_attn6_softmax_exp32_int(int opposite_delta) {
#pragma HLS INLINE
  static const int table[32] = {
      32768, 17064, 8886, 4627, 2410, 1255, 653, 340,
      177, 92, 48, 25, 13, 6, 3, 1,
      0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<32>(opposite_delta, 8, 4, 31, table);
}

static inline bool hgtxr_hgpipe_attn6_softmax_recip_uses_table_two(int acc) {
#pragma HLS INLINE
  return ((acc + -29038) >> 13) > 63;
}

static inline int hgtxr_hgpipe_attn6_softmax_recip_int(int acc) {
#pragma HLS INLINE
  static const int table_one[64] = {
      173, 139, 116, 99, 87, 77, 69, 63,
      58, 53, 50, 46, 43, 41, 38, 36,
      35, 33, 31, 30, 29, 28, 26, 25,
      25, 24, 23, 22, 21, 21, 20, 20,
      19, 18, 18, 18, 17, 17, 16, 16,
      15, 15, 15, 14, 14, 14, 14, 13,
      13, 13, 13, 12, 12, 12, 12, 11,
      11, 11, 11, 11, 10, 10, 10, 10
  };
  static const int table_two[64] = {
      173, 154, 139, 126, 116, 107, 99, 93,
      87, 82, 77, 73, 69, 66, 63, 60,
      58, 55, 53, 51, 50, 48, 46, 45,
      43, 42, 41, 40, 38, 37, 36, 35,
      35, 34, 33, 32, 31, 31, 30, 29,
      29, 28, 28, 27, 26, 26, 25, 25,
      25, 24, 24, 23, 23, 23, 22, 22,
      21, 21, 21, 20, 20, 20, 20, 19
  };
#pragma HLS bind_storage variable=table_one type=rom_2p impl=lutram
#pragma HLS bind_storage variable=table_two type=rom_2p impl=lutram
  if (hgtxr_hgpipe_attn6_softmax_recip_uses_table_two(acc)) {
    return hgtxr_hgpipe_int_table_lookup<64>(acc, -499119, 16, 63, table_two);
  }
  return hgtxr_hgpipe_int_table_lookup<64>(acc, -29038, 13, 63, table_one);
}

static inline int hgtxr_hgpipe_attn6_softmax_requant_uint3_int(int exp_score,
                                                               int recip,
                                                               bool in_table_two) {
#pragma HLS INLINE
  int rel_b = in_table_two ? 524288 : 32768;
  int rel_s = in_table_two ? 20 : 16;
  int rel = ((exp_score * recip) + rel_b) >> rel_s;
  return hgtxr_clamp_int(rel, 0, 7);
}

static inline int hgtxr_hgpipe_attn7_softmax_exp32_int(int opposite_delta) {
#pragma HLS INLINE
  static const int table[32] = {
      32768, 22306, 15185, 10337, 7036, 4790, 3261, 2219,
      1511, 1028, 700, 476, 324, 220, 150, 102,
      69, 47, 32, 21, 14, 10, 6, 4,
      3, 2, 1, 1, 0, 0, 0, 0
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<32>(opposite_delta, 4, 3, 31, table);
}

static inline bool hgtxr_hgpipe_attn7_softmax_recip_uses_table_two(int acc) {
#pragma HLS INLINE
  return ((acc + -28773) >> 13) > 63;
}

static inline int hgtxr_hgpipe_attn7_softmax_recip_int(int acc) {
#pragma HLS INLINE
  static const int table_one[64] = {
      254, 203, 169, 145, 127, 113, 102, 92,
      85, 78, 72, 68, 63, 60, 56, 53,
      51, 48, 46, 44, 42, 40, 39, 37,
      36, 35, 34, 32, 31, 30, 30, 29,
      28, 27, 26, 26, 25, 24, 24, 23,
      23, 22, 22, 21, 21, 20, 20, 20,
      19, 19, 18, 18, 18, 17, 17, 17,
      17, 16, 16, 16, 15, 15, 15, 15
  };
  static const int table_two[64] = {
      200, 167, 143, 126, 112, 101, 92, 84,
      78, 72, 67, 63, 59, 56, 53, 50,
      48, 46, 44, 42, 40, 39, 37, 36,
      35, 33, 32, 31, 30, 29, 29, 28,
      27, 26, 26, 25, 24, 24, 23, 23,
      22, 22, 21, 21, 20, 20, 19, 19,
      19, 18, 18, 18, 17, 17, 17, 16,
      16, 16, 16, 15, 15, 15, 15, 15
  };
#pragma HLS bind_storage variable=table_one type=rom_2p impl=lutram
#pragma HLS bind_storage variable=table_two type=rom_2p impl=lutram
  if (hgtxr_hgpipe_attn7_softmax_recip_uses_table_two(acc)) {
    return hgtxr_hgpipe_int_table_lookup<64>(acc, -301274, 16, 63, table_two);
  }
  return hgtxr_hgpipe_int_table_lookup<64>(acc, -28773, 13, 63, table_one);
}

static inline int hgtxr_hgpipe_attn7_softmax_requant_uint3_int(int exp_score,
                                                               int recip,
                                                               bool in_table_two) {
#pragma HLS INLINE
  int rel_b = in_table_two ? 262144 : 32768;
  int rel_s = in_table_two ? 19 : 16;
  int rel = ((exp_score * recip) + rel_b) >> rel_s;
  return hgtxr_clamp_int(rel, 0, 7);
}

static inline int hgtxr_hgpipe_attn8_softmax_exp32_int(int opposite_delta) {
#pragma HLS INLINE
  static const int table[32] = {
      32768, 19519, 11627, 6926, 4126, 2457, 1464, 872,
      519, 309, 184, 109, 65, 38, 23, 13,
      8, 4, 2, 1, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<32>(opposite_delta, 8, 4, 31, table);
}

static inline bool hgtxr_hgpipe_attn8_softmax_recip_uses_table_two(int acc) {
#pragma HLS INLINE
  return ((acc + -28744) >> 13) > 63;
}

static inline int hgtxr_hgpipe_attn8_softmax_recip_int(int acc) {
#pragma HLS INLINE
  static const int table_one[64] = {
      162, 129, 108, 92, 81, 72, 65, 59,
      54, 50, 46, 43, 40, 38, 36, 34,
      32, 30, 29, 28, 27, 26, 25, 24,
      23, 22, 21, 20, 20, 19, 19, 18,
      18, 17, 17, 16, 16, 15, 15, 15,
      14, 14, 14, 13, 13, 13, 13, 12,
      12, 12, 12, 11, 11, 11, 11, 11,
      10, 10, 10, 10, 10, 10, 9, 9
  };
  static const int table_two[64] = {
      181, 159, 142, 128, 116, 107, 98, 91,
      85, 80, 75, 71, 67, 64, 61, 58,
      56, 53, 51, 49, 47, 46, 44, 43,
      41, 40, 39, 38, 37, 35, 35, 34,
      33, 32, 31, 30, 30, 29, 28, 28,
      27, 27, 26, 25, 25, 24, 24, 24,
      23, 23, 22, 22, 21, 21, 21, 20,
      20, 20, 19, 19, 19, 19, 18, 18
  };
#pragma HLS bind_storage variable=table_one type=rom_2p impl=lutram
#pragma HLS bind_storage variable=table_two type=rom_2p impl=lutram
  if (hgtxr_hgpipe_attn8_softmax_recip_uses_table_two(acc)) {
    return hgtxr_hgpipe_int_table_lookup<64>(acc, -436489, 16, 63, table_two);
  }
  return hgtxr_hgpipe_int_table_lookup<64>(acc, -28744, 13, 63, table_one);
}

static inline int hgtxr_hgpipe_attn8_softmax_requant_uint3_int(int exp_score,
                                                               int recip,
                                                               bool in_table_two) {
#pragma HLS INLINE
  int rel_b = in_table_two ? 524288 : 32768;
  int rel_s = in_table_two ? 20 : 16;
  int rel = ((exp_score * recip) + rel_b) >> rel_s;
  return hgtxr_clamp_int(rel, 0, 7);
}

static inline int hgtxr_hgpipe_attn9_softmax_exp32_int(int opposite_delta) {
#pragma HLS INLINE
  static const int table[32] = {
      32768, 18416, 10350, 5817, 3269, 1837, 1032, 580,
      326, 183, 103, 57, 32, 18, 10, 5,
      3, 1, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<32>(opposite_delta, 8, 4, 31, table);
}

static inline bool hgtxr_hgpipe_attn9_softmax_recip_uses_table_two(int acc) {
#pragma HLS INLINE
  return ((acc + -28802) >> 13) > 63;
}

static inline int hgtxr_hgpipe_attn9_softmax_recip_int(int acc) {
#pragma HLS INLINE
  static const int table_one[64] = {
      136, 109, 91, 78, 68, 60, 54, 49,
      45, 42, 39, 36, 34, 32, 30, 28,
      27, 26, 24, 23, 22, 21, 21, 20,
      19, 18, 18, 17, 17, 16, 16, 15,
      15, 14, 14, 14, 13, 13, 13, 12,
      12, 12, 11, 11, 11, 11, 10, 10,
      10, 10, 10, 9, 9, 9, 9, 9,
      9, 8, 8, 8, 8, 8, 8, 8
  };
  static const int table_two[64] = {
      187, 160, 139, 123, 111, 101, 92, 85,
      79, 73, 69, 65, 61, 58, 55, 52,
      50, 47, 45, 44, 42, 40, 39, 38,
      36, 35, 34, 33, 32, 31, 30, 29,
      28, 28, 27, 26, 26, 25, 25, 24,
      23, 23, 22, 22, 21, 21, 21, 20,
      20, 19, 19, 19, 18, 18, 18, 18,
      17, 17, 17, 16, 16, 16, 16, 15
  };
#pragma HLS bind_storage variable=table_one type=rom_2p impl=lutram
#pragma HLS bind_storage variable=table_two type=rom_2p impl=lutram
  if (hgtxr_hgpipe_attn9_softmax_recip_uses_table_two(acc)) {
    return hgtxr_hgpipe_int_table_lookup<64>(acc, -350846, 16, 63, table_two);
  }
  return hgtxr_hgpipe_int_table_lookup<64>(acc, -28802, 13, 63, table_one);
}

static inline int hgtxr_hgpipe_attn9_softmax_requant_uint3_int(int exp_score,
                                                               int recip,
                                                               bool in_table_two) {
#pragma HLS INLINE
  int rel_b = in_table_two ? 262144 : 16384;
  int rel_s = in_table_two ? 19 : 15;
  int rel = ((exp_score * recip) + rel_b) >> rel_s;
  return hgtxr_clamp_int(rel, 0, 7);
}

static inline int hgtxr_hgpipe_attn10_softmax_exp32_int(int opposite_delta) {
#pragma HLS INLINE
  static const int table[32] = {
      32768, 21869, 14595, 9740, 6500, 4338, 2895, 1932,
      1289, 860, 574, 383, 255, 170, 113, 76,
      50, 33, 22, 15, 10, 6, 4, 2,
      1, 1, 0, 0, 0, 0, 0, 0
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<32>(opposite_delta, 8, 4, 31, table);
}

static inline bool hgtxr_hgpipe_attn10_softmax_recip_uses_table_two(int acc) {
#pragma HLS INLINE
  return ((acc + -28706) >> 13) > 63;
}

static inline int hgtxr_hgpipe_attn10_softmax_recip_int(int acc) {
#pragma HLS INLINE
  static const int table_one[64] = {
      132, 106, 88, 75, 66, 59, 53, 48,
      44, 40, 37, 35, 33, 31, 29, 28,
      26, 25, 24, 23, 22, 21, 20, 19,
      19, 18, 17, 17, 16, 16, 15, 15,
      14, 14, 14, 13, 13, 12, 12, 12,
      12, 11, 11, 11, 11, 10, 10, 10,
      10, 10, 9, 9, 9, 9, 9, 9,
      8, 8, 8, 8, 8, 8, 8, 7
  };
  static const int table_two[64] = {
      177, 152, 133, 118, 106, 96, 88, 81,
      76, 70, 66, 62, 59, 56, 53, 50,
      48, 46, 44, 42, 40, 39, 38, 36,
      35, 34, 33, 32, 31, 30, 29, 28,
      28, 27, 26, 25, 25, 24, 24, 23,
      23, 22, 22, 21, 21, 20, 20, 20,
      19, 19, 19, 18, 18, 18, 17, 17,
      17, 16, 16, 16, 16, 15, 15, 15
  };
#pragma HLS bind_storage variable=table_one type=rom_2p impl=lutram
#pragma HLS bind_storage variable=table_two type=rom_2p impl=lutram
  if (hgtxr_hgpipe_attn10_softmax_recip_uses_table_two(acc)) {
    return hgtxr_hgpipe_int_table_lookup<64>(acc, -359889, 16, 63, table_two);
  }
  return hgtxr_hgpipe_int_table_lookup<64>(acc, -28706, 13, 63, table_one);
}

static inline int hgtxr_hgpipe_attn10_softmax_requant_uint3_int(int exp_score,
                                                                int recip,
                                                                bool in_table_two) {
#pragma HLS INLINE
  int rel_b = in_table_two ? 262144 : 16384;
  int rel_s = in_table_two ? 19 : 15;
  int rel = ((exp_score * recip) + rel_b) >> rel_s;
  return hgtxr_clamp_int(rel, 0, 7);
}

static inline int hgtxr_hgpipe_attn11_softmax_exp32_int(int opposite_delta) {
#pragma HLS INLINE
  static const int table[32] = {
      32768, 19181, 11227, 6572, 3847, 2252, 1318, 771,
      451, 264, 154, 90, 53, 31, 18, 10,
      6, 3, 2, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<32>(opposite_delta, 16, 5, 31, table);
}

static inline bool hgtxr_hgpipe_attn11_softmax_recip_uses_table_two(int acc) {
#pragma HLS INLINE
  return ((acc + -33678) >> 13) > 63;
}

static inline int hgtxr_hgpipe_attn11_softmax_recip_int(int acc) {
#pragma HLS INLINE
  static const int table_one[64] = {
      191, 157, 133, 115, 102, 91, 83, 76,
      69, 64, 60, 56, 53, 50, 47, 45,
      42, 40, 39, 37, 35, 34, 33, 31,
      30, 29, 28, 27, 27, 26, 25, 24,
      24, 23, 22, 22, 21, 21, 20, 20,
      19, 19, 18, 18, 18, 17, 17, 17,
      16, 16, 16, 15, 15, 15, 15, 14,
      14, 14, 14, 13, 13, 13, 13, 13
  };
  static const int table_two[64] = {
      129, 112, 100, 89, 81, 74, 68, 63,
      59, 55, 52, 49, 46, 44, 42, 40,
      38, 37, 35, 34, 32, 31, 30, 29,
      28, 27, 26, 26, 25, 24, 23, 23,
      22, 22, 21, 21, 20, 20, 19, 19,
      18, 18, 18, 17, 17, 17, 16, 16,
      16, 15, 15, 15, 15, 14, 14, 14,
      14, 13, 13, 13, 13, 13, 12, 12
  };
#pragma HLS bind_storage variable=table_one type=rom_2p impl=lutram
#pragma HLS bind_storage variable=table_two type=rom_2p impl=lutram
  if (hgtxr_hgpipe_attn11_softmax_recip_uses_table_two(acc)) {
    return hgtxr_hgpipe_int_table_lookup<64>(acc, -414060, 16, 63, table_two);
  }
  return hgtxr_hgpipe_int_table_lookup<64>(acc, -33678, 13, 63, table_one);
}

static inline int hgtxr_hgpipe_attn11_softmax_requant_uint3_int(int exp_score,
                                                                int recip,
                                                                bool in_table_two) {
#pragma HLS INLINE
  int rel_b = in_table_two ? 262144 : 32768;
  int rel_s = in_table_two ? 19 : 16;
  int rel = ((exp_score * recip) + rel_b) >> rel_s;
  return hgtxr_clamp_int(rel, 0, 7);
}

// Isolated HG-PIPE LayerNorm integer helpers generated from case/refs/*_lnq_*.

static inline int hgtxr_hgpipe_attn0_lnq_rsqrt128_int(int variance_sum) {
#pragma HLS INLINE
  static const int table[128] = {
      2503, 2240, 2045, 1894, 1771, 1670, 1585, 1511,
      1447, 1390, 1340, 1294, 1253, 1216, 1181, 1150,
      1121, 1094, 1069, 1045, 1023, 1003, 983, 965,
      947, 931, 915, 900, 886, 873, 860, 847,
      835, 824, 813, 803, 793, 783, 773, 764,
      756, 747, 739, 731, 723, 716, 709, 702,
      695, 688, 682, 676, 670, 664, 658, 653,
      647, 642, 637, 631, 626, 622, 617, 612,
      608, 603, 599, 595, 591, 587, 583, 579,
      575, 571, 567, 564, 560, 557, 553, 550,
      547, 544, 540, 537, 534, 531, 528, 525,
      522, 520, 517, 514, 511, 509, 506, 504,
      501, 499, 496, 494, 491, 489, 487, 484,
      482, 480, 478, 476, 473, 471, 469, 467,
      465, 463, 461, 459, 457, 456, 454, 452,
      450, 448, 446, 445, 443, 441, 439, 438
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<128>(variance_sum, -1842833, 19, 127, table);
}

static inline int hgtxr_hgpipe_attn0_lnq_mean_int(int sum) {
#pragma HLS INLINE
  long long tmp = static_cast<long long>(sum) * 43691;
  tmp += (1LL << (23 - 1));
  return static_cast<int>(tmp >> 23);
}

static inline int hgtxr_hgpipe_attn0_lnq_requant_int(int input_value,
                                                     int mean,
                                                     int rsqrt,
                                                     int lnw,
                                                     long long lnb) {
#pragma HLS INLINE
  long long diff = static_cast<long long>(input_value - mean);
  long long val = diff * static_cast<long long>(rsqrt) * static_cast<long long>(lnw) + lnb;
  long long rel = val >> 31;
  return hgtxr_clamp_int(static_cast<int>(rel), -4, 3);
}

static inline int hgtxr_hgpipe_attn1_lnq_rsqrt128_int(int variance_sum) {
#pragma HLS INLINE
  static const int table[128] = {
      2406, 2279, 2170, 2075, 1991, 1917, 1851, 1791,
      1736, 1686, 1641, 1598, 1559, 1523, 1489, 1457,
      1427, 1399, 1373, 1348, 1324, 1302, 1280, 1260,
      1240, 1222, 1204, 1187, 1171, 1155, 1140, 1126,
      1112, 1098, 1085, 1073, 1061, 1049, 1038, 1027,
      1016, 1006, 996, 986, 977, 968, 959, 950,
      942, 934, 926, 918, 910, 903, 896, 888,
      882, 875, 868, 862, 856, 849, 843, 837,
      832, 826, 820, 815, 810, 804, 799, 794,
      789, 784, 780, 775, 770, 766, 761, 757,
      753, 749, 744, 740, 736, 732, 729, 725,
      721, 717, 714, 710, 707, 703, 700, 696,
      693, 690, 686, 683, 680, 677, 674, 671,
      668, 665, 662, 659, 656, 653, 651, 648,
      645, 643, 640, 637, 635, 632, 630, 627,
      625, 622, 620, 618, 615, 613, 611, 609
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<128>(variance_sum, -8585116, 20, 127, table);
}

static inline int hgtxr_hgpipe_attn1_lnq_mean_int(int sum) {
#pragma HLS INLINE
  long long tmp = static_cast<long long>(sum) * 43691;
  tmp += (1LL << (23 - 1));
  return static_cast<int>(tmp >> 23);
}

static inline int hgtxr_hgpipe_attn1_lnq_requant_int(int input_value,
                                                     int mean,
                                                     int rsqrt,
                                                     int lnw,
                                                     long long lnb) {
#pragma HLS INLINE
  long long diff = static_cast<long long>(input_value - mean);
  long long val = diff * static_cast<long long>(rsqrt) * static_cast<long long>(lnw) + lnb;
  long long rel = val >> 33;
  return hgtxr_clamp_int(static_cast<int>(rel), -4, 3);
}

static inline int hgtxr_hgpipe_attn2_lnq_rsqrt128_int(int variance_sum) {
#pragma HLS INLINE
  static const int table[128] = {
      2878, 2767, 2667, 2578, 2497, 2423, 2355, 2293,
      2235, 2181, 2132, 2085, 2041, 2000, 1962, 1925,
      1891, 1858, 1827, 1797, 1769, 1742, 1717, 1692,
      1668, 1646, 1624, 1603, 1583, 1564, 1545, 1527,
      1510, 1493, 1476, 1461, 1445, 1431, 1416, 1402,
      1389, 1376, 1363, 1351, 1339, 1327, 1315, 1304,
      1293, 1283, 1272, 1262, 1252, 1243, 1233, 1224,
      1215, 1206, 1197, 1189, 1181, 1173, 1165, 1157,
      1149, 1142, 1134, 1127, 1120, 1113, 1106, 1100,
      1093, 1087, 1080, 1074, 1068, 1062, 1056, 1050,
      1045, 1039, 1034, 1028, 1023, 1017, 1012, 1007,
      1002, 997, 992, 987, 983, 978, 973, 969,
      964, 960, 955, 951, 947, 943, 939, 935,
      930, 926, 923, 919, 915, 911, 907, 904,
      900, 896, 893, 889, 886, 882, 879, 876,
      872, 869, 866, 863, 859, 856, 853, 850
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<128>(variance_sum, -6105211, 19, 127, table);
}

static inline int hgtxr_hgpipe_attn2_lnq_mean_int(int sum) {
#pragma HLS INLINE
  long long tmp = static_cast<long long>(sum) * 43691;
  tmp += (1LL << (23 - 1));
  return static_cast<int>(tmp >> 23);
}

static inline int hgtxr_hgpipe_attn2_lnq_requant_int(int input_value,
                                                     int mean,
                                                     int rsqrt,
                                                     int lnw,
                                                     long long lnb) {
#pragma HLS INLINE
  long long diff = static_cast<long long>(input_value - mean);
  long long val = diff * static_cast<long long>(rsqrt) * static_cast<long long>(lnw) + lnb;
  long long rel = val >> 33;
  return hgtxr_clamp_int(static_cast<int>(rel), -4, 3);
}

static inline int hgtxr_hgpipe_attn3_lnq_rsqrt128_int(int variance_sum) {
#pragma HLS INLINE
  static const int table[128] = {
      3121, 3048, 2980, 2916, 2857, 2800, 2747, 2697,
      2650, 2605, 2562, 2521, 2482, 2445, 2410, 2376,
      2343, 2312, 2281, 2253, 2225, 2198, 2172, 2147,
      2123, 2099, 2077, 2055, 2034, 2013, 1993, 1974,
      1955, 1936, 1919, 1901, 1884, 1868, 1852, 1836,
      1821, 1806, 1792, 1778, 1764, 1751, 1737, 1724,
      1712, 1700, 1687, 1676, 1664, 1653, 1642, 1631,
      1620, 1610, 1599, 1589, 1579, 1570, 1560, 1551,
      1542, 1533, 1524, 1515, 1506, 1498, 1490, 1482,
      1474, 1466, 1458, 1450, 1443, 1435, 1428, 1421,
      1414, 1407, 1400, 1393, 1386, 1380, 1373, 1367,
      1361, 1354, 1348, 1342, 1336, 1330, 1325, 1319,
      1313, 1308, 1302, 1297, 1291, 1286, 1281, 1275,
      1270, 1265, 1260, 1255, 1250, 1246, 1241, 1236,
      1231, 1227, 1222, 1218, 1213, 1209, 1204, 1200,
      1196, 1192, 1187, 1183, 1179, 1175, 1171, 1167
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<128>(variance_sum, -5285370, 18, 127, table);
}

static inline int hgtxr_hgpipe_attn3_lnq_mean_int(int sum) {
#pragma HLS INLINE
  long long tmp = static_cast<long long>(sum) * 43691;
  tmp += (1LL << (23 - 1));
  return static_cast<int>(tmp >> 23);
}

static inline int hgtxr_hgpipe_attn3_lnq_requant_int(int input_value,
                                                     int mean,
                                                     int rsqrt,
                                                     int lnw,
                                                     long long lnb) {
#pragma HLS INLINE
  long long diff = static_cast<long long>(input_value - mean);
  long long val = diff * static_cast<long long>(rsqrt) * static_cast<long long>(lnw) + lnb;
  long long rel = val >> 33;
  return hgtxr_clamp_int(static_cast<int>(rel), -4, 3);
}

static inline int hgtxr_hgpipe_attn4_lnq_rsqrt128_int(int variance_sum) {
#pragma HLS INLINE
  static const int table[128] = {
      3058, 2989, 2925, 2865, 2808, 2754, 2704, 2656,
      2611, 2568, 2527, 2487, 2450, 2414, 2380, 2347,
      2316, 2286, 2256, 2228, 2201, 2175, 2150, 2126,
      2102, 2080, 2058, 2036, 2016, 1996, 1976, 1957,
      1939, 1921, 1904, 1887, 1870, 1854, 1839, 1823,
      1808, 1794, 1780, 1766, 1752, 1739, 1726, 1714,
      1701, 1689, 1677, 1666, 1654, 1643, 1632, 1621,
      1611, 1601, 1591, 1581, 1571, 1561, 1552, 1543,
      1534, 1525, 1516, 1508, 1499, 1491, 1483, 1475,
      1467, 1459, 1451, 1444, 1436, 1429, 1422, 1415,
      1408, 1401, 1394, 1387, 1381, 1374, 1368, 1362,
      1355, 1349, 1343, 1337, 1331, 1325, 1320, 1314,
      1308, 1303, 1297, 1292, 1287, 1281, 1276, 1271,
      1266, 1261, 1256, 1251, 1246, 1241, 1237, 1232,
      1227, 1223, 1218, 1214, 1209, 1205, 1201, 1196,
      1192, 1188, 1184, 1180, 1176, 1172, 1168, 1164
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<128>(variance_sum, -5511642, 18, 127, table);
}

static inline int hgtxr_hgpipe_attn4_lnq_mean_int(int sum) {
#pragma HLS INLINE
  long long tmp = static_cast<long long>(sum) * 43691;
  tmp += (1LL << (23 - 1));
  return static_cast<int>(tmp >> 23);
}

static inline int hgtxr_hgpipe_attn4_lnq_requant_int(int input_value,
                                                     int mean,
                                                     int rsqrt,
                                                     int lnw,
                                                     long long lnb) {
#pragma HLS INLINE
  long long diff = static_cast<long long>(input_value - mean);
  long long val = diff * static_cast<long long>(rsqrt) * static_cast<long long>(lnw) + lnb;
  long long rel = val >> 33;
  return hgtxr_clamp_int(static_cast<int>(rel), -4, 3);
}

static inline int hgtxr_hgpipe_attn5_lnq_rsqrt128_int(int variance_sum) {
#pragma HLS INLINE
  static const int table[128] = {
      3017, 2951, 2889, 2831, 2776, 2724, 2675, 2629,
      2585, 2543, 2503, 2465, 2429, 2394, 2361, 2329,
      2298, 2268, 2240, 2212, 2186, 2160, 2136, 2112,
      2089, 2067, 2045, 2024, 2004, 1984, 1965, 1946,
      1928, 1911, 1894, 1877, 1861, 1845, 1830, 1815,
      1800, 1786, 1772, 1758, 1745, 1731, 1719, 1706,
      1694, 1682, 1670, 1659, 1648, 1637, 1626, 1615,
      1605, 1595, 1585, 1575, 1565, 1556, 1547, 1537,
      1529, 1520, 1511, 1503, 1494, 1486, 1478, 1470,
      1462, 1454, 1447, 1439, 1432, 1425, 1418, 1411,
      1404, 1397, 1390, 1383, 1377, 1370, 1364, 1358,
      1352, 1346, 1340, 1334, 1328, 1322, 1316, 1311,
      1305, 1300, 1294, 1289, 1284, 1278, 1273, 1268,
      1263, 1258, 1253, 1248, 1243, 1239, 1234, 1229,
      1225, 1220, 1216, 1211, 1207, 1202, 1198, 1194,
      1190, 1186, 1181, 1177, 1173, 1169, 1165, 1161
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<128>(variance_sum, -5666093, 18, 127, table);
}

static inline int hgtxr_hgpipe_attn5_lnq_mean_int(int sum) {
#pragma HLS INLINE
  long long tmp = static_cast<long long>(sum) * 43691;
  tmp += (1LL << (23 - 1));
  return static_cast<int>(tmp >> 23);
}

static inline int hgtxr_hgpipe_attn5_lnq_requant_int(int input_value,
                                                     int mean,
                                                     int rsqrt,
                                                     int lnw,
                                                     long long lnb) {
#pragma HLS INLINE
  long long diff = static_cast<long long>(input_value - mean);
  long long val = diff * static_cast<long long>(rsqrt) * static_cast<long long>(lnw) + lnb;
  long long rel = val >> 33;
  return hgtxr_clamp_int(static_cast<int>(rel), -4, 3);
}

static inline int hgtxr_hgpipe_attn6_lnq_rsqrt128_int(int variance_sum) {
#pragma HLS INLINE
  static const int table[128] = {
      3602, 3491, 3390, 3297, 3211, 3132, 3059, 2990,
      2926, 2865, 2809, 2755, 2705, 2657, 2611, 2568,
      2527, 2488, 2450, 2415, 2380, 2348, 2316, 2286,
      2257, 2229, 2202, 2176, 2150, 2126, 2103, 2080,
      2058, 2037, 2016, 1996, 1976, 1957, 1939, 1921,
      1904, 1887, 1870, 1854, 1839, 1823, 1809, 1794,
      1780, 1766, 1753, 1739, 1726, 1714, 1701, 1689,
      1677, 1666, 1654, 1643, 1632, 1622, 1611, 1601,
      1591, 1581, 1571, 1562, 1552, 1543, 1534, 1525,
      1516, 1508, 1499, 1491, 1483, 1475, 1467, 1459,
      1451, 1444, 1436, 1429, 1422, 1415, 1408, 1401,
      1394, 1387, 1381, 1374, 1368, 1362, 1355, 1349,
      1343, 1337, 1331, 1325, 1320, 1314, 1308, 1303,
      1297, 1292, 1287, 1281, 1276, 1271, 1266, 1261,
      1256, 1251, 1246, 1242, 1237, 1232, 1227, 1223,
      1218, 1214, 1209, 1205, 1201, 1196, 1192, 1188
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<128>(variance_sum, -3935884, 18, 127, table);
}

static inline int hgtxr_hgpipe_attn6_lnq_mean_int(int sum) {
#pragma HLS INLINE
  long long tmp = static_cast<long long>(sum) * 43691;
  tmp += (1LL << (23 - 1));
  return static_cast<int>(tmp >> 23);
}

static inline int hgtxr_hgpipe_attn6_lnq_requant_int(int input_value,
                                                     int mean,
                                                     int rsqrt,
                                                     int lnw,
                                                     long long lnb) {
#pragma HLS INLINE
  long long diff = static_cast<long long>(input_value - mean);
  long long val = diff * static_cast<long long>(rsqrt) * static_cast<long long>(lnw) + lnb;
  long long rel = val >> 33;
  return hgtxr_clamp_int(static_cast<int>(rel), -4, 3);
}

static inline int hgtxr_hgpipe_attn7_lnq_rsqrt128_int(int variance_sum) {
#pragma HLS INLINE
  static const int table[128] = {
      2077, 1919, 1792, 1688, 1600, 1524, 1458, 1400,
      1349, 1302, 1260, 1222, 1188, 1156, 1126, 1099,
      1073, 1049, 1027, 1006, 987, 968, 950, 934,
      918, 903, 889, 875, 862, 850, 838, 826,
      815, 805, 794, 785, 775, 766, 757, 749,
      741, 733, 725, 717, 710, 703, 696, 690,
      683, 677, 671, 665, 659, 654, 648, 643,
      637, 632, 627, 623, 618, 613, 609, 604,
      600, 596, 591, 587, 583, 579, 576, 572,
      568, 565, 561, 557, 554, 551, 547, 544,
      541, 538, 535, 532, 529, 526, 523, 520,
      517, 515, 512, 509, 507, 504, 502, 499,
      497, 494, 492, 489, 487, 485, 483, 480,
      478, 476, 474, 472, 470, 468, 466, 464,
      462, 460, 458, 456, 454, 452, 450, 449,
      447, 445, 443, 441, 440, 438, 436, 435
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<128>(variance_sum, -2793443, 19, 127, table);
}

static inline int hgtxr_hgpipe_attn7_lnq_mean_int(int sum) {
#pragma HLS INLINE
  long long tmp = static_cast<long long>(sum) * 43691;
  tmp += (1LL << (23 - 1));
  return static_cast<int>(tmp >> 23);
}

static inline int hgtxr_hgpipe_attn7_lnq_requant_int(int input_value,
                                                     int mean,
                                                     int rsqrt,
                                                     int lnw,
                                                     long long lnb) {
#pragma HLS INLINE
  long long diff = static_cast<long long>(input_value - mean);
  long long val = diff * static_cast<long long>(rsqrt) * static_cast<long long>(lnw) + lnb;
  long long rel = val >> 32;
  return hgtxr_clamp_int(static_cast<int>(rel), -4, 3);
}

static inline int hgtxr_hgpipe_attn8_lnq_rsqrt128_int(int variance_sum) {
#pragma HLS INLINE
  static const int table[128] = {
      2338, 2119, 1952, 1819, 1710, 1618, 1540, 1472,
      1413, 1360, 1312, 1270, 1231, 1195, 1163, 1133,
      1105, 1079, 1055, 1032, 1011, 991, 972, 954,
      938, 922, 906, 892, 878, 865, 852, 840,
      829, 818, 807, 797, 787, 777, 768, 759,
      751, 742, 734, 727, 719, 712, 705, 698,
      691, 685, 678, 672, 666, 660, 655, 649,
      644, 639, 634, 629, 624, 619, 614, 610,
      605, 601, 597, 592, 588, 584, 580, 576,
      573, 569, 565, 562, 558, 555, 551, 548,
      545, 542, 539, 535, 532, 529, 527, 524,
      521, 518, 515, 513, 510, 507, 505, 502,
      500, 497, 495, 492, 490, 488, 485, 483,
      481, 479, 477, 474, 472, 470, 468, 466,
      464, 462, 460, 458, 456, 454, 453, 451,
      449, 447, 445, 444, 442, 440, 438, 437
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<128>(variance_sum, -2150450, 19, 127, table);
}

static inline int hgtxr_hgpipe_attn8_lnq_mean_int(int sum) {
#pragma HLS INLINE
  long long tmp = static_cast<long long>(sum) * 43691;
  tmp += (1LL << (23 - 1));
  return static_cast<int>(tmp >> 23);
}

static inline int hgtxr_hgpipe_attn8_lnq_requant_int(int input_value,
                                                     int mean,
                                                     int rsqrt,
                                                     int lnw,
                                                     long long lnb) {
#pragma HLS INLINE
  long long diff = static_cast<long long>(input_value - mean);
  long long val = diff * static_cast<long long>(rsqrt) * static_cast<long long>(lnw) + lnb;
  long long rel = val >> 32;
  return hgtxr_clamp_int(static_cast<int>(rel), -4, 3);
}

static inline int hgtxr_hgpipe_attn9_lnq_rsqrt128_int(int variance_sum) {
#pragma HLS INLINE
  static const int table[128] = {
      2731, 2549, 2398, 2272, 2164, 2070, 1987, 1913,
      1847, 1787, 1733, 1684, 1638, 1596, 1557, 1521,
      1487, 1455, 1426, 1398, 1371, 1346, 1323, 1300,
      1279, 1259, 1239, 1221, 1203, 1186, 1170, 1154,
      1139, 1125, 1111, 1098, 1085, 1072, 1060, 1048,
      1037, 1026, 1016, 1005, 995, 986, 976, 967,
      958, 950, 941, 933, 925, 917, 910, 902,
      895, 888, 881, 874, 868, 861, 855, 849,
      843, 837, 831, 826, 820, 815, 809, 804,
      799, 794, 789, 784, 779, 775, 770, 766,
      761, 757, 753, 748, 744, 740, 736, 732,
      728, 725, 721, 717, 713, 710, 706, 703,
      699, 696, 693, 689, 686, 683, 680, 677,
      674, 671, 668, 665, 662, 659, 656, 653,
      651, 648, 645, 642, 640, 637, 635, 632,
      630, 627, 625, 622, 620, 618, 615, 613
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<128>(variance_sum, -1637294, 18, 127, table);
}

static inline int hgtxr_hgpipe_attn9_lnq_mean_int(int sum) {
#pragma HLS INLINE
  long long tmp = static_cast<long long>(sum) * 43691;
  tmp += (1LL << (23 - 1));
  return static_cast<int>(tmp >> 23);
}

static inline int hgtxr_hgpipe_attn9_lnq_requant_int(int input_value,
                                                     int mean,
                                                     int rsqrt,
                                                     int lnw,
                                                     long long lnb) {
#pragma HLS INLINE
  long long diff = static_cast<long long>(input_value - mean);
  long long val = diff * static_cast<long long>(rsqrt) * static_cast<long long>(lnw) + lnb;
  long long rel = val >> 32;
  return hgtxr_clamp_int(static_cast<int>(rel), -4, 3);
}

static inline int hgtxr_hgpipe_attn10_lnq_rsqrt128_int(int variance_sum) {
#pragma HLS INLINE
  static const int table[128] = {
      3102, 2842, 2638, 2473, 2335, 2218, 2117, 2028,
      1950, 1880, 1818, 1761, 1709, 1661, 1617, 1577,
      1539, 1504, 1472, 1441, 1412, 1385, 1359, 1335,
      1312, 1290, 1269, 1249, 1230, 1212, 1195, 1178,
      1162, 1147, 1132, 1118, 1104, 1091, 1079, 1066,
      1054, 1043, 1032, 1021, 1011, 1001, 991, 981,
      972, 963, 954, 946, 937, 929, 921, 914,
      906, 899, 892, 885, 878, 871, 865, 858,
      852, 846, 840, 834, 829, 823, 817, 812,
      807, 802, 797, 792, 787, 782, 777, 773,
      768, 764, 759, 755, 751, 746, 742, 738,
      734, 730, 727, 723, 719, 715, 712, 708,
      705, 701, 698, 694, 691, 688, 685, 681,
      678, 675, 672, 669, 666, 663, 660, 658,
      655, 652, 649, 647, 644, 641, 639, 636,
      633, 631, 628, 626, 624, 621, 619, 616
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<128>(variance_sum, -1239294, 18, 127, table);
}

static inline int hgtxr_hgpipe_attn10_lnq_mean_int(int sum) {
#pragma HLS INLINE
  long long tmp = static_cast<long long>(sum) * 43691;
  tmp += (1LL << (23 - 1));
  return static_cast<int>(tmp >> 23);
}

static inline int hgtxr_hgpipe_attn10_lnq_requant_int(int input_value,
                                                      int mean,
                                                      int rsqrt,
                                                      int lnw,
                                                      long long lnb) {
#pragma HLS INLINE
  long long diff = static_cast<long long>(input_value - mean);
  long long val = diff * static_cast<long long>(rsqrt) * static_cast<long long>(lnw) + lnb;
  long long rel = val >> 32;
  return hgtxr_clamp_int(static_cast<int>(rel), -4, 3);
}

static inline int hgtxr_hgpipe_attn11_lnq_rsqrt128_int(int variance_sum) {
#pragma HLS INLINE
  static const int table[128] = {
      2138, 1967, 1831, 1720, 1627, 1548, 1479, 1418,
      1365, 1317, 1274, 1234, 1199, 1166, 1135, 1107,
      1081, 1057, 1034, 1013, 993, 974, 956, 939,
      923, 908, 893, 879, 866, 854, 841, 830,
      819, 808, 798, 788, 778, 769, 760, 752,
      743, 735, 727, 720, 713, 705, 699, 692,
      685, 679, 673, 667, 661, 655, 650, 644,
      639, 634, 629, 624, 619, 615, 610, 606,
      601, 597, 593, 589, 585, 581, 577, 573,
      569, 566, 562, 559, 555, 552, 548, 545,
      542, 539, 536, 533, 530, 527, 524, 521,
      518, 516, 513, 510, 508, 505, 502, 500,
      497, 495, 493, 490, 488, 486, 483, 481,
      479, 477, 475, 472, 470, 468, 466, 464,
      462, 460, 458, 456, 455, 453, 451, 449,
      447, 446, 444, 442, 440, 439, 437, 435
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<128>(variance_sum, -655549, 17, 127, table);
}

static inline int hgtxr_hgpipe_attn11_lnq_mean_int(int sum) {
#pragma HLS INLINE
  long long tmp = static_cast<long long>(sum) * 43691;
  tmp += (1LL << (23 - 1));
  return static_cast<int>(tmp >> 23);
}

static inline int hgtxr_hgpipe_attn11_lnq_requant_int(int input_value,
                                                      int mean,
                                                      int rsqrt,
                                                      int lnw,
                                                      long long lnb) {
#pragma HLS INLINE
  long long diff = static_cast<long long>(input_value - mean);
  long long val = diff * static_cast<long long>(rsqrt) * static_cast<long long>(lnw) + lnb;
  long long rel = val >> 31;
  return hgtxr_clamp_int(static_cast<int>(rel), -4, 3);
}

static inline int hgtxr_hgpipe_mlp0_lnq_rsqrt128_int(int variance_sum) {
#pragma HLS INLINE
  static const int table[128] = {
      2449, 1752, 1436, 1246, 1116, 1019, 944, 884,
      833, 791, 754, 722, 694, 669, 646, 626,
      607, 590, 574, 560, 546, 534, 522, 511,
      501, 491, 482, 473, 465, 457, 450, 443,
      436, 429, 423, 417, 412, 406, 401, 396,
      391, 386, 382, 377, 373, 369, 365, 361,
      358, 354, 351, 347, 344, 341, 338, 335,
      332, 329, 326, 323, 321, 318, 315, 313,
      310, 308, 306, 304, 301, 299, 297, 295,
      293, 291, 289, 287, 285, 283, 282, 280,
      278, 276, 275, 273, 271, 270, 268, 267,
      265, 264, 262, 261, 260, 258, 257, 255,
      254, 253, 252, 250, 249, 248, 247, 245,
      244, 243, 242, 241, 240, 239, 238, 236,
      235, 234, 233, 232, 231, 230, 229, 228,
      227, 227, 226, 225, 224, 223, 222, 221
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<128>(variance_sum, -4602380, 23, 127, table);
}

static inline int hgtxr_hgpipe_mlp0_lnq_mean_int(int sum) {
#pragma HLS INLINE
  long long tmp = static_cast<long long>(sum) * 43691;
  tmp += (1LL << (23 - 1));
  return static_cast<int>(tmp >> 23);
}

static inline int hgtxr_hgpipe_mlp0_lnq_requant_int(int input_value,
                                                    int mean,
                                                    int rsqrt,
                                                    int lnw,
                                                    long long lnb) {
#pragma HLS INLINE
  long long diff = static_cast<long long>(input_value - mean);
  long long val = diff * static_cast<long long>(rsqrt) * static_cast<long long>(lnw) + lnb;
  long long rel = val >> 32;
  return hgtxr_clamp_int(static_cast<int>(rel), -4, 3);
}

static inline int hgtxr_hgpipe_mlp1_lnq_rsqrt128_int(int variance_sum) {
#pragma HLS INLINE
  static const int table[128] = {
      3109, 2642, 2338, 2119, 1952, 1819, 1710, 1618,
      1540, 1472, 1413, 1360, 1312, 1270, 1231, 1195,
      1163, 1133, 1105, 1079, 1055, 1032, 1011, 991,
      972, 954, 937, 922, 906, 892, 878, 865,
      852, 840, 829, 818, 807, 797, 787, 777,
      768, 759, 751, 742, 734, 727, 719, 712,
      705, 698, 691, 685, 678, 672, 666, 660,
      655, 649, 644, 639, 634, 629, 624, 619,
      614, 610, 605, 601, 597, 592, 588, 584,
      580, 576, 573, 569, 565, 562, 558, 555,
      551, 548, 545, 542, 539, 535, 532, 529,
      527, 524, 521, 518, 515, 513, 510, 507,
      505, 502, 500, 497, 495, 492, 490, 488,
      485, 483, 481, 479, 477, 474, 472, 470,
      468, 466, 464, 462, 460, 458, 456, 454,
      453, 451, 449, 447, 445, 444, 442, 440
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<128>(variance_sum, -4409734, 21, 127, table);
}

static inline int hgtxr_hgpipe_mlp1_lnq_mean_int(int sum) {
#pragma HLS INLINE
  long long tmp = static_cast<long long>(sum) * 43691;
  tmp += (1LL << (23 - 1));
  return static_cast<int>(tmp >> 23);
}

static inline int hgtxr_hgpipe_mlp1_lnq_requant_int(int input_value,
                                                    int mean,
                                                    int rsqrt,
                                                    int lnw,
                                                    long long lnb) {
#pragma HLS INLINE
  long long diff = static_cast<long long>(input_value - mean);
  long long val = diff * static_cast<long long>(rsqrt) * static_cast<long long>(lnw) + lnb;
  long long rel = val >> 33;
  return hgtxr_clamp_int(static_cast<int>(rel), -4, 3);
}

static inline int hgtxr_hgpipe_mlp2_lnq_rsqrt128_int(int variance_sum) {
#pragma HLS INLINE
  static const int table[128] = {
      2365, 2243, 2139, 2048, 1967, 1896, 1831, 1773,
      1720, 1672, 1627, 1586, 1548, 1512, 1479, 1448,
      1418, 1391, 1365, 1340, 1317, 1295, 1274, 1254,
      1235, 1216, 1199, 1182, 1166, 1150, 1136, 1121,
      1108, 1094, 1081, 1069, 1057, 1046, 1034, 1024,
      1013, 1003, 993, 983, 974, 965, 956, 948,
      939, 931, 923, 915, 908, 901, 893, 886,
      879, 873, 866, 860, 854, 847, 841, 836,
      830, 824, 819, 813, 808, 803, 798, 793,
      788, 783, 778, 774, 769, 765, 760, 756,
      752, 747, 743, 739, 735, 731, 727, 724,
      720, 716, 713, 709, 705, 702, 699, 695,
      692, 689, 685, 682, 679, 676, 673, 670,
      667, 664, 661, 658, 655, 653, 650, 647,
      644, 642, 639, 637, 634, 632, 629, 627,
      624, 622, 619, 617, 615, 612, 610, 608
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<128>(variance_sum, -8909606, 20, 127, table);
}

static inline int hgtxr_hgpipe_mlp2_lnq_mean_int(int sum) {
#pragma HLS INLINE
  long long tmp = static_cast<long long>(sum) * 43691;
  tmp += (1LL << (23 - 1));
  return static_cast<int>(tmp >> 23);
}

static inline int hgtxr_hgpipe_mlp2_lnq_requant_int(int input_value,
                                                    int mean,
                                                    int rsqrt,
                                                    int lnw,
                                                    long long lnb) {
#pragma HLS INLINE
  long long diff = static_cast<long long>(input_value - mean);
  long long val = diff * static_cast<long long>(rsqrt) * static_cast<long long>(lnw) + lnb;
  long long rel = val >> 33;
  return hgtxr_clamp_int(static_cast<int>(rel), -4, 3);
}

static inline int hgtxr_hgpipe_mlp3_lnq_rsqrt128_int(int variance_sum) {
#pragma HLS INLINE
  static const int table[128] = {
      2963, 2842, 2734, 2638, 2551, 2472, 2400, 2334,
      2274, 2217, 2165, 2116, 2071, 2028, 1988, 1950,
      1914, 1880, 1848, 1817, 1788, 1760, 1734, 1709,
      1684, 1661, 1639, 1617, 1597, 1577, 1558, 1539,
      1521, 1504, 1488, 1471, 1456, 1441, 1426, 1412,
      1398, 1385, 1372, 1359, 1347, 1335, 1323, 1312,
      1301, 1290, 1279, 1269, 1259, 1249, 1240, 1230,
      1221, 1212, 1203, 1195, 1186, 1178, 1170, 1162,
      1154, 1147, 1139, 1132, 1125, 1118, 1111, 1104,
      1098, 1091, 1085, 1078, 1072, 1066, 1060, 1054,
      1049, 1043, 1037, 1032, 1026, 1021, 1016, 1011,
      1006, 1000, 996, 991, 986, 981, 976, 972,
      967, 963, 958, 954, 950, 946, 941, 937,
      933, 929, 925, 921, 917, 914, 910, 906,
      902, 899, 895, 892, 888, 885, 881, 878,
      875, 871, 868, 865, 862, 858, 855, 852
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<128>(variance_sum, -5747680, 19, 127, table);
}

static inline int hgtxr_hgpipe_mlp3_lnq_mean_int(int sum) {
#pragma HLS INLINE
  long long tmp = static_cast<long long>(sum) * 43691;
  tmp += (1LL << (23 - 1));
  return static_cast<int>(tmp >> 23);
}

static inline int hgtxr_hgpipe_mlp3_lnq_requant_int(int input_value,
                                                    int mean,
                                                    int rsqrt,
                                                    int lnw,
                                                    long long lnb) {
#pragma HLS INLINE
  long long diff = static_cast<long long>(input_value - mean);
  long long val = diff * static_cast<long long>(rsqrt) * static_cast<long long>(lnw) + lnb;
  long long rel = val >> 33;
  return hgtxr_clamp_int(static_cast<int>(rel), -4, 3);
}

static inline int hgtxr_hgpipe_mlp4_lnq_rsqrt128_int(int variance_sum) {
#pragma HLS INLINE
  static const int table[128] = {
      2756, 2705, 2657, 2612, 2569, 2528, 2488, 2451,
      2415, 2381, 2348, 2317, 2286, 2257, 2229, 2202,
      2176, 2151, 2126, 2103, 2080, 2058, 2037, 2016,
      1996, 1977, 1958, 1939, 1921, 1904, 1887, 1871,
      1855, 1839, 1824, 1809, 1794, 1780, 1766, 1753,
      1739, 1727, 1714, 1701, 1689, 1678, 1666, 1655,
      1643, 1632, 1622, 1611, 1601, 1591, 1581, 1571,
      1562, 1552, 1543, 1534, 1525, 1516, 1508, 1499,
      1491, 1483, 1475, 1467, 1459, 1451, 1444, 1436,
      1429, 1422, 1415, 1408, 1401, 1394, 1388, 1381,
      1374, 1368, 1362, 1355, 1349, 1343, 1337, 1331,
      1326, 1320, 1314, 1308, 1303, 1297, 1292, 1287,
      1281, 1276, 1271, 1266, 1261, 1256, 1251, 1246,
      1242, 1237, 1232, 1228, 1223, 1218, 1214, 1210,
      1205, 1201, 1197, 1192, 1188, 1184, 1180, 1176,
      1172, 1168, 1164, 1160, 1156, 1152, 1148, 1145
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<128>(variance_sum, -6816111, 18, 127, table);
}

static inline int hgtxr_hgpipe_mlp4_lnq_mean_int(int sum) {
#pragma HLS INLINE
  long long tmp = static_cast<long long>(sum) * 43691;
  tmp += (1LL << (23 - 1));
  return static_cast<int>(tmp >> 23);
}

static inline int hgtxr_hgpipe_mlp4_lnq_requant_int(int input_value,
                                                    int mean,
                                                    int rsqrt,
                                                    int lnw,
                                                    long long lnb) {
#pragma HLS INLINE
  long long diff = static_cast<long long>(input_value - mean);
  long long val = diff * static_cast<long long>(rsqrt) * static_cast<long long>(lnw) + lnb;
  long long rel = val >> 33;
  return hgtxr_clamp_int(static_cast<int>(rel), -4, 3);
}

static inline int hgtxr_hgpipe_mlp5_lnq_rsqrt128_int(int variance_sum) {
#pragma HLS INLINE
  static const int table[128] = {
      2707, 2659, 2613, 2570, 2529, 2490, 2452, 2416,
      2382, 2349, 2318, 2287, 2258, 2230, 2203, 2177,
      2152, 2127, 2104, 2081, 2059, 2038, 2017, 1997,
      1977, 1958, 1940, 1922, 1905, 1888, 1871, 1855,
      1839, 1824, 1809, 1795, 1781, 1767, 1753, 1740,
      1727, 1714, 1702, 1690, 1678, 1666, 1655, 1644,
      1633, 1622, 1612, 1601, 1591, 1581, 1572, 1562,
      1553, 1543, 1534, 1525, 1517, 1508, 1500, 1491,
      1483, 1475, 1467, 1459, 1452, 1444, 1437, 1429,
      1422, 1415, 1408, 1401, 1394, 1388, 1381, 1375,
      1368, 1362, 1356, 1349, 1343, 1337, 1332, 1326,
      1320, 1314, 1309, 1303, 1298, 1292, 1287, 1282,
      1276, 1271, 1266, 1261, 1256, 1251, 1247, 1242,
      1237, 1232, 1228, 1223, 1219, 1214, 1210, 1205,
      1201, 1197, 1192, 1188, 1184, 1180, 1176, 1172,
      1168, 1164, 1160, 1156, 1152, 1149, 1145, 1141
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<128>(variance_sum, -7069397, 18, 127, table);
}

static inline int hgtxr_hgpipe_mlp5_lnq_mean_int(int sum) {
#pragma HLS INLINE
  long long tmp = static_cast<long long>(sum) * 43691;
  tmp += (1LL << (23 - 1));
  return static_cast<int>(tmp >> 23);
}

static inline int hgtxr_hgpipe_mlp5_lnq_requant_int(int input_value,
                                                    int mean,
                                                    int rsqrt,
                                                    int lnw,
                                                    long long lnb) {
#pragma HLS INLINE
  long long diff = static_cast<long long>(input_value - mean);
  long long val = diff * static_cast<long long>(rsqrt) * static_cast<long long>(lnw) + lnb;
  long long rel = val >> 33;
  return hgtxr_clamp_int(static_cast<int>(rel), -4, 3);
}

static inline int hgtxr_hgpipe_mlp6_lnq_rsqrt128_int(int variance_sum) {
#pragma HLS INLINE
  static const int table[128] = {
      2894, 2780, 2679, 2589, 2506, 2432, 2363, 2300,
      2242, 2188, 2138, 2091, 2047, 2005, 1967, 1930,
      1895, 1862, 1831, 1801, 1773, 1746, 1720, 1695,
      1671, 1649, 1627, 1606, 1586, 1566, 1547, 1529,
      1512, 1495, 1479, 1463, 1447, 1433, 1418, 1404,
      1391, 1377, 1365, 1352, 1340, 1328, 1317, 1306,
      1295, 1284, 1274, 1263, 1253, 1244, 1234, 1225,
      1216, 1207, 1199, 1190, 1182, 1174, 1166, 1158,
      1150, 1143, 1135, 1128, 1121, 1114, 1107, 1101,
      1094, 1088, 1081, 1075, 1069, 1063, 1057, 1051,
      1045, 1040, 1034, 1029, 1023, 1018, 1013, 1008,
      1003, 998, 993, 988, 983, 979, 974, 969,
      965, 960, 956, 952, 947, 943, 939, 935,
      931, 927, 923, 919, 915, 912, 908, 904,
      900, 897, 893, 890, 886, 883, 879, 876,
      873, 869, 866, 863, 860, 857, 854, 850
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<128>(variance_sum, -6037980, 19, 127, table);
}

static inline int hgtxr_hgpipe_mlp6_lnq_mean_int(int sum) {
#pragma HLS INLINE
  long long tmp = static_cast<long long>(sum) * 43691;
  tmp += (1LL << (23 - 1));
  return static_cast<int>(tmp >> 23);
}

static inline int hgtxr_hgpipe_mlp6_lnq_requant_int(int input_value,
                                                    int mean,
                                                    int rsqrt,
                                                    int lnw,
                                                    long long lnb) {
#pragma HLS INLINE
  long long diff = static_cast<long long>(input_value - mean);
  long long val = diff * static_cast<long long>(rsqrt) * static_cast<long long>(lnw) + lnb;
  long long rel = val >> 33;
  return hgtxr_clamp_int(static_cast<int>(rel), -4, 3);
}

static inline int hgtxr_hgpipe_mlp7_lnq_rsqrt128_int(int variance_sum) {
#pragma HLS INLINE
  static const int table[128] = {
      3632, 3415, 3232, 3077, 2941, 2823, 2717, 2623,
      2537, 2460, 2389, 2324, 2264, 2208, 2157, 2109,
      2063, 2021, 1981, 1944, 1908, 1875, 1843, 1812,
      1783, 1756, 1730, 1704, 1680, 1657, 1635, 1614,
      1593, 1574, 1554, 1536, 1518, 1501, 1485, 1469,
      1453, 1438, 1424, 1410, 1396, 1382, 1370, 1357,
      1345, 1333, 1321, 1310, 1299, 1288, 1278, 1267,
      1257, 1248, 1238, 1229, 1220, 1211, 1202, 1193,
      1185, 1177, 1169, 1161, 1153, 1146, 1138, 1131,
      1124, 1117, 1110, 1103, 1097, 1090, 1084, 1077,
      1071, 1065, 1059, 1053, 1048, 1042, 1036, 1031,
      1025, 1020, 1015, 1010, 1005, 1000, 995, 990,
      985, 980, 976, 971, 967, 962, 958, 953,
      949, 945, 941, 937, 933, 929, 925, 921,
      917, 913, 909, 906, 902, 898, 895, 891,
      888, 884, 881, 877, 874, 871, 867, 864
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<128>(variance_sum, -3738640, 19, 127, table);
}

static inline int hgtxr_hgpipe_mlp7_lnq_mean_int(int sum) {
#pragma HLS INLINE
  long long tmp = static_cast<long long>(sum) * 43691;
  tmp += (1LL << (23 - 1));
  return static_cast<int>(tmp >> 23);
}

static inline int hgtxr_hgpipe_mlp7_lnq_requant_int(int input_value,
                                                    int mean,
                                                    int rsqrt,
                                                    int lnw,
                                                    long long lnb) {
#pragma HLS INLINE
  long long diff = static_cast<long long>(input_value - mean);
  long long val = diff * static_cast<long long>(rsqrt) * static_cast<long long>(lnw) + lnb;
  long long rel = val >> 33;
  return hgtxr_clamp_int(static_cast<int>(rel), -4, 3);
}

static inline int hgtxr_hgpipe_mlp8_lnq_rsqrt128_int(int variance_sum) {
#pragma HLS INLINE
  static const int table[128] = {
      3532, 3331, 3162, 3015, 2888, 2775, 2675, 2584,
      2502, 2428, 2360, 2297, 2239, 2185, 2135, 2088,
      2045, 2003, 1965, 1928, 1893, 1860, 1829, 1800,
      1771, 1744, 1718, 1694, 1670, 1647, 1626, 1605,
      1585, 1565, 1546, 1528, 1511, 1494, 1478, 1462,
      1447, 1432, 1417, 1403, 1390, 1377, 1364, 1352,
      1339, 1328, 1316, 1305, 1294, 1283, 1273, 1263,
      1253, 1243, 1234, 1225, 1216, 1207, 1198, 1190,
      1181, 1173, 1165, 1158, 1150, 1142, 1135, 1128,
      1121, 1114, 1107, 1100, 1094, 1087, 1081, 1075,
      1069, 1063, 1057, 1051, 1045, 1040, 1034, 1029,
      1023, 1018, 1013, 1008, 1002, 997, 993, 988,
      983, 978, 974, 969, 965, 960, 956, 952,
      947, 943, 939, 935, 931, 927, 923, 919,
      915, 911, 908, 904, 900, 897, 893, 890,
      886, 883, 879, 876, 873, 869, 866, 863
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<128>(variance_sum, -3967626, 19, 127, table);
}

static inline int hgtxr_hgpipe_mlp8_lnq_mean_int(int sum) {
#pragma HLS INLINE
  long long tmp = static_cast<long long>(sum) * 43691;
  tmp += (1LL << (23 - 1));
  return static_cast<int>(tmp >> 23);
}

static inline int hgtxr_hgpipe_mlp8_lnq_requant_int(int input_value,
                                                    int mean,
                                                    int rsqrt,
                                                    int lnw,
                                                    long long lnb) {
#pragma HLS INLINE
  long long diff = static_cast<long long>(input_value - mean);
  long long val = diff * static_cast<long long>(rsqrt) * static_cast<long long>(lnw) + lnb;
  long long rel = val >> 33;
  return hgtxr_clamp_int(static_cast<int>(rel), -4, 3);
}

static inline int hgtxr_hgpipe_mlp9_lnq_rsqrt128_int(int variance_sum) {
#pragma HLS INLINE
  static const int table[128] = {
      2832, 2630, 2466, 2329, 2213, 2112, 2025, 1947,
      1877, 1815, 1758, 1706, 1659, 1615, 1575, 1538,
      1503, 1470, 1439, 1411, 1384, 1358, 1334, 1311,
      1289, 1268, 1248, 1229, 1211, 1194, 1177, 1162,
      1146, 1132, 1117, 1104, 1091, 1078, 1066, 1054,
      1042, 1031, 1021, 1010, 1000, 990, 981, 971,
      963, 954, 945, 937, 929, 921, 913, 906,
      899, 891, 884, 878, 871, 864, 858, 852,
      846, 840, 834, 828, 823, 817, 812, 807,
      801, 796, 791, 786, 782, 777, 772, 768,
      763, 759, 755, 750, 746, 742, 738, 734,
      730, 726, 723, 719, 715, 712, 708, 704,
      701, 698, 694, 691, 688, 684, 681, 678,
      675, 672, 669, 666, 663, 660, 657, 655,
      652, 649, 646, 644, 641, 638, 636, 633,
      631, 628, 626, 623, 621, 619, 616, 614
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<128>(variance_sum, -1513477, 18, 127, table);
}

static inline int hgtxr_hgpipe_mlp9_lnq_mean_int(int sum) {
#pragma HLS INLINE
  long long tmp = static_cast<long long>(sum) * 43691;
  tmp += (1LL << (23 - 1));
  return static_cast<int>(tmp >> 23);
}

static inline int hgtxr_hgpipe_mlp9_lnq_requant_int(int input_value,
                                                    int mean,
                                                    int rsqrt,
                                                    int lnw,
                                                    long long lnb) {
#pragma HLS INLINE
  long long diff = static_cast<long long>(input_value - mean);
  long long val = diff * static_cast<long long>(rsqrt) * static_cast<long long>(lnw) + lnb;
  long long rel = val >> 32;
  return hgtxr_clamp_int(static_cast<int>(rel), -4, 3);
}

static inline int hgtxr_hgpipe_mlp10_lnq_rsqrt128_int(int variance_sum) {
#pragma HLS INLINE
  static const int table[128] = {
      3558, 3180, 2902, 2686, 2512, 2368, 2246, 2141,
      2050, 1969, 1897, 1833, 1775, 1721, 1673, 1628,
      1587, 1549, 1513, 1480, 1448, 1419, 1392, 1365,
      1341, 1317, 1295, 1274, 1254, 1235, 1217, 1199,
      1182, 1166, 1151, 1136, 1122, 1108, 1095, 1082,
      1069, 1057, 1046, 1035, 1024, 1013, 1003, 993,
      984, 974, 965, 956, 948, 939, 931, 923,
      916, 908, 901, 893, 886, 880, 873, 866,
      860, 854, 848, 842, 836, 830, 824, 819,
      813, 808, 803, 798, 793, 788, 783, 778,
      774, 769, 765, 760, 756, 752, 747, 743,
      739, 735, 731, 727, 724, 720, 716, 713,
      709, 706, 702, 699, 695, 692, 689, 685,
      682, 679, 676, 673, 670, 667, 664, 661,
      658, 655, 653, 650, 647, 645, 642, 639,
      637, 634, 632, 629, 627, 624, 622, 619
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<128>(variance_sum, -910942, 18, 127, table);
}

static inline int hgtxr_hgpipe_mlp10_lnq_mean_int(int sum) {
#pragma HLS INLINE
  long long tmp = static_cast<long long>(sum) * 43691;
  tmp += (1LL << (23 - 1));
  return static_cast<int>(tmp >> 23);
}

static inline int hgtxr_hgpipe_mlp10_lnq_requant_int(int input_value,
                                                     int mean,
                                                     int rsqrt,
                                                     int lnw,
                                                     long long lnb) {
#pragma HLS INLINE
  long long diff = static_cast<long long>(input_value - mean);
  long long val = diff * static_cast<long long>(rsqrt) * static_cast<long long>(lnw) + lnb;
  long long rel = val >> 32;
  return hgtxr_clamp_int(static_cast<int>(rel), -4, 3);
}

static inline int hgtxr_hgpipe_mlp11_lnq_rsqrt128_int(int variance_sum) {
#pragma HLS INLINE
  static const int table[128] = {
      3843, 3379, 3050, 2802, 2606, 2446, 2313, 2199,
      2100, 2014, 1937, 1869, 1807, 1751, 1700, 1653,
      1610, 1570, 1533, 1498, 1466, 1436, 1407, 1380,
      1355, 1331, 1308, 1286, 1265, 1246, 1227, 1209,
      1192, 1175, 1159, 1144, 1130, 1116, 1102, 1089,
      1076, 1064, 1052, 1041, 1030, 1019, 1009, 999,
      989, 979, 970, 961, 953, 944, 936, 928,
      920, 912, 905, 898, 890, 884, 877, 870,
      864, 857, 851, 845, 839, 833, 828, 822,
      816, 811, 806, 801, 796, 791, 786, 781,
      776, 772, 767, 763, 758, 754, 750, 746,
      742, 738, 734, 730, 726, 722, 718, 715,
      711, 708, 704, 701, 697, 694, 691, 687,
      684, 681, 678, 675, 672, 669, 666, 663,
      660, 657, 654, 651, 649, 646, 643, 641,
      638, 636, 633, 631, 628, 626, 623, 621
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<128>(variance_sum, -762143, 18, 127, table);
}

static inline int hgtxr_hgpipe_mlp11_lnq_mean_int(int sum) {
#pragma HLS INLINE
  long long tmp = static_cast<long long>(sum) * 43691;
  tmp += (1LL << (23 - 1));
  return static_cast<int>(tmp >> 23);
}

static inline int hgtxr_hgpipe_mlp11_lnq_requant_int(int input_value,
                                                     int mean,
                                                     int rsqrt,
                                                     int lnw,
                                                     long long lnb) {
#pragma HLS INLINE
  long long diff = static_cast<long long>(input_value - mean);
  long long val = diff * static_cast<long long>(rsqrt) * static_cast<long long>(lnw) + lnb;
  long long rel = val >> 32;
  return hgtxr_clamp_int(static_cast<int>(rel), -4, 3);
}

static inline int hgtxr_hgpipe_head_lnq_rsqrt128_int(int variance_sum) {
#pragma HLS INLINE
  static const int table[128] = {
      2130, 1961, 1826, 1716, 1623, 1544, 1476, 1416,
      1363, 1315, 1272, 1233, 1197, 1164, 1134, 1106,
      1080, 1056, 1033, 1012, 992, 973, 955, 938,
      922, 907, 893, 879, 866, 853, 841, 829,
      818, 808, 797, 787, 778, 769, 760, 751,
      743, 735, 727, 720, 712, 705, 698, 692,
      685, 679, 673, 667, 661, 655, 650, 644,
      639, 634, 629, 624, 619, 614, 610, 605,
      601, 597, 593, 588, 584, 581, 577, 573,
      569, 566, 562, 558, 555, 552, 548, 545,
      542, 539, 536, 533, 530, 527, 524, 521,
      518, 515, 513, 510, 507, 505, 502, 500,
      497, 495, 492, 490, 488, 486, 483, 481,
      479, 477, 474, 472, 470, 468, 466, 464,
      462, 460, 458, 456, 455, 453, 451, 449,
      447, 445, 444, 442, 440, 439, 437, 435
  };
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  return hgtxr_hgpipe_int_table_lookup<128>(variance_sum, -660994, 17, 127, table);
}

static inline int hgtxr_hgpipe_head_lnq_mean_int(int sum) {
#pragma HLS INLINE
  long long tmp = static_cast<long long>(sum) * 43691;
  tmp += (1LL << (23 - 1));
  return static_cast<int>(tmp >> 23);
}

static inline int hgtxr_hgpipe_head_lnq_requant_int(int input_value,
                                                   int mean,
                                                   int rsqrt,
                                                   int lnw,
                                                   long long lnb) {
#pragma HLS INLINE
  long long diff = static_cast<long long>(input_value - mean);
  long long val = diff * static_cast<long long>(rsqrt) * static_cast<long long>(lnw) + lnb;
  long long rel = val >> 27;
  return hgtxr_clamp_int(static_cast<int>(rel), -128, 127);
}

static inline HgtxrDataT hgtxr_hgpipe_gelu_lut(HgtxrDataT x) {
#pragma HLS INLINE
  static const HgtxrDataT table[16] = {
      static_cast<HgtxrDataT>(-0.0001), static_cast<HgtxrDataT>(-0.0020),
      static_cast<HgtxrDataT>(-0.0151), static_cast<HgtxrDataT>(-0.0715),
      static_cast<HgtxrDataT>(-0.1543), static_cast<HgtxrDataT>(-0.1588),
      static_cast<HgtxrDataT>(-0.0701), static_cast<HgtxrDataT>(0.1614),
      static_cast<HgtxrDataT>(0.5714), static_cast<HgtxrDataT>(1.1299),
      static_cast<HgtxrDataT>(1.7588), static_cast<HgtxrDataT>(2.3788),
      static_cast<HgtxrDataT>(2.9952), static_cast<HgtxrDataT>(3.6038),
      static_cast<HgtxrDataT>(4.2020), static_cast<HgtxrDataT>(4.8000)};
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  HgtxrDataT scaled = (x + static_cast<HgtxrDataT>(4)) * static_cast<HgtxrDataT>(1.875);
  int cursor = hgtxr_clamp_int(static_cast<int>(scaled), 0, 15);
  return table[cursor];
}

static inline HgtxrScoreT hgtxr_hgpipe_exp_lut(HgtxrScoreT shifted_score) {
#pragma HLS INLINE
  static const HgtxrScoreT table[16] = {
      static_cast<HgtxrScoreT>(1.000000), static_cast<HgtxrScoreT>(0.586646),
      static_cast<HgtxrScoreT>(0.344154), static_cast<HgtxrScoreT>(0.201897),
      static_cast<HgtxrScoreT>(0.118442), static_cast<HgtxrScoreT>(0.069483),
      static_cast<HgtxrScoreT>(0.040762), static_cast<HgtxrScoreT>(0.023916),
      static_cast<HgtxrScoreT>(0.014025), static_cast<HgtxrScoreT>(0.008230),
      static_cast<HgtxrScoreT>(0.004827), static_cast<HgtxrScoreT>(0.002831),
      static_cast<HgtxrScoreT>(0.001661), static_cast<HgtxrScoreT>(0.000974),
      static_cast<HgtxrScoreT>(0.000572), static_cast<HgtxrScoreT>(0.000335)};
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  HgtxrScoreT opposite_delta = 0;
  if (shifted_score < 0) {
    opposite_delta = static_cast<HgtxrScoreT>(-shifted_score);
  }
  HgtxrScoreT scaled = opposite_delta * static_cast<HgtxrScoreT>(1.875);
  int cursor = hgtxr_clamp_int(static_cast<int>(scaled), 0, 15);
  return table[cursor];
}

static inline HgtxrDataT hgtxr_hgpipe_rsqrt_lut(HgtxrAccumT variance_mean) {
#pragma HLS INLINE
  static const HgtxrDataT table[32] = {
      static_cast<HgtxrDataT>(11.313708), static_cast<HgtxrDataT>(0.695971),
      static_cast<HgtxrDataT>(0.492126), static_cast<HgtxrDataT>(0.401842),
      static_cast<HgtxrDataT>(0.347985), static_cast<HgtxrDataT>(0.311264),
      static_cast<HgtxrDataT>(0.284190), static_cast<HgtxrDataT>(0.263166),
      static_cast<HgtxrDataT>(0.246063), static_cast<HgtxrDataT>(0.231908),
      static_cast<HgtxrDataT>(0.219682), static_cast<HgtxrDataT>(0.209100),
      static_cast<HgtxrDataT>(0.199647), static_cast<HgtxrDataT>(0.191337),
      static_cast<HgtxrDataT>(0.183974), static_cast<HgtxrDataT>(0.177386),
      static_cast<HgtxrDataT>(0.171444), static_cast<HgtxrDataT>(0.166047),
      static_cast<HgtxrDataT>(0.161117), static_cast<HgtxrDataT>(0.156591),
      static_cast<HgtxrDataT>(0.152416), static_cast<HgtxrDataT>(0.148550),
      static_cast<HgtxrDataT>(0.144956), static_cast<HgtxrDataT>(0.141604),
      static_cast<HgtxrDataT>(0.138470), static_cast<HgtxrDataT>(0.135531),
      static_cast<HgtxrDataT>(0.132770), static_cast<HgtxrDataT>(0.130171),
      static_cast<HgtxrDataT>(0.127719), static_cast<HgtxrDataT>(0.125402),
      static_cast<HgtxrDataT>(0.123210), static_cast<HgtxrDataT>(0.125000)};
#pragma HLS bind_storage variable=table type=rom_2p impl=lutram
  HgtxrAccumT clipped = variance_mean;
  if (clipped < static_cast<HgtxrAccumT>(0.0078125)) {
    clipped = static_cast<HgtxrAccumT>(0.0078125);
  }
  if (clipped > static_cast<HgtxrAccumT>(64)) {
    clipped = static_cast<HgtxrAccumT>(64);
  }
  HgtxrAccumT scaled = clipped * static_cast<HgtxrAccumT>(0.484375);
  int cursor = hgtxr_clamp_int(static_cast<int>(scaled), 0, 31);
  return table[cursor];
}

static inline HgtxrDataT hgtxr_hgpipe_quantize_clamp(HgtxrAccumT value,
                                                     int clamp_bits,
                                                     bool signed_output) {
#pragma HLS INLINE
  HgtxrAccumT max_value = signed_output
      ? static_cast<HgtxrAccumT>((1 << (clamp_bits - 1)) - 1)
      : static_cast<HgtxrAccumT>((1 << clamp_bits) - 1);
  HgtxrAccumT min_value = signed_output
      ? static_cast<HgtxrAccumT>(-(1 << (clamp_bits - 1)))
      : static_cast<HgtxrAccumT>(0);
  if (value > max_value) {
    return static_cast<HgtxrDataT>(max_value);
  }
  if (value < min_value) {
    return static_cast<HgtxrDataT>(min_value);
  }
  return static_cast<HgtxrDataT>(value);
}

static inline HgtxrDataT hgtxr_rsqrt_approx(HgtxrAccumT variance_mean) {
#pragma HLS INLINE
#if HGTXR_USE_HGPIPE_LUT_MATH
  return hgtxr_hgpipe_rsqrt_lut(variance_mean);
#else
  HgtxrAccumT denom = static_cast<HgtxrAccumT>(1) + variance_mean;
  return static_cast<HgtxrDataT>(static_cast<HgtxrAccumT>(1) / denom);
#endif
}

static inline HgtxrDataT hgtxr_relu(HgtxrDataT x) {
#pragma HLS INLINE
  return x > 0 ? x : static_cast<HgtxrDataT>(0);
}

static inline HgtxrDataT hgtxr_hard_sigmoid(HgtxrDataT x) {
#pragma HLS INLINE
  HgtxrDataT y = (x >> 2) + static_cast<HgtxrDataT>(0.5);
  if (y < 0) {
    return 0;
  }
  if (y > 1) {
    return 1;
  }
  return y;
}

static inline HgtxrDataT hgtxr_gelu_approx(HgtxrDataT x) {
#pragma HLS INLINE
#if HGTXR_USE_HGPIPE_LUT_MATH
  return hgtxr_hgpipe_gelu_lut(x);
#else
  return x * hgtxr_hard_sigmoid(x);
#endif
}

static inline HgtxrScoreT hgtxr_score_exp_approx(HgtxrScoreT x) {
#pragma HLS INLINE
#if HGTXR_USE_HGPIPE_LUT_MATH
  return hgtxr_hgpipe_exp_lut(x);
#else
  if (x <= static_cast<HgtxrScoreT>(-8)) {
    return 0;
  }
  if (x >= 0) {
    return 1;
  }
  HgtxrScoreT one = 1;
  HgtxrScoreT y = one + (x >> 3);
  return y > 0 ? y : static_cast<HgtxrScoreT>(0);
#endif
}

template <int TM, int TN>
void row_max_tile(const HgtxrScoreT in[TM][TN], HgtxrScoreT row_max[TM]) {
#pragma HLS INLINE off
  for (int i = 0; i < TM; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_SEQ
    HgtxrScoreT max_value = in[i][0];
    for (int j = 1; j < TN; ++j) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_SEQ
#pragma HLS PIPELINE II=1
      if (in[i][j] > max_value) {
        max_value = in[i][j];
      }
    }
    row_max[i] = max_value;
  }
}

template <int TM, int TN>
void softmax_approx_tile(const HgtxrScoreT scores[TM][TN],
                         const HgtxrScoreT row_max[TM],
                         HgtxrScoreT probs[TM][TN],
                         HgtxrAccumT row_sum[TM]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=row_max cyclic factor=HGTXR_PAR_ATTN dim=1
#pragma HLS ARRAY_PARTITION variable=row_sum cyclic factor=HGTXR_PAR_ATTN dim=1

  for (int i = 0; i < TM; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_SEQ
    HgtxrAccumT sum = 0;
    for (int j = 0; j < TN; ++j) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_SEQ
#pragma HLS PIPELINE II=1
      HgtxrScoreT shifted = scores[i][j] - row_max[i];
      HgtxrScoreT e = hgtxr_score_exp_approx(shifted);
      probs[i][j] = e;
      sum += static_cast<HgtxrAccumT>(e);
    }
    row_sum[i] = sum;
  }
}

template <int TM, int TN>
void softmax_normalize_tile(const HgtxrScoreT probs_in[TM][TN],
                            const HgtxrAccumT row_sum[TM],
                            HgtxrDataT probs_out[TM][TN]) {
#pragma HLS INLINE off
  for (int i = 0; i < TM; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_SEQ
    HgtxrAccumT denom = row_sum[i] == 0 ? static_cast<HgtxrAccumT>(1) : row_sum[i];
    for (int j = 0; j < TN; ++j) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_SEQ
#pragma HLS PIPELINE II=1
      probs_out[i][j] = static_cast<HgtxrDataT>(
          static_cast<HgtxrAccumT>(probs_in[i][j]) / denom);
    }
  }
}

}  // namespace cyclic_transformer
}  // namespace hgtxr

#endif  // HGTXR_CYCLIC_MATH_HPP
