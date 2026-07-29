#include "../include/common.h"

#ifndef HGTXR_ENABLE_CYCLIC_TRANSFORMER_TOP
#define HGTXR_ENABLE_CYCLIC_TRANSFORMER_TOP 0
#endif

#ifndef HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS
#define HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS 0
#endif

#ifndef HGTXR_ENABLE_CYCLIC_S2_FIRST_STEP
#define HGTXR_ENABLE_CYCLIC_S2_FIRST_STEP 0
#endif
#ifndef HGTXR_ENABLE_CYCLIC_S2_QKV_FIRST_STEP
#define HGTXR_ENABLE_CYCLIC_S2_QKV_FIRST_STEP 0
#endif
#ifndef HGTXR_ENABLE_CYCLIC_S2_ATTN_FIRST_STEP
#define HGTXR_ENABLE_CYCLIC_S2_ATTN_FIRST_STEP 0
#endif
#ifndef HGTXR_ENABLE_CYCLIC_S2_MLP_FIRST_STEP
#define HGTXR_ENABLE_CYCLIC_S2_MLP_FIRST_STEP 0
#endif
#ifndef HGTXR_ENABLE_CYCLIC_S2_BLOCK_FIRST_STEP
#define HGTXR_ENABLE_CYCLIC_S2_BLOCK_FIRST_STEP 0
#endif
#ifndef HGTXR_ENABLE_CYCLIC_S2_BLOCK_PRE_LN
#define HGTXR_ENABLE_CYCLIC_S2_BLOCK_PRE_LN 0
#endif

#if HGTXR_ENABLE_CYCLIC_S2_FIRST_STEP && !HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS
#error "HGTXR_ENABLE_CYCLIC_S2_FIRST_STEP requires HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS"
#endif
#if HGTXR_ENABLE_CYCLIC_S2_QKV_FIRST_STEP && !HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS
#error "HGTXR_ENABLE_CYCLIC_S2_QKV_FIRST_STEP requires HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS"
#endif
#if HGTXR_ENABLE_CYCLIC_S2_ATTN_FIRST_STEP && !HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS
#error "HGTXR_ENABLE_CYCLIC_S2_ATTN_FIRST_STEP requires HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS"
#endif
#if HGTXR_ENABLE_CYCLIC_S2_MLP_FIRST_STEP && !HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS
#error "HGTXR_ENABLE_CYCLIC_S2_MLP_FIRST_STEP requires HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS"
#endif
#if HGTXR_ENABLE_CYCLIC_S2_BLOCK_FIRST_STEP && !HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS
#error "HGTXR_ENABLE_CYCLIC_S2_BLOCK_FIRST_STEP requires HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS"
#endif
#if (HGTXR_ENABLE_CYCLIC_S2_FIRST_STEP + HGTXR_ENABLE_CYCLIC_S2_QKV_FIRST_STEP + HGTXR_ENABLE_CYCLIC_S2_ATTN_FIRST_STEP + HGTXR_ENABLE_CYCLIC_S2_MLP_FIRST_STEP + HGTXR_ENABLE_CYCLIC_S2_BLOCK_FIRST_STEP) > 1
#error "Enable only one S2 first-step top mode at a time"
#endif
#if (HGTXR_ENABLE_CYCLIC_S2_ATTN_FIRST_STEP || HGTXR_ENABLE_CYCLIC_S2_BLOCK_FIRST_STEP) && HGTXR_CYCLIC_REQUIRE_HEAD_ALIGNED_ATTENTION
#if HGTXR_TILE_CHANNELS != HGTXR_CYCLIC_HEAD_DIM
#error "Head-aligned S2 attention requires HGTXR_TILE_CHANNELS == HGTXR_CYCLIC_HEAD_DIM"
#endif
#if ((HGTXR_CYCLIC_MODEL_DIM + HGTXR_TILE_CHANNELS - 1) / HGTXR_TILE_CHANNELS) != HGTXR_CYCLIC_HEADS
#error "Head-aligned S2 attention requires one channel tile per attention head"
#endif
#endif

#if HGTXR_ENABLE_CYCLIC_TRANSFORMER_TOP
#include "../include/hgtxr_cyclic_transformer_block.hpp"
#if HGTXR_ENABLE_CYCLIC_S2_FIRST_STEP || HGTXR_ENABLE_CYCLIC_S2_QKV_FIRST_STEP || HGTXR_ENABLE_CYCLIC_S2_ATTN_FIRST_STEP || HGTXR_ENABLE_CYCLIC_S2_MLP_FIRST_STEP || HGTXR_ENABLE_CYCLIC_S2_BLOCK_FIRST_STEP
#include "../include/hgtxr_cyclic_s2_projection.hpp"
#endif

namespace {

using namespace hgtxr::cyclic_transformer;

static constexpr int HGTXR_CYCLIC_TM = HGTXR_TILE_SEQ;
static constexpr int HGTXR_CYCLIC_TC = HGTXR_TILE_MODEL_DIM;
static constexpr int HGTXR_CYCLIC_TF = HGTXR_TILE_FF_DIM;

void hgtxr_init_cyclic_transformer_weights(
    HgtxrDataT norm1_gamma[HGTXR_CYCLIC_TC],
    HgtxrDataT norm1_beta[HGTXR_CYCLIC_TC],
    HgtxrDataT norm2_gamma[HGTXR_CYCLIC_TC],
    HgtxrDataT norm2_beta[HGTXR_CYCLIC_TC],
    HgtxrDataT wq[HGTXR_CYCLIC_TC][HGTXR_CYCLIC_TC],
    HgtxrDataT wk[HGTXR_CYCLIC_TC][HGTXR_CYCLIC_TC],
    HgtxrDataT wv[HGTXR_CYCLIC_TC][HGTXR_CYCLIC_TC],
    HgtxrDataT wo[HGTXR_CYCLIC_TC][HGTXR_CYCLIC_TC],
    HgtxrDataT w1[HGTXR_CYCLIC_TC][HGTXR_CYCLIC_TF],
    HgtxrDataT w2[HGTXR_CYCLIC_TF][HGTXR_CYCLIC_TC]) {
#pragma HLS INLINE off
  for (int c = 0; c < HGTXR_CYCLIC_TC; ++c) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_MODEL_DIM
#pragma HLS PIPELINE II=1
    norm1_gamma[c] = 1;
    norm1_beta[c] = 0;
    norm2_gamma[c] = 1;
    norm2_beta[c] = 0;
  }

  for (int r = 0; r < HGTXR_CYCLIC_TC; ++r) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_MODEL_DIM
    for (int c = 0; c < HGTXR_CYCLIC_TC; ++c) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_MODEL_DIM
#pragma HLS PIPELINE II=1
      HgtxrDataT diag = (r == c) ? (HgtxrDataT)1 : (HgtxrDataT)0;
      wq[r][c] = diag;
      wk[r][c] = diag;
      wv[r][c] = diag;
      wo[r][c] = diag;
    }
  }

  for (int c = 0; c < HGTXR_CYCLIC_TC; ++c) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_MODEL_DIM
    for (int f = 0; f < HGTXR_CYCLIC_TF; ++f) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_FF_DIM
#pragma HLS PIPELINE II=1
      w1[c][f] = (c == f) ? (HgtxrDataT)1 : (HgtxrDataT)0;
    }
  }

  for (int f = 0; f < HGTXR_CYCLIC_TF; ++f) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_FF_DIM
    for (int c = 0; c < HGTXR_CYCLIC_TC; ++c) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_MODEL_DIM
#pragma HLS PIPELINE II=1
      w2[f][c] = (f == c) ? (HgtxrDataT)1 : (HgtxrDataT)0;
    }
  }
}


#if HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS
HgtxrDataT hgtxr_unpack_cyclic_weight_word(HgtxrAxiWordT word, int lane) {
#pragma HLS INLINE
  HgtxrAxiWordT shifted = word >> (lane * HGTXR_WEIGHT_WIDTH);
  HgtxrWeightIntT bits = shifted.range(HGTXR_WEIGHT_WIDTH - 1, 0);
  HgtxrWeightT value = 0;
  value.range(HGTXR_WEIGHT_WIDTH - 1, 0) = bits.range(HGTXR_WEIGHT_WIDTH - 1, 0);
  return (HgtxrDataT)value;
}

HgtxrDataT hgtxr_load_cyclic_weight_elem(const HgtxrAxiWordT *cyclic_weights,
                                         int elem_offset) {
#pragma HLS INLINE
  int word_idx = elem_offset / HGTXR_AXI_WEIGHT_LANES;
  int lane_idx = elem_offset % HGTXR_AXI_WEIGHT_LANES;
  return hgtxr_unpack_cyclic_weight_word(cyclic_weights[word_idx], lane_idx);
}

void hgtxr_load_cyclic_transformer_weights_from_packed(
    const HgtxrAxiWordT *cyclic_weights,
    int block_idx,
    HgtxrDataT norm1_gamma[HGTXR_CYCLIC_TC],
    HgtxrDataT norm1_beta[HGTXR_CYCLIC_TC],
    HgtxrDataT norm2_gamma[HGTXR_CYCLIC_TC],
    HgtxrDataT norm2_beta[HGTXR_CYCLIC_TC],
    HgtxrDataT wq[HGTXR_CYCLIC_TC][HGTXR_CYCLIC_TC],
    HgtxrDataT wk[HGTXR_CYCLIC_TC][HGTXR_CYCLIC_TC],
    HgtxrDataT wv[HGTXR_CYCLIC_TC][HGTXR_CYCLIC_TC],
    HgtxrDataT wo[HGTXR_CYCLIC_TC][HGTXR_CYCLIC_TC],
    HgtxrDataT w1[HGTXR_CYCLIC_TC][HGTXR_CYCLIC_TF],
    HgtxrDataT w2[HGTXR_CYCLIC_TF][HGTXR_CYCLIC_TC]) {
#pragma HLS INLINE off
  const HgtxrAxiWordT *block_weights =
      cyclic_weights + block_idx * HGTXR_CYCLIC_WEIGHT_LAYER_STRIDE_WORDS;

  for (int c = 0; c < HGTXR_CYCLIC_TC; ++c) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_MODEL_DIM
#pragma HLS PIPELINE II=1
    norm1_gamma[c] = hgtxr_load_cyclic_weight_elem(block_weights, HGTXR_CYCLIC_WEIGHT_NORM1_GAMMA_OFFSET + c);
    norm1_beta[c] = hgtxr_load_cyclic_weight_elem(block_weights, HGTXR_CYCLIC_WEIGHT_NORM1_BETA_OFFSET + c);
    norm2_gamma[c] = hgtxr_load_cyclic_weight_elem(block_weights, HGTXR_CYCLIC_WEIGHT_NORM2_GAMMA_OFFSET + c);
    norm2_beta[c] = hgtxr_load_cyclic_weight_elem(block_weights, HGTXR_CYCLIC_WEIGHT_NORM2_BETA_OFFSET + c);
  }

  for (int r = 0; r < HGTXR_CYCLIC_TC; ++r) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_MODEL_DIM
    for (int c = 0; c < HGTXR_CYCLIC_TC; ++c) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_MODEL_DIM
#pragma HLS PIPELINE II=1
      int elem = r * HGTXR_CYCLIC_TC + c;
      wq[r][c] = hgtxr_load_cyclic_weight_elem(block_weights, HGTXR_CYCLIC_WEIGHT_WQ_OFFSET + elem);
      wk[r][c] = hgtxr_load_cyclic_weight_elem(block_weights, HGTXR_CYCLIC_WEIGHT_WK_OFFSET + elem);
      wv[r][c] = hgtxr_load_cyclic_weight_elem(block_weights, HGTXR_CYCLIC_WEIGHT_WV_OFFSET + elem);
      wo[r][c] = hgtxr_load_cyclic_weight_elem(block_weights, HGTXR_CYCLIC_WEIGHT_WO_OFFSET + elem);
    }
  }

  for (int c = 0; c < HGTXR_CYCLIC_TC; ++c) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_MODEL_DIM
    for (int f = 0; f < HGTXR_CYCLIC_TF; ++f) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_FF_DIM
#pragma HLS PIPELINE II=1
      w1[c][f] = hgtxr_load_cyclic_weight_elem(block_weights, HGTXR_CYCLIC_WEIGHT_W1_OFFSET + c * HGTXR_CYCLIC_TF + f);
    }
  }

  for (int f = 0; f < HGTXR_CYCLIC_TF; ++f) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_FF_DIM
    for (int c = 0; c < HGTXR_CYCLIC_TC; ++c) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_MODEL_DIM
#pragma HLS PIPELINE II=1
      w2[f][c] = hgtxr_load_cyclic_weight_elem(block_weights, HGTXR_CYCLIC_WEIGHT_W2_OFFSET + f * HGTXR_CYCLIC_TC + c);
    }
  }
}
#endif

#if HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS && (HGTXR_ENABLE_CYCLIC_S2_ATTN_FIRST_STEP || HGTXR_ENABLE_CYCLIC_S2_BLOCK_FIRST_STEP)
template <int ACTIVE_TOKENS>
void hgtxr_cyclic_s2_attention_first_step_stage(
    token_buffer_t tokens, const HgtxrAxiWordT *cyclic_weights, int layer_idx) {
#pragma HLS INLINE off
  HgtxrDataT x_tile[HGTXR_CYCLIC_TM][HGTXR_CYCLIC_TC];
  HgtxrDataT q_tile[HGTXR_CYCLIC_TM][HGTXR_CYCLIC_TC];
  HgtxrDataT k_tile[HGTXR_CYCLIC_TM][HGTXR_CYCLIC_TC];
  HgtxrDataT v_tile[HGTXR_CYCLIC_TM][HGTXR_CYCLIC_TC];
  HgtxrDataT attn_tiles[HGTXR_CYCLIC_CHANNEL_TILES][HGTXR_CYCLIC_TM][HGTXR_CYCLIC_TC];
  HgtxrAccumT q_acc[HGTXR_CYCLIC_TM][HGTXR_CYCLIC_TC];
  HgtxrAccumT k_acc[HGTXR_CYCLIC_TM][HGTXR_CYCLIC_TC];
  HgtxrAccumT v_acc[HGTXR_CYCLIC_TM][HGTXR_CYCLIC_TC];
  HgtxrAccumT out_acc[HGTXR_CYCLIC_TM][HGTXR_CYCLIC_TC];
#if HGTXR_CYCLIC_SMALL_TILE_LUTRAM
#pragma HLS bind_storage variable=x_tile type=ram_2p impl=lutram
#pragma HLS bind_storage variable=q_tile type=ram_2p impl=lutram
#pragma HLS bind_storage variable=k_tile type=ram_2p impl=lutram
#pragma HLS bind_storage variable=v_tile type=ram_2p impl=lutram
#else
#pragma HLS bind_storage variable=x_tile type=ram_2p impl=bram
#pragma HLS bind_storage variable=q_tile type=ram_2p impl=bram
#pragma HLS bind_storage variable=k_tile type=ram_2p impl=bram
#pragma HLS bind_storage variable=v_tile type=ram_2p impl=bram
#endif
#if HGTXR_CYCLIC_FORCE_URAM_LARGE_TEMPS
#pragma HLS bind_storage variable=attn_tiles type=ram_2p impl=uram
#else
#pragma HLS bind_storage variable=attn_tiles type=ram_2p impl=bram
#endif
#pragma HLS bind_storage variable=q_acc type=ram_2p impl=bram
#pragma HLS bind_storage variable=k_acc type=ram_2p impl=bram
#pragma HLS bind_storage variable=v_acc type=ram_2p impl=bram
#pragma HLS bind_storage variable=out_acc type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=x_tile cyclic factor=HGTXR_PAR_QK dim=2
#pragma HLS ARRAY_PARTITION variable=q_tile cyclic factor=HGTXR_PAR_FFN dim=2
#pragma HLS ARRAY_PARTITION variable=k_tile cyclic factor=HGTXR_PAR_FFN dim=2
#pragma HLS ARRAY_PARTITION variable=v_tile cyclic factor=HGTXR_PAR_FFN dim=2
#pragma HLS ARRAY_PARTITION variable=q_acc cyclic factor=HGTXR_PAR_FFN dim=2
#pragma HLS ARRAY_PARTITION variable=k_acc cyclic factor=HGTXR_PAR_FFN dim=2
#pragma HLS ARRAY_PARTITION variable=v_acc cyclic factor=HGTXR_PAR_FFN dim=2
#pragma HLS ARRAY_PARTITION variable=out_acc cyclic factor=HGTXR_PAR_FFN dim=2

  for (int token_base = 0; token_base < ACTIVE_TOKENS;
       token_base += HGTXR_CYCLIC_TM) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TOKENS
    for (int qkv_tile = 0; qkv_tile < HGTXR_CYCLIC_CHANNEL_TILES;
         ++qkv_tile) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_CYCLIC_CHANNEL_TILES
      for (int input_tile = 0; input_tile < HGTXR_CYCLIC_CHANNEL_TILES;
           ++input_tile) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_CYCLIC_CHANNEL_TILES
        const int input_channel_base = input_tile * HGTXR_CYCLIC_TC;
        for (int tm = 0; tm < HGTXR_CYCLIC_TM; ++tm) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_SEQ
          for (int tc = 0; tc < HGTXR_CYCLIC_TC; ++tc) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_MODEL_DIM
#pragma HLS PIPELINE II=1
            const int token_idx = token_base + tm;
            const int channel_idx = input_channel_base + tc;
            if (token_idx < ACTIVE_TOKENS && channel_idx < HGTXR_EMBED) {
              x_tile[tm][tc] =
                  static_cast<HgtxrDataT>(tokens[token_idx][channel_idx]);
            } else {
              x_tile[tm][tc] = 0;
            }
          }
        }

        hgtxr_s2_qkv_project_pair_accum<HGTXR_CYCLIC_TM, HGTXR_CYCLIC_TC,
                                        HGTXR_CYCLIC_TC>(
            cyclic_weights, layer_idx, qkv_tile, input_tile,
            HGTXR_CYCLIC_CHANNEL_TILES, x_tile, q_acc, k_acc, v_acc,
            input_tile == 0);
      }

      accum_to_data_tile<HGTXR_CYCLIC_TM, HGTXR_CYCLIC_TC>(q_acc, q_tile, 0);
      accum_to_data_tile<HGTXR_CYCLIC_TM, HGTXR_CYCLIC_TC>(k_acc, k_tile, 0);
      accum_to_data_tile<HGTXR_CYCLIC_TM, HGTXR_CYCLIC_TC>(v_acc, v_tile, 0);
      attention_tile<HGTXR_CYCLIC_TM, HGTXR_CYCLIC_TM, HGTXR_CYCLIC_TC>(
          q_tile, k_tile, v_tile, attn_tiles[qkv_tile],
          HGTXR_CYCLIC_ATTN_SCORE_SCALE_SHIFT, 0);
    }

    for (int output_tile = 0; output_tile < HGTXR_CYCLIC_CHANNEL_TILES;
         ++output_tile) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_CYCLIC_CHANNEL_TILES
      for (int input_tile = 0; input_tile < HGTXR_CYCLIC_CHANNEL_TILES;
           ++input_tile) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_CYCLIC_CHANNEL_TILES
        hgtxr_s2_project_pair_accum<HGTXR_CYCLIC_TM, HGTXR_CYCLIC_TC,
                                    HGTXR_CYCLIC_TC>(
            cyclic_weights, layer_idx, output_tile, input_tile,
            HGTXR_CYCLIC_CHANNEL_TILES, HGTXR_CYCLIC_WEIGHT_WO_OFFSET,
            attn_tiles[input_tile], out_acc, input_tile == 0);
      }

      const int output_channel_base = output_tile * HGTXR_CYCLIC_TC;
      for (int tm = 0; tm < HGTXR_CYCLIC_TM; ++tm) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_SEQ
        for (int tc = 0; tc < HGTXR_CYCLIC_TC; ++tc) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_MODEL_DIM
#pragma HLS PIPELINE II=1
          const int token_idx = token_base + tm;
          const int channel_idx = output_channel_base + tc;
          if (token_idx < ACTIVE_TOKENS && channel_idx < HGTXR_EMBED) {
            tokens[token_idx][channel_idx] =
                static_cast<hgtxr_data_t>(out_acc[tm][tc]);
          }
        }
      }
    }
  }
}
#endif


#if HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS && (HGTXR_ENABLE_CYCLIC_S2_MLP_FIRST_STEP || HGTXR_ENABLE_CYCLIC_S2_BLOCK_FIRST_STEP)
template <int ACTIVE_TOKENS>
void hgtxr_cyclic_s2_mlp_first_step_stage(
    token_buffer_t tokens, const HgtxrAxiWordT *cyclic_weights, int layer_idx) {
#pragma HLS INLINE off
  HgtxrDataT x_tile[HGTXR_CYCLIC_TM][HGTXR_CYCLIC_TC];
  HgtxrAccumT hidden_acc[HGTXR_CYCLIC_TM][HGTXR_CYCLIC_TF];
  HgtxrDataT hidden_tiles[HGTXR_CYCLIC_HIDDEN_TILES][HGTXR_CYCLIC_TM]
                          [HGTXR_CYCLIC_TF];
  HgtxrAccumT out_acc[HGTXR_CYCLIC_TM][HGTXR_CYCLIC_TC];
#if HGTXR_CYCLIC_SMALL_TILE_LUTRAM
#pragma HLS bind_storage variable=x_tile type=ram_2p impl=lutram
#else
#pragma HLS bind_storage variable=x_tile type=ram_2p impl=bram
#endif
#pragma HLS bind_storage variable=hidden_acc type=ram_2p impl=bram
#if HGTXR_CYCLIC_FORCE_URAM_LARGE_TEMPS
#pragma HLS bind_storage variable=hidden_tiles type=ram_2p impl=uram
#else
#pragma HLS bind_storage variable=hidden_tiles type=ram_2p impl=bram
#endif
#pragma HLS bind_storage variable=out_acc type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=x_tile cyclic factor=HGTXR_PAR_QK dim=2
#pragma HLS ARRAY_PARTITION variable=hidden_acc cyclic factor=HGTXR_PAR_FFN dim=2
#pragma HLS ARRAY_PARTITION variable=out_acc cyclic factor=HGTXR_PAR_FFN dim=2

  for (int token_base = 0; token_base < ACTIVE_TOKENS;
       token_base += HGTXR_CYCLIC_TM) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TOKENS
    for (int hidden_tile = 0; hidden_tile < HGTXR_CYCLIC_HIDDEN_TILES;
         ++hidden_tile) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_CYCLIC_HIDDEN_TILES
      for (int input_tile = 0; input_tile < HGTXR_CYCLIC_CHANNEL_TILES;
           ++input_tile) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_CYCLIC_CHANNEL_TILES
        const int input_channel_base = input_tile * HGTXR_CYCLIC_TC;
        for (int tm = 0; tm < HGTXR_CYCLIC_TM; ++tm) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_SEQ
          for (int tc = 0; tc < HGTXR_CYCLIC_TC; ++tc) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_MODEL_DIM
#pragma HLS PIPELINE II=1
            const int token_idx = token_base + tm;
            const int channel_idx = input_channel_base + tc;
            if (token_idx < ACTIVE_TOKENS && channel_idx < HGTXR_EMBED) {
              x_tile[tm][tc] =
                  static_cast<HgtxrDataT>(tokens[token_idx][channel_idx]);
            } else {
              x_tile[tm][tc] = 0;
            }
          }
        }

        hgtxr_s2_mlp_w1_pair_accum<HGTXR_CYCLIC_TM, HGTXR_CYCLIC_TC,
                                   HGTXR_CYCLIC_TF>(
            cyclic_weights, layer_idx, hidden_tile, input_tile,
            HGTXR_CYCLIC_CHANNEL_TILES, HGTXR_CYCLIC_HIDDEN_TILES, x_tile,
            hidden_acc, input_tile == 0);
      }

      for (int tm = 0; tm < HGTXR_CYCLIC_TM; ++tm) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_SEQ
        for (int tf = 0; tf < HGTXR_CYCLIC_TF; ++tf) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_FF_DIM
#pragma HLS PIPELINE II=1
          hidden_tiles[hidden_tile][tm][tf] =
              hgtxr_gelu_approx(static_cast<HgtxrDataT>(hidden_acc[tm][tf]));
        }
      }
    }

    for (int output_tile = 0; output_tile < HGTXR_CYCLIC_CHANNEL_TILES;
         ++output_tile) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_CYCLIC_CHANNEL_TILES
      for (int hidden_tile = 0; hidden_tile < HGTXR_CYCLIC_HIDDEN_TILES;
           ++hidden_tile) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_CYCLIC_HIDDEN_TILES
        hgtxr_s2_mlp_w2_pair_accum<HGTXR_CYCLIC_TM, HGTXR_CYCLIC_TF,
                                   HGTXR_CYCLIC_TC>(
            cyclic_weights, layer_idx, output_tile, hidden_tile,
            HGTXR_CYCLIC_CHANNEL_TILES, HGTXR_CYCLIC_HIDDEN_TILES,
            hidden_tiles[hidden_tile], out_acc, hidden_tile == 0);
      }

      const int output_channel_base = output_tile * HGTXR_CYCLIC_TC;
      for (int tm = 0; tm < HGTXR_CYCLIC_TM; ++tm) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_SEQ
        for (int tc = 0; tc < HGTXR_CYCLIC_TC; ++tc) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_MODEL_DIM
#pragma HLS PIPELINE II=1
          const int token_idx = token_base + tm;
          const int channel_idx = output_channel_base + tc;
          if (token_idx < ACTIVE_TOKENS && channel_idx < HGTXR_EMBED) {
            tokens[token_idx][channel_idx] =
                static_cast<hgtxr_data_t>(out_acc[tm][tc]);
          }
        }
      }
    }
  }
}
#endif


#if HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS && HGTXR_ENABLE_CYCLIC_S2_BLOCK_FIRST_STEP
template <int ACTIVE_TOKENS>
void hgtxr_copy_tokens(token_buffer_t src,
                       hgtxr_data_t dst[ACTIVE_TOKENS][HGTXR_EMBED]) {
#pragma HLS INLINE off
  for (int token_idx = 0; token_idx < ACTIVE_TOKENS; ++token_idx) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TOKENS
    for (int channel_idx = 0; channel_idx < HGTXR_EMBED; ++channel_idx) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_EMBED
#pragma HLS PIPELINE II=1
      dst[token_idx][channel_idx] = src[token_idx][channel_idx];
    }
  }
}

template <int ACTIVE_TOKENS>
void hgtxr_add_residual_to_tokens(
    token_buffer_t tokens, const hgtxr_data_t residual[ACTIVE_TOKENS][HGTXR_EMBED]) {
#pragma HLS INLINE off
  for (int token_idx = 0; token_idx < ACTIVE_TOKENS; ++token_idx) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TOKENS
    for (int channel_idx = 0; channel_idx < HGTXR_EMBED; ++channel_idx) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_EMBED
#pragma HLS PIPELINE II=1
      HgtxrAccumT sum = static_cast<HgtxrAccumT>(tokens[token_idx][channel_idx]) +
                        static_cast<HgtxrAccumT>(residual[token_idx][channel_idx]);
      tokens[token_idx][channel_idx] =
          static_cast<hgtxr_data_t>(static_cast<HgtxrDataT>(sum));
    }
  }
}

#if HGTXR_ENABLE_CYCLIC_S2_BLOCK_PRE_LN
HgtxrAccumT hgtxr_s2_block_rsqrt_approx(HgtxrAccumT x) {
#pragma HLS INLINE
  if (x <= static_cast<HgtxrAccumT>(0.0001)) {
    return static_cast<HgtxrAccumT>(32);
  }
  if (x <= static_cast<HgtxrAccumT>(0.0625)) {
    return static_cast<HgtxrAccumT>(4);
  }
  if (x <= static_cast<HgtxrAccumT>(0.125)) {
    return static_cast<HgtxrAccumT>(3.375);
  }
  if (x <= static_cast<HgtxrAccumT>(0.25)) {
    return static_cast<HgtxrAccumT>(2.5);
  }
  if (x <= static_cast<HgtxrAccumT>(0.5)) {
    return static_cast<HgtxrAccumT>(1.625);
  }
  if (x <= static_cast<HgtxrAccumT>(1)) {
    return static_cast<HgtxrAccumT>(1);
  }
  if (x <= static_cast<HgtxrAccumT>(2)) {
    return static_cast<HgtxrAccumT>(0.75);
  }
  if (x <= static_cast<HgtxrAccumT>(4)) {
    return static_cast<HgtxrAccumT>(0.5);
  }
  if (x <= static_cast<HgtxrAccumT>(8)) {
    return static_cast<HgtxrAccumT>(0.375);
  }
  if (x <= static_cast<HgtxrAccumT>(16)) {
    return static_cast<HgtxrAccumT>(0.25);
  }
  return static_cast<HgtxrAccumT>(0.125);
}

HgtxrDataT hgtxr_load_s2_block_norm_elem(const HgtxrAxiWordT *cyclic_weights,
                                         int layer_idx,
                                         int channel_idx,
                                         bool norm2,
                                         bool beta) {
#pragma HLS INLINE
  const int channel_pair_blocks =
      HGTXR_CYCLIC_CHANNEL_TILES * HGTXR_CYCLIC_CHANNEL_TILES;
  const int mlp_pair_blocks =
      2 * HGTXR_CYCLIC_CHANNEL_TILES * HGTXR_CYCLIC_HIDDEN_TILES;
  const int output_tile = channel_idx / HGTXR_CYCLIC_TC;
  const int local_channel = channel_idx % HGTXR_CYCLIC_TC;
  const int block_idx =
      layer_idx * (channel_pair_blocks + mlp_pair_blocks) +
      output_tile * HGTXR_CYCLIC_CHANNEL_TILES;
  const int elem_offset =
      (norm2
           ? (beta ? HGTXR_CYCLIC_WEIGHT_NORM2_BETA_OFFSET
                   : HGTXR_CYCLIC_WEIGHT_NORM2_GAMMA_OFFSET)
           : (beta ? HGTXR_CYCLIC_WEIGHT_NORM1_BETA_OFFSET
                   : HGTXR_CYCLIC_WEIGHT_NORM1_GAMMA_OFFSET)) +
      local_channel;
  return hgtxr_load_cyclic_weight_elem(
      cyclic_weights + block_idx * HGTXR_CYCLIC_WEIGHT_LAYER_STRIDE_WORDS,
      elem_offset);
}

template <int ACTIVE_TOKENS>
void hgtxr_apply_s2_block_pre_layernorm(
    const hgtxr_data_t src[ACTIVE_TOKENS][HGTXR_EMBED],
    token_buffer_t dst,
    const HgtxrAxiWordT *cyclic_weights,
    int layer_idx,
    bool norm2) {
#pragma HLS INLINE off
  for (int token_idx = 0; token_idx < ACTIVE_TOKENS; ++token_idx) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TOKENS
    HgtxrAccumT mean = 0;
    HgtxrAccumT var = 0;

    for (int channel_idx = 0; channel_idx < HGTXR_EMBED; ++channel_idx) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_EMBED
#pragma HLS PIPELINE II=1
      mean += static_cast<HgtxrAccumT>(src[token_idx][channel_idx]);
    }
    mean = mean / static_cast<HgtxrAccumT>(HGTXR_EMBED);

    for (int channel_idx = 0; channel_idx < HGTXR_EMBED; ++channel_idx) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_EMBED
#pragma HLS PIPELINE II=1
      HgtxrAccumT diff =
          static_cast<HgtxrAccumT>(src[token_idx][channel_idx]) - mean;
      var += diff * diff;
    }
    HgtxrAccumT inv_std = hgtxr_s2_block_rsqrt_approx(
        var / static_cast<HgtxrAccumT>(HGTXR_EMBED) +
        static_cast<HgtxrAccumT>(0.0001));

    for (int channel_idx = 0; channel_idx < HGTXR_EMBED; ++channel_idx) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_EMBED
#pragma HLS PIPELINE II=1
      HgtxrAccumT centered =
          (static_cast<HgtxrAccumT>(src[token_idx][channel_idx]) - mean) *
          inv_std;
      HgtxrAccumT gamma = static_cast<HgtxrAccumT>(
          hgtxr_load_s2_block_norm_elem(cyclic_weights, layer_idx, channel_idx,
                                        norm2, false));
      HgtxrAccumT beta = static_cast<HgtxrAccumT>(
          hgtxr_load_s2_block_norm_elem(cyclic_weights, layer_idx, channel_idx,
                                        norm2, true));
      dst[token_idx][channel_idx] = static_cast<hgtxr_data_t>(
          static_cast<HgtxrDataT>(centered * gamma + beta));
    }
  }
}
#endif

template <int ACTIVE_TOKENS>
void hgtxr_cyclic_s2_block_first_step_stage(
    token_buffer_t tokens, const HgtxrAxiWordT *cyclic_weights, int layer_idx) {
#pragma HLS INLINE off
  hgtxr_data_t residual0[ACTIVE_TOKENS][HGTXR_EMBED];
  hgtxr_data_t residual1[ACTIVE_TOKENS][HGTXR_EMBED];
#if HGTXR_CYCLIC_FORCE_URAM_LARGE_TEMPS
#pragma HLS bind_storage variable=residual0 type=ram_2p impl=uram
#pragma HLS bind_storage variable=residual1 type=ram_2p impl=uram
#else
#pragma HLS bind_storage variable=residual0 type=ram_2p impl=bram
#pragma HLS bind_storage variable=residual1 type=ram_2p impl=bram
#endif

  const int channel_pair_blocks =
      HGTXR_CYCLIC_CHANNEL_TILES * HGTXR_CYCLIC_CHANNEL_TILES;
  const int mlp_pair_blocks =
      2 * HGTXR_CYCLIC_CHANNEL_TILES * HGTXR_CYCLIC_HIDDEN_TILES;
  const int layer_block_stride_words =
      (channel_pair_blocks + mlp_pair_blocks) *
      HGTXR_CYCLIC_WEIGHT_LAYER_STRIDE_WORDS;
  const int attn_word_shift =
      layer_idx * mlp_pair_blocks * HGTXR_CYCLIC_WEIGHT_LAYER_STRIDE_WORDS;
  const int mlp_word_shift =
      (layer_idx + 1) * channel_pair_blocks *
      HGTXR_CYCLIC_WEIGHT_LAYER_STRIDE_WORDS;
  (void)layer_block_stride_words;

  hgtxr_copy_tokens<ACTIVE_TOKENS>(tokens, residual0);
#if HGTXR_ENABLE_CYCLIC_S2_BLOCK_PRE_LN
  hgtxr_apply_s2_block_pre_layernorm<ACTIVE_TOKENS>(
      residual0, tokens, cyclic_weights, layer_idx, false);
#endif
  hgtxr_cyclic_s2_attention_first_step_stage<ACTIVE_TOKENS>(
      tokens, cyclic_weights + attn_word_shift, layer_idx);
  hgtxr_add_residual_to_tokens<ACTIVE_TOKENS>(tokens, residual0);

  hgtxr_copy_tokens<ACTIVE_TOKENS>(tokens, residual1);
#if HGTXR_ENABLE_CYCLIC_S2_BLOCK_PRE_LN
  hgtxr_apply_s2_block_pre_layernorm<ACTIVE_TOKENS>(
      residual1, tokens, cyclic_weights, layer_idx, true);
#endif
  hgtxr_cyclic_s2_mlp_first_step_stage<ACTIVE_TOKENS>(
      tokens, cyclic_weights + mlp_word_shift, layer_idx);
  hgtxr_add_residual_to_tokens<ACTIVE_TOKENS>(tokens, residual1);
}
#endif

#if HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS && HGTXR_ENABLE_CYCLIC_S2_QKV_FIRST_STEP
template <int ACTIVE_TOKENS>
void hgtxr_cyclic_s2_qkv_first_step_stage(token_buffer_t tokens,
                                          const HgtxrAxiWordT *cyclic_weights,
                                          int layer_idx) {
#pragma HLS INLINE off
  HgtxrDataT x_tile[HGTXR_CYCLIC_TM][HGTXR_CYCLIC_TC];
  HgtxrAccumT q_acc[HGTXR_CYCLIC_TM][HGTXR_CYCLIC_TC];
  HgtxrAccumT k_acc[HGTXR_CYCLIC_TM][HGTXR_CYCLIC_TC];
  HgtxrAccumT v_acc[HGTXR_CYCLIC_TM][HGTXR_CYCLIC_TC];
#if HGTXR_CYCLIC_SMALL_TILE_LUTRAM
#pragma HLS bind_storage variable=x_tile type=ram_2p impl=lutram
#else
#pragma HLS bind_storage variable=x_tile type=ram_2p impl=bram
#endif
#pragma HLS bind_storage variable=q_acc type=ram_2p impl=bram
#pragma HLS bind_storage variable=k_acc type=ram_2p impl=bram
#pragma HLS bind_storage variable=v_acc type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=x_tile cyclic factor=HGTXR_PAR_QK dim=2
#pragma HLS ARRAY_PARTITION variable=q_acc cyclic factor=HGTXR_PAR_FFN dim=2
#pragma HLS ARRAY_PARTITION variable=k_acc cyclic factor=HGTXR_PAR_FFN dim=2
#pragma HLS ARRAY_PARTITION variable=v_acc cyclic factor=HGTXR_PAR_FFN dim=2

  for (int token_base = 0; token_base < ACTIVE_TOKENS;
       token_base += HGTXR_CYCLIC_TM) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TOKENS
    for (int output_tile = 0; output_tile < HGTXR_CYCLIC_CHANNEL_TILES;
         ++output_tile) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_CYCLIC_CHANNEL_TILES
      for (int input_tile = 0; input_tile < HGTXR_CYCLIC_CHANNEL_TILES;
           ++input_tile) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_CYCLIC_CHANNEL_TILES
        const int input_channel_base = input_tile * HGTXR_CYCLIC_TC;
        for (int tm = 0; tm < HGTXR_CYCLIC_TM; ++tm) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_SEQ
          for (int tc = 0; tc < HGTXR_CYCLIC_TC; ++tc) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_MODEL_DIM
#pragma HLS PIPELINE II=1
            const int token_idx = token_base + tm;
            const int channel_idx = input_channel_base + tc;
            if (token_idx < ACTIVE_TOKENS && channel_idx < HGTXR_EMBED) {
              x_tile[tm][tc] =
                  static_cast<HgtxrDataT>(tokens[token_idx][channel_idx]);
            } else {
              x_tile[tm][tc] = 0;
            }
          }
        }

        hgtxr_s2_qkv_project_pair_accum<HGTXR_CYCLIC_TM, HGTXR_CYCLIC_TC,
                                        HGTXR_CYCLIC_TC>(
            cyclic_weights, layer_idx, output_tile, input_tile,
            HGTXR_CYCLIC_CHANNEL_TILES, x_tile, q_acc, k_acc, v_acc,
            input_tile == 0);
      }

      const int output_channel_base = output_tile * HGTXR_CYCLIC_TC;
      for (int tm = 0; tm < HGTXR_CYCLIC_TM; ++tm) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_SEQ
        for (int tc = 0; tc < HGTXR_CYCLIC_TC; ++tc) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_MODEL_DIM
#pragma HLS PIPELINE II=1
          const int token_idx = token_base + tm;
          const int channel_idx = output_channel_base + tc;
          if (token_idx < ACTIVE_TOKENS && channel_idx < HGTXR_EMBED) {
            HgtxrAccumT merged = q_acc[tm][tc] + k_acc[tm][tc] + v_acc[tm][tc];
            tokens[token_idx][channel_idx] = static_cast<hgtxr_data_t>(
                static_cast<HgtxrDataT>(merged));
          }
        }
      }
    }
  }
}
#endif

#if HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS && HGTXR_ENABLE_CYCLIC_S2_FIRST_STEP
template <int ACTIVE_TOKENS>
void hgtxr_cyclic_s2_wo_first_step_stage(token_buffer_t tokens,
                                         const HgtxrAxiWordT *cyclic_weights,
                                         int layer_idx) {
#pragma HLS INLINE off
  HgtxrDataT x_tile[HGTXR_CYCLIC_TM][HGTXR_CYCLIC_TC];
  HgtxrAccumT out_acc[HGTXR_CYCLIC_TM][HGTXR_CYCLIC_TC];
#if HGTXR_CYCLIC_SMALL_TILE_LUTRAM
#pragma HLS bind_storage variable=x_tile type=ram_2p impl=lutram
#else
#pragma HLS bind_storage variable=x_tile type=ram_2p impl=bram
#endif
#pragma HLS bind_storage variable=out_acc type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=x_tile cyclic factor=HGTXR_PAR_QK dim=2
#pragma HLS ARRAY_PARTITION variable=out_acc cyclic factor=HGTXR_PAR_FFN dim=2

  for (int token_base = 0; token_base < ACTIVE_TOKENS;
       token_base += HGTXR_CYCLIC_TM) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TOKENS
    for (int output_tile = 0; output_tile < HGTXR_CYCLIC_CHANNEL_TILES;
         ++output_tile) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_CYCLIC_CHANNEL_TILES
      for (int input_tile = 0; input_tile < HGTXR_CYCLIC_CHANNEL_TILES;
           ++input_tile) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_CYCLIC_CHANNEL_TILES
        const int input_channel_base = input_tile * HGTXR_CYCLIC_TC;
        for (int tm = 0; tm < HGTXR_CYCLIC_TM; ++tm) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_SEQ
          for (int tc = 0; tc < HGTXR_CYCLIC_TC; ++tc) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_MODEL_DIM
#pragma HLS PIPELINE II=1
            const int token_idx = token_base + tm;
            const int channel_idx = input_channel_base + tc;
            if (token_idx < ACTIVE_TOKENS && channel_idx < HGTXR_EMBED) {
              x_tile[tm][tc] =
                  static_cast<HgtxrDataT>(tokens[token_idx][channel_idx]);
            } else {
              x_tile[tm][tc] = 0;
            }
          }
        }

        hgtxr_s2_project_pair_accum<HGTXR_CYCLIC_TM, HGTXR_CYCLIC_TC,
                                    HGTXR_CYCLIC_TC>(
            cyclic_weights, layer_idx, output_tile, input_tile,
            HGTXR_CYCLIC_CHANNEL_TILES, HGTXR_CYCLIC_WEIGHT_WO_OFFSET, x_tile,
            out_acc, input_tile == 0);
      }

      const int output_channel_base = output_tile * HGTXR_CYCLIC_TC;
      for (int tm = 0; tm < HGTXR_CYCLIC_TM; ++tm) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_SEQ
        for (int tc = 0; tc < HGTXR_CYCLIC_TC; ++tc) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_MODEL_DIM
#pragma HLS PIPELINE II=1
          const int token_idx = token_base + tm;
          const int channel_idx = output_channel_base + tc;
          if (token_idx < ACTIVE_TOKENS && channel_idx < HGTXR_EMBED) {
            tokens[token_idx][channel_idx] =
                static_cast<hgtxr_data_t>(out_acc[tm][tc]);
          }
        }
      }
    }
  }
}
#endif

template <int ACTIVE_TOKENS>
#if HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS
void hgtxr_cyclic_transformer_stage(token_buffer_t tokens, const HgtxrAxiWordT *cyclic_weights, int block_idx) {
#else
void hgtxr_cyclic_transformer_stage(token_buffer_t tokens, int block_idx) {
#endif
#pragma HLS INLINE off
#if HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS && HGTXR_ENABLE_CYCLIC_S2_BLOCK_FIRST_STEP
  hgtxr_cyclic_s2_block_first_step_stage<ACTIVE_TOKENS>(
      tokens, cyclic_weights, block_idx);
  return;
#endif
#if HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS && HGTXR_ENABLE_CYCLIC_S2_MLP_FIRST_STEP
  hgtxr_cyclic_s2_mlp_first_step_stage<ACTIVE_TOKENS>(
      tokens, cyclic_weights, block_idx);
  return;
#endif
#if HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS && HGTXR_ENABLE_CYCLIC_S2_ATTN_FIRST_STEP
  hgtxr_cyclic_s2_attention_first_step_stage<ACTIVE_TOKENS>(
      tokens, cyclic_weights, block_idx);
  return;
#endif

#if HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS && HGTXR_ENABLE_CYCLIC_S2_QKV_FIRST_STEP
  hgtxr_cyclic_s2_qkv_first_step_stage<ACTIVE_TOKENS>(tokens, cyclic_weights,
                                                      block_idx);
  return;
#endif
#if HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS && HGTXR_ENABLE_CYCLIC_S2_FIRST_STEP
  hgtxr_cyclic_s2_wo_first_step_stage<ACTIVE_TOKENS>(tokens, cyclic_weights,
                                                     block_idx);
  return;
#endif
  HgtxrDataT norm1_gamma[HGTXR_CYCLIC_TC];
  HgtxrDataT norm1_beta[HGTXR_CYCLIC_TC];
  HgtxrDataT norm2_gamma[HGTXR_CYCLIC_TC];
  HgtxrDataT norm2_beta[HGTXR_CYCLIC_TC];
  HgtxrDataT wq[HGTXR_CYCLIC_TC][HGTXR_CYCLIC_TC];
  HgtxrDataT wk[HGTXR_CYCLIC_TC][HGTXR_CYCLIC_TC];
  HgtxrDataT wv[HGTXR_CYCLIC_TC][HGTXR_CYCLIC_TC];
  HgtxrDataT wo[HGTXR_CYCLIC_TC][HGTXR_CYCLIC_TC];
  HgtxrDataT w1[HGTXR_CYCLIC_TC][HGTXR_CYCLIC_TF];
  HgtxrDataT w2[HGTXR_CYCLIC_TF][HGTXR_CYCLIC_TC];
  HgtxrDataT x_tile[HGTXR_CYCLIC_TM][HGTXR_CYCLIC_TC];
  HgtxrDataT out_tile[HGTXR_CYCLIC_TM][HGTXR_CYCLIC_TC];
#if HGTXR_CYCLIC_FORCE_URAM_WEIGHT_TILES
#pragma HLS bind_storage variable=wq type=ram_2p impl=uram
#pragma HLS bind_storage variable=wk type=ram_2p impl=uram
#pragma HLS bind_storage variable=wv type=ram_2p impl=uram
#pragma HLS bind_storage variable=wo type=ram_2p impl=uram
#pragma HLS bind_storage variable=w1 type=ram_2p impl=uram
#pragma HLS bind_storage variable=w2 type=ram_2p impl=uram
#else
#pragma HLS bind_storage variable=wq type=ram_2p impl=bram
#pragma HLS bind_storage variable=wk type=ram_2p impl=bram
#pragma HLS bind_storage variable=wv type=ram_2p impl=bram
#pragma HLS bind_storage variable=wo type=ram_2p impl=bram
#pragma HLS bind_storage variable=w1 type=ram_2p impl=bram
#pragma HLS bind_storage variable=w2 type=ram_2p impl=bram
#endif
#if HGTXR_CYCLIC_SMALL_TILE_LUTRAM
#pragma HLS bind_storage variable=x_tile type=ram_2p impl=lutram
#pragma HLS bind_storage variable=out_tile type=ram_2p impl=lutram
#else
#pragma HLS bind_storage variable=x_tile type=ram_2p impl=bram
#pragma HLS bind_storage variable=out_tile type=ram_2p impl=bram
#endif

#if HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS
  hgtxr_load_cyclic_transformer_weights_from_packed(
      cyclic_weights, block_idx, norm1_gamma, norm1_beta, norm2_gamma, norm2_beta,
      wq, wk, wv, wo, w1, w2);
#else
  (void)block_idx;
  hgtxr_init_cyclic_transformer_weights(norm1_gamma, norm1_beta, norm2_gamma,
                                        norm2_beta, wq, wk, wv, wo, w1, w2);
#endif

  for (int token_base = 0; token_base < ACTIVE_TOKENS; token_base += HGTXR_CYCLIC_TM) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TOKENS
    for (int channel_base = 0; channel_base < HGTXR_EMBED; channel_base += HGTXR_CYCLIC_TC) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_EMBED
      for (int tm = 0; tm < HGTXR_CYCLIC_TM; ++tm) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_SEQ
        for (int tc = 0; tc < HGTXR_CYCLIC_TC; ++tc) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_MODEL_DIM
#pragma HLS PIPELINE II=1
          int token_idx = token_base + tm;
          int channel_idx = channel_base + tc;
          if (token_idx < ACTIVE_TOKENS && channel_idx < HGTXR_EMBED) {
            x_tile[tm][tc] = static_cast<HgtxrDataT>(tokens[token_idx][channel_idx]);
          } else {
            x_tile[tm][tc] = 0;
          }
        }
      }

      transformer_block_tile<HGTXR_CYCLIC_TM, HGTXR_CYCLIC_TC, HGTXR_CYCLIC_TF>(
          x_tile, norm1_gamma, norm1_beta, norm2_gamma, norm2_beta, wq, wk, wv,
          wo, w1, w2, out_tile, 0, 0, 0, 0, 0, false, true);

      for (int tm = 0; tm < HGTXR_CYCLIC_TM; ++tm) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_SEQ
        for (int tc = 0; tc < HGTXR_CYCLIC_TC; ++tc) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TILE_MODEL_DIM
#pragma HLS PIPELINE II=1
          int token_idx = token_base + tm;
          int channel_idx = channel_base + tc;
          if (token_idx < ACTIVE_TOKENS && channel_idx < HGTXR_EMBED) {
            tokens[token_idx][channel_idx] = static_cast<hgtxr_data_t>(out_tile[tm][tc]);
          }
        }
      }
    }
  }
}

}  // namespace
#endif

void hgtxr_top(const hgtxr_data_t frame[HGTXR_HEIGHT][HGTXR_WIDTH], const hgtxr_data_t event_pos[HGTXR_HEIGHT][HGTXR_WIDTH], const hgtxr_data_t event_neg[HGTXR_HEIGHT][HGTXR_WIDTH], const hgtxr_data_t prev_state[HGTXR_STATE],
#if HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS
               const hgtxr::cyclic_transformer::HgtxrAxiWordT *cyclic_weights,
#endif
               hgtxr_data_t out_state[HGTXR_STATE], int *runtime_state) {
#pragma HLS INTERFACE m_axi port=frame offset=slave bundle=gmem_frame depth=65536 max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi port=event_pos offset=slave bundle=gmem_event_pos depth=65536 max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi port=event_neg offset=slave bundle=gmem_event_neg depth=65536 max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi port=prev_state offset=slave bundle=gmem_prev_state depth=6 max_read_burst_length=16 num_read_outstanding=2
#pragma HLS INTERFACE m_axi port=out_state offset=slave bundle=gmem_out_state depth=6 max_write_burst_length=16 num_write_outstanding=2
#pragma HLS INTERFACE m_axi port=runtime_state offset=slave bundle=gmem_runtime_state depth=1 max_write_burst_length=2 num_write_outstanding=2
#if HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS
#pragma HLS INTERFACE m_axi port=cyclic_weights offset=slave bundle=gmem_cyclic_weights depth=HGTXR_CYCLIC_WEIGHT_TOTAL_WORDS max_read_burst_length=64 num_read_outstanding=8
#endif
#pragma HLS INTERFACE s_axilite port=frame bundle=control
#pragma HLS INTERFACE s_axilite port=event_pos bundle=control
#pragma HLS INTERFACE s_axilite port=event_neg bundle=control
#pragma HLS INTERFACE s_axilite port=prev_state bundle=control
#pragma HLS INTERFACE s_axilite port=out_state bundle=control
#pragma HLS INTERFACE s_axilite port=runtime_state bundle=control
#if HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS
#pragma HLS INTERFACE s_axilite port=cyclic_weights bundle=control
#endif
#pragma HLS INTERFACE s_axilite port=return bundle=control
  token_buffer_t frame_tokens;
  token_buffer_t event_tokens;
  token_buffer_t routed_tokens;
  token_buffer_t global_tokens;
  pooled_buffer_t frame_pooled;
  pooled_buffer_t event_pooled;
  hgtxr_data_t search_logits[HGTXR_SEARCH_LOGITS];
  hgtxr_data_t fused[HGTXR_EMBED * 2];
  hgtxr_data_t track_state[HGTXR_STATE];

  int mode = prev_state[0] == (hgtxr_data_t)0 ? HGTXR_MODE_SEARCH : HGTXR_MODE_TRACK;
  int prefetch_valid = 0;
  weight_prefetcher(mode, 0, &prefetch_valid);
  frame_patch_embed(frame, frame_tokens);
  event_patch_embed(event_pos, event_neg, event_tokens);
  global_buffer_write_search(frame_tokens, global_tokens);
  noc_route_tokens(global_tokens, routed_tokens, HGTXR_MODE_SEARCH);
#if HGTXR_ENABLE_CYCLIC_TRANSFORMER_TOP
  for (int i = 0; i < controller_depth_limit(HGTXR_MODE_SEARCH) / 2; ++i) {
#if HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS
    hgtxr_cyclic_transformer_stage<HGTXR_SEARCH_TOKENS>(routed_tokens, cyclic_weights, i);
#else
    hgtxr_cyclic_transformer_stage<HGTXR_SEARCH_TOKENS>(routed_tokens, i);
#endif
  }
#else
  for (int i = 0; i < controller_depth_limit(HGTXR_MODE_SEARCH) / 2; ++i) { attention_stage(routed_tokens); mlp_stage(routed_tokens); }
#endif
  pool_tokens(routed_tokens, frame_pooled);
  global_buffer_write_track(event_tokens, global_tokens);
  noc_route_tokens(global_tokens, routed_tokens, HGTXR_MODE_TRACK);
#if HGTXR_ENABLE_CYCLIC_TRANSFORMER_TOP
  for (int i = 0; i < controller_depth_limit(HGTXR_MODE_TRACK) / 2; ++i) {
#if HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS
    hgtxr_cyclic_transformer_stage<HGTXR_TRACK_TOKENS>(
        routed_tokens, cyclic_weights, HGTXR_SEARCH_DEPTH / 2 + i);
#else
    hgtxr_cyclic_transformer_stage<HGTXR_TRACK_TOKENS>(routed_tokens, HGTXR_SEARCH_DEPTH / 2 + i);
#endif
  }
#else
  for (int i = 0; i < controller_depth_limit(HGTXR_MODE_TRACK) / 2; ++i) { attention_stage(routed_tokens); mlp_stage(routed_tokens); }
#endif
  pool_tokens(routed_tokens, event_pooled);
  search_head(frame_pooled, search_logits);
  fusion(frame_pooled, event_pooled, prev_state, fused);
  track_head(fused, prev_state, track_state);

  HGTXRDecision d = runtime_fsm(search_logits[6], (hgtxr_data_t)1, (hgtxr_data_t)1, (hgtxr_data_t)1, (hgtxr_data_t)1, false);
  *runtime_state = d.state;
  for (int i = 0; i < HGTXR_STATE; ++i) out_state[i] = d.state == 1 ? track_state[i] : search_logits[i];
}
