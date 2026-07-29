#ifndef HGTXR_CYCLIC_TRANSFORMER_PARAMS_HPP
#define HGTXR_CYCLIC_TRANSFORMER_PARAMS_HPP

#include <ap_fixed.h>
#include <ap_int.h>


// Sweep overrides. Keep macro names separate from the internal constexpr names
// so generated -D flags can tune builds without breaking existing source code.
#ifndef HGTXR_TILING_FACTOR
#define HGTXR_TILING_FACTOR 2
#endif
#ifndef HGTXR_PARALLELISM_FACTOR
#define HGTXR_PARALLELISM_FACTOR 2
#endif
#ifndef HGTXR_BUS_WIDTH
#define HGTXR_BUS_WIDTH 256
#endif
#ifndef HGTXR_BIT_WIDTH
#define HGTXR_BIT_WIDTH 16
#endif
#ifndef HGTXR_WEIGHT_BIT_WIDTH
#define HGTXR_WEIGHT_BIT_WIDTH HGTXR_BIT_WIDTH
#endif
#ifndef HGTXR_BUFFER_SIZE
#define HGTXR_BUFFER_SIZE 512
#endif
#ifndef HGTXR_FIFO_DEPTH
#define HGTXR_FIFO_DEPTH 128
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
#ifndef HGTXR_CYCLIC_MODEL_DIM
#define HGTXR_CYCLIC_MODEL_DIM 192
#endif
#ifndef HGTXR_CYCLIC_MLP_RATIO
#define HGTXR_CYCLIC_MLP_RATIO 4
#endif
#ifndef HGTXR_CYCLIC_FF_DIM
#define HGTXR_CYCLIC_FF_DIM (HGTXR_CYCLIC_MODEL_DIM * HGTXR_CYCLIC_MLP_RATIO)
#endif
#ifndef HGTXR_CYCLIC_HEADS
#define HGTXR_CYCLIC_HEADS 3
#endif
#ifndef HGTXR_CYCLIC_HEAD_DIM
#define HGTXR_CYCLIC_HEAD_DIM (HGTXR_CYCLIC_MODEL_DIM / HGTXR_CYCLIC_HEADS)
#endif
#ifndef HGTXR_CYCLIC_REQUIRE_HEAD_ALIGNED_ATTENTION
#define HGTXR_CYCLIC_REQUIRE_HEAD_ALIGNED_ATTENTION 0
#endif
#ifndef HGTXR_TILE_TOKENS
#define HGTXR_TILE_TOKENS (16 * HGTXR_TILING_FACTOR)
#endif
#ifndef HGTXR_TILE_CHANNELS
#define HGTXR_TILE_CHANNELS (32 * HGTXR_TILING_FACTOR)
#endif
#ifndef HGTXR_TILE_K
#define HGTXR_TILE_K HGTXR_TILE_CHANNELS
#endif
#ifndef HGTXR_CYCLIC_ATTN_SCORE_SCALE_SHIFT
#if (HGTXR_TILE_CHANNELS == HGTXR_CYCLIC_HEAD_DIM) && (HGTXR_CYCLIC_HEAD_DIM == 64)
#define HGTXR_CYCLIC_ATTN_SCORE_SCALE_SHIFT 3
#else
#define HGTXR_CYCLIC_ATTN_SCORE_SCALE_SHIFT 0
#endif
#endif
#ifndef HGTXR_HEAD_PAR
#define HGTXR_HEAD_PAR HGTXR_PARALLELISM_FACTOR
#endif
#ifndef HGTXR_PE_PAR
#define HGTXR_PE_PAR HGTXR_PARALLELISM_FACTOR
#endif
#ifndef HGTXR_UNROLL_K
#define HGTXR_UNROLL_K HGTXR_PE_PAR
#endif
#ifndef HGTXR_PIPELINE_STAGES
#define HGTXR_PIPELINE_STAGES 2
#endif
#ifndef HGTXR_BUS_WIDTH_BITS
#define HGTXR_BUS_WIDTH_BITS HGTXR_BUS_WIDTH
#endif
#ifndef HGTXR_DATA_W
#define HGTXR_DATA_W HGTXR_BIT_WIDTH
#endif
#ifndef HGTXR_DATA_I
#if HGTXR_DATA_W <= 8
#define HGTXR_DATA_I 4
#else
#define HGTXR_DATA_I 6
#endif
#endif
#ifndef HGTXR_WEIGHT_INT_WIDTH
#if HGTXR_WEIGHT_BIT_WIDTH <= 4
#define HGTXR_WEIGHT_INT_WIDTH 2
#else
#define HGTXR_WEIGHT_INT_WIDTH HGTXR_DATA_I
#endif
#endif
#ifndef HGTXR_ACC_W
#define HGTXR_ACC_W 32
#endif
#ifndef HGTXR_ACC_I
#define HGTXR_ACC_I 12
#endif
#ifndef HGTXR_LOCAL_BUFFER_DEPTH
#define HGTXR_LOCAL_BUFFER_DEPTH (((HGTXR_BUFFER_SIZE) * 1024 * 8) / HGTXR_DATA_W)
#endif
#ifndef HGTXR_STREAM_FIFO_DEPTH
#define HGTXR_STREAM_FIFO_DEPTH HGTXR_FIFO_DEPTH
#endif
#ifndef HGTXR_CYCLIC_FORCE_URAM_WEIGHT_TILES
#define HGTXR_CYCLIC_FORCE_URAM_WEIGHT_TILES 1
#endif
#ifndef HGTXR_CYCLIC_FORCE_URAM_LARGE_TEMPS
#define HGTXR_CYCLIC_FORCE_URAM_LARGE_TEMPS 1
#endif
#ifndef HGTXR_CYCLIC_SMALL_TILE_LUTRAM
#define HGTXR_CYCLIC_SMALL_TILE_LUTRAM 1
#endif

namespace hgtxr {
namespace cyclic_transformer {

// Tile policy. These are conservative ZCU104-fit defaults; sweep scripts should
// override them through compiler definitions or generated config headers.
static constexpr int HGTXR_TILE_BATCH = 1;
static constexpr int HGTXR_TILE_SEQ = HGTXR_TILE_TOKENS;
static constexpr int HGTXR_TILE_MODEL_DIM = HGTXR_TILE_CHANNELS;
static constexpr int HGTXR_TILE_HEAD_DIM = HGTXR_TILE_K;
static constexpr int HGTXR_TILE_HEADS = HGTXR_HEAD_PAR;
static constexpr int HGTXR_TILE_FF_DIM = HGTXR_TILE_CHANNELS;
static constexpr int HGTXR_TILE_LAYER_CHUNK = 1;
static constexpr int HGTXR_CYCLIC_NUM_HEADS = HGTXR_CYCLIC_HEADS;
static constexpr int HGTXR_CYCLIC_PER_HEAD_DIM = HGTXR_CYCLIC_HEAD_DIM;
static constexpr int HGTXR_CYCLIC_SCORE_SCALE_SHIFT = HGTXR_CYCLIC_ATTN_SCORE_SCALE_SHIFT;

// Parallelism policy.
static constexpr int HGTXR_PAR_BATCH = 1;
static constexpr int HGTXR_PAR_HEADS = HGTXR_HEAD_PAR;
static constexpr int HGTXR_PAR_QK = HGTXR_PE_PAR;
static constexpr int HGTXR_PAR_ATTN = HGTXR_PE_PAR;
static constexpr int HGTXR_PAR_FFN = HGTXR_PE_PAR;
static constexpr int HGTXR_PAR_PIPELINE_STAGES = HGTXR_PIPELINE_STAGES;

// AXI/PYNQ bus policy.
static constexpr int HGTXR_AXI_ADDR_WIDTH = 64;
static constexpr int HGTXR_AXI_DATA_WIDTH = HGTXR_BUS_WIDTH_BITS;
static constexpr int HGTXR_AXI_STRB_WIDTH = HGTXR_AXI_DATA_WIDTH / 8;
static constexpr int HGTXR_AXI_BURST_MAX = 16;
static constexpr int HGTXR_AXI_BYTES_PER_BEAT = HGTXR_AXI_DATA_WIDTH / 8;
static constexpr int HGTXR_WEIGHT_WIDTH = HGTXR_WEIGHT_BIT_WIDTH;
static constexpr int HGTXR_WEIGHT_INT = HGTXR_WEIGHT_INT_WIDTH;
static constexpr int HGTXR_WEIGHT_FRAC = HGTXR_WEIGHT_WIDTH - HGTXR_WEIGHT_INT;
static constexpr int HGTXR_AXI_WEIGHT_LANES = HGTXR_AXI_DATA_WIDTH / HGTXR_WEIGHT_WIDTH;
static constexpr int HGTXR_CYCLIC_CHANNEL_TILES =
    (HGTXR_CYCLIC_MODEL_DIM + HGTXR_TILE_MODEL_DIM - 1) /
    HGTXR_TILE_MODEL_DIM;
static constexpr int HGTXR_CYCLIC_HIDDEN_TILES =
    (HGTXR_CYCLIC_FF_DIM + HGTXR_TILE_FF_DIM - 1) / HGTXR_TILE_FF_DIM;
static constexpr int HGTXR_CYCLIC_WEIGHT_NORM1_GAMMA_OFFSET = 0;
static constexpr int HGTXR_CYCLIC_WEIGHT_NORM1_BETA_OFFSET = HGTXR_CYCLIC_WEIGHT_NORM1_GAMMA_OFFSET + HGTXR_TILE_MODEL_DIM;
static constexpr int HGTXR_CYCLIC_WEIGHT_NORM2_GAMMA_OFFSET = HGTXR_CYCLIC_WEIGHT_NORM1_BETA_OFFSET + HGTXR_TILE_MODEL_DIM;
static constexpr int HGTXR_CYCLIC_WEIGHT_NORM2_BETA_OFFSET = HGTXR_CYCLIC_WEIGHT_NORM2_GAMMA_OFFSET + HGTXR_TILE_MODEL_DIM;
static constexpr int HGTXR_CYCLIC_WEIGHT_WQ_OFFSET = HGTXR_CYCLIC_WEIGHT_NORM2_BETA_OFFSET + HGTXR_TILE_MODEL_DIM;
static constexpr int HGTXR_CYCLIC_WEIGHT_WK_OFFSET = HGTXR_CYCLIC_WEIGHT_WQ_OFFSET + HGTXR_TILE_MODEL_DIM * HGTXR_TILE_MODEL_DIM;
static constexpr int HGTXR_CYCLIC_WEIGHT_WV_OFFSET = HGTXR_CYCLIC_WEIGHT_WK_OFFSET + HGTXR_TILE_MODEL_DIM * HGTXR_TILE_MODEL_DIM;
static constexpr int HGTXR_CYCLIC_WEIGHT_WO_OFFSET = HGTXR_CYCLIC_WEIGHT_WV_OFFSET + HGTXR_TILE_MODEL_DIM * HGTXR_TILE_MODEL_DIM;
static constexpr int HGTXR_CYCLIC_WEIGHT_W1_OFFSET = HGTXR_CYCLIC_WEIGHT_WO_OFFSET + HGTXR_TILE_MODEL_DIM * HGTXR_TILE_MODEL_DIM;
static constexpr int HGTXR_CYCLIC_WEIGHT_W2_OFFSET = HGTXR_CYCLIC_WEIGHT_W1_OFFSET + HGTXR_TILE_MODEL_DIM * HGTXR_TILE_FF_DIM;
static constexpr int HGTXR_CYCLIC_WEIGHT_TILE_ELEMS = HGTXR_CYCLIC_WEIGHT_W2_OFFSET + HGTXR_TILE_FF_DIM * HGTXR_TILE_MODEL_DIM;
static constexpr int HGTXR_CYCLIC_WEIGHT_TILE_WORDS = (HGTXR_CYCLIC_WEIGHT_TILE_ELEMS + HGTXR_AXI_WEIGHT_LANES - 1) / HGTXR_AXI_WEIGHT_LANES;
#ifndef HGTXR_CYCLIC_WEIGHT_BLOCKS
#define HGTXR_CYCLIC_WEIGHT_BLOCKS 6
#endif
static constexpr int HGTXR_CYCLIC_WEIGHT_LAYER_STRIDE_WORDS = HGTXR_CYCLIC_WEIGHT_TILE_WORDS;
static constexpr int HGTXR_CYCLIC_WEIGHT_TOTAL_BLOCKS =
#if HGTXR_ENABLE_CYCLIC_S2_BLOCK_FIRST_STEP
    HGTXR_CYCLIC_WEIGHT_BLOCKS *
    (HGTXR_CYCLIC_CHANNEL_TILES * HGTXR_CYCLIC_CHANNEL_TILES +
     2 * HGTXR_CYCLIC_CHANNEL_TILES * HGTXR_CYCLIC_HIDDEN_TILES);
#elif HGTXR_ENABLE_CYCLIC_S2_MLP_FIRST_STEP
    HGTXR_CYCLIC_WEIGHT_BLOCKS * 2 * HGTXR_CYCLIC_CHANNEL_TILES *
    HGTXR_CYCLIC_HIDDEN_TILES;
#elif HGTXR_ENABLE_CYCLIC_S2_FIRST_STEP || HGTXR_ENABLE_CYCLIC_S2_QKV_FIRST_STEP || HGTXR_ENABLE_CYCLIC_S2_ATTN_FIRST_STEP
    HGTXR_CYCLIC_WEIGHT_BLOCKS * HGTXR_CYCLIC_CHANNEL_TILES *
    HGTXR_CYCLIC_CHANNEL_TILES;
#else
    HGTXR_CYCLIC_WEIGHT_BLOCKS;
#endif
static constexpr int HGTXR_CYCLIC_WEIGHT_TOTAL_WORDS =
    HGTXR_CYCLIC_WEIGHT_LAYER_STRIDE_WORDS * HGTXR_CYCLIC_WEIGHT_TOTAL_BLOCKS;
static constexpr int HGTXR_AXI_CTRL_ADDR_WIDTH = 32;
static constexpr int HGTXR_PYNQ_ALIGN_BYTES = HGTXR_AXI_BYTES_PER_BEAT;

// Fixed-point policy.
static constexpr int HGTXR_DATA_WIDTH = HGTXR_DATA_W;
static constexpr int HGTXR_DATA_INT = HGTXR_DATA_I;
static constexpr int HGTXR_DATA_FRAC = HGTXR_DATA_WIDTH - HGTXR_DATA_INT;

static constexpr int HGTXR_ACC_WIDTH = HGTXR_ACC_W;
static constexpr int HGTXR_ACC_INT = HGTXR_ACC_I;
static constexpr int HGTXR_ACC_FRAC = HGTXR_ACC_WIDTH - HGTXR_ACC_INT;

static constexpr int HGTXR_SCORE_WIDTH = 24;
static constexpr int HGTXR_SCORE_INT = 8;
static constexpr int HGTXR_SCORE_FRAC = HGTXR_SCORE_WIDTH - HGTXR_SCORE_INT;

static constexpr int HGTXR_INDEX_WIDTH = 16;
static constexpr int HGTXR_PTR_WIDTH = 64;

// Local buffer and FIFO defaults.
static constexpr int HGTXR_BUFFER_INPUT_WORDS = HGTXR_LOCAL_BUFFER_DEPTH;
static constexpr int HGTXR_BUFFER_WEIGHT_WORDS = HGTXR_LOCAL_BUFFER_DEPTH;
static constexpr int HGTXR_BUFFER_KV_WORDS = HGTXR_LOCAL_BUFFER_DEPTH;
static constexpr int HGTXR_BUFFER_FFN_WORDS = HGTXR_LOCAL_BUFFER_DEPTH;
static constexpr int HGTXR_BUFFER_OUTPUT_WORDS = HGTXR_LOCAL_BUFFER_DEPTH;

static constexpr int HGTXR_FIFO_INPUT_DEPTH = HGTXR_STREAM_FIFO_DEPTH;
static constexpr int HGTXR_FIFO_QK_DEPTH = HGTXR_STREAM_FIFO_DEPTH;
static constexpr int HGTXR_FIFO_PROJ_DEPTH = HGTXR_STREAM_FIFO_DEPTH;
static constexpr int HGTXR_FIFO_ATTN_DEPTH = HGTXR_STREAM_FIFO_DEPTH;
static constexpr int HGTXR_FIFO_OUT_DEPTH = HGTXR_STREAM_FIFO_DEPTH;

typedef ap_fixed<HGTXR_DATA_WIDTH, HGTXR_DATA_INT, AP_TRN, AP_SAT> HgtxrDataT;
typedef ap_fixed<HGTXR_WEIGHT_WIDTH, HGTXR_WEIGHT_INT, AP_TRN, AP_SAT> HgtxrWeightT;
typedef ap_fixed<HGTXR_ACC_WIDTH, HGTXR_ACC_INT, AP_TRN, AP_SAT> HgtxrAccumT;
typedef ap_fixed<HGTXR_SCORE_WIDTH, HGTXR_SCORE_INT, AP_TRN, AP_SAT> HgtxrScoreT;

typedef ap_int<HGTXR_DATA_WIDTH> HgtxrDataIntT;
typedef ap_int<HGTXR_WEIGHT_WIDTH> HgtxrWeightIntT;
typedef ap_uint<HGTXR_INDEX_WIDTH> HgtxrIndexT;
typedef ap_uint<HGTXR_AXI_DATA_WIDTH> HgtxrAxiWordT;

struct HgtxrRuntimeConfig {
  ap_uint<16> batch_size;
  ap_uint<16> seq_len;
  ap_uint<16> model_dim;
  ap_uint<16> num_layers;
  ap_uint<8> num_attention_heads;
  ap_uint<8> num_kv_heads;
  ap_uint<16> head_dim;
  ap_uint<16> ffn_hidden_dim;
  ap_uint<16> max_cyclic_tokens;

  ap_uint<16> tile_batch;
  ap_uint<16> tile_seq;
  ap_uint<16> tile_model_dim;
  ap_uint<16> tile_head_dim;
  ap_uint<8> tile_heads;
  ap_uint<8> parallel_heads;
  ap_uint<8> parallel_qk;
  ap_uint<8> parallel_ffn;
  ap_uint<8> pipeline_stages;

  ap_uint<2> precision_code;
  ap_int<8> data_scale_shift;
  ap_int<8> accum_scale_shift;
  ap_uint<1> use_dropout;
  ap_uint<1> apply_rotary;
  ap_uint<1> causal_mask;
  ap_uint<1> use_residual;
  ap_uint<1> use_layernorm;

  ap_uint<8> axi_burst_len;
  ap_uint<16> fifo_input_depth;
  ap_uint<16> fifo_qk_depth;
  ap_uint<16> fifo_ffn_depth;
  ap_uint<16> fifo_output_depth;

  ap_uint<HGTXR_AXI_ADDR_WIDTH> input_base_addr;
  ap_uint<HGTXR_AXI_ADDR_WIDTH> output_base_addr;
  ap_uint<HGTXR_AXI_ADDR_WIDTH> w_q_base_addr;
  ap_uint<HGTXR_AXI_ADDR_WIDTH> w_k_base_addr;
  ap_uint<HGTXR_AXI_ADDR_WIDTH> w_v_base_addr;
  ap_uint<HGTXR_AXI_ADDR_WIDTH> w_o_base_addr;
  ap_uint<HGTXR_AXI_ADDR_WIDTH> w_ffn1_base_addr;
  ap_uint<HGTXR_AXI_ADDR_WIDTH> w_ffn2_base_addr;
  ap_uint<HGTXR_AXI_ADDR_WIDTH> kv_cache_k_base_addr;
  ap_uint<HGTXR_AXI_ADDR_WIDTH> kv_cache_v_base_addr;

  ap_uint<1> start;
  ap_uint<1> clear_kv_cache;
};

static_assert(HGTXR_AXI_DATA_WIDTH % HGTXR_DATA_WIDTH == 0,
              "AXI data width must be an integer multiple of numeric data width");
static_assert(HGTXR_AXI_DATA_WIDTH % HGTXR_WEIGHT_WIDTH == 0,
              "AXI data width must be an integer multiple of packed weight width");
static_assert(HGTXR_AXI_WEIGHT_LANES >= 1,
              "AXI data width must be at least one numeric lane");
static_assert(HGTXR_DATA_WIDTH < HGTXR_ACC_WIDTH,
              "Accumulator width should be wider than data width");
static_assert(HGTXR_AXI_BURST_MAX <= 16,
              "Keep burst length conservative for broad AXI compatibility");

}  // namespace cyclic_transformer
}  // namespace hgtxr

#endif  // HGTXR_CYCLIC_TRANSFORMER_PARAMS_HPP
