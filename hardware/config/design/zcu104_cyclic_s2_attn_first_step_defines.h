#ifndef HGTXR_ZCU104_CYCLIC_S2_ATTN_FIRST_STEP_DEFINES_H
#define HGTXR_ZCU104_CYCLIC_S2_ATTN_FIRST_STEP_DEFINES_H

// S2 attention first-step validates the gated packed channel-pair path through
// Q/K/V projection, tile-local attention, and S2 WO projection. It remains an
// intermediate mode before residual, LayerNorm, and MLP hidden-tile integration.
#define HGTXR_TILING_FACTOR 1
#define HGTXR_TILE_CHANNELS 64
#define HGTXR_PARALLELISM_FACTOR 1
#define HGTXR_BUS_WIDTH 128
#define HGTXR_BIT_WIDTH 16
#define HGTXR_BUFFER_SIZE 64
#define HGTXR_FIFO_DEPTH 16
#define HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS 1
#define HGTXR_ENABLE_CYCLIC_S2_ATTN_FIRST_STEP 1
#define HGTXR_CYCLIC_MODEL_DIM 192
#define HGTXR_CYCLIC_HEADS 3
#define HGTXR_CYCLIC_HEAD_DIM 64
#define HGTXR_CYCLIC_ATTN_SCORE_SCALE_SHIFT 3
#define HGTXR_CYCLIC_REQUIRE_HEAD_ALIGNED_ATTENTION 1
#define HGTXR_CYCLIC_WEIGHT_BLOCKS 6

#endif  // HGTXR_ZCU104_CYCLIC_S2_ATTN_FIRST_STEP_DEFINES_H
