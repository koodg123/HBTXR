#ifndef HGTXR_ZCU104_CYCLIC_S2_QKV_FIRST_STEP_DEFINES_H
#define HGTXR_ZCU104_CYCLIC_S2_QKV_FIRST_STEP_DEFINES_H

// S2 QKV first-step keeps conservative ZCU104-fit tiling and enables only the
// packed channel-pair Q/K/V projection accumulation path. It is a gated
// intermediate mode, not a full Transformer block replacement yet.
#define HGTXR_TILING_FACTOR 1
#define HGTXR_PARALLELISM_FACTOR 1
#define HGTXR_BUS_WIDTH 128
#define HGTXR_BIT_WIDTH 16
#define HGTXR_BUFFER_SIZE 64
#define HGTXR_FIFO_DEPTH 16
#define HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS 1
#define HGTXR_ENABLE_CYCLIC_S2_QKV_FIRST_STEP 1
#define HGTXR_CYCLIC_MODEL_DIM 192
#define HGTXR_CYCLIC_WEIGHT_BLOCKS 6

#endif  // HGTXR_ZCU104_CYCLIC_S2_QKV_FIRST_STEP_DEFINES_H
