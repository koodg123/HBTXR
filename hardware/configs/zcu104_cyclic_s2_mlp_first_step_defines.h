#ifndef HGTXR_ZCU104_CYCLIC_S2_MLP_FIRST_STEP_DEFINES_H
#define HGTXR_ZCU104_CYCLIC_S2_MLP_FIRST_STEP_DEFINES_H

// S2 MLP first-step validates DeiT-style mlp_ratio=4 hidden expansion with
// explicit hidden-tile traversal over packed W1 and W2 schedules. It remains
// an intermediate mode before combining attention, residual, and LayerNorm.
#define HGTXR_TILING_FACTOR 1
#define HGTXR_PARALLELISM_FACTOR 1
#define HGTXR_BUS_WIDTH 128
#define HGTXR_BIT_WIDTH 16
#define HGTXR_BUFFER_SIZE 64
#define HGTXR_FIFO_DEPTH 16
#define HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS 1
#define HGTXR_ENABLE_CYCLIC_S2_MLP_FIRST_STEP 1
#define HGTXR_CYCLIC_MODEL_DIM 192
#define HGTXR_CYCLIC_MLP_RATIO 4
#define HGTXR_CYCLIC_WEIGHT_BLOCKS 6

#endif  // HGTXR_ZCU104_CYCLIC_S2_MLP_FIRST_STEP_DEFINES_H
