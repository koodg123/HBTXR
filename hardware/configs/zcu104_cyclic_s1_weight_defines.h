#ifndef HGTXR_ZCU104_CYCLIC_S1_WEIGHT_DEFINES_H
#define HGTXR_ZCU104_CYCLIC_S1_WEIGHT_DEFINES_H

// S1 keeps the active-token-specialized cyclic top and enables the optional
// packed weight AXI port. Tiling/parallelism remain conservative for the first
// weight-port synthesis so resource growth is attributable to weight loading.
#define HGTXR_TILING_FACTOR 1
#define HGTXR_PARALLELISM_FACTOR 1
#define HGTXR_BUS_WIDTH 128
#define HGTXR_BIT_WIDTH 16
#define HGTXR_BUFFER_SIZE 64
#define HGTXR_FIFO_DEPTH 16
#define HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS 1
#define HGTXR_CYCLIC_WEIGHT_BLOCKS 6

#endif  // HGTXR_ZCU104_CYCLIC_S1_WEIGHT_DEFINES_H
