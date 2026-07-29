#pragma once

#if defined(__SYNTHESIS__) || defined(HGTXR_HLS_FIXED_CSIM)
#include <ap_fixed.h>
#ifndef HGTXR_BIT_WIDTH
#define HGTXR_BIT_WIDTH 16
#endif
#ifndef HGTXR_DATA_I
#if HGTXR_BIT_WIDTH <= 8
#define HGTXR_DATA_I 4
#else
#define HGTXR_DATA_I 6
#endif
#endif
typedef ap_fixed<HGTXR_BIT_WIDTH, HGTXR_DATA_I> hgtxr_data_t;
#ifndef HGTXR_ACC_W
#define HGTXR_ACC_W 32
#endif
#ifndef HGTXR_ACC_I
#define HGTXR_ACC_I 12
#endif
typedef ap_fixed<HGTXR_ACC_W, HGTXR_ACC_I> hgtxr_acc_t;
#else
typedef float hgtxr_data_t;
typedef float hgtxr_acc_t;
#endif
