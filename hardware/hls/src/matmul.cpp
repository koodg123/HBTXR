#include "../include/common.h"

static const int HGTXR_MM_TILE_M = 8;
static const int HGTXR_MM_TILE_N = 8;
static const int HGTXR_MM_TILE_K = 16;

void matmul(const hgtxr_data_t *a, const hgtxr_data_t *b, hgtxr_data_t *c, int m, int n, int k) {
  hgtxr_acc_t acc_tile[HGTXR_MM_TILE_M][HGTXR_MM_TILE_N];
#pragma HLS array_partition variable=acc_tile complete dim=0

  for (int mi = 0; mi < m; mi += HGTXR_MM_TILE_M) {
    for (int nj = 0; nj < n; nj += HGTXR_MM_TILE_N) {
      for (int tm = 0; tm < HGTXR_MM_TILE_M; ++tm) {
        for (int tn = 0; tn < HGTXR_MM_TILE_N; ++tn) {
#pragma HLS unroll
          acc_tile[tm][tn] = 0;
        }
      }

      for (int kk = 0; kk < k; kk += HGTXR_MM_TILE_K) {
        for (int tk = 0; tk < HGTXR_MM_TILE_K; ++tk) {
          int p = kk + tk;
          if (p >= k) continue;
          for (int tm = 0; tm < HGTXR_MM_TILE_M; ++tm) {
            int row = mi + tm;
            if (row >= m) continue;
            hgtxr_data_t a_val = a[row * k + p];
            for (int tn = 0; tn < HGTXR_MM_TILE_N; ++tn) {
#pragma HLS pipeline II=1
              int col = nj + tn;
              if (col < n) acc_tile[tm][tn] += a_val * b[p * n + col];
            }
          }
        }
      }

      for (int tm = 0; tm < HGTXR_MM_TILE_M; ++tm) {
        int row = mi + tm;
        if (row >= m) continue;
        for (int tn = 0; tn < HGTXR_MM_TILE_N; ++tn) {
#pragma HLS pipeline II=1
          int col = nj + tn;
          if (col < n) c[row * n + col] = clamp_data(acc_tile[tm][tn]);
        }
      }
    }
  }
}
