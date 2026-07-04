#ifndef HGTXR_CYCLIC_SCHEDULER_HPP
#define HGTXR_CYCLIC_SCHEDULER_HPP

#include "hgtxr_cyclic_transformer_params.hpp"

namespace hgtxr {
namespace cyclic_transformer {

typedef ap_uint<16> HgtxrSchedIndexT;
typedef ap_uint<5> HgtxrSchedStateT;

enum HgtxrSchedulerState {
  HGTXR_SCHED_IDLE = 0,
  HGTXR_SCHED_LOAD_X_TILE = 1,
  HGTXR_SCHED_LN1 = 2,
  HGTXR_SCHED_QKV_COMPUTE = 3,
  HGTXR_SCHED_ATTN_QK_TILE = 4,
  HGTXR_SCHED_SOFTMAX_ACCUM = 5,
  HGTXR_SCHED_ATTN_AV_TILE = 6,
  HGTXR_SCHED_WO_PROJECT = 7,
  HGTXR_SCHED_RESID_ADD1 = 8,
  HGTXR_SCHED_LN2 = 9,
  HGTXR_SCHED_MLP1 = 10,
  HGTXR_SCHED_ACT_GELU = 11,
  HGTXR_SCHED_MLP2 = 12,
  HGTXR_SCHED_RESID_ADD2 = 13,
  HGTXR_SCHED_STORE = 14,
  HGTXR_SCHED_DONE = 15
};

struct HgtxrSchedulerLimits {
  HgtxrSchedIndexT tiles_per_layer;
  HgtxrSchedIndexT kv_tiles_per_attention;
  HgtxrSchedIndexT heads_per_layer;
  HgtxrSchedIndexT num_layers;
};

struct HgtxrSchedulerCounters {
  HgtxrSchedIndexT tile_idx;
  HgtxrSchedIndexT kv_tile_idx;
  HgtxrSchedIndexT head_idx;
  HgtxrSchedIndexT layer_idx;
};

static inline bool hgtxr_is_last(HgtxrSchedIndexT idx, HgtxrSchedIndexT limit) {
#pragma HLS INLINE
  return (limit != 0) && (idx == (limit - 1));
}

static inline HgtxrSchedulerState hgtxr_scheduler_next_state(
    HgtxrSchedulerState state,
    const HgtxrSchedulerCounters &ctr,
    const HgtxrSchedulerLimits &lim) {
#pragma HLS INLINE
  switch (state) {
    case HGTXR_SCHED_IDLE:
      return HGTXR_SCHED_LOAD_X_TILE;
    case HGTXR_SCHED_LOAD_X_TILE:
      return HGTXR_SCHED_LN1;
    case HGTXR_SCHED_LN1:
      return HGTXR_SCHED_QKV_COMPUTE;
    case HGTXR_SCHED_QKV_COMPUTE:
      return HGTXR_SCHED_ATTN_QK_TILE;
    case HGTXR_SCHED_ATTN_QK_TILE:
      return hgtxr_is_last(ctr.kv_tile_idx, lim.kv_tiles_per_attention)
                 ? HGTXR_SCHED_SOFTMAX_ACCUM
                 : HGTXR_SCHED_ATTN_QK_TILE;
    case HGTXR_SCHED_SOFTMAX_ACCUM:
      return HGTXR_SCHED_ATTN_AV_TILE;
    case HGTXR_SCHED_ATTN_AV_TILE:
      return hgtxr_is_last(ctr.kv_tile_idx, lim.kv_tiles_per_attention)
                 ? HGTXR_SCHED_WO_PROJECT
                 : HGTXR_SCHED_ATTN_QK_TILE;
    case HGTXR_SCHED_WO_PROJECT:
      return HGTXR_SCHED_RESID_ADD1;
    case HGTXR_SCHED_RESID_ADD1:
      return HGTXR_SCHED_LN2;
    case HGTXR_SCHED_LN2:
      return HGTXR_SCHED_MLP1;
    case HGTXR_SCHED_MLP1:
      return HGTXR_SCHED_ACT_GELU;
    case HGTXR_SCHED_ACT_GELU:
      return HGTXR_SCHED_MLP2;
    case HGTXR_SCHED_MLP2:
      return HGTXR_SCHED_RESID_ADD2;
    case HGTXR_SCHED_RESID_ADD2:
      return HGTXR_SCHED_STORE;
    case HGTXR_SCHED_STORE:
      if (hgtxr_is_last(ctr.layer_idx, lim.num_layers) &&
          hgtxr_is_last(ctr.tile_idx, lim.tiles_per_layer) &&
          hgtxr_is_last(ctr.head_idx, lim.heads_per_layer)) {
        return HGTXR_SCHED_DONE;
      }
      return HGTXR_SCHED_LOAD_X_TILE;
    case HGTXR_SCHED_DONE:
      return HGTXR_SCHED_DONE;
    default:
      return HGTXR_SCHED_IDLE;
  }
}

static inline void hgtxr_scheduler_advance_counters(
    HgtxrSchedulerState state,
    const HgtxrSchedulerLimits &lim,
    HgtxrSchedulerCounters &ctr) {
#pragma HLS INLINE
  if (state == HGTXR_SCHED_ATTN_QK_TILE || state == HGTXR_SCHED_ATTN_AV_TILE) {
    if (!hgtxr_is_last(ctr.kv_tile_idx, lim.kv_tiles_per_attention)) {
      ctr.kv_tile_idx++;
      return;
    }
    ctr.kv_tile_idx = 0;
  }

  if (state != HGTXR_SCHED_STORE) {
    return;
  }

  if (!hgtxr_is_last(ctr.head_idx, lim.heads_per_layer)) {
    ctr.head_idx++;
    return;
  }
  ctr.head_idx = 0;

  if (!hgtxr_is_last(ctr.tile_idx, lim.tiles_per_layer)) {
    ctr.tile_idx++;
    return;
  }
  ctr.tile_idx = 0;

  if (!hgtxr_is_last(ctr.layer_idx, lim.num_layers)) {
    ctr.layer_idx++;
  }
}

}  // namespace cyclic_transformer
}  // namespace hgtxr

#endif  // HGTXR_CYCLIC_SCHEDULER_HPP

