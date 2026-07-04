#pragma once

#include "config.h"
#include "cyclic_config.h"
#include "register_map.h"
#include "fixed_types.h"
#include "quant.h"
#include "stream_types.h"
#include "tensor_buffer.h"

#ifndef HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS
#define HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS 0
#endif

#if HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS
#include "hgtxr_cyclic_transformer_params.hpp"
#endif

void frame_patch_embed(const hgtxr_data_t frame[HGTXR_HEIGHT][HGTXR_WIDTH], token_buffer_t tokens);
void event_patch_embed(const hgtxr_data_t event_pos[HGTXR_HEIGHT][HGTXR_WIDTH], const hgtxr_data_t event_neg[HGTXR_HEIGHT][HGTXR_WIDTH], token_buffer_t tokens);
void pool_tokens(const token_buffer_t tokens, pooled_buffer_t pooled);
void matmul(const hgtxr_data_t *a, const hgtxr_data_t *b, hgtxr_data_t *c, int m, int n, int k);
void attention_stage(token_buffer_t tokens);
void mlp_stage(token_buffer_t tokens);
void fusion(const pooled_buffer_t frame, const pooled_buffer_t event, const hgtxr_data_t prev_state[HGTXR_STATE], hgtxr_data_t fused[HGTXR_EMBED * 2]);
void search_head(const pooled_buffer_t pooled, hgtxr_data_t out[HGTXR_SEARCH_LOGITS]);
void track_head(const hgtxr_data_t fused[HGTXR_EMBED * 2], const hgtxr_data_t prev_state[HGTXR_STATE], hgtxr_data_t out[HGTXR_STATE]);
HGTXRDecision runtime_fsm(hgtxr_data_t search_conf, hgtxr_data_t track_conf, hgtxr_data_t track_quality, hgtxr_data_t similarity, hgtxr_data_t event_density, bool closed_eye);
#if HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS
void hgtxr_top(const hgtxr_data_t frame[HGTXR_HEIGHT][HGTXR_WIDTH], const hgtxr_data_t event_pos[HGTXR_HEIGHT][HGTXR_WIDTH], const hgtxr_data_t event_neg[HGTXR_HEIGHT][HGTXR_WIDTH], const hgtxr_data_t prev_state[HGTXR_STATE], const hgtxr::cyclic_transformer::HgtxrAxiWordT *cyclic_weights, hgtxr_data_t out_state[HGTXR_STATE], int *runtime_state);
#else
void hgtxr_top(const hgtxr_data_t frame[HGTXR_HEIGHT][HGTXR_WIDTH], const hgtxr_data_t event_pos[HGTXR_HEIGHT][HGTXR_WIDTH], const hgtxr_data_t event_neg[HGTXR_HEIGHT][HGTXR_WIDTH], const hgtxr_data_t prev_state[HGTXR_STATE], hgtxr_data_t out_state[HGTXR_STATE], int *runtime_state);
#endif

void global_buffer_write_search(const token_buffer_t in, token_buffer_t out);
void global_buffer_write_track(const token_buffer_t in, token_buffer_t out);
void noc_route_tokens(const token_buffer_t in, token_buffer_t out, int mode);
void weight_prefetcher(int mode, int block_idx, int *prefetch_valid);
int controller_active_tokens(int mode);
int controller_depth_limit(int mode);
void layernorm_stage(token_buffer_t tokens);
void softmax_stage(token_buffer_t tokens);
void gelu_stage(token_buffer_t tokens);
void rmu_projection_stage(token_buffer_t tokens);
void smu_relation_stage(token_buffer_t tokens);
