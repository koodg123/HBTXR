#ifndef HGTXR_E2E_VIT_HPP
#define HGTXR_E2E_VIT_HPP

#include "common.h"
#include "hgtxr_cyclic_math.hpp"
#include "hgtxr_cyclic_transformer_params.hpp"

#include <ap_axi_sdata.h>
#include <hls_stream.h>
#ifndef __SYNTHESIS__
#include <cassert>
#include <cstdio>
#endif

#ifndef HGTXR_E2E_BLOCKS
#define HGTXR_E2E_BLOCKS 6
#endif
#ifndef HGTXR_E2E_ACTIVE_TOKENS
#define HGTXR_E2E_ACTIVE_TOKENS HGTXR_TOKENS
#endif
#ifndef HGTXR_E2E_PATCH_GRID_H
#define HGTXR_E2E_PATCH_GRID_H HGTXR_GRID_H
#endif
#ifndef HGTXR_E2E_PATCH_GRID_W
#define HGTXR_E2E_PATCH_GRID_W HGTXR_GRID_W
#endif
#ifndef HGTXR_E2E_HEADS
#define HGTXR_E2E_HEADS 3
#endif
#ifndef HGTXR_E2E_HEAD_DIM
#define HGTXR_E2E_HEAD_DIM (HGTXR_EMBED / HGTXR_E2E_HEADS)
#endif
#ifndef HGTXR_E2E_FF_DIM
#define HGTXR_E2E_FF_DIM (HGTXR_EMBED * HGTXR_CYCLIC_MLP_RATIO)
#endif
#ifndef HGTXR_E2E_DENSE_PAR
#define HGTXR_E2E_DENSE_PAR HGTXR_PARALLELISM_FACTOR
#endif
#ifndef HGTXR_E2E_HEAD_PAR
#define HGTXR_E2E_HEAD_PAR HGTXR_E2E_DENSE_PAR
#endif
#ifndef HGTXR_E2E_PATCH_PAR
#define HGTXR_E2E_PATCH_PAR 1
#endif
#ifndef HGTXR_E2E_PATCH_TOKEN_LOOP
#define HGTXR_E2E_PATCH_TOKEN_LOOP 0
#endif
#ifndef HGTXR_E2E_MEM_BANK_PAR
#define HGTXR_E2E_MEM_BANK_PAR HGTXR_E2E_DENSE_PAR
#endif
#ifndef HGTXR_E2E_TOKEN_BANK_PAR
#define HGTXR_E2E_TOKEN_BANK_PAR 1
#endif
#ifndef HGTXR_E2E_ACC_SCALE
#define HGTXR_E2E_ACC_SCALE 16
#endif
#ifndef HGTXR_E2E_USE_HGPIPE_INT_GELUQ
#define HGTXR_E2E_USE_HGPIPE_INT_GELUQ 0
#endif
#ifndef HGTXR_E2E_HGPIPE_GELUQ_INPUT_SCALE
#define HGTXR_E2E_HGPIPE_GELUQ_INPUT_SCALE 16
#endif
#ifndef HGTXR_E2E_HGPIPE_GELUQ_OUTPUT_SCALE
#define HGTXR_E2E_HGPIPE_GELUQ_OUTPUT_SCALE 4
#endif
#ifndef HGTXR_E2E_USE_HGPIPE_INT_SOFTMAXQ
#define HGTXR_E2E_USE_HGPIPE_INT_SOFTMAXQ 0
#endif
#ifndef HGTXR_E2E_HGPIPE_SOFTMAX_INPUT_SCALE
#define HGTXR_E2E_HGPIPE_SOFTMAX_INPUT_SCALE 16
#endif
#ifndef HGTXR_E2E_HGPIPE_SOFTMAX_PROB_SCALE
#define HGTXR_E2E_HGPIPE_SOFTMAX_PROB_SCALE 4
#endif
#ifndef HGTXR_E2E_USE_HGPIPE_INT_LAYERNORMQ
#define HGTXR_E2E_USE_HGPIPE_INT_LAYERNORMQ 0
#endif
#ifndef HGTXR_E2E_HGPIPE_LAYERNORM_INPUT_SCALE
#define HGTXR_E2E_HGPIPE_LAYERNORM_INPUT_SCALE 16
#endif
#ifndef HGTXR_E2E_HGPIPE_LAYERNORM_OUTPUT_SCALE
#define HGTXR_E2E_HGPIPE_LAYERNORM_OUTPUT_SCALE 4
#endif
#ifndef HGTXR_E2E_HGPIPE_LAYERNORM_BIAS_SHIFT
#define HGTXR_E2E_HGPIPE_LAYERNORM_BIAS_SHIFT 33
#endif
#ifndef HGTXR_E2E_FORCE_DSP_MUL
#define HGTXR_E2E_FORCE_DSP_MUL 1
#endif
#ifndef HGTXR_E2E_FORCE_WIDE_DSP_MUL
#define HGTXR_E2E_FORCE_WIDE_DSP_MUL 0
#endif
#ifndef HGTXR_E2E_DSP_MUL_LATENCY
#define HGTXR_E2E_DSP_MUL_LATENCY 0
#endif
#ifndef HGTXR_E2E_FABRIC_MUL_LATENCY
#define HGTXR_E2E_FABRIC_MUL_LATENCY 0
#endif
#ifndef HGTXR_E2E_STAGE_PREFETCH_LOAD
#define HGTXR_E2E_STAGE_PREFETCH_LOAD 0
#endif
#ifndef HGTXR_E2E_HEAD_FABRIC_MUL
#define HGTXR_E2E_HEAD_FABRIC_MUL 0
#endif
#ifndef HGTXR_E2E_HEAD_RAW_SHIFTADD_MUL
#define HGTXR_E2E_HEAD_RAW_SHIFTADD_MUL 0
#endif
#ifndef HGTXR_E2E_PATCH_FABRIC_MUL
#define HGTXR_E2E_PATCH_FABRIC_MUL 0
#endif
#ifndef HGTXR_E2E_LAYERNORM_FABRIC_MUL
#define HGTXR_E2E_LAYERNORM_FABRIC_MUL 0
#endif
#ifndef HGTXR_E2E_FASTPATH_FABRIC_MUL
#define HGTXR_E2E_FASTPATH_FABRIC_MUL 0
#endif
#ifndef HGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL
#define HGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL 0
#endif
#ifndef HGTXR_E2E_CORE_ALL_FABRIC_MUL
#define HGTXR_E2E_CORE_ALL_FABRIC_MUL 0
#endif
#ifndef HGTXR_E2E_CORE_LANE_CT_SWITCH
#define HGTXR_E2E_CORE_LANE_CT_SWITCH 0
#endif
#ifndef HGTXR_E2E_CORE_FABRIC_TAIL_LANES
#define HGTXR_E2E_CORE_FABRIC_TAIL_LANES 0
#endif
#ifndef HGTXR_E2E_SHARE_RUNTIME_UNITS
#define HGTXR_E2E_SHARE_RUNTIME_UNITS 0
#endif
#ifndef HGTXR_E2E_PREFETCH_ADDR_SWITCH
#define HGTXR_E2E_PREFETCH_ADDR_SWITCH 0
#endif
#ifndef HGTXR_E2E_FORCE_URAM_BUFFERS
#define HGTXR_E2E_FORCE_URAM_BUFFERS 1
#endif
#ifndef HGTXR_E2E_URAM_FRAME_TOKENS
#define HGTXR_E2E_URAM_FRAME_TOKENS HGTXR_E2E_FORCE_URAM_BUFFERS
#endif
#ifndef HGTXR_E2E_URAM_GB_TOKENS
#define HGTXR_E2E_URAM_GB_TOKENS HGTXR_E2E_FORCE_URAM_BUFFERS
#endif
#ifndef HGTXR_E2E_URAM_NORM
#define HGTXR_E2E_URAM_NORM HGTXR_E2E_FORCE_URAM_BUFFERS
#endif
#ifndef HGTXR_E2E_URAM_Q
#define HGTXR_E2E_URAM_Q HGTXR_E2E_FORCE_URAM_BUFFERS
#endif
#ifndef HGTXR_E2E_URAM_K
#define HGTXR_E2E_URAM_K HGTXR_E2E_FORCE_URAM_BUFFERS
#endif
#ifndef HGTXR_E2E_URAM_V
#define HGTXR_E2E_URAM_V HGTXR_E2E_FORCE_URAM_BUFFERS
#endif
#ifndef HGTXR_E2E_URAM_ATTN
#define HGTXR_E2E_URAM_ATTN HGTXR_E2E_FORCE_URAM_BUFFERS
#endif
#ifndef HGTXR_E2E_URAM_HIDDEN
#define HGTXR_E2E_URAM_HIDDEN HGTXR_E2E_FORCE_URAM_BUFFERS
#endif
#ifndef HGTXR_E2E_WEIGHT_VEC_CACHE
#define HGTXR_E2E_WEIGHT_VEC_CACHE 1
#endif
#ifndef HGTXR_E2E_WEIGHT_VEC_ALIGNED_FASTPATH
#define HGTXR_E2E_WEIGHT_VEC_ALIGNED_FASTPATH 1
#endif
#ifndef HGTXR_E2E_MLP_TOKEN_PAR
#define HGTXR_E2E_MLP_TOKEN_PAR 1
#endif
#ifndef HGTXR_E2E_DENSE_TOKEN_PAR
#define HGTXR_E2E_DENSE_TOKEN_PAR 1
#endif
#ifndef HGTXR_E2E_ATTN_QUERY_PAR
#define HGTXR_E2E_ATTN_QUERY_PAR 1
#endif
#ifndef HGTXR_E2E_MLP_FUSED_W2
#define HGTXR_E2E_MLP_FUSED_W2 0
#endif
#ifndef HGTXR_E2E_MLP_FUSED_W2_BANKED_ACC
#define HGTXR_E2E_MLP_FUSED_W2_BANKED_ACC 0
#endif
#ifndef HGTXR_E2E_MLP_FUSED_W2_SKIP_HIDDEN_STORE
#define HGTXR_E2E_MLP_FUSED_W2_SKIP_HIDDEN_STORE 0
#endif
#ifndef HGTXR_E2E_MLP_FUSED_W2_HP_PAR
#define HGTXR_E2E_MLP_FUSED_W2_HP_PAR 1
#endif
#ifndef HGTXR_E2E_MLP_W2_FABRIC_HP_LANES
#define HGTXR_E2E_MLP_W2_FABRIC_HP_LANES 0
#endif
#ifndef HGTXR_E2E_MLP_W1_C_PAR
#define HGTXR_E2E_MLP_W1_C_PAR 1
#endif
#ifndef HGTXR_E2E_W1_WEIGHT_CACHE_BANKS
#define HGTXR_E2E_W1_WEIGHT_CACHE_BANKS 1
#endif
#ifndef HGTXR_E2E_MLP_W2_HIDDEN_BANK_CACHE
#define HGTXR_E2E_MLP_W2_HIDDEN_BANK_CACHE 0
#endif
#ifndef HGTXR_E2E_MLP_W2_HGROUP_BANK_CACHE
#define HGTXR_E2E_MLP_W2_HGROUP_BANK_CACHE 0
#endif
#ifndef HGTXR_E2E_W2_WEIGHT_CACHE_BANKS
#define HGTXR_E2E_W2_WEIGHT_CACHE_BANKS 1
#endif
#ifndef HGTXR_E2E_SMALL_MEM_LUTRAM
#define HGTXR_E2E_SMALL_MEM_LUTRAM 1
#endif
#ifndef HGTXR_E2E_LUTRAM_FRAME_TOKENS
#define HGTXR_E2E_LUTRAM_FRAME_TOKENS 0
#endif
#ifndef HGTXR_E2E_LUTRAM_GB_TOKENS
#define HGTXR_E2E_LUTRAM_GB_TOKENS 0
#endif
#ifndef HGTXR_E2E_LUTRAM_NORM
#define HGTXR_E2E_LUTRAM_NORM 0
#endif
#ifndef HGTXR_E2E_LUTRAM_Q
#define HGTXR_E2E_LUTRAM_Q 0
#endif
#ifndef HGTXR_E2E_LUTRAM_K
#define HGTXR_E2E_LUTRAM_K 0
#endif
#ifndef HGTXR_E2E_LUTRAM_V
#define HGTXR_E2E_LUTRAM_V 0
#endif
#ifndef HGTXR_E2E_LUTRAM_ATTN
#define HGTXR_E2E_LUTRAM_ATTN 0
#endif
#ifndef HGTXR_E2E_LUTRAM_HIDDEN
#define HGTXR_E2E_LUTRAM_HIDDEN 0
#endif
#ifndef HGTXR_E2E_NONLINEAR_ROM_LUTRAM
#define HGTXR_E2E_NONLINEAR_ROM_LUTRAM 0
#endif
#ifndef HGTXR_E2E_GELU_ROM_PARTITION
#define HGTXR_E2E_GELU_ROM_PARTITION 0
#endif
#ifndef HGTXR_E2E_SOFTMAX_EXP_ROM_PARTITION
#define HGTXR_E2E_SOFTMAX_EXP_ROM_PARTITION 0
#endif
#ifndef HGTXR_E2E_URAM_QKV_WEIGHT_CACHE
#define HGTXR_E2E_URAM_QKV_WEIGHT_CACHE 0
#endif
#ifndef HGTXR_E2E_LN_PARAM_CACHE
#define HGTXR_E2E_LN_PARAM_CACHE 1
#endif
#ifndef HGTXR_E2E_QKV_WEIGHT_CACHE
#define HGTXR_E2E_QKV_WEIGHT_CACHE 1
#endif
#ifndef HGTXR_E2E_REPORT_MODE_STATE
#define HGTXR_E2E_REPORT_MODE_STATE 0
#endif
#ifndef HGTXR_E2E_REQUIRE_MODE_TARGET
#define HGTXR_E2E_REQUIRE_MODE_TARGET 0
#endif
#ifndef HGTXR_E2E_MODE_DEPTH_COUNTS_LAYERS
#define HGTXR_E2E_MODE_DEPTH_COUNTS_LAYERS 0
#endif
#ifndef HGTXR_E2E_USE_MODE_PROFILE_FASTPATH
#define HGTXR_E2E_USE_MODE_PROFILE_FASTPATH 0
#endif
#ifndef HGTXR_E2E_OMIT_WEIGHT_AXI
#define HGTXR_E2E_OMIT_WEIGHT_AXI 0
#endif
#ifndef HGTXR_E2E_USE_ONCHIP_PARAM_ROM
#define HGTXR_E2E_USE_ONCHIP_PARAM_ROM 0
#endif
#ifndef HGTXR_E2E_USE_ONCHIP_NONLINEAR_ROM
#define HGTXR_E2E_USE_ONCHIP_NONLINEAR_ROM 0
#endif
#ifndef HGTXR_E2E_USE_SEARCH_WEIGHT_DISPATCHER
#define HGTXR_E2E_USE_SEARCH_WEIGHT_DISPATCHER 0
#endif
#ifndef HGTXR_E2E_OBSERVE_DATAPATH
#define HGTXR_E2E_OBSERVE_DATAPATH 0
#endif
#ifndef HGTXR_E2E_LOW_FANOUT_WEIGHT_AXI
#define HGTXR_E2E_LOW_FANOUT_WEIGHT_AXI 0
#endif
#ifndef HGTXR_E2E_DISPATCH_PREFETCH_WORDS
#define HGTXR_E2E_DISPATCH_PREFETCH_WORDS 16
#endif
#ifndef HGTXR_E2E_DISPATCH_PREFETCH_FULL_BLOCK
#define HGTXR_E2E_DISPATCH_PREFETCH_FULL_BLOCK 0
#endif
#ifndef HGTXR_E2E_DISPATCH_PREFETCH_BANKS
#define HGTXR_E2E_DISPATCH_PREFETCH_BANKS 2
#endif
#ifndef HGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE
#define HGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE 0
#endif
#ifndef HGTXR_E2E_URAM_DISPATCH_PREFETCH
#define HGTXR_E2E_URAM_DISPATCH_PREFETCH 0
#endif
#ifndef HGTXR_E2E_FORCE_RUNTIME_MODE
#define HGTXR_E2E_FORCE_RUNTIME_MODE -1
#endif
#ifndef HGTXR_E2E_CSIM_PREFETCH_TRACE
#define HGTXR_E2E_CSIM_PREFETCH_TRACE 0
#endif
#ifndef HGTXR_E2E_CSIM_STRICT_IMMEDIATE_START
#define HGTXR_E2E_CSIM_STRICT_IMMEDIATE_START HGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE
#endif
#ifndef HGTXR_E2E_STAGE_LAYERNORM_WRITE
#define HGTXR_E2E_STAGE_LAYERNORM_WRITE 0
#endif
#ifndef HGTXR_E2E_STRUCTURED_PARAM_ROM
#define HGTXR_E2E_STRUCTURED_PARAM_ROM 0
#endif
#ifndef HGTXR_E2E_STRUCTURED_FAST_MATH
#define HGTXR_E2E_STRUCTURED_FAST_MATH HGTXR_E2E_STRUCTURED_PARAM_ROM
#endif
#ifndef HGTXR_E2E_AVOID_RUNTIME_DIVIDERS
#define HGTXR_E2E_AVOID_RUNTIME_DIVIDERS 0
#endif

namespace hgtxr {
namespace e2e {

using hgtxr::cyclic_transformer::HgtxrAccumT;
using hgtxr::cyclic_transformer::HgtxrAxiWordT;
using hgtxr::cyclic_transformer::HgtxrDataT;
using hgtxr::cyclic_transformer::HgtxrScoreT;
using hgtxr::cyclic_transformer::HgtxrWeightIntT;
using hgtxr::cyclic_transformer::HgtxrWeightT;
using HgtxrAxisWord = ap_axiu<HGTXR_BUS_WIDTH, 0, 0, 0>;
#if HGTXR_E2E_FORCE_WIDE_DSP_MUL
using HgtxrDspOperandT = ap_fixed<18, 6, AP_TRN, AP_SAT>;
#endif

static constexpr int kActiveTokens = HGTXR_E2E_ACTIVE_TOKENS;
static constexpr int kEmbed = HGTXR_EMBED;
static constexpr int kPatchElems = HGTXR_PATCH * HGTXR_PATCH;
static constexpr int kHeads = HGTXR_E2E_HEADS;
static constexpr int kHeadDim = HGTXR_E2E_HEAD_DIM;
static constexpr int kFfDim = HGTXR_E2E_FF_DIM;
static constexpr int kWeightLanes = hgtxr::cyclic_transformer::HGTXR_AXI_WEIGHT_LANES;
static constexpr int kDensePar = HGTXR_E2E_DENSE_PAR;
static constexpr int kHeadPar = HGTXR_E2E_HEAD_PAR;
static constexpr int kPatchPar = HGTXR_E2E_PATCH_PAR;
static constexpr int kMlpTokenPar = HGTXR_E2E_MLP_TOKEN_PAR;
static constexpr int kDenseTokenPar = HGTXR_E2E_DENSE_TOKEN_PAR;
static constexpr int kAttnQueryPar = HGTXR_E2E_ATTN_QUERY_PAR;
static constexpr int kMlpFusedW2HpPar = HGTXR_E2E_MLP_FUSED_W2_HP_PAR;
static constexpr int kMlpW2FabricHpLanes = HGTXR_E2E_MLP_W2_FABRIC_HP_LANES;
static constexpr int kMlpW1CPar = HGTXR_E2E_MLP_W1_C_PAR;
static constexpr int kW1WeightCacheBanks = HGTXR_E2E_W1_WEIGHT_CACHE_BANKS;
static constexpr int kW2WeightCacheBanks = HGTXR_E2E_W2_WEIGHT_CACHE_BANKS;

static_assert(kActiveTokens > 0 && kActiveTokens <= HGTXR_TOKENS,
              "E2E active token count must fit token buffers");
static_assert(kHeads > 0, "E2E head count must be positive");
static_assert(kHeadDim * kHeads == kEmbed,
              "E2E head dimension must exactly cover embedding dimension");
static_assert(kDensePar > 0, "E2E dense parallelism must be positive");
static_assert(kHeadPar > 0, "E2E head parallelism must be positive");
static_assert(kPatchPar > 0, "E2E patch parallelism must be positive");
static_assert(kMlpTokenPar > 0, "E2E MLP token parallelism must be positive");
static_assert(kDenseTokenPar > 0, "E2E dense token parallelism must be positive");
static_assert(kAttnQueryPar > 0, "E2E attention query parallelism must be positive");
static_assert(kMlpFusedW2HpPar > 0,
              "E2E fused W2 hidden-lane parallelism must be positive");
static_assert(kMlpW1CPar > 0,
              "E2E MLP W1 input-channel parallelism must be positive");
static_assert(kW2WeightCacheBanks > 0,
              "E2E W2 weight cache bank count must be positive");
static_assert(kEmbed % kDensePar == 0,
              "E2E dense parallelism must divide embedding dimension");
static_assert(kEmbed % kHeadPar == 0,
              "E2E head parallelism must divide embedding dimension");
static_assert(kPatchElems % kPatchPar == 0,
              "E2E patch parallelism must divide patch element count");
static_assert(kFfDim > 0 && kFfDim % kDensePar == 0,
              "E2E dense parallelism must divide feed-forward dimension");
static_assert(kDensePar % kMlpFusedW2HpPar == 0,
              "E2E fused W2 hidden-lane parallelism must divide dense parallelism");
static_assert(kWeightLanes > 0, "E2E packed weight lanes must be positive");
static_assert(kWeightLanes % kDensePar == 0,
              "E2E dense parallelism must divide packed weight lanes");
static_assert(HGTXR_E2E_DISPATCH_PREFETCH_BANKS > 0,
              "E2E dispatcher prefetch bank count must be positive");
static_assert(HGTXR_E2E_TOKEN_BANK_PAR > 0,
              "E2E token bank partition factor must be positive");
static_assert(HGTXR_TOKENS % HGTXR_E2E_TOKEN_BANK_PAR == 0,
              "E2E token bank partition factor must divide token count");
static_assert(HGTXR_BUS_WIDTH == hgtxr::cyclic_transformer::HGTXR_AXI_DATA_WIDTH,
              "E2E AXIS width must match cyclic AXI data width");
#if HGTXR_E2E_REQUIRE_MODE_TARGET
static_assert(HGTXR_E2E_ACTIVE_TOKENS >= HGTXR_SEARCH_TOKENS,
              "Runtime Search mode requires HGTXR_E2E_ACTIVE_TOKENS >= HGTXR_SEARCH_TOKENS");
static_assert(HGTXR_E2E_BLOCKS >= HGTXR_SEARCH_DEPTH,
              "Runtime Search mode requires HGTXR_E2E_BLOCKS >= HGTXR_SEARCH_DEPTH");
#endif

static constexpr int kPatchWeightElemBase = 0;
static constexpr int kPatchWeightElems = kEmbed * kPatchElems;
static constexpr int kEventPatchWeightElemBase =
    kPatchWeightElemBase + kPatchWeightElems;
static constexpr int kEventPatchWeightElems = kEmbed * kPatchElems;
static constexpr int kBlockWeightElemBase =
    kEventPatchWeightElemBase + kEventPatchWeightElems;
static constexpr int kLnParamElems = kEmbed;
static constexpr int kDenseModelElems = kEmbed * kEmbed;
static constexpr int kMlpW1Elems = kEmbed * kFfDim;
static constexpr int kMlpW2Elems = kFfDim * kEmbed;
static constexpr int kBlockLn1Gamma = 0;
static constexpr int kBlockLn1Beta = kBlockLn1Gamma + kLnParamElems;
static constexpr int kBlockLn2Gamma = kBlockLn1Beta + kLnParamElems;
static constexpr int kBlockLn2Beta = kBlockLn2Gamma + kLnParamElems;
static constexpr int kBlockWq = kBlockLn2Beta + kLnParamElems;
static constexpr int kBlockWk = kBlockWq + kDenseModelElems;
static constexpr int kBlockWv = kBlockWk + kDenseModelElems;
static constexpr int kBlockWo = kBlockWv + kDenseModelElems;
static constexpr int kBlockW1 = kBlockWo + kDenseModelElems;
static constexpr int kBlockW2 = kBlockW1 + kMlpW1Elems;
static constexpr int kBlockWeightElems = kBlockW2 + kMlpW2Elems;
static constexpr int kHeadWeightElemBase =
    kBlockWeightElemBase + HGTXR_E2E_BLOCKS * kBlockWeightElems;
static constexpr int kHeadWeightElems = HGTXR_STATE * kEmbed;
static constexpr int kRequiredWeightElems = kHeadWeightElemBase + kHeadWeightElems;
static constexpr int kRequiredWeightWords =
    (kRequiredWeightElems + kWeightLanes - 1) / kWeightLanes;
static_assert(kRequiredWeightWords <= HGTXR_E2E_WEIGHT_DEPTH,
              "E2E weight depth must cover frame/event conv, Transformer blocks, and head");
static constexpr int kLnParamWords = (kLnParamElems + kWeightLanes - 1) / kWeightLanes;
static constexpr int kDenseModelWords =
    (kDenseModelElems + kWeightLanes - 1) / kWeightLanes;
static constexpr int kMlpW1Words =
    (kMlpW1Elems + kWeightLanes - 1) / kWeightLanes;
static constexpr int kMlpW2Words =
    (kMlpW2Elems + kWeightLanes - 1) / kWeightLanes;
static constexpr int kMlpW2ColWords =
    (kEmbed + kWeightLanes - 1) / kWeightLanes;
static constexpr int kMlpW2HiddenRows =
    (kFfDim + kDensePar - 1) / kDensePar;
static constexpr int kMlpW2HiddenBankWords =
    kMlpW2HiddenRows * kMlpW2ColWords;
static_assert(kMlpW1Words % kW1WeightCacheBanks == 0,
              "E2E W1 weight cache words must divide cache bank count");
static_assert(kMlpW2Words % kW2WeightCacheBanks == 0,
              "E2E W2 weight cache words must divide cache bank count");
#if HGTXR_E2E_MLP_W2_HIDDEN_BANK_CACHE
static_assert(HGTXR_E2E_MLP_FUSED_W2,
              "E2E hidden-bank W2 cache requires fused W2 MLP");
static_assert(kEmbed % kWeightLanes == 0,
              "E2E hidden-bank W2 cache expects row-aligned W2 words");
#endif
#if HGTXR_E2E_MLP_W2_HGROUP_BANK_CACHE
static_assert(HGTXR_E2E_MLP_FUSED_W2,
              "E2E hgroup-bank W2 cache requires fused W2 MLP");
static_assert(kEmbed % kWeightLanes == 0,
              "E2E hgroup-bank W2 cache expects row-aligned W2 words");
#endif
static_assert(!(HGTXR_E2E_MLP_W2_HIDDEN_BANK_CACHE &&
                HGTXR_E2E_MLP_W2_HGROUP_BANK_CACHE),
              "E2E W2 hidden-bank and hgroup-bank caches are mutually exclusive");
static constexpr int kBlockWeightWords =
    (kBlockWeightElems + kWeightLanes - 1) / kWeightLanes;
static constexpr int kDispatchPrefetchWords =
    HGTXR_E2E_DISPATCH_PREFETCH_FULL_BLOCK ? kBlockWeightWords
                                           : HGTXR_E2E_DISPATCH_PREFETCH_WORDS;
static constexpr int kTrackRomBlockPairsRequested =
    HGTXR_E2E_MODE_DEPTH_COUNTS_LAYERS ? ((HGTXR_TRACK_CUT_DEPTH + 1) / 2)
                                       : HGTXR_TRACK_CUT_DEPTH;
static constexpr int kTrackRomBlockPairs =
    kTrackRomBlockPairsRequested < HGTXR_E2E_BLOCKS ? kTrackRomBlockPairsRequested
                                                    : HGTXR_E2E_BLOCKS;

struct HgtxrGlobalBuffer {
  token_buffer_t tokens;
  pooled_buffer_t pooled;
  hgtxr_data_t norm[HGTXR_TOKENS][HGTXR_EMBED];
  hgtxr_data_t q[HGTXR_TOKENS][HGTXR_EMBED];
  hgtxr_data_t k[HGTXR_TOKENS][HGTXR_EMBED];
  hgtxr_data_t v[HGTXR_TOKENS][HGTXR_EMBED];
  hgtxr_data_t attn[HGTXR_TOKENS][HGTXR_EMBED];
  hgtxr_data_t hidden[HGTXR_TOKENS][HGTXR_E2E_FF_DIM];
  HgtxrAxiWordT dispatch_prefetch[HGTXR_E2E_DISPATCH_PREFETCH_BANKS][kDispatchPrefetchWords];
  int dispatch_prefetch_base_word[HGTXR_E2E_DISPATCH_PREFETCH_BANKS];
  int dispatch_prefetch_valid_words[HGTXR_E2E_DISPATCH_PREFETCH_BANKS];
  int dispatch_prefetch_block[HGTXR_E2E_DISPATCH_PREFETCH_BANKS];
  int dispatch_prefetch_ready[HGTXR_E2E_DISPATCH_PREFETCH_BANKS];
};

struct HgtxrRuntimeSchedule {
  int mode;
  int active_tokens;
  int depth_limit;
  int block_pairs;
  int search_probability_percent;
};

static inline int hgtxr_e2e_min_int(int a, int b) {
#pragma HLS INLINE
  return a < b ? a : b;
}

static inline int hgtxr_e2e_runtime_mode_from_control(int control) {
#pragma HLS INLINE
#if HGTXR_E2E_FORCE_RUNTIME_MODE == 0
  (void)control;
  return HGTXR_MODE_SEARCH;
#elif HGTXR_E2E_FORCE_RUNTIME_MODE == 1
  (void)control;
  return HGTXR_MODE_TRACK;
#else
  if (control == HGTXR_MODE_TRACK || control == HGTXR_TRACK_H * HGTXR_TRACK_W) {
    return HGTXR_MODE_TRACK;
  }
  return HGTXR_MODE_SEARCH;
#endif
}

static inline int hgtxr_e2e_mode_active_tokens(int mode) {
#pragma HLS INLINE
  const int requested =
      mode == HGTXR_MODE_TRACK ? HGTXR_TRACK_TOKENS : HGTXR_SEARCH_TOKENS;
  return hgtxr_e2e_min_int(requested, kActiveTokens);
}

static inline int hgtxr_e2e_mode_depth_limit(int mode) {
#pragma HLS INLINE
  const int requested =
      mode == HGTXR_MODE_TRACK ? HGTXR_TRACK_CUT_DEPTH : HGTXR_SEARCH_DEPTH;
  return hgtxr_e2e_min_int(requested, HGTXR_E2E_BLOCKS);
}

static inline HgtxrRuntimeSchedule hgtxr_e2e_make_runtime_schedule(int control) {
#pragma HLS INLINE
  HgtxrRuntimeSchedule schedule;
  schedule.mode = hgtxr_e2e_runtime_mode_from_control(control);
  schedule.active_tokens = hgtxr_e2e_mode_active_tokens(schedule.mode);
  schedule.depth_limit = hgtxr_e2e_mode_depth_limit(schedule.mode);
  schedule.block_pairs = schedule.depth_limit;
#if HGTXR_E2E_MODE_DEPTH_COUNTS_LAYERS
  schedule.block_pairs = (schedule.depth_limit + 1) / 2;
#endif
  schedule.search_probability_percent = 10;
  return schedule;
}

static inline hgtxr_data_t hgtxr_e2e_data_from_scaled_int(int raw, int scale) {
#pragma HLS INLINE
  hgtxr_acc_t value = static_cast<hgtxr_acc_t>(raw);
  value /= static_cast<hgtxr_acc_t>(scale);
  return static_cast<hgtxr_data_t>(value);
}

template <int SCALE>
static inline hgtxr_data_t hgtxr_e2e_data_from_scaled_int_ct(int raw) {
#pragma HLS INLINE
  hgtxr_acc_t value = static_cast<hgtxr_acc_t>(raw);
  value *= static_cast<hgtxr_acc_t>(1.0 / SCALE);
  return static_cast<hgtxr_data_t>(value);
}

static inline int hgtxr_e2e_data_to_scaled_int(hgtxr_data_t value, int scale) {
#pragma HLS INLINE
  hgtxr_acc_t scaled = static_cast<hgtxr_acc_t>(value);
  scaled *= static_cast<hgtxr_acc_t>(scale);
  return static_cast<int>(scaled);
}

template <int SCALE>
static inline int hgtxr_e2e_data_to_scaled_int_ct(hgtxr_data_t value) {
#pragma HLS INLINE
  hgtxr_acc_t scaled = static_cast<hgtxr_acc_t>(value);
  scaled *= static_cast<hgtxr_acc_t>(SCALE);
  return static_cast<int>(scaled);
}

static inline hgtxr_data_t hgtxr_e2e_data_div_int(hgtxr_data_t value, int scale) {
#pragma HLS INLINE
  hgtxr_acc_t scaled = static_cast<hgtxr_acc_t>(value);
  scaled /= static_cast<hgtxr_acc_t>(scale);
  return static_cast<hgtxr_data_t>(scaled);
}

template <int SCALE>
static inline hgtxr_data_t hgtxr_e2e_data_div_int_ct(hgtxr_data_t value) {
#pragma HLS INLINE
  hgtxr_acc_t scaled = static_cast<hgtxr_acc_t>(value);
  scaled *= static_cast<hgtxr_acc_t>(1.0 / SCALE);
  return static_cast<hgtxr_data_t>(scaled);
}

template <int SCALE>
static inline hgtxr_acc_t hgtxr_e2e_acc_div_int_ct(hgtxr_acc_t value) {
#pragma HLS INLINE
  value *= static_cast<hgtxr_acc_t>(1.0 / SCALE);
  return value;
}

static inline hgtxr_data_t hgtxr_e2e_data_from_token_count(int raw,
                                                          int token_count) {
#pragma HLS INLINE
#if HGTXR_E2E_AVOID_RUNTIME_DIVIDERS
  if (token_count == HGTXR_TRACK_TOKENS) {
    return hgtxr_e2e_data_from_scaled_int_ct<HGTXR_TRACK_TOKENS>(raw);
  }
  if (token_count == HGTXR_SEARCH_TOKENS) {
    return hgtxr_e2e_data_from_scaled_int_ct<HGTXR_SEARCH_TOKENS>(raw);
  }
  return hgtxr_e2e_data_from_scaled_int_ct<kActiveTokens>(raw);
#else
  return hgtxr_e2e_data_from_scaled_int(raw, token_count);
#endif
}

static inline hgtxr_data_t hgtxr_e2e_data_div_token_count(hgtxr_data_t value,
                                                         int token_count) {
#pragma HLS INLINE
#if HGTXR_E2E_AVOID_RUNTIME_DIVIDERS
  if (token_count == HGTXR_TRACK_TOKENS) {
    return hgtxr_e2e_data_div_int_ct<HGTXR_TRACK_TOKENS>(value);
  }
  if (token_count == HGTXR_SEARCH_TOKENS) {
    return hgtxr_e2e_data_div_int_ct<HGTXR_SEARCH_TOKENS>(value);
  }
  return hgtxr_e2e_data_div_int_ct<kActiveTokens>(value);
#else
  return hgtxr_e2e_data_div_int(value, token_count);
#endif
}

static inline hgtxr_acc_t hgtxr_e2e_acc_div_token_count(hgtxr_acc_t value,
                                                       int token_count) {
#pragma HLS INLINE
#if HGTXR_E2E_AVOID_RUNTIME_DIVIDERS
  if (token_count == HGTXR_TRACK_TOKENS) {
    return hgtxr_e2e_acc_div_int_ct<HGTXR_TRACK_TOKENS>(value);
  }
  if (token_count == HGTXR_SEARCH_TOKENS) {
    return hgtxr_e2e_acc_div_int_ct<HGTXR_SEARCH_TOKENS>(value);
  }
  return hgtxr_e2e_acc_div_int_ct<kActiveTokens>(value);
#else
  return value / static_cast<hgtxr_acc_t>(token_count);
#endif
}

static inline hgtxr_data_t hgtxr_axis_to_data(HgtxrAxisWord word) {
#pragma HLS INLINE
  ap_uint<8> raw = word.data.range(7, 0);
  return hgtxr_e2e_data_from_scaled_int_ct<128>(static_cast<int>(raw));
}

static inline HgtxrAxisWord hgtxr_data_to_axis(hgtxr_data_t value, bool last) {
#pragma HLS INLINE
  HgtxrAxisWord word;
  ap_uint<HGTXR_BUS_WIDTH> packed = 0;
#if defined(__SYNTHESIS__) || defined(HGTXR_HLS_FIXED_CSIM)
  ap_uint<HGTXR_BIT_WIDTH> raw_bits;
  raw_bits.range(HGTXR_BIT_WIDTH - 1, 0) =
      value.range(HGTXR_BIT_WIDTH - 1, 0);
#if HGTXR_BIT_WIDTH >= 16
  packed.range(15, 0) = raw_bits.range(15, 0);
#else
  packed.range(HGTXR_BIT_WIDTH - 1, 0) =
      raw_bits.range(HGTXR_BIT_WIDTH - 1, 0);
#endif
#else
  int scaled = static_cast<int>(value * static_cast<hgtxr_data_t>(16));
  packed.range(15, 0) = static_cast<ap_int<16> >(scaled).range(15, 0);
#endif
  word.data = packed;
  word.keep = -1;
  word.strb = -1;
  word.last = last ? 1 : 0;
  return word;
}

static inline HgtxrWeightIntT hgtxr_e2e_extract_weight_bits(HgtxrAxiWordT word,
                                                            int lane) {
#pragma HLS INLINE
  HgtxrWeightIntT bits = 0;
#if HGTXR_WEIGHT_BIT_WIDTH == 4 && HGTXR_BUS_WIDTH_BITS == 256
  switch (lane) {
    case 0: bits = word.range(3, 0); break;
    case 1: bits = word.range(7, 4); break;
    case 2: bits = word.range(11, 8); break;
    case 3: bits = word.range(15, 12); break;
    case 4: bits = word.range(19, 16); break;
    case 5: bits = word.range(23, 20); break;
    case 6: bits = word.range(27, 24); break;
    case 7: bits = word.range(31, 28); break;
    case 8: bits = word.range(35, 32); break;
    case 9: bits = word.range(39, 36); break;
    case 10: bits = word.range(43, 40); break;
    case 11: bits = word.range(47, 44); break;
    case 12: bits = word.range(51, 48); break;
    case 13: bits = word.range(55, 52); break;
    case 14: bits = word.range(59, 56); break;
    case 15: bits = word.range(63, 60); break;
    case 16: bits = word.range(67, 64); break;
    case 17: bits = word.range(71, 68); break;
    case 18: bits = word.range(75, 72); break;
    case 19: bits = word.range(79, 76); break;
    case 20: bits = word.range(83, 80); break;
    case 21: bits = word.range(87, 84); break;
    case 22: bits = word.range(91, 88); break;
    case 23: bits = word.range(95, 92); break;
    case 24: bits = word.range(99, 96); break;
    case 25: bits = word.range(103, 100); break;
    case 26: bits = word.range(107, 104); break;
    case 27: bits = word.range(111, 108); break;
    case 28: bits = word.range(115, 112); break;
    case 29: bits = word.range(119, 116); break;
    case 30: bits = word.range(123, 120); break;
    case 31: bits = word.range(127, 124); break;
    case 32: bits = word.range(131, 128); break;
    case 33: bits = word.range(135, 132); break;
    case 34: bits = word.range(139, 136); break;
    case 35: bits = word.range(143, 140); break;
    case 36: bits = word.range(147, 144); break;
    case 37: bits = word.range(151, 148); break;
    case 38: bits = word.range(155, 152); break;
    case 39: bits = word.range(159, 156); break;
    case 40: bits = word.range(163, 160); break;
    case 41: bits = word.range(167, 164); break;
    case 42: bits = word.range(171, 168); break;
    case 43: bits = word.range(175, 172); break;
    case 44: bits = word.range(179, 176); break;
    case 45: bits = word.range(183, 180); break;
    case 46: bits = word.range(187, 184); break;
    case 47: bits = word.range(191, 188); break;
    case 48: bits = word.range(195, 192); break;
    case 49: bits = word.range(199, 196); break;
    case 50: bits = word.range(203, 200); break;
    case 51: bits = word.range(207, 204); break;
    case 52: bits = word.range(211, 208); break;
    case 53: bits = word.range(215, 212); break;
    case 54: bits = word.range(219, 216); break;
    case 55: bits = word.range(223, 220); break;
    case 56: bits = word.range(227, 224); break;
    case 57: bits = word.range(231, 228); break;
    case 58: bits = word.range(235, 232); break;
    case 59: bits = word.range(239, 236); break;
    case 60: bits = word.range(243, 240); break;
    case 61: bits = word.range(247, 244); break;
    case 62: bits = word.range(251, 248); break;
    default: bits = word.range(255, 252); break;
  }
#else
  HgtxrAxiWordT shifted = word >> (lane * HGTXR_WEIGHT_BIT_WIDTH);
  bits = shifted.range(HGTXR_WEIGHT_BIT_WIDTH - 1, 0);
#endif
  return bits;
}

static inline hgtxr_data_t hgtxr_e2e_unpack_weight_word(HgtxrAxiWordT word,
                                                        int lane) {
#pragma HLS INLINE
  HgtxrWeightIntT bits = hgtxr_e2e_extract_weight_bits(word, lane);
  HgtxrWeightT value = 0;
  value.range(HGTXR_WEIGHT_BIT_WIDTH - 1, 0) =
      bits.range(HGTXR_WEIGHT_BIT_WIDTH - 1, 0);
  return static_cast<hgtxr_data_t>(static_cast<HgtxrDataT>(value));
}

static inline hgtxr_data_t hgtxr_e2e_raw_to_weight(int raw) {
#pragma HLS INLINE
  HgtxrWeightT value = 0;
  ap_int<HGTXR_WEIGHT_BIT_WIDTH> packed = raw;
  value.range(HGTXR_WEIGHT_BIT_WIDTH - 1, 0) =
      packed.range(HGTXR_WEIGHT_BIT_WIDTH - 1, 0);
  return static_cast<hgtxr_data_t>(static_cast<HgtxrDataT>(value));
}

static inline int hgtxr_e2e_param_pattern_rom_raw(int elem_offset, int salt) {
#pragma HLS INLINE
  static const ap_int<HGTXR_WEIGHT_BIT_WIDTH> kParamRom[64] = {
      1, 0, -1, 2, -2, 1, 0, -1, 3, -3, 1, -1, 2, 0, -2, 1,
      -1, 1, 0, 2, -2, 0, 1, -1, 3, 1, -3, 0, 2, -1, 1, -2,
      0, 1, -1, 2, 0, -2, 1, 3, -1, -3, 2, 1, 0, -1, 1, -2,
      2, 0, -1, 1, -2, 3, 0, -3, 1, -1, 2, 0, -2, 1, 0, -1};
#pragma HLS bind_storage variable=kParamRom type=rom_1p impl=bram
#pragma HLS ARRAY_PARTITION variable=kParamRom complete dim=1
  const int idx = (elem_offset * 13 + salt * 7) & 63;
  return static_cast<int>(kParamRom[idx]);
}

static inline int hgtxr_e2e_structured_block_rom_raw(int block_rel) {
#pragma HLS INLINE
  if (block_rel >= kBlockLn1Gamma && block_rel < kBlockLn1Beta) {
    return 1;
  }
  if (block_rel >= kBlockLn1Beta && block_rel < kBlockLn2Gamma) {
    return 0;
  }
  if (block_rel >= kBlockLn2Gamma && block_rel < kBlockLn2Beta) {
    return 1;
  }
  if (block_rel >= kBlockLn2Beta && block_rel < kBlockWq) {
    return 0;
  }
  if (block_rel >= kBlockWq && block_rel < kBlockWv) {
    return 0;
  }
  if (block_rel >= kBlockWv && block_rel < kBlockWo) {
    const int rel = block_rel - kBlockWv;
    const int input = rel / HGTXR_EMBED;
    const int output = rel - input * HGTXR_EMBED;
    return input == output ? 1 : 0;
  }
  if (block_rel >= kBlockWo && block_rel < kBlockW1) {
    const int rel = block_rel - kBlockWo;
    const int input = rel / HGTXR_EMBED;
    const int output = rel - input * HGTXR_EMBED;
    return input == output ? 1 : 0;
  }
  if (block_rel >= kBlockW1 && block_rel < kBlockW2) {
    const int rel = block_rel - kBlockW1;
    const int input = rel / kFfDim;
    const int hidden = rel - input * kFfDim;
    return hidden == input ? 1 : 0;
  }
  if (block_rel >= kBlockW2 && block_rel < kBlockWeightElems) {
    const int rel = block_rel - kBlockW2;
    const int hidden = rel / HGTXR_EMBED;
    const int output = rel - hidden * HGTXR_EMBED;
    return hidden == output ? 1 : 0;
  }
  return 0;
}

static inline bool hgtxr_e2e_is_track_rom_block_weight(int elem_offset) {
#pragma HLS INLINE
  if (elem_offset < kBlockWeightElemBase || elem_offset >= kHeadWeightElemBase) {
    return false;
  }
  const int rel = elem_offset - kBlockWeightElemBase;
  const int block_idx = rel / kBlockWeightElems;
  return block_idx < kTrackRomBlockPairs;
}

static inline bool hgtxr_e2e_is_onchip_param_rom_weight(int elem_offset) {
#pragma HLS INLINE
#if HGTXR_E2E_OMIT_WEIGHT_AXI
  return elem_offset >= 0 && elem_offset < kRequiredWeightElems;
#else
  const bool patch_weight =
      elem_offset >= kPatchWeightElemBase && elem_offset < kBlockWeightElemBase;
  const bool head_weight =
      elem_offset >= kHeadWeightElemBase && elem_offset < kRequiredWeightElems;
  return patch_weight || head_weight ||
         hgtxr_e2e_is_track_rom_block_weight(elem_offset);
#endif
}

static inline int hgtxr_e2e_onchip_param_rom_raw(int elem_offset) {
#pragma HLS INLINE
#if HGTXR_E2E_STRUCTURED_PARAM_ROM
  if (elem_offset >= kBlockWeightElemBase && elem_offset < kHeadWeightElemBase) {
    const int rel = elem_offset - kBlockWeightElemBase;
    const int block_rel = rel % kBlockWeightElems;
    return hgtxr_e2e_structured_block_rom_raw(block_rel);
  }
#endif
  int salt = 1;
  if (elem_offset < kEventPatchWeightElemBase) {
    salt = 3;
  } else if (elem_offset < kBlockWeightElemBase) {
    salt = 5;
  } else if (elem_offset >= kHeadWeightElemBase) {
    salt = 11;
  } else {
    const int rel = elem_offset - kBlockWeightElemBase;
    const int block_idx = rel / kBlockWeightElems;
    salt = 17 + block_idx;
  }
  return hgtxr_e2e_param_pattern_rom_raw(elem_offset, salt);
}

static inline int hgtxr_e2e_load_weight_axi_raw(
    const volatile HgtxrAxiWordT *weights, int elem_offset) {
#pragma HLS INLINE
  int word_idx = elem_offset / kWeightLanes;
  int lane_idx = elem_offset - word_idx * kWeightLanes;
  HgtxrAxiWordT word = weights[word_idx];
  return static_cast<int>(hgtxr_e2e_extract_weight_bits(word, lane_idx));
}

static inline int hgtxr_e2e_load_weight_raw(
    const volatile HgtxrAxiWordT *weights, int elem_offset) {
#pragma HLS INLINE
#if HGTXR_E2E_USE_ONCHIP_PARAM_ROM && HGTXR_E2E_OMIT_WEIGHT_AXI
  (void)weights;
  return hgtxr_e2e_onchip_param_rom_raw(elem_offset);
#else
#if HGTXR_E2E_USE_ONCHIP_PARAM_ROM
  if (hgtxr_e2e_is_onchip_param_rom_weight(elem_offset)) {
    return hgtxr_e2e_onchip_param_rom_raw(elem_offset);
  }
#endif
#if HGTXR_E2E_OMIT_WEIGHT_AXI
  return 0;
#else
  return hgtxr_e2e_load_weight_axi_raw(weights, elem_offset);
#endif
#endif
}

static inline hgtxr_data_t hgtxr_e2e_load_weight(
    const volatile HgtxrAxiWordT *weights, int elem_offset) {
#pragma HLS INLINE
  return hgtxr_e2e_raw_to_weight(
      hgtxr_e2e_load_weight_raw(weights, elem_offset));
}

static inline HgtxrAxiWordT hgtxr_e2e_pack_weight_word_from_rom(
    int elem_base) {
#pragma HLS INLINE
  HgtxrAxiWordT word = 0;
  for (int lane = 0; lane < kWeightLanes; ++lane) {
#pragma HLS UNROLL
    HgtxrWeightIntT bits = hgtxr_e2e_onchip_param_rom_raw(elem_base + lane);
    HgtxrAxiWordT lane_bits = 0;
    lane_bits.range(HGTXR_WEIGHT_BIT_WIDTH - 1, 0) =
        bits.range(HGTXR_WEIGHT_BIT_WIDTH - 1, 0);
    word |= lane_bits << (lane * HGTXR_WEIGHT_BIT_WIDTH);
  }
  return word;
}

static inline HgtxrAxiWordT hgtxr_e2e_load_weight_word(
    const volatile HgtxrAxiWordT *weights, int word_idx) {
#pragma HLS INLINE
#if HGTXR_E2E_USE_ONCHIP_PARAM_ROM && HGTXR_E2E_OMIT_WEIGHT_AXI
  (void)weights;
  return hgtxr_e2e_pack_weight_word_from_rom(word_idx * kWeightLanes);
#else
#if HGTXR_E2E_USE_ONCHIP_PARAM_ROM
  const int elem_base = word_idx * kWeightLanes;
  if (hgtxr_e2e_is_onchip_param_rom_weight(elem_base)) {
    return hgtxr_e2e_pack_weight_word_from_rom(elem_base);
  }
#endif
#if HGTXR_E2E_OMIT_WEIGHT_AXI
  return 0;
#else
  return weights[word_idx];
#endif
#endif
}

static inline int hgtxr_e2e_dispatch_prefetch_bank(int block_idx) {
#pragma HLS INLINE
  return block_idx % HGTXR_E2E_DISPATCH_PREFETCH_BANKS;
}

#ifndef __SYNTHESIS__
#if HGTXR_E2E_CSIM_PREFETCH_TRACE
static int g_prefetch_trace_seq = 0;
static int g_prefetch_trace_done_seq[HGTXR_E2E_BLOCKS];
static int g_prefetch_trace_first_load_seq[HGTXR_E2E_BLOCKS];
static int g_prefetch_trace_block_begin_seq[HGTXR_E2E_BLOCKS];
static int g_prefetch_trace_block_end_seq[HGTXR_E2E_BLOCKS];
static int g_prefetch_trace_violations = 0;
static int g_prefetch_trace_not_ready = 0;
static int g_prefetch_trace_immediate_gaps = 0;

inline void hgtxr_e2e_prefetch_trace_reset() {
  g_prefetch_trace_seq = 0;
  g_prefetch_trace_violations = 0;
  g_prefetch_trace_not_ready = 0;
  g_prefetch_trace_immediate_gaps = 0;
  for (int block = 0; block < HGTXR_E2E_BLOCKS; ++block) {
    g_prefetch_trace_done_seq[block] = 0;
    g_prefetch_trace_first_load_seq[block] = 0;
    g_prefetch_trace_block_begin_seq[block] = 0;
    g_prefetch_trace_block_end_seq[block] = 0;
  }
}

inline void hgtxr_e2e_prefetch_trace_done(int block_idx, int words) {
  if (block_idx < 0 || block_idx >= HGTXR_E2E_BLOCKS) {
    return;
  }
  g_prefetch_trace_done_seq[block_idx] = ++g_prefetch_trace_seq;
  std::printf("prefetch_trace_done block=%d seq=%d words=%d\n", block_idx,
              g_prefetch_trace_done_seq[block_idx], words);
}

inline void hgtxr_e2e_prefetch_trace_load_hit(int block_idx) {
  if (block_idx < 0 || block_idx >= HGTXR_E2E_BLOCKS ||
      g_prefetch_trace_first_load_seq[block_idx] != 0) {
    return;
  }
  g_prefetch_trace_first_load_seq[block_idx] = ++g_prefetch_trace_seq;
  const int done_seq = g_prefetch_trace_done_seq[block_idx];
  const bool ordered = done_seq > 0 &&
                       done_seq < g_prefetch_trace_first_load_seq[block_idx];
  std::printf("prefetch_trace_first_load block=%d seq=%d done_seq=%d ordered=%d\n",
              block_idx, g_prefetch_trace_first_load_seq[block_idx], done_seq,
              ordered ? 1 : 0);
  if (!ordered) {
    ++g_prefetch_trace_violations;
  }
}

inline void hgtxr_e2e_prefetch_trace_block_begin(int block_idx,
                                                int block_pairs) {
  if (block_idx < 0 || block_idx >= HGTXR_E2E_BLOCKS) {
    return;
  }
  g_prefetch_trace_block_begin_seq[block_idx] = ++g_prefetch_trace_seq;
  const int done_seq = g_prefetch_trace_done_seq[block_idx];
  const bool requires_prefetch =
      block_pairs > kTrackRomBlockPairs && block_idx >= kTrackRomBlockPairs;
  const bool ready = done_seq > 0 &&
                     done_seq < g_prefetch_trace_block_begin_seq[block_idx];
  std::printf("prefetch_trace_block_begin block=%d seq=%d done_seq=%d "
              "requires_prefetch=%d ready=%d\n",
              block_idx, g_prefetch_trace_block_begin_seq[block_idx],
              done_seq, requires_prefetch ? 1 : 0, ready ? 1 : 0);
  if (requires_prefetch && !ready) {
    ++g_prefetch_trace_not_ready;
  }
  if (block_idx > 0 && g_prefetch_trace_block_end_seq[block_idx - 1] > 0) {
    const bool immediate =
        g_prefetch_trace_block_end_seq[block_idx - 1] + 1 ==
        g_prefetch_trace_block_begin_seq[block_idx];
    std::printf("prefetch_trace_interblock prev=%d next=%d prev_end_seq=%d "
                "next_begin_seq=%d immediate=%d\n",
                block_idx - 1, block_idx,
                g_prefetch_trace_block_end_seq[block_idx - 1],
                g_prefetch_trace_block_begin_seq[block_idx],
                immediate ? 1 : 0);
    if (!immediate) {
      ++g_prefetch_trace_immediate_gaps;
    }
  }
}

inline void hgtxr_e2e_prefetch_trace_block_end(int block_idx) {
  if (block_idx < 0 || block_idx >= HGTXR_E2E_BLOCKS) {
    return;
  }
  g_prefetch_trace_block_end_seq[block_idx] = ++g_prefetch_trace_seq;
  std::printf("prefetch_trace_block_end block=%d seq=%d\n", block_idx,
              g_prefetch_trace_block_end_seq[block_idx]);
}

inline void hgtxr_e2e_prefetch_trace_report(int block_pairs) {
  std::printf("prefetch_trace_summary block_pairs=%d violations=%d "
              "not_ready=%d immediate_gaps=%d\n",
              block_pairs, g_prefetch_trace_violations,
              g_prefetch_trace_not_ready, g_prefetch_trace_immediate_gaps);
  std::fflush(stdout);
  assert(g_prefetch_trace_violations == 0);
#if HGTXR_E2E_CSIM_STRICT_IMMEDIATE_START
  assert(g_prefetch_trace_not_ready == 0);
  assert(g_prefetch_trace_immediate_gaps == 0);
#endif
}
#endif
#endif

static inline HgtxrAxiWordT hgtxr_e2e_load_weight_word_prefetched(
    const volatile HgtxrAxiWordT *weights,
    HgtxrGlobalBuffer &gb,
    int block_idx,
    int word_idx) {
#if HGTXR_E2E_STAGE_PREFETCH_LOAD
#pragma HLS INLINE off
#else
#pragma HLS INLINE
#endif
#if HGTXR_E2E_USE_SEARCH_WEIGHT_DISPATCHER
  const int bank = hgtxr_e2e_dispatch_prefetch_bank(block_idx);
  const int base = gb.dispatch_prefetch_base_word[bank];
  const int offset = word_idx - base;
  if (gb.dispatch_prefetch_ready[bank] &&
      gb.dispatch_prefetch_block[bank] == block_idx &&
      offset >= 0 && offset < gb.dispatch_prefetch_valid_words[bank]) {
#ifndef __SYNTHESIS__
#if HGTXR_E2E_CSIM_PREFETCH_TRACE
    hgtxr_e2e_prefetch_trace_load_hit(block_idx);
#endif
#endif
    HgtxrAxiWordT prefetched_word = gb.dispatch_prefetch[bank][offset];
    return prefetched_word;
  }
#else
  (void)gb;
  (void)block_idx;
#endif
  return hgtxr_e2e_load_weight_word(weights, word_idx);
}

static inline void hgtxr_e2e_load_weight_vec(
    const volatile HgtxrAxiWordT *weights, int elem_base,
    hgtxr_data_t out[kDensePar]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=out complete dim=1
#if HGTXR_E2E_USE_ONCHIP_PARAM_ROM && HGTXR_E2E_OMIT_WEIGHT_AXI
  for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
    out[lane] = hgtxr_e2e_load_weight(weights, elem_base + lane);
  }
#elif HGTXR_E2E_USE_ONCHIP_PARAM_ROM
  if (hgtxr_e2e_is_onchip_param_rom_weight(elem_base)) {
    for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
      out[lane] = hgtxr_e2e_load_weight(weights, elem_base + lane);
    }
  } else {
    int word_idx = elem_base / kWeightLanes;
    int lane_idx = elem_base - word_idx * kWeightLanes;
    HgtxrAxiWordT word0 = hgtxr_e2e_load_weight_word(weights, word_idx);
    HgtxrAxiWordT word1 = word0;
    if (lane_idx + kDensePar > kWeightLanes) {
      word1 = hgtxr_e2e_load_weight_word(weights, word_idx + 1);
    }
    for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
      int packed_lane = lane_idx + lane;
      if (packed_lane < kWeightLanes) {
        out[lane] = hgtxr_e2e_unpack_weight_word(word0, packed_lane);
      } else {
        out[lane] =
            hgtxr_e2e_unpack_weight_word(word1, packed_lane - kWeightLanes);
      }
    }
  }
#elif HGTXR_E2E_WEIGHT_VEC_CACHE && HGTXR_E2E_WEIGHT_VEC_ALIGNED_FASTPATH && \
    (((HGTXR_BUS_WIDTH_BITS / HGTXR_WEIGHT_BIT_WIDTH) % HGTXR_E2E_DENSE_PAR) == 0) && \
    ((HGTXR_EMBED % HGTXR_E2E_DENSE_PAR) == 0) && \
    ((HGTXR_E2E_FF_DIM % HGTXR_E2E_DENSE_PAR) == 0)
  int word_idx = elem_base / kWeightLanes;
  int lane_idx = elem_base - word_idx * kWeightLanes;
  HgtxrAxiWordT word0 = hgtxr_e2e_load_weight_word(weights, word_idx);
  for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
    out[lane] = hgtxr_e2e_unpack_weight_word(word0, lane_idx + lane);
  }
#elif HGTXR_E2E_WEIGHT_VEC_CACHE
  int word_idx = elem_base / kWeightLanes;
  int lane_idx = elem_base - word_idx * kWeightLanes;
  HgtxrAxiWordT word0 = hgtxr_e2e_load_weight_word(weights, word_idx);
  HgtxrAxiWordT word1 = word0;
  if (lane_idx + kDensePar > kWeightLanes) {
    word1 = hgtxr_e2e_load_weight_word(weights, word_idx + 1);
  }
  for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
    int packed_lane = lane_idx + lane;
    if (packed_lane < kWeightLanes) {
      out[lane] = hgtxr_e2e_unpack_weight_word(word0, packed_lane);
    } else {
      out[lane] =
          hgtxr_e2e_unpack_weight_word(word1, packed_lane - kWeightLanes);
    }
  }
#else
  for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
    out[lane] = hgtxr_e2e_load_weight(weights, elem_base + lane);
  }
#endif
}

template <int CACHE_WORDS>
static inline void hgtxr_e2e_load_weight_vec_from_cache(
    const HgtxrAxiWordT (&cache)[CACHE_WORDS], int elem_base,
    hgtxr_data_t out[kDensePar]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=out complete dim=1
#if HGTXR_E2E_WEIGHT_VEC_ALIGNED_FASTPATH && \
    (((HGTXR_BUS_WIDTH_BITS / HGTXR_WEIGHT_BIT_WIDTH) % HGTXR_E2E_DENSE_PAR) == 0) && \
    ((HGTXR_EMBED % HGTXR_E2E_DENSE_PAR) == 0) && \
    ((HGTXR_E2E_FF_DIM % HGTXR_E2E_DENSE_PAR) == 0)
  int word_idx = elem_base / kWeightLanes;
  int lane_idx = elem_base - word_idx * kWeightLanes;
  HgtxrAxiWordT word0 = cache[word_idx];
  for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
    out[lane] = hgtxr_e2e_unpack_weight_word(word0, lane_idx + lane);
  }
#else
  int word_idx = elem_base / kWeightLanes;
  int lane_idx = elem_base - word_idx * kWeightLanes;
  HgtxrAxiWordT word0 = cache[word_idx];
  HgtxrAxiWordT word1 = word0;
  if (lane_idx + kDensePar > kWeightLanes) {
    word1 = cache[word_idx + 1];
  }
  for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
    int packed_lane = lane_idx + lane;
    if (packed_lane < kWeightLanes) {
      out[lane] = hgtxr_e2e_unpack_weight_word(word0, packed_lane);
    } else {
      out[lane] =
          hgtxr_e2e_unpack_weight_word(word1, packed_lane - kWeightLanes);
    }
  }
#endif
}

template <int BANK_WORDS>
static inline void hgtxr_e2e_load_w2_vec_from_hidden_bank(
    const HgtxrAxiWordT (&cache)[kDensePar][BANK_WORDS], int h, int c0,
    hgtxr_data_t out[kDensePar]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=out complete dim=1
  const int hp = h - (h / kDensePar) * kDensePar;
  const int row = h / kDensePar;
  const int word_col = c0 / kWeightLanes;
  const int lane_idx = c0 - word_col * kWeightLanes;
  const int word_idx = row * kMlpW2ColWords + word_col;
  HgtxrAxiWordT word0 = cache[hp][word_idx];
  HgtxrAxiWordT word1 = word0;
  if (lane_idx + kDensePar > kWeightLanes) {
    word1 = cache[hp][word_idx + 1];
  }
  for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
    const int packed_lane = lane_idx + lane;
    if (packed_lane < kWeightLanes) {
      out[lane] = hgtxr_e2e_unpack_weight_word(word0, packed_lane);
    } else {
      out[lane] =
          hgtxr_e2e_unpack_weight_word(word1, packed_lane - kWeightLanes);
    }
  }
}

template <int COL_WORDS>
static inline void hgtxr_e2e_load_w2_vec_from_hgroup_bank(
    const HgtxrAxiWordT (&cache)[kDensePar][COL_WORDS], int h, int c0,
    hgtxr_data_t out[kDensePar]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=out complete dim=1
  const int hp = h - (h / kDensePar) * kDensePar;
  const int word_col = c0 / kWeightLanes;
  const int lane_idx = c0 - word_col * kWeightLanes;
  HgtxrAxiWordT word0 = cache[hp][word_col];
  HgtxrAxiWordT word1 = word0;
  if (lane_idx + kDensePar > kWeightLanes) {
    word1 = cache[hp][word_col + 1];
  }
  for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
    const int packed_lane = lane_idx + lane;
    if (packed_lane < kWeightLanes) {
      out[lane] = hgtxr_e2e_unpack_weight_word(word0, packed_lane);
    } else {
      out[lane] =
          hgtxr_e2e_unpack_weight_word(word1, packed_lane - kWeightLanes);
    }
  }
}

static inline int hgtxr_e2e_load_weight_raw_from_cache(
    const HgtxrAxiWordT cache[kLnParamWords], int elem_offset) {
#pragma HLS INLINE
  int word_idx = elem_offset / kWeightLanes;
  int lane_idx = elem_offset - word_idx * kWeightLanes;
  return static_cast<int>(
      hgtxr_e2e_extract_weight_bits(cache[word_idx], lane_idx));
}

static inline hgtxr_data_t hgtxr_e2e_load_weight_from_cache(
    const HgtxrAxiWordT cache[kLnParamWords], int elem_offset) {
#pragma HLS INLINE
  int word_idx = elem_offset / kWeightLanes;
  int lane_idx = elem_offset - word_idx * kWeightLanes;
  return hgtxr_e2e_unpack_weight_word(cache[word_idx], lane_idx);
}

static inline hgtxr_acc_t hgtxr_e2e_dsp_mul(hgtxr_data_t lhs,
                                            hgtxr_data_t rhs) {
#pragma HLS INLINE
  hgtxr_acc_t product;
#if HGTXR_E2E_FORCE_DSP_MUL
#if HGTXR_E2E_DSP_MUL_LATENCY > 0
#pragma HLS bind_op variable=product op=mul impl=dsp latency=HGTXR_E2E_DSP_MUL_LATENCY
#else
#pragma HLS bind_op variable=product op=mul impl=dsp
#endif
#endif
#if HGTXR_E2E_FORCE_WIDE_DSP_MUL
  HgtxrDspOperandT lhs_w = static_cast<HgtxrDspOperandT>(lhs);
  HgtxrDspOperandT rhs_w = static_cast<HgtxrDspOperandT>(rhs);
  product = static_cast<hgtxr_acc_t>(lhs_w * rhs_w);
#else
  product = static_cast<hgtxr_acc_t>(lhs) * static_cast<hgtxr_acc_t>(rhs);
#endif
  return product;
}

static inline hgtxr_acc_t hgtxr_e2e_dsp_mul_acc(hgtxr_acc_t lhs,
                                                hgtxr_acc_t rhs) {
#pragma HLS INLINE
  hgtxr_acc_t product;
#if HGTXR_E2E_FORCE_DSP_MUL
#if HGTXR_E2E_DSP_MUL_LATENCY > 0
#pragma HLS bind_op variable=product op=mul impl=dsp latency=HGTXR_E2E_DSP_MUL_LATENCY
#else
#pragma HLS bind_op variable=product op=mul impl=dsp
#endif
#endif
  product = lhs * rhs;
  return product;
}

static inline hgtxr_acc_t hgtxr_e2e_fabric_mul(hgtxr_data_t lhs,
                                               hgtxr_data_t rhs) {
#pragma HLS INLINE
  hgtxr_acc_t product;
#if HGTXR_E2E_FABRIC_MUL_LATENCY > 0
#pragma HLS bind_op variable=product op=mul impl=fabric latency=HGTXR_E2E_FABRIC_MUL_LATENCY
#else
#pragma HLS bind_op variable=product op=mul impl=fabric
#endif
#if HGTXR_E2E_FORCE_WIDE_DSP_MUL
  HgtxrDspOperandT lhs_w = static_cast<HgtxrDspOperandT>(lhs);
  HgtxrDspOperandT rhs_w = static_cast<HgtxrDspOperandT>(rhs);
  product = static_cast<hgtxr_acc_t>(lhs_w * rhs_w);
#else
  product = static_cast<hgtxr_acc_t>(lhs) * static_cast<hgtxr_acc_t>(rhs);
#endif
  return product;
}

static inline hgtxr_acc_t hgtxr_e2e_patch_mul(hgtxr_data_t lhs,
                                              hgtxr_data_t rhs) {
#pragma HLS INLINE
#if HGTXR_E2E_PATCH_FABRIC_MUL
  return hgtxr_e2e_fabric_mul(lhs, rhs);
#else
  return hgtxr_e2e_dsp_mul(lhs, rhs);
#endif
}

static inline hgtxr_acc_t hgtxr_e2e_layernorm_mul_acc(hgtxr_acc_t lhs,
                                                      hgtxr_acc_t rhs) {
#pragma HLS INLINE
  hgtxr_acc_t product;
#if HGTXR_E2E_LAYERNORM_FABRIC_MUL
#if HGTXR_E2E_FABRIC_MUL_LATENCY > 0
#pragma HLS bind_op variable=product op=mul impl=fabric latency=HGTXR_E2E_FABRIC_MUL_LATENCY
#else
#pragma HLS bind_op variable=product op=mul impl=fabric
#endif
#else
#if HGTXR_E2E_FORCE_DSP_MUL
#if HGTXR_E2E_DSP_MUL_LATENCY > 0
#pragma HLS bind_op variable=product op=mul impl=dsp latency=HGTXR_E2E_DSP_MUL_LATENCY
#else
#pragma HLS bind_op variable=product op=mul impl=dsp
#endif
#endif
#endif
  product = lhs * rhs;
  return product;
}

static inline hgtxr_acc_t hgtxr_e2e_fastpath_mul(hgtxr_data_t lhs,
                                                 hgtxr_data_t rhs) {
#pragma HLS INLINE
#if HGTXR_E2E_FASTPATH_FABRIC_MUL
  return hgtxr_e2e_fabric_mul(lhs, rhs);
#else
  return hgtxr_e2e_dsp_mul(lhs, rhs);
#endif
}

template <int LANE>
static inline hgtxr_acc_t hgtxr_e2e_core_lane_mul_ct(hgtxr_data_t lhs,
                                                     hgtxr_data_t rhs) {
#pragma HLS INLINE
#if HGTXR_E2E_CORE_ALL_FABRIC_MUL
  return hgtxr_e2e_fabric_mul(lhs, rhs);
#elif HGTXR_E2E_CORE_FABRIC_TAIL_LANES > 0
  if (LANE >= kDensePar - HGTXR_E2E_CORE_FABRIC_TAIL_LANES) {
    return hgtxr_e2e_fabric_mul(lhs, rhs);
  }
#elif HGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL
  if (LANE == kDensePar - 1) {
    return hgtxr_e2e_fabric_mul(lhs, rhs);
  }
#endif
  return hgtxr_e2e_dsp_mul(lhs, rhs);
}

static inline hgtxr_acc_t hgtxr_e2e_core_lane_mul(hgtxr_data_t lhs,
                                                  hgtxr_data_t rhs,
                                                  int lane) {
#pragma HLS INLINE
#if HGTXR_E2E_CORE_LANE_CT_SWITCH
  switch (lane) {
    case 0:
      return hgtxr_e2e_core_lane_mul_ct<0>(lhs, rhs);
    case 1:
      return hgtxr_e2e_core_lane_mul_ct<1>(lhs, rhs);
    case 2:
      return hgtxr_e2e_core_lane_mul_ct<2>(lhs, rhs);
    case 3:
      return hgtxr_e2e_core_lane_mul_ct<3>(lhs, rhs);
    case 4:
      return hgtxr_e2e_core_lane_mul_ct<4>(lhs, rhs);
    case 5:
      return hgtxr_e2e_core_lane_mul_ct<5>(lhs, rhs);
    case 6:
      return hgtxr_e2e_core_lane_mul_ct<6>(lhs, rhs);
    case 7:
      return hgtxr_e2e_core_lane_mul_ct<7>(lhs, rhs);
    case 8:
      return hgtxr_e2e_core_lane_mul_ct<8>(lhs, rhs);
    case 9:
      return hgtxr_e2e_core_lane_mul_ct<9>(lhs, rhs);
    case 10:
      return hgtxr_e2e_core_lane_mul_ct<10>(lhs, rhs);
    case 11:
      return hgtxr_e2e_core_lane_mul_ct<11>(lhs, rhs);
    case 12:
      return hgtxr_e2e_core_lane_mul_ct<12>(lhs, rhs);
    case 13:
      return hgtxr_e2e_core_lane_mul_ct<13>(lhs, rhs);
    case 14:
      return hgtxr_e2e_core_lane_mul_ct<14>(lhs, rhs);
    case 15:
      return hgtxr_e2e_core_lane_mul_ct<15>(lhs, rhs);
    case 16:
      return hgtxr_e2e_core_lane_mul_ct<16>(lhs, rhs);
    case 17:
      return hgtxr_e2e_core_lane_mul_ct<17>(lhs, rhs);
    case 18:
      return hgtxr_e2e_core_lane_mul_ct<18>(lhs, rhs);
    case 19:
      return hgtxr_e2e_core_lane_mul_ct<19>(lhs, rhs);
    case 20:
      return hgtxr_e2e_core_lane_mul_ct<20>(lhs, rhs);
    case 21:
      return hgtxr_e2e_core_lane_mul_ct<21>(lhs, rhs);
    case 22:
      return hgtxr_e2e_core_lane_mul_ct<22>(lhs, rhs);
    case 23:
      return hgtxr_e2e_core_lane_mul_ct<23>(lhs, rhs);
    case 24:
      return hgtxr_e2e_core_lane_mul_ct<24>(lhs, rhs);
    case 25:
      return hgtxr_e2e_core_lane_mul_ct<25>(lhs, rhs);
    case 26:
      return hgtxr_e2e_core_lane_mul_ct<26>(lhs, rhs);
    case 27:
      return hgtxr_e2e_core_lane_mul_ct<27>(lhs, rhs);
    case 28:
      return hgtxr_e2e_core_lane_mul_ct<28>(lhs, rhs);
    case 29:
      return hgtxr_e2e_core_lane_mul_ct<29>(lhs, rhs);
    case 30:
      return hgtxr_e2e_core_lane_mul_ct<30>(lhs, rhs);
    default:
      return hgtxr_e2e_core_lane_mul_ct<31>(lhs, rhs);
  }
#elif HGTXR_E2E_CORE_ALL_FABRIC_MUL
  (void)lane;
  return hgtxr_e2e_fabric_mul(lhs, rhs);
#elif HGTXR_E2E_CORE_FABRIC_TAIL_LANES > 0
  if (lane >= kDensePar - HGTXR_E2E_CORE_FABRIC_TAIL_LANES) {
    return hgtxr_e2e_fabric_mul(lhs, rhs);
  }
#elif HGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL
  if (lane == kDensePar - 1) {
    return hgtxr_e2e_fabric_mul(lhs, rhs);
  }
#else
  (void)lane;
#endif
  return hgtxr_e2e_dsp_mul(lhs, rhs);
}

static inline hgtxr_acc_t hgtxr_e2e_mlp_w2_hp_lane_mul(hgtxr_data_t lhs,
                                                       hgtxr_data_t rhs,
                                                       int hp_lane,
                                                       int out_lane) {
#pragma HLS INLINE
#if HGTXR_E2E_MLP_W2_FABRIC_HP_LANES > 0
  if (hp_lane >= kMlpFusedW2HpPar - kMlpW2FabricHpLanes) {
    return hgtxr_e2e_fabric_mul(lhs, rhs);
  }
#endif
  return hgtxr_e2e_core_lane_mul(lhs, rhs, out_lane);
}

static inline hgtxr_acc_t hgtxr_e2e_head_mul(hgtxr_data_t lhs,
                                             hgtxr_data_t rhs) {
#pragma HLS INLINE
#if HGTXR_E2E_HEAD_FABRIC_MUL
  return hgtxr_e2e_fabric_mul(lhs, rhs);
#else
  return hgtxr_e2e_dsp_mul(lhs, rhs);
#endif
}

static inline hgtxr_acc_t hgtxr_e2e_head_raw_shiftadd_mul(hgtxr_data_t lhs,
                                                          int raw_weight) {
#pragma HLS INLINE
  hgtxr_acc_t x = static_cast<hgtxr_acc_t>(lhs);
  hgtxr_acc_t scaled = 0;
  switch (raw_weight) {
    case -8:
      scaled = -(x + x + x + x + x + x + x + x);
      break;
    case -7:
      scaled = -(x + x + x + x + x + x + x);
      break;
    case -6:
      scaled = -(x + x + x + x + x + x);
      break;
    case -5:
      scaled = -(x + x + x + x + x);
      break;
    case -4:
      scaled = -(x + x + x + x);
      break;
    case -3:
      scaled = -(x + x + x);
      break;
    case -2:
      scaled = -(x + x);
      break;
    case -1:
      scaled = -x;
      break;
    case 0:
      scaled = 0;
      break;
    case 1:
      scaled = x;
      break;
    case 2:
      scaled = x + x;
      break;
    case 3:
      scaled = x + x + x;
      break;
    case 4:
      scaled = x + x + x + x;
      break;
    case 5:
      scaled = x + x + x + x + x;
      break;
    case 6:
      scaled = x + x + x + x + x + x;
      break;
    default:
      scaled = x + x + x + x + x + x + x;
      break;
  }
  return hgtxr_e2e_acc_div_int_ct<4>(scaled);
}

static inline int hgtxr_e2e_clip_lut_index(int value, int offset, int limit) {
#pragma HLS INLINE
  int idx = value + offset;
  if (idx < 0) {
    idx = 0;
  }
  if (idx >= limit) {
    idx = limit - 1;
  }
  return idx;
}

static inline hgtxr_data_t hgtxr_e2e_gelu_rom(hgtxr_data_t x) {
#pragma HLS INLINE
  static const int kGeluRom[32] = {
      -1, -1, -1, -1, -1, -1, 0, 0,
      0, 0, 0, 1, 1, 2, 3, 4,
      5, 6, 7, 8, 9, 10, 12, 13,
      14, 15, 16, 17, 18, 19, 20, 21};
#if HGTXR_E2E_GELU_ROM_PARTITION
#pragma HLS ARRAY_PARTITION variable=kGeluRom complete dim=1
#elif HGTXR_E2E_NONLINEAR_ROM_LUTRAM
#pragma HLS bind_storage variable=kGeluRom type=rom_1p impl=lutram
#else
#pragma HLS bind_storage variable=kGeluRom type=rom_1p impl=bram
#endif
  const int raw = hgtxr_e2e_data_to_scaled_int_ct<8>(x);
  const int idx = hgtxr_e2e_clip_lut_index(raw, 16, 32);
  return hgtxr_e2e_data_from_scaled_int_ct<8>(kGeluRom[idx]);
}

static inline hgtxr_data_t hgtxr_e2e_softmax_exp_rom(hgtxr_data_t x) {
#pragma HLS INLINE
  static const int kExpRom[32] = {
      0, 0, 0, 0, 0, 1, 1, 1,
      1, 2, 2, 3, 3, 4, 5, 6,
      7, 8, 9, 10, 11, 12, 13, 14,
      15, 16, 16, 16, 16, 16, 16, 16};
#if HGTXR_E2E_SOFTMAX_EXP_ROM_PARTITION
#pragma HLS ARRAY_PARTITION variable=kExpRom complete dim=1
#elif HGTXR_E2E_NONLINEAR_ROM_LUTRAM
#pragma HLS bind_storage variable=kExpRom type=rom_1p impl=lutram
#else
#pragma HLS bind_storage variable=kExpRom type=rom_1p impl=bram
#endif
  const int raw = hgtxr_e2e_data_to_scaled_int_ct<4>(x);
  const int idx = hgtxr_e2e_clip_lut_index(raw, 24, 32);
  return hgtxr_e2e_data_from_scaled_int_ct<16>(kExpRom[idx]);
}

static inline hgtxr_data_t hgtxr_e2e_layernorm_rsqrt_rom(hgtxr_acc_t var_mean) {
#pragma HLS INLINE
  static const int kRsqrtRom[32] = {
      32, 28, 25, 23, 21, 19, 18, 17,
      16, 15, 14, 13, 13, 12, 12, 11,
      11, 10, 10, 10, 9, 9, 9, 8,
      8, 8, 8, 7, 7, 7, 7, 7};
#if HGTXR_E2E_NONLINEAR_ROM_LUTRAM
#pragma HLS bind_storage variable=kRsqrtRom type=rom_1p impl=lutram
#else
#pragma HLS bind_storage variable=kRsqrtRom type=rom_1p impl=bram
#endif
  int idx = static_cast<int>(var_mean * static_cast<hgtxr_acc_t>(2));
  if (idx < 0) {
    idx = 0;
  }
  if (idx > 31) {
    idx = 31;
  }
  return hgtxr_e2e_data_from_scaled_int_ct<16>(kRsqrtRom[idx]);
}

static inline bool hgtxr_e2e_search_dispatcher_uses_axi(int block_idx,
                                                        int block_pairs) {
#pragma HLS INLINE
  return block_pairs > kTrackRomBlockPairs && block_idx >= kTrackRomBlockPairs;
}

static inline int hgtxr_e2e_prefetch_block_elem_base(int block_idx) {
#pragma HLS INLINE
#if HGTXR_E2E_PREFETCH_ADDR_SWITCH
  switch (block_idx) {
    case 0:
      return kBlockWeightElemBase;
    case 1:
      return kBlockWeightElemBase + kBlockWeightElems;
    case 2:
      return kBlockWeightElemBase + 2 * kBlockWeightElems;
    case 3:
      return kBlockWeightElemBase + 3 * kBlockWeightElems;
    case 4:
      return kBlockWeightElemBase + 4 * kBlockWeightElems;
    case 5:
      return kBlockWeightElemBase + 5 * kBlockWeightElems;
    case 6:
      return kBlockWeightElemBase + 6 * kBlockWeightElems;
    default:
      return kBlockWeightElemBase + 7 * kBlockWeightElems;
  }
#else
  return kBlockWeightElemBase + block_idx * kBlockWeightElems;
#endif
}

static inline int hgtxr_e2e_prefetch_block_word_base(int block_idx) {
#pragma HLS INLINE
#if HGTXR_E2E_PREFETCH_ADDR_SWITCH
  switch (block_idx) {
    case 0:
      return kBlockWeightElemBase / kWeightLanes;
    case 1:
      return (kBlockWeightElemBase + kBlockWeightElems) / kWeightLanes;
    case 2:
      return (kBlockWeightElemBase + 2 * kBlockWeightElems) / kWeightLanes;
    case 3:
      return (kBlockWeightElemBase + 3 * kBlockWeightElems) / kWeightLanes;
    case 4:
      return (kBlockWeightElemBase + 4 * kBlockWeightElems) / kWeightLanes;
    case 5:
      return (kBlockWeightElemBase + 5 * kBlockWeightElems) / kWeightLanes;
    case 6:
      return (kBlockWeightElemBase + 6 * kBlockWeightElems) / kWeightLanes;
    default:
      return (kBlockWeightElemBase + 7 * kBlockWeightElems) / kWeightLanes;
  }
#else
  return hgtxr_e2e_prefetch_block_elem_base(block_idx) / kWeightLanes;
#endif
}

inline void hgtxr_e2e_search_weight_dispatch_prefetch(
    const volatile HgtxrAxiWordT *weights,
    HgtxrGlobalBuffer &gb,
    int block_idx,
    int block_pairs) {
#pragma HLS INLINE off
#if HGTXR_E2E_USE_SEARCH_WEIGHT_DISPATCHER
  const int bank = hgtxr_e2e_dispatch_prefetch_bank(block_idx);
  if (hgtxr_e2e_search_dispatcher_uses_axi(block_idx, block_pairs)) {
    const int block_word_base = hgtxr_e2e_prefetch_block_word_base(block_idx);
    gb.dispatch_prefetch_base_word[bank] = block_word_base;
    gb.dispatch_prefetch_valid_words[bank] = kDispatchPrefetchWords;
    gb.dispatch_prefetch_block[bank] = block_idx;
    for (int w = 0; w < kDispatchPrefetchWords; ++w) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=8192
#pragma HLS PIPELINE II=1
      gb.dispatch_prefetch[bank][w] =
          hgtxr_e2e_load_weight_word(weights, block_word_base + w);
    }
    gb.dispatch_prefetch_ready[bank] = 1;
#ifndef __SYNTHESIS__
#if HGTXR_E2E_CSIM_PREFETCH_TRACE
    hgtxr_e2e_prefetch_trace_done(block_idx, kDispatchPrefetchWords);
#endif
#endif
  } else {
    gb.dispatch_prefetch_base_word[bank] = 0;
    gb.dispatch_prefetch_valid_words[bank] = 0;
    gb.dispatch_prefetch_block[bank] = block_idx;
    for (int w = 0; w < kDispatchPrefetchWords; ++w) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=8192
#pragma HLS PIPELINE II=1
      gb.dispatch_prefetch[bank][w] = 0;
    }
    gb.dispatch_prefetch_ready[bank] = 0;
  }
#else
  (void)weights;
  (void)block_pairs;
  const int bank = hgtxr_e2e_dispatch_prefetch_bank(block_idx);
  gb.dispatch_prefetch_base_word[bank] = 0;
  gb.dispatch_prefetch_valid_words[bank] = 0;
  gb.dispatch_prefetch_block[bank] = block_idx;
  for (int w = 0; w < kDispatchPrefetchWords; ++w) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS PIPELINE II=1
    gb.dispatch_prefetch[bank][w] = 0;
  }
  gb.dispatch_prefetch_ready[bank] = 0;
#endif
}

static inline int hgtxr_e2e_hgpipe_geluq_raw(int value, int block_idx) {
#pragma HLS INLINE
  switch (block_idx % 12) {
    case 0:
      return hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp0_geluq64_int(value);
    case 1:
      return hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp1_geluq64_int(value);
    case 2:
      return hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp2_geluq64_int(value);
    case 3:
      return hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp3_geluq64_int(value);
    case 4:
      return hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp4_geluq64_int(value);
    case 5:
      return hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp5_geluq64_int(value);
    case 6:
      return hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp6_geluq64_int(value);
    case 7:
      return hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp7_geluq64_int(value);
    case 8:
      return hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp8_geluq64_int(value);
    case 9:
      return hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp9_geluq64_int(value);
    case 10:
      return hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp10_geluq64_int(value);
    default:
      return hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp11_geluq64_int(value);
  }
}

static inline hgtxr_data_t hgtxr_e2e_gelu(hgtxr_data_t x, int block_idx) {
#pragma HLS INLINE
#if HGTXR_E2E_USE_ONCHIP_NONLINEAR_ROM
  (void)block_idx;
  return hgtxr_e2e_gelu_rom(x);
#elif HGTXR_E2E_USE_HGPIPE_INT_GELUQ
  int raw = hgtxr_e2e_data_to_scaled_int_ct<
      HGTXR_E2E_HGPIPE_GELUQ_INPUT_SCALE>(x);
  int quantized = hgtxr_e2e_hgpipe_geluq_raw(raw, block_idx);
  return hgtxr_e2e_data_from_scaled_int_ct<
      HGTXR_E2E_HGPIPE_GELUQ_OUTPUT_SCALE>(quantized);
#else
  (void)block_idx;
  return static_cast<hgtxr_data_t>(
      hgtxr::cyclic_transformer::hgtxr_gelu_approx(static_cast<HgtxrDataT>(x)));
#endif
}

static inline hgtxr_data_t hgtxr_e2e_exp_approx(hgtxr_data_t x) {
#pragma HLS INLINE
#if HGTXR_E2E_USE_ONCHIP_NONLINEAR_ROM
  return hgtxr_e2e_softmax_exp_rom(x);
#elif HGTXR_USE_HGPIPE_LUT_MATH
  return static_cast<hgtxr_data_t>(
      hgtxr::cyclic_transformer::hgtxr_score_exp_approx(static_cast<HgtxrScoreT>(x)));
#else
  if (x <= static_cast<hgtxr_data_t>(-8)) {
    return 0;
  }
  if (x >= static_cast<hgtxr_data_t>(0)) {
    return 1;
  }
  hgtxr_data_t y =
      static_cast<hgtxr_data_t>(1) + hgtxr_e2e_data_div_int_ct<8>(x);
  return y > 0 ? y : static_cast<hgtxr_data_t>(0);
#endif
}

static inline int hgtxr_e2e_hgpipe_softmax_exp_raw(int opposite_delta, int block_idx) {
#pragma HLS INLINE
  switch (block_idx % 12) {
    case 0: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn0_softmax_exp32_int(opposite_delta);
    case 1: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn1_softmax_exp32_int(opposite_delta);
    case 2: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn2_softmax_exp32_int(opposite_delta);
    case 3: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn3_softmax_exp32_int(opposite_delta);
    case 4: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn4_softmax_exp32_int(opposite_delta);
    case 5: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn5_softmax_exp32_int(opposite_delta);
    case 6: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn6_softmax_exp32_int(opposite_delta);
    case 7: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn7_softmax_exp32_int(opposite_delta);
    case 8: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn8_softmax_exp32_int(opposite_delta);
    case 9: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn9_softmax_exp32_int(opposite_delta);
    case 10: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn10_softmax_exp32_int(opposite_delta);
    default: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn11_softmax_exp32_int(opposite_delta);
  }
}

static inline bool hgtxr_e2e_hgpipe_softmax_uses_table_two(int acc, int block_idx) {
#pragma HLS INLINE
  switch (block_idx % 12) {
    case 0: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn0_softmax_recip_uses_table_two(acc);
    case 1: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn1_softmax_recip_uses_table_two(acc);
    case 2: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn2_softmax_recip_uses_table_two(acc);
    case 3: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn3_softmax_recip_uses_table_two(acc);
    case 4: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn4_softmax_recip_uses_table_two(acc);
    case 5: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn5_softmax_recip_uses_table_two(acc);
    case 6: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn6_softmax_recip_uses_table_two(acc);
    case 7: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn7_softmax_recip_uses_table_two(acc);
    case 8: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn8_softmax_recip_uses_table_two(acc);
    case 9: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn9_softmax_recip_uses_table_two(acc);
    case 10: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn10_softmax_recip_uses_table_two(acc);
    default: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn11_softmax_recip_uses_table_two(acc);
  }
}

static inline int hgtxr_e2e_hgpipe_softmax_recip_raw(int acc, int block_idx) {
#pragma HLS INLINE
  switch (block_idx % 12) {
    case 0: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn0_softmax_recip_int(acc);
    case 1: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn1_softmax_recip_int(acc);
    case 2: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn2_softmax_recip_int(acc);
    case 3: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn3_softmax_recip_int(acc);
    case 4: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn4_softmax_recip_int(acc);
    case 5: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn5_softmax_recip_int(acc);
    case 6: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn6_softmax_recip_int(acc);
    case 7: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn7_softmax_recip_int(acc);
    case 8: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn8_softmax_recip_int(acc);
    case 9: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn9_softmax_recip_int(acc);
    case 10: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn10_softmax_recip_int(acc);
    default: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn11_softmax_recip_int(acc);
  }
}

static inline int hgtxr_e2e_hgpipe_softmax_requant_raw(int exp_score,
                                                       int recip,
                                                       bool in_table_two,
                                                       int block_idx) {
#pragma HLS INLINE
  switch (block_idx % 12) {
    case 0: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn0_softmax_requant_uint3_int(exp_score, recip, in_table_two);
    case 1: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn1_softmax_requant_uint3_int(exp_score, recip, in_table_two);
    case 2: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn2_softmax_requant_uint3_int(exp_score, recip, in_table_two);
    case 3: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn3_softmax_requant_uint3_int(exp_score, recip, in_table_two);
    case 4: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn4_softmax_requant_uint3_int(exp_score, recip, in_table_two);
    case 5: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn5_softmax_requant_uint3_int(exp_score, recip, in_table_two);
    case 6: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn6_softmax_requant_uint3_int(exp_score, recip, in_table_two);
    case 7: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn7_softmax_requant_uint3_int(exp_score, recip, in_table_two);
    case 8: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn8_softmax_requant_uint3_int(exp_score, recip, in_table_two);
    case 9: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn9_softmax_requant_uint3_int(exp_score, recip, in_table_two);
    case 10: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn10_softmax_requant_uint3_int(exp_score, recip, in_table_two);
    default: return hgtxr::cyclic_transformer::hgtxr_hgpipe_attn11_softmax_requant_uint3_int(exp_score, recip, in_table_two);
  }
}

static inline int hgtxr_e2e_hgpipe_lnq_mean_raw(int sum,
                                                int block_idx,
                                                bool mlp_layernorm) {
#pragma HLS INLINE
  switch (block_idx % 12) {
    case 0: return mlp_layernorm ? hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp0_lnq_mean_int(sum) : hgtxr::cyclic_transformer::hgtxr_hgpipe_attn0_lnq_mean_int(sum);
    case 1: return mlp_layernorm ? hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp1_lnq_mean_int(sum) : hgtxr::cyclic_transformer::hgtxr_hgpipe_attn1_lnq_mean_int(sum);
    case 2: return mlp_layernorm ? hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp2_lnq_mean_int(sum) : hgtxr::cyclic_transformer::hgtxr_hgpipe_attn2_lnq_mean_int(sum);
    case 3: return mlp_layernorm ? hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp3_lnq_mean_int(sum) : hgtxr::cyclic_transformer::hgtxr_hgpipe_attn3_lnq_mean_int(sum);
    case 4: return mlp_layernorm ? hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp4_lnq_mean_int(sum) : hgtxr::cyclic_transformer::hgtxr_hgpipe_attn4_lnq_mean_int(sum);
    case 5: return mlp_layernorm ? hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp5_lnq_mean_int(sum) : hgtxr::cyclic_transformer::hgtxr_hgpipe_attn5_lnq_mean_int(sum);
    case 6: return mlp_layernorm ? hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp6_lnq_mean_int(sum) : hgtxr::cyclic_transformer::hgtxr_hgpipe_attn6_lnq_mean_int(sum);
    case 7: return mlp_layernorm ? hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp7_lnq_mean_int(sum) : hgtxr::cyclic_transformer::hgtxr_hgpipe_attn7_lnq_mean_int(sum);
    case 8: return mlp_layernorm ? hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp8_lnq_mean_int(sum) : hgtxr::cyclic_transformer::hgtxr_hgpipe_attn8_lnq_mean_int(sum);
    case 9: return mlp_layernorm ? hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp9_lnq_mean_int(sum) : hgtxr::cyclic_transformer::hgtxr_hgpipe_attn9_lnq_mean_int(sum);
    case 10: return mlp_layernorm ? hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp10_lnq_mean_int(sum) : hgtxr::cyclic_transformer::hgtxr_hgpipe_attn10_lnq_mean_int(sum);
    default: return mlp_layernorm ? hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp11_lnq_mean_int(sum) : hgtxr::cyclic_transformer::hgtxr_hgpipe_attn11_lnq_mean_int(sum);
  }
}

static inline int hgtxr_e2e_hgpipe_lnq_rsqrt_raw(int variance_sum,
                                                 int block_idx,
                                                 bool mlp_layernorm) {
#pragma HLS INLINE
  switch (block_idx % 12) {
    case 0: return mlp_layernorm ? hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp0_lnq_rsqrt128_int(variance_sum) : hgtxr::cyclic_transformer::hgtxr_hgpipe_attn0_lnq_rsqrt128_int(variance_sum);
    case 1: return mlp_layernorm ? hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp1_lnq_rsqrt128_int(variance_sum) : hgtxr::cyclic_transformer::hgtxr_hgpipe_attn1_lnq_rsqrt128_int(variance_sum);
    case 2: return mlp_layernorm ? hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp2_lnq_rsqrt128_int(variance_sum) : hgtxr::cyclic_transformer::hgtxr_hgpipe_attn2_lnq_rsqrt128_int(variance_sum);
    case 3: return mlp_layernorm ? hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp3_lnq_rsqrt128_int(variance_sum) : hgtxr::cyclic_transformer::hgtxr_hgpipe_attn3_lnq_rsqrt128_int(variance_sum);
    case 4: return mlp_layernorm ? hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp4_lnq_rsqrt128_int(variance_sum) : hgtxr::cyclic_transformer::hgtxr_hgpipe_attn4_lnq_rsqrt128_int(variance_sum);
    case 5: return mlp_layernorm ? hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp5_lnq_rsqrt128_int(variance_sum) : hgtxr::cyclic_transformer::hgtxr_hgpipe_attn5_lnq_rsqrt128_int(variance_sum);
    case 6: return mlp_layernorm ? hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp6_lnq_rsqrt128_int(variance_sum) : hgtxr::cyclic_transformer::hgtxr_hgpipe_attn6_lnq_rsqrt128_int(variance_sum);
    case 7: return mlp_layernorm ? hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp7_lnq_rsqrt128_int(variance_sum) : hgtxr::cyclic_transformer::hgtxr_hgpipe_attn7_lnq_rsqrt128_int(variance_sum);
    case 8: return mlp_layernorm ? hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp8_lnq_rsqrt128_int(variance_sum) : hgtxr::cyclic_transformer::hgtxr_hgpipe_attn8_lnq_rsqrt128_int(variance_sum);
    case 9: return mlp_layernorm ? hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp9_lnq_rsqrt128_int(variance_sum) : hgtxr::cyclic_transformer::hgtxr_hgpipe_attn9_lnq_rsqrt128_int(variance_sum);
    case 10: return mlp_layernorm ? hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp10_lnq_rsqrt128_int(variance_sum) : hgtxr::cyclic_transformer::hgtxr_hgpipe_attn10_lnq_rsqrt128_int(variance_sum);
    default: return mlp_layernorm ? hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp11_lnq_rsqrt128_int(variance_sum) : hgtxr::cyclic_transformer::hgtxr_hgpipe_attn11_lnq_rsqrt128_int(variance_sum);
  }
}

static inline int hgtxr_e2e_hgpipe_lnq_requant_raw(int input_value,
                                                   int mean,
                                                   int rsqrt,
                                                   int lnw,
                                                   long long lnb,
                                                   int block_idx,
                                                   bool mlp_layernorm) {
#pragma HLS INLINE
  switch (block_idx % 12) {
    case 0: return mlp_layernorm ? hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp0_lnq_requant_int(input_value, mean, rsqrt, lnw, lnb) : hgtxr::cyclic_transformer::hgtxr_hgpipe_attn0_lnq_requant_int(input_value, mean, rsqrt, lnw, lnb);
    case 1: return mlp_layernorm ? hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp1_lnq_requant_int(input_value, mean, rsqrt, lnw, lnb) : hgtxr::cyclic_transformer::hgtxr_hgpipe_attn1_lnq_requant_int(input_value, mean, rsqrt, lnw, lnb);
    case 2: return mlp_layernorm ? hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp2_lnq_requant_int(input_value, mean, rsqrt, lnw, lnb) : hgtxr::cyclic_transformer::hgtxr_hgpipe_attn2_lnq_requant_int(input_value, mean, rsqrt, lnw, lnb);
    case 3: return mlp_layernorm ? hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp3_lnq_requant_int(input_value, mean, rsqrt, lnw, lnb) : hgtxr::cyclic_transformer::hgtxr_hgpipe_attn3_lnq_requant_int(input_value, mean, rsqrt, lnw, lnb);
    case 4: return mlp_layernorm ? hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp4_lnq_requant_int(input_value, mean, rsqrt, lnw, lnb) : hgtxr::cyclic_transformer::hgtxr_hgpipe_attn4_lnq_requant_int(input_value, mean, rsqrt, lnw, lnb);
    case 5: return mlp_layernorm ? hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp5_lnq_requant_int(input_value, mean, rsqrt, lnw, lnb) : hgtxr::cyclic_transformer::hgtxr_hgpipe_attn5_lnq_requant_int(input_value, mean, rsqrt, lnw, lnb);
    case 6: return mlp_layernorm ? hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp6_lnq_requant_int(input_value, mean, rsqrt, lnw, lnb) : hgtxr::cyclic_transformer::hgtxr_hgpipe_attn6_lnq_requant_int(input_value, mean, rsqrt, lnw, lnb);
    case 7: return mlp_layernorm ? hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp7_lnq_requant_int(input_value, mean, rsqrt, lnw, lnb) : hgtxr::cyclic_transformer::hgtxr_hgpipe_attn7_lnq_requant_int(input_value, mean, rsqrt, lnw, lnb);
    case 8: return mlp_layernorm ? hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp8_lnq_requant_int(input_value, mean, rsqrt, lnw, lnb) : hgtxr::cyclic_transformer::hgtxr_hgpipe_attn8_lnq_requant_int(input_value, mean, rsqrt, lnw, lnb);
    case 9: return mlp_layernorm ? hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp9_lnq_requant_int(input_value, mean, rsqrt, lnw, lnb) : hgtxr::cyclic_transformer::hgtxr_hgpipe_attn9_lnq_requant_int(input_value, mean, rsqrt, lnw, lnb);
    case 10: return mlp_layernorm ? hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp10_lnq_requant_int(input_value, mean, rsqrt, lnw, lnb) : hgtxr::cyclic_transformer::hgtxr_hgpipe_attn10_lnq_requant_int(input_value, mean, rsqrt, lnw, lnb);
    default: return mlp_layernorm ? hgtxr::cyclic_transformer::hgtxr_hgpipe_mlp11_lnq_requant_int(input_value, mean, rsqrt, lnw, lnb) : hgtxr::cyclic_transformer::hgtxr_hgpipe_attn11_lnq_requant_int(input_value, mean, rsqrt, lnw, lnb);
  }
}

inline void hgtxr_axis_read_frame(hls::stream<HgtxrAxisWord> &axis_in,
                                  hgtxr_data_t frame[HGTXR_HEIGHT][HGTXR_WIDTH],
                                  int read_h = HGTXR_HEIGHT,
                                  int read_w = HGTXR_WIDTH) {
#pragma HLS INLINE off
  for (int y = 0; y < read_h; ++y) {
#pragma HLS LOOP_TRIPCOUNT min=256 max=256
    for (int x = 0; x < read_w; ++x) {
#pragma HLS LOOP_TRIPCOUNT min=256 max=256
#pragma HLS PIPELINE II=1
      frame[y][x] = hgtxr_axis_to_data(axis_in.read());
    }
  }
}

inline void hgtxr_conv_patch_embedding(
    const hgtxr_data_t frame[HGTXR_HEIGHT][HGTXR_WIDTH],
    const volatile HgtxrAxiWordT *weights,
    token_buffer_t tokens,
    int active_tokens = kActiveTokens) {
#pragma HLS INLINE off
  for (int t = 0; t < kActiveTokens; ++t) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
    for (int c = 0; c < HGTXR_EMBED; ++c) {
#pragma HLS LOOP_TRIPCOUNT min=192 max=192
#pragma HLS PIPELINE II=1
      tokens[t][c] = 0;
    }
  }

#if HGTXR_E2E_PATCH_TOKEN_LOOP
  for (int token = 0; token < active_tokens; ++token) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=64
    const int gy = token / HGTXR_E2E_PATCH_GRID_W;
    const int gx = token - gy * HGTXR_E2E_PATCH_GRID_W;
    for (int c = 0; c < HGTXR_EMBED; ++c) {
#pragma HLS LOOP_TRIPCOUNT min=192 max=192
      hgtxr_acc_t acc_lane[kPatchPar];
#pragma HLS ARRAY_PARTITION variable=acc_lane complete dim=1
      for (int lane = 0; lane < kPatchPar; ++lane) {
#pragma HLS UNROLL
        acc_lane[lane] = 0;
      }
      for (int p0 = 0; p0 < kPatchElems; p0 += kPatchPar) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
#pragma HLS PIPELINE II=1
        for (int lane = 0; lane < kPatchPar; ++lane) {
#pragma HLS UNROLL
          const int p = p0 + lane;
          const int py = p / HGTXR_PATCH;
          const int px = p - py * HGTXR_PATCH;
          hgtxr_data_t w = hgtxr_e2e_load_weight(
              weights, kPatchWeightElemBase + c * kPatchElems + p);
          hgtxr_data_t pix =
              frame[gy * HGTXR_PATCH + py][gx * HGTXR_PATCH + px];
          acc_lane[lane] += hgtxr_e2e_patch_mul(pix, w);
        }
      }
      hgtxr_acc_t acc = 0;
      for (int lane = 0; lane < kPatchPar; ++lane) {
#pragma HLS UNROLL
        acc += acc_lane[lane];
      }
      hgtxr_data_t pos =
          hgtxr_e2e_data_from_scaled_int_ct<64>((c & 15) - 8);
      tokens[token][c] =
          static_cast<hgtxr_data_t>(hgtxr_e2e_acc_div_int_ct<32>(acc)) +
          pos;
    }
  }
#else
  for (int gy = 0; gy < HGTXR_E2E_PATCH_GRID_H; ++gy) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=16
    for (int gx = 0; gx < HGTXR_E2E_PATCH_GRID_W; ++gx) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=16
      const int token = gy * HGTXR_E2E_PATCH_GRID_W + gx;
      if (token < active_tokens) {
        for (int c = 0; c < HGTXR_EMBED; ++c) {
#pragma HLS LOOP_TRIPCOUNT min=192 max=192
          hgtxr_acc_t acc_lane[kPatchPar];
#pragma HLS ARRAY_PARTITION variable=acc_lane complete dim=1
          for (int lane = 0; lane < kPatchPar; ++lane) {
#pragma HLS UNROLL
            acc_lane[lane] = 0;
          }
          for (int p0 = 0; p0 < kPatchElems; p0 += kPatchPar) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
#pragma HLS PIPELINE II=1
            for (int lane = 0; lane < kPatchPar; ++lane) {
#pragma HLS UNROLL
              const int p = p0 + lane;
              const int py = p / HGTXR_PATCH;
              const int px = p - py * HGTXR_PATCH;
              hgtxr_data_t w = hgtxr_e2e_load_weight(
                  weights, kPatchWeightElemBase + c * kPatchElems + p);
              hgtxr_data_t pix =
                  frame[gy * HGTXR_PATCH + py][gx * HGTXR_PATCH + px];
              acc_lane[lane] += hgtxr_e2e_patch_mul(pix, w);
            }
          }
          hgtxr_acc_t acc = 0;
          for (int lane = 0; lane < kPatchPar; ++lane) {
#pragma HLS UNROLL
            acc += acc_lane[lane];
          }
          hgtxr_data_t pos =
              hgtxr_e2e_data_from_scaled_int_ct<64>((c & 15) - 8);
          tokens[token][c] =
              static_cast<hgtxr_data_t>(hgtxr_e2e_acc_div_int_ct<32>(acc)) +
              pos;
        }
      }
    }
  }
#endif
}

inline void hgtxr_event_conv_patch_embedding(
    const hgtxr_data_t frame[HGTXR_HEIGHT][HGTXR_WIDTH],
    const volatile HgtxrAxiWordT *weights,
    token_buffer_t tokens,
    int active_tokens = kActiveTokens) {
#pragma HLS INLINE off
  for (int t = 0; t < kActiveTokens; ++t) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
    for (int c = 0; c < HGTXR_EMBED; ++c) {
#pragma HLS LOOP_TRIPCOUNT min=192 max=192
#pragma HLS PIPELINE II=1
      tokens[t][c] = 0;
    }
  }

#if HGTXR_E2E_PATCH_TOKEN_LOOP
  for (int token = 0; token < active_tokens; ++token) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=64
    const int gy = token / HGTXR_E2E_PATCH_GRID_W;
    const int gx = token - gy * HGTXR_E2E_PATCH_GRID_W;
    for (int c = 0; c < HGTXR_EMBED; ++c) {
#pragma HLS LOOP_TRIPCOUNT min=192 max=192
      hgtxr_acc_t acc_lane[kPatchPar];
#pragma HLS ARRAY_PARTITION variable=acc_lane complete dim=1
      for (int lane = 0; lane < kPatchPar; ++lane) {
#pragma HLS UNROLL
        acc_lane[lane] = 0;
      }
      for (int p0 = 0; p0 < kPatchElems; p0 += kPatchPar) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
#pragma HLS PIPELINE II=1
        for (int lane = 0; lane < kPatchPar; ++lane) {
#pragma HLS UNROLL
          const int p = p0 + lane;
          const int py = p / HGTXR_PATCH;
          const int px = p - py * HGTXR_PATCH;
          const int y = gy * HGTXR_PATCH + py;
          const int x = gx * HGTXR_PATCH + px;
          const hgtxr_data_t cur = frame[y][x];
          hgtxr_data_t ref = 0;
          if (x > 0) {
            ref = frame[y][x - 1];
          } else if (y > 0) {
            ref = frame[y - 1][x];
          }
          const hgtxr_data_t event_delta = cur - ref;
          hgtxr_data_t w = hgtxr_e2e_load_weight(
              weights, kEventPatchWeightElemBase + c * kPatchElems + p);
          acc_lane[lane] += hgtxr_e2e_patch_mul(event_delta, w);
        }
      }
      hgtxr_acc_t acc = 0;
      for (int lane = 0; lane < kPatchPar; ++lane) {
#pragma HLS UNROLL
        acc += acc_lane[lane];
      }
      hgtxr_data_t pos =
          hgtxr_e2e_data_from_scaled_int_ct<64>((c & 7) - 4);
      tokens[token][c] =
          static_cast<hgtxr_data_t>(hgtxr_e2e_acc_div_int_ct<32>(acc)) +
          pos;
    }
  }
#else
  for (int gy = 0; gy < HGTXR_E2E_PATCH_GRID_H; ++gy) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=16
    for (int gx = 0; gx < HGTXR_E2E_PATCH_GRID_W; ++gx) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=16
      const int token = gy * HGTXR_E2E_PATCH_GRID_W + gx;
      if (token < active_tokens) {
        for (int c = 0; c < HGTXR_EMBED; ++c) {
#pragma HLS LOOP_TRIPCOUNT min=192 max=192
          hgtxr_acc_t acc_lane[kPatchPar];
#pragma HLS ARRAY_PARTITION variable=acc_lane complete dim=1
          for (int lane = 0; lane < kPatchPar; ++lane) {
#pragma HLS UNROLL
            acc_lane[lane] = 0;
          }
          for (int p0 = 0; p0 < kPatchElems; p0 += kPatchPar) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
#pragma HLS PIPELINE II=1
            for (int lane = 0; lane < kPatchPar; ++lane) {
#pragma HLS UNROLL
              const int p = p0 + lane;
              const int py = p / HGTXR_PATCH;
              const int px = p - py * HGTXR_PATCH;
              const int y = gy * HGTXR_PATCH + py;
              const int x = gx * HGTXR_PATCH + px;
              const hgtxr_data_t cur = frame[y][x];
              hgtxr_data_t ref = 0;
              if (x > 0) {
                ref = frame[y][x - 1];
              } else if (y > 0) {
                ref = frame[y - 1][x];
              }
              const hgtxr_data_t event_delta = cur - ref;
              hgtxr_data_t w = hgtxr_e2e_load_weight(
                  weights, kEventPatchWeightElemBase + c * kPatchElems + p);
              acc_lane[lane] += hgtxr_e2e_patch_mul(event_delta, w);
            }
          }
          hgtxr_acc_t acc = 0;
          for (int lane = 0; lane < kPatchPar; ++lane) {
#pragma HLS UNROLL
            acc += acc_lane[lane];
          }
          hgtxr_data_t pos =
              hgtxr_e2e_data_from_scaled_int_ct<64>((c & 7) - 4);
          tokens[token][c] =
              static_cast<hgtxr_data_t>(hgtxr_e2e_acc_div_int_ct<32>(acc)) +
              pos;
        }
      }
    }
  }
#endif
}

inline void hgtxr_global_buffer_load(token_buffer_t tokens, HgtxrGlobalBuffer &gb,
                                     int active_tokens = kActiveTokens) {
#pragma HLS INLINE off
  for (int t = 0; t < active_tokens; ++t) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
    for (int c = 0; c < HGTXR_EMBED; ++c) {
#pragma HLS LOOP_TRIPCOUNT min=192 max=192
#pragma HLS PIPELINE II=1
      gb.tokens[t][c] = tokens[t][c];
    }
  }
}

inline void hgtxr_e2e_layernorm(HgtxrGlobalBuffer &gb,
                                const volatile HgtxrAxiWordT *weights,
                                int gamma_base,
                                int beta_base,
                                int block_idx,
                                bool mlp_layernorm,
                                int active_tokens = kActiveTokens) {
#pragma HLS INLINE off
#if HGTXR_E2E_USE_HGPIPE_INT_LAYERNORMQ
  const long long bias_unit =
      1LL << HGTXR_E2E_HGPIPE_LAYERNORM_BIAS_SHIFT;
#if HGTXR_E2E_LN_PARAM_CACHE
  HgtxrAxiWordT gamma_cache[kLnParamWords];
  HgtxrAxiWordT beta_cache[kLnParamWords];
#pragma HLS ARRAY_PARTITION variable=gamma_cache complete dim=1
#pragma HLS ARRAY_PARTITION variable=beta_cache complete dim=1
  const int gamma_word_base = gamma_base / kWeightLanes;
  const int beta_word_base = beta_base / kWeightLanes;
  for (int w = 0; w < kLnParamWords; ++w) {
#pragma HLS LOOP_TRIPCOUNT min=3 max=3
#pragma HLS PIPELINE II=1
    gamma_cache[w] =
        hgtxr_e2e_load_weight_word_prefetched(weights, gb, block_idx,
                                              gamma_word_base + w);
    beta_cache[w] =
        hgtxr_e2e_load_weight_word_prefetched(weights, gb, block_idx,
                                              beta_word_base + w);
  }
#endif
  for (int t = 0; t < active_tokens; ++t) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
#if HGTXR_E2E_STAGE_LAYERNORM_WRITE
    hgtxr_data_t norm_row[HGTXR_EMBED];
#pragma HLS bind_storage variable=norm_row type=ram_2p impl=lutram
#endif
    int sum_raw = 0;
    for (int c = 0; c < HGTXR_EMBED; ++c) {
#pragma HLS LOOP_TRIPCOUNT min=192 max=192
#pragma HLS PIPELINE II=1
      int raw = hgtxr_e2e_data_to_scaled_int_ct<
          HGTXR_E2E_HGPIPE_LAYERNORM_INPUT_SCALE>(gb.tokens[t][c]);
      sum_raw += raw;
    }
    int mean_raw = hgtxr_e2e_hgpipe_lnq_mean_raw(
        sum_raw, block_idx, mlp_layernorm);

    int variance_sum = 0;
    for (int c = 0; c < HGTXR_EMBED; ++c) {
#pragma HLS LOOP_TRIPCOUNT min=192 max=192
#pragma HLS PIPELINE II=1
      int raw = hgtxr_e2e_data_to_scaled_int_ct<
          HGTXR_E2E_HGPIPE_LAYERNORM_INPUT_SCALE>(gb.tokens[t][c]);
      int diff = raw - mean_raw;
      variance_sum += diff * diff;
    }
    int rsqrt = hgtxr_e2e_hgpipe_lnq_rsqrt_raw(
        variance_sum, block_idx, mlp_layernorm);

    for (int c = 0; c < HGTXR_EMBED; ++c) {
#pragma HLS LOOP_TRIPCOUNT min=192 max=192
#pragma HLS PIPELINE II=1
      int raw = hgtxr_e2e_data_to_scaled_int_ct<
          HGTXR_E2E_HGPIPE_LAYERNORM_INPUT_SCALE>(gb.tokens[t][c]);
#if HGTXR_E2E_LN_PARAM_CACHE
      int gamma_raw = hgtxr_e2e_load_weight_raw_from_cache(gamma_cache, c);
      int beta_raw = hgtxr_e2e_load_weight_raw_from_cache(beta_cache, c);
#else
      int gamma_raw = hgtxr_e2e_load_weight_raw(weights, gamma_base + c);
      int beta_raw = hgtxr_e2e_load_weight_raw(weights, beta_base + c);
#endif
      long long lnb = static_cast<long long>(beta_raw) * bias_unit;
      int quantized = hgtxr_e2e_hgpipe_lnq_requant_raw(
          raw, mean_raw, rsqrt, gamma_raw, lnb, block_idx, mlp_layernorm);
#if HGTXR_E2E_STAGE_LAYERNORM_WRITE
      norm_row[c] = hgtxr_e2e_data_from_scaled_int_ct<
          HGTXR_E2E_HGPIPE_LAYERNORM_OUTPUT_SCALE>(quantized);
#else
      gb.norm[t][c] = hgtxr_e2e_data_from_scaled_int_ct<
          HGTXR_E2E_HGPIPE_LAYERNORM_OUTPUT_SCALE>(quantized);
#endif
    }
#if HGTXR_E2E_STAGE_LAYERNORM_WRITE
    for (int c = 0; c < HGTXR_EMBED; ++c) {
#pragma HLS LOOP_TRIPCOUNT min=192 max=192
#pragma HLS PIPELINE II=1
      gb.norm[t][c] = norm_row[c];
    }
#endif
  }
#else
  (void)block_idx;
  (void)mlp_layernorm;
#if HGTXR_E2E_LN_PARAM_CACHE
  HgtxrAxiWordT gamma_cache[kLnParamWords];
  HgtxrAxiWordT beta_cache[kLnParamWords];
#pragma HLS ARRAY_PARTITION variable=gamma_cache complete dim=1
#pragma HLS ARRAY_PARTITION variable=beta_cache complete dim=1
  const int gamma_word_base = gamma_base / kWeightLanes;
  const int beta_word_base = beta_base / kWeightLanes;
  for (int w = 0; w < kLnParamWords; ++w) {
#pragma HLS LOOP_TRIPCOUNT min=3 max=3
#pragma HLS PIPELINE II=1
    gamma_cache[w] =
        hgtxr_e2e_load_weight_word_prefetched(weights, gb, block_idx,
                                              gamma_word_base + w);
    beta_cache[w] =
        hgtxr_e2e_load_weight_word_prefetched(weights, gb, block_idx,
                                              beta_word_base + w);
  }
#endif
  for (int t = 0; t < active_tokens; ++t) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
#if HGTXR_E2E_STAGE_LAYERNORM_WRITE
    hgtxr_data_t norm_row[HGTXR_EMBED];
#pragma HLS bind_storage variable=norm_row type=ram_2p impl=lutram
#endif
    hgtxr_acc_t mean = 0;
    for (int c = 0; c < HGTXR_EMBED; ++c) {
#pragma HLS LOOP_TRIPCOUNT min=192 max=192
#pragma HLS PIPELINE II=1
      mean += gb.tokens[t][c];
    }
    mean /= static_cast<hgtxr_acc_t>(HGTXR_EMBED);

    hgtxr_acc_t var = 0;
    for (int c = 0; c < HGTXR_EMBED; ++c) {
#pragma HLS LOOP_TRIPCOUNT min=192 max=192
#pragma HLS PIPELINE II=1
      hgtxr_acc_t d = static_cast<hgtxr_acc_t>(gb.tokens[t][c]) - mean;
      var += hgtxr_e2e_layernorm_mul_acc(d, d);
    }
    hgtxr_acc_t var_mean = var / static_cast<hgtxr_acc_t>(HGTXR_EMBED);
#if HGTXR_E2E_USE_ONCHIP_NONLINEAR_ROM
    hgtxr_data_t inv_std = hgtxr_e2e_layernorm_rsqrt_rom(var_mean);
#else
    hgtxr_data_t inv_std = hgtxr::cyclic_transformer::hgtxr_rsqrt_approx(var_mean);
#endif

    for (int c = 0; c < HGTXR_EMBED; ++c) {
#pragma HLS LOOP_TRIPCOUNT min=192 max=192
#pragma HLS PIPELINE II=1
#if HGTXR_E2E_LN_PARAM_CACHE
      hgtxr_data_t gamma = hgtxr_e2e_load_weight_from_cache(gamma_cache, c);
      hgtxr_data_t beta = hgtxr_e2e_load_weight_from_cache(beta_cache, c);
#else
      hgtxr_data_t gamma = hgtxr_e2e_load_weight(weights, gamma_base + c);
      hgtxr_data_t beta = hgtxr_e2e_load_weight(weights, beta_base + c);
#endif
      hgtxr_acc_t centered = static_cast<hgtxr_acc_t>(gb.tokens[t][c]) - mean;
      hgtxr_acc_t scaled = hgtxr_e2e_layernorm_mul_acc(
          centered, static_cast<hgtxr_acc_t>(inv_std));
      hgtxr_acc_t affine = hgtxr_e2e_layernorm_mul_acc(
          scaled, static_cast<hgtxr_acc_t>(gamma));
#if HGTXR_E2E_STAGE_LAYERNORM_WRITE
      norm_row[c] = static_cast<hgtxr_data_t>(affine) + beta;
#else
      gb.norm[t][c] = static_cast<hgtxr_data_t>(affine) + beta;
#endif
    }
#if HGTXR_E2E_STAGE_LAYERNORM_WRITE
    for (int c = 0; c < HGTXR_EMBED; ++c) {
#pragma HLS LOOP_TRIPCOUNT min=192 max=192
#pragma HLS PIPELINE II=1
      gb.norm[t][c] = norm_row[c];
    }
#endif
  }
#endif
}

inline void hgtxr_e2e_project_qkv(HgtxrGlobalBuffer &gb,
                                  const volatile HgtxrAxiWordT *weights,
                                  int wq_base,
                                  int wk_base,
                                  int wv_base,
                                  int block_idx,
                                  int active_tokens = kActiveTokens) {
#pragma HLS INLINE off
#if HGTXR_E2E_QKV_WEIGHT_CACHE
  HgtxrAxiWordT q_weight_cache[kDenseModelWords];
  HgtxrAxiWordT k_weight_cache[kDenseModelWords];
  HgtxrAxiWordT v_weight_cache[kDenseModelWords];
#if HGTXR_E2E_URAM_QKV_WEIGHT_CACHE
#pragma HLS bind_storage variable=q_weight_cache type=ram_2p impl=uram
#pragma HLS bind_storage variable=k_weight_cache type=ram_2p impl=uram
#pragma HLS bind_storage variable=v_weight_cache type=ram_2p impl=uram
#else
#pragma HLS bind_storage variable=q_weight_cache type=ram_2p impl=bram
#pragma HLS bind_storage variable=k_weight_cache type=ram_2p impl=bram
#pragma HLS bind_storage variable=v_weight_cache type=ram_2p impl=bram
#endif
  const int wq_word_base = wq_base / kWeightLanes;
  const int wk_word_base = wk_base / kWeightLanes;
  const int wv_word_base = wv_base / kWeightLanes;
  for (int w = 0; w < kDenseModelWords; ++w) {
#pragma HLS LOOP_TRIPCOUNT min=576 max=576
#pragma HLS PIPELINE II=1
    q_weight_cache[w] =
        hgtxr_e2e_load_weight_word_prefetched(weights, gb, block_idx,
                                              wq_word_base + w);
  }
  for (int w = 0; w < kDenseModelWords; ++w) {
#pragma HLS LOOP_TRIPCOUNT min=576 max=576
#pragma HLS PIPELINE II=1
    k_weight_cache[w] =
        hgtxr_e2e_load_weight_word_prefetched(weights, gb, block_idx,
                                              wk_word_base + w);
  }
  for (int w = 0; w < kDenseModelWords; ++w) {
#pragma HLS LOOP_TRIPCOUNT min=576 max=576
#pragma HLS PIPELINE II=1
    v_weight_cache[w] =
        hgtxr_e2e_load_weight_word_prefetched(weights, gb, block_idx,
                                              wv_word_base + w);
  }
#endif
#if HGTXR_E2E_DENSE_TOKEN_PAR > 1
  for (int t0 = 0; t0 < active_tokens; t0 += kDenseTokenPar) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
    for (int o0 = 0; o0 < HGTXR_EMBED; o0 += kDensePar) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=192
      hgtxr_acc_t q_acc[kDenseTokenPar][kDensePar];
      hgtxr_acc_t k_acc[kDenseTokenPar][kDensePar];
      hgtxr_acc_t v_acc[kDenseTokenPar][kDensePar];
#pragma HLS ARRAY_PARTITION variable=q_acc complete dim=1
#pragma HLS ARRAY_PARTITION variable=q_acc complete dim=2
#pragma HLS ARRAY_PARTITION variable=k_acc complete dim=1
#pragma HLS ARRAY_PARTITION variable=k_acc complete dim=2
#pragma HLS ARRAY_PARTITION variable=v_acc complete dim=1
#pragma HLS ARRAY_PARTITION variable=v_acc complete dim=2
      for (int tl = 0; tl < kDenseTokenPar; ++tl) {
#pragma HLS UNROLL
        for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
          q_acc[tl][lane] = 0;
          k_acc[tl][lane] = 0;
          v_acc[tl][lane] = 0;
        }
      }
      for (int c = 0; c < HGTXR_EMBED; ++c) {
#pragma HLS LOOP_TRIPCOUNT min=192 max=192
#pragma HLS PIPELINE II=1
        hgtxr_data_t wq_vec[kDensePar];
        hgtxr_data_t wk_vec[kDensePar];
        hgtxr_data_t wv_vec[kDensePar];
#pragma HLS ARRAY_PARTITION variable=wq_vec complete dim=1
#pragma HLS ARRAY_PARTITION variable=wk_vec complete dim=1
#pragma HLS ARRAY_PARTITION variable=wv_vec complete dim=1
#if HGTXR_E2E_QKV_WEIGHT_CACHE
        const int weight_elem = c * HGTXR_EMBED + o0;
        hgtxr_e2e_load_weight_vec_from_cache(q_weight_cache, weight_elem,
                                             wq_vec);
        hgtxr_e2e_load_weight_vec_from_cache(k_weight_cache, weight_elem,
                                             wk_vec);
        hgtxr_e2e_load_weight_vec_from_cache(v_weight_cache, weight_elem,
                                             wv_vec);
#else
        hgtxr_e2e_load_weight_vec(weights, wq_base + c * HGTXR_EMBED + o0,
                                  wq_vec);
        hgtxr_e2e_load_weight_vec(weights, wk_base + c * HGTXR_EMBED + o0,
                                  wk_vec);
        hgtxr_e2e_load_weight_vec(weights, wv_base + c * HGTXR_EMBED + o0,
                                  wv_vec);
#endif
        for (int tl = 0; tl < kDenseTokenPar; ++tl) {
#pragma HLS UNROLL
          const int t = t0 + tl;
          const bool valid_t = t < active_tokens;
          hgtxr_data_t x = valid_t ? gb.norm[t][c] : hgtxr_data_t(0);
          for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
            int o = o0 + lane;
            if (o < HGTXR_EMBED) {
              q_acc[tl][lane] += hgtxr_e2e_core_lane_mul(x, wq_vec[lane], lane);
              k_acc[tl][lane] += hgtxr_e2e_core_lane_mul(x, wk_vec[lane], lane);
              v_acc[tl][lane] += hgtxr_e2e_core_lane_mul(x, wv_vec[lane], lane);
            }
          }
        }
      }
      for (int tl = 0; tl < kDenseTokenPar; ++tl) {
#pragma HLS UNROLL
        const int t = t0 + tl;
        const bool valid_t = t < active_tokens;
        for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
          int o = o0 + lane;
          if (valid_t && o < HGTXR_EMBED) {
            gb.q[t][o] = static_cast<hgtxr_data_t>(
                hgtxr_e2e_acc_div_int_ct<HGTXR_E2E_ACC_SCALE>(q_acc[tl][lane]));
            gb.k[t][o] = static_cast<hgtxr_data_t>(
                hgtxr_e2e_acc_div_int_ct<HGTXR_E2E_ACC_SCALE>(k_acc[tl][lane]));
            gb.v[t][o] = static_cast<hgtxr_data_t>(
                hgtxr_e2e_acc_div_int_ct<HGTXR_E2E_ACC_SCALE>(v_acc[tl][lane]));
          }
        }
      }
    }
  }
#else
  for (int t = 0; t < active_tokens; ++t) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
    for (int o0 = 0; o0 < HGTXR_EMBED; o0 += kDensePar) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=192
      hgtxr_acc_t q_acc[kDensePar];
      hgtxr_acc_t k_acc[kDensePar];
      hgtxr_acc_t v_acc[kDensePar];
#pragma HLS ARRAY_PARTITION variable=q_acc complete dim=1
#pragma HLS ARRAY_PARTITION variable=k_acc complete dim=1
#pragma HLS ARRAY_PARTITION variable=v_acc complete dim=1
      for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
        q_acc[lane] = 0;
        k_acc[lane] = 0;
        v_acc[lane] = 0;
      }
      for (int c = 0; c < HGTXR_EMBED; ++c) {
#pragma HLS LOOP_TRIPCOUNT min=192 max=192
#pragma HLS PIPELINE II=1
        hgtxr_data_t x = gb.norm[t][c];
        hgtxr_data_t wq_vec[kDensePar];
        hgtxr_data_t wk_vec[kDensePar];
        hgtxr_data_t wv_vec[kDensePar];
#pragma HLS ARRAY_PARTITION variable=wq_vec complete dim=1
#pragma HLS ARRAY_PARTITION variable=wk_vec complete dim=1
#pragma HLS ARRAY_PARTITION variable=wv_vec complete dim=1
#if HGTXR_E2E_QKV_WEIGHT_CACHE
        const int weight_elem = c * HGTXR_EMBED + o0;
        hgtxr_e2e_load_weight_vec_from_cache(q_weight_cache, weight_elem,
                                             wq_vec);
        hgtxr_e2e_load_weight_vec_from_cache(k_weight_cache, weight_elem,
                                             wk_vec);
        hgtxr_e2e_load_weight_vec_from_cache(v_weight_cache, weight_elem,
                                             wv_vec);
#else
        hgtxr_e2e_load_weight_vec(weights, wq_base + c * HGTXR_EMBED + o0,
                                  wq_vec);
        hgtxr_e2e_load_weight_vec(weights, wk_base + c * HGTXR_EMBED + o0,
                                  wk_vec);
        hgtxr_e2e_load_weight_vec(weights, wv_base + c * HGTXR_EMBED + o0,
                                  wv_vec);
#endif
        for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
          int o = o0 + lane;
          if (o < HGTXR_EMBED) {
            q_acc[lane] += hgtxr_e2e_core_lane_mul(x, wq_vec[lane], lane);
            k_acc[lane] += hgtxr_e2e_core_lane_mul(x, wk_vec[lane], lane);
            v_acc[lane] += hgtxr_e2e_core_lane_mul(x, wv_vec[lane], lane);
          }
        }
      }
      for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
        int o = o0 + lane;
        if (o < HGTXR_EMBED) {
          gb.q[t][o] = static_cast<hgtxr_data_t>(
              hgtxr_e2e_acc_div_int_ct<HGTXR_E2E_ACC_SCALE>(q_acc[lane]));
          gb.k[t][o] = static_cast<hgtxr_data_t>(
              hgtxr_e2e_acc_div_int_ct<HGTXR_E2E_ACC_SCALE>(k_acc[lane]));
          gb.v[t][o] = static_cast<hgtxr_data_t>(
              hgtxr_e2e_acc_div_int_ct<HGTXR_E2E_ACC_SCALE>(v_acc[lane]));
        }
      }
    }
  }
#endif
}

inline void hgtxr_e2e_attention_core(HgtxrGlobalBuffer &gb, int block_idx,
                                     int active_tokens = kActiveTokens) {
#pragma HLS INLINE off
#if HGTXR_E2E_ATTN_QUERY_PAR > 1
  hgtxr_data_t score[kAttnQueryPar][kActiveTokens];
  hgtxr_data_t prob[kAttnQueryPar][kActiveTokens];
  int exp_raw[kAttnQueryPar][kActiveTokens];
#pragma HLS ARRAY_PARTITION variable=score complete dim=1
#pragma HLS ARRAY_PARTITION variable=prob complete dim=1
#pragma HLS ARRAY_PARTITION variable=exp_raw complete dim=1
#if HGTXR_E2E_SMALL_MEM_LUTRAM
#pragma HLS bind_storage variable=score type=ram_2p impl=lutram
#pragma HLS bind_storage variable=prob type=ram_2p impl=lutram
#pragma HLS bind_storage variable=exp_raw type=ram_2p impl=lutram
#else
#pragma HLS bind_storage variable=score type=ram_2p impl=bram
#pragma HLS bind_storage variable=prob type=ram_2p impl=bram
#pragma HLS bind_storage variable=exp_raw type=ram_2p impl=bram
#endif

  for (int h = 0; h < kHeads; ++h) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=3
    const int base = h * kHeadDim;
    for (int tq0 = 0; tq0 < active_tokens; tq0 += kAttnQueryPar) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
      hgtxr_data_t row_max[kAttnQueryPar];
#pragma HLS ARRAY_PARTITION variable=row_max complete dim=1
      for (int ql = 0; ql < kAttnQueryPar; ++ql) {
#pragma HLS UNROLL
        row_max[ql] = static_cast<hgtxr_data_t>(-16);
      }

      for (int tk = 0; tk < active_tokens; ++tk) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
        hgtxr_acc_t lane_acc[kAttnQueryPar][kDensePar];
#pragma HLS ARRAY_PARTITION variable=lane_acc complete dim=1
#pragma HLS ARRAY_PARTITION variable=lane_acc complete dim=2
        for (int ql = 0; ql < kAttnQueryPar; ++ql) {
#pragma HLS UNROLL
          for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
            lane_acc[ql][lane] = 0;
          }
        }
        for (int d0 = 0; d0 < kHeadDim; d0 += kDensePar) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=64
#pragma HLS PIPELINE II=1
          for (int ql = 0; ql < kAttnQueryPar; ++ql) {
#pragma HLS UNROLL
            const int tq = tq0 + ql;
            const bool valid_q = tq < active_tokens;
            for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
              int d = d0 + lane;
              if (valid_q && d < kHeadDim) {
                lane_acc[ql][lane] += hgtxr_e2e_core_lane_mul(
                    gb.q[tq][base + d], gb.k[tk][base + d], lane);
              }
            }
          }
        }
        for (int ql = 0; ql < kAttnQueryPar; ++ql) {
#pragma HLS UNROLL
          const int tq = tq0 + ql;
          hgtxr_acc_t acc = 0;
          for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
            acc += lane_acc[ql][lane];
          }
          hgtxr_data_t s =
              static_cast<hgtxr_data_t>(hgtxr_e2e_acc_div_int_ct<8>(acc));
          if (tq < active_tokens) {
            score[ql][tk] = s;
            if (s > row_max[ql]) {
              row_max[ql] = s;
            }
          }
        }
      }

#if HGTXR_E2E_USE_HGPIPE_INT_SOFTMAXQ
      int row_sum_raw[kAttnQueryPar];
#pragma HLS ARRAY_PARTITION variable=row_sum_raw complete dim=1
      for (int ql = 0; ql < kAttnQueryPar; ++ql) {
#pragma HLS UNROLL
        row_sum_raw[ql] = 0;
      }
      for (int tk = 0; tk < active_tokens; ++tk) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
#pragma HLS PIPELINE II=1
        for (int ql = 0; ql < kAttnQueryPar; ++ql) {
#pragma HLS UNROLL
          const int tq = tq0 + ql;
          if (tq < active_tokens) {
            hgtxr_data_t opposite = row_max[ql] - score[ql][tk];
            int opposite_raw = hgtxr_e2e_data_to_scaled_int_ct<
                HGTXR_E2E_HGPIPE_SOFTMAX_INPUT_SCALE>(opposite);
            if (opposite_raw < 0) {
              opposite_raw = 0;
            }
            int e = hgtxr_e2e_hgpipe_softmax_exp_raw(opposite_raw, block_idx);
            exp_raw[ql][tk] = e;
            row_sum_raw[ql] += e;
          }
        }
      }
      bool in_table_two[kAttnQueryPar];
      int recip[kAttnQueryPar];
#pragma HLS ARRAY_PARTITION variable=in_table_two complete dim=1
#pragma HLS ARRAY_PARTITION variable=recip complete dim=1
      for (int ql = 0; ql < kAttnQueryPar; ++ql) {
#pragma HLS UNROLL
        if (row_sum_raw[ql] == 0) {
          row_sum_raw[ql] = 1;
        }
        in_table_two[ql] =
            hgtxr_e2e_hgpipe_softmax_uses_table_two(row_sum_raw[ql],
                                                    block_idx);
        recip[ql] = hgtxr_e2e_hgpipe_softmax_recip_raw(row_sum_raw[ql],
                                                       block_idx);
      }
      for (int tk = 0; tk < active_tokens; ++tk) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
#pragma HLS PIPELINE II=1
        for (int ql = 0; ql < kAttnQueryPar; ++ql) {
#pragma HLS UNROLL
          const int tq = tq0 + ql;
          if (tq < active_tokens) {
            int qprob = hgtxr_e2e_hgpipe_softmax_requant_raw(
                exp_raw[ql][tk], recip[ql], in_table_two[ql], block_idx);
            prob[ql][tk] = hgtxr_e2e_data_from_scaled_int_ct<
                HGTXR_E2E_HGPIPE_SOFTMAX_PROB_SCALE>(qprob);
          }
        }
      }
#else
      hgtxr_acc_t row_sum[kAttnQueryPar];
#pragma HLS ARRAY_PARTITION variable=row_sum complete dim=1
      for (int ql = 0; ql < kAttnQueryPar; ++ql) {
#pragma HLS UNROLL
        row_sum[ql] = 0;
      }
      for (int tk = 0; tk < active_tokens; ++tk) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
#pragma HLS PIPELINE II=1
        for (int ql = 0; ql < kAttnQueryPar; ++ql) {
#pragma HLS UNROLL
          const int tq = tq0 + ql;
          if (tq < active_tokens) {
            hgtxr_data_t e = hgtxr_e2e_exp_approx(score[ql][tk] - row_max[ql]);
            exp_raw[ql][tk] = 0;
            prob[ql][tk] = e;
            row_sum[ql] += e;
          }
        }
      }
      for (int ql = 0; ql < kAttnQueryPar; ++ql) {
#pragma HLS UNROLL
        if (row_sum[ql] == 0) {
          row_sum[ql] = 1;
        }
      }
      for (int tk = 0; tk < active_tokens; ++tk) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
#pragma HLS PIPELINE II=1
        for (int ql = 0; ql < kAttnQueryPar; ++ql) {
#pragma HLS UNROLL
          const int tq = tq0 + ql;
          if (tq < active_tokens) {
            prob[ql][tk] = static_cast<hgtxr_data_t>(prob[ql][tk] / row_sum[ql]);
          }
        }
      }
#endif

      for (int d0 = 0; d0 < kHeadDim; d0 += kDensePar) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=64
        hgtxr_acc_t acc[kAttnQueryPar][kDensePar];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc complete dim=2
        for (int ql = 0; ql < kAttnQueryPar; ++ql) {
#pragma HLS UNROLL
          for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
            acc[ql][lane] = 0;
          }
        }
        for (int tk = 0; tk < active_tokens; ++tk) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
#pragma HLS PIPELINE II=1
          for (int ql = 0; ql < kAttnQueryPar; ++ql) {
#pragma HLS UNROLL
            const int tq = tq0 + ql;
            hgtxr_data_t p = prob[ql][tk];
            for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
              int d = d0 + lane;
              if (tq < active_tokens && d < kHeadDim) {
                acc[ql][lane] +=
                    hgtxr_e2e_core_lane_mul(p, gb.v[tk][base + d], lane);
              }
            }
          }
        }
        for (int ql = 0; ql < kAttnQueryPar; ++ql) {
#pragma HLS UNROLL
          const int tq = tq0 + ql;
          for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
            int d = d0 + lane;
            if (tq < active_tokens && d < kHeadDim) {
              gb.attn[tq][base + d] =
                  static_cast<hgtxr_data_t>(acc[ql][lane]);
            }
          }
        }
      }
    }
  }
#else
  hgtxr_data_t score[kActiveTokens];
  hgtxr_data_t prob[kActiveTokens];
  int exp_raw[kActiveTokens];
#if HGTXR_E2E_SMALL_MEM_LUTRAM
#pragma HLS bind_storage variable=score type=ram_2p impl=lutram
#pragma HLS bind_storage variable=prob type=ram_2p impl=lutram
#pragma HLS bind_storage variable=exp_raw type=ram_2p impl=lutram
#else
#pragma HLS bind_storage variable=score type=ram_2p impl=bram
#pragma HLS bind_storage variable=prob type=ram_2p impl=bram
#pragma HLS bind_storage variable=exp_raw type=ram_2p impl=bram
#endif

  for (int h = 0; h < kHeads; ++h) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=3
    const int base = h * kHeadDim;
    for (int tq = 0; tq < active_tokens; ++tq) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
      hgtxr_data_t row_max = static_cast<hgtxr_data_t>(-16);
      for (int tk = 0; tk < active_tokens; ++tk) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
        hgtxr_acc_t lane_acc[kDensePar];
#pragma HLS ARRAY_PARTITION variable=lane_acc complete dim=1
        for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
          lane_acc[lane] = 0;
        }
        for (int d0 = 0; d0 < kHeadDim; d0 += kDensePar) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=64
#pragma HLS PIPELINE II=1
          for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
            int d = d0 + lane;
            if (d < kHeadDim) {
              lane_acc[lane] += hgtxr_e2e_core_lane_mul(
                  gb.q[tq][base + d], gb.k[tk][base + d], lane);
            }
          }
        }
        hgtxr_acc_t acc = 0;
        for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
          acc += lane_acc[lane];
        }
        hgtxr_data_t s =
            static_cast<hgtxr_data_t>(hgtxr_e2e_acc_div_int_ct<8>(acc));
        score[tk] = s;
        if (s > row_max) {
          row_max = s;
        }
      }

#if HGTXR_E2E_USE_HGPIPE_INT_SOFTMAXQ
      int row_sum_raw = 0;
      for (int tk = 0; tk < active_tokens; ++tk) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
#pragma HLS PIPELINE II=1
        hgtxr_data_t opposite = row_max - score[tk];
        int opposite_raw = hgtxr_e2e_data_to_scaled_int_ct<
            HGTXR_E2E_HGPIPE_SOFTMAX_INPUT_SCALE>(opposite);
        if (opposite_raw < 0) {
          opposite_raw = 0;
        }
        int e = hgtxr_e2e_hgpipe_softmax_exp_raw(opposite_raw, block_idx);
        exp_raw[tk] = e;
        row_sum_raw += e;
      }
      if (row_sum_raw == 0) {
        row_sum_raw = 1;
      }
      bool in_table_two = hgtxr_e2e_hgpipe_softmax_uses_table_two(row_sum_raw, block_idx);
      int recip = hgtxr_e2e_hgpipe_softmax_recip_raw(row_sum_raw, block_idx);
      for (int tk = 0; tk < active_tokens; ++tk) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
#pragma HLS PIPELINE II=1
        int qprob = hgtxr_e2e_hgpipe_softmax_requant_raw(
            exp_raw[tk], recip, in_table_two, block_idx);
        prob[tk] = hgtxr_e2e_data_from_scaled_int_ct<
            HGTXR_E2E_HGPIPE_SOFTMAX_PROB_SCALE>(qprob);
      }
#else
      hgtxr_acc_t row_sum = 0;
      for (int tk = 0; tk < active_tokens; ++tk) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
#pragma HLS PIPELINE II=1
        hgtxr_data_t e = hgtxr_e2e_exp_approx(score[tk] - row_max);
        exp_raw[tk] = 0;
        prob[tk] = e;
        row_sum += e;
      }
      if (row_sum == 0) {
        row_sum = 1;
      }
      for (int tk = 0; tk < active_tokens; ++tk) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
#pragma HLS PIPELINE II=1
        prob[tk] = static_cast<hgtxr_data_t>(prob[tk] / row_sum);
      }
#endif

      for (int d0 = 0; d0 < kHeadDim; d0 += kDensePar) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=64
        hgtxr_acc_t acc[kDensePar];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
        for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
          acc[lane] = 0;
        }
        for (int tk = 0; tk < active_tokens; ++tk) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
#pragma HLS PIPELINE II=1
          hgtxr_data_t p = prob[tk];
          for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
            int d = d0 + lane;
            if (d < kHeadDim) {
              acc[lane] +=
                  hgtxr_e2e_core_lane_mul(p, gb.v[tk][base + d], lane);
            }
          }
        }
        for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
          int d = d0 + lane;
          if (d < kHeadDim) {
            gb.attn[tq][base + d] = static_cast<hgtxr_data_t>(acc[lane]);
          }
        }
      }
    }
  }
#endif
}

inline void hgtxr_e2e_output_projection(HgtxrGlobalBuffer &gb,
                                        const volatile HgtxrAxiWordT *weights,
                                        int wo_base,
                                        int block_idx,
                                        int active_tokens = kActiveTokens) {
#pragma HLS INLINE off
#if HGTXR_E2E_WEIGHT_VEC_CACHE
  HgtxrAxiWordT wo_weight_cache[kDenseModelWords];
#pragma HLS bind_storage variable=wo_weight_cache type=ram_2p impl=bram
  const int wo_word_base = wo_base / kWeightLanes;
  for (int w = 0; w < kDenseModelWords; ++w) {
#pragma HLS LOOP_TRIPCOUNT min=576 max=576
#pragma HLS PIPELINE II=1
    wo_weight_cache[w] =
        hgtxr_e2e_load_weight_word_prefetched(weights, gb, block_idx,
                                              wo_word_base + w);
  }
#endif
#if HGTXR_E2E_DENSE_TOKEN_PAR > 1
  for (int t0 = 0; t0 < active_tokens; t0 += kDenseTokenPar) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
    for (int o0 = 0; o0 < HGTXR_EMBED; o0 += kDensePar) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=192
      hgtxr_acc_t acc[kDenseTokenPar][kDensePar];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc complete dim=2
      for (int tl = 0; tl < kDenseTokenPar; ++tl) {
#pragma HLS UNROLL
        for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
          acc[tl][lane] = 0;
        }
      }
      for (int c = 0; c < HGTXR_EMBED; ++c) {
#pragma HLS LOOP_TRIPCOUNT min=192 max=192
#pragma HLS PIPELINE II=1
        hgtxr_data_t wo_vec[kDensePar];
#pragma HLS ARRAY_PARTITION variable=wo_vec complete dim=1
#if HGTXR_E2E_WEIGHT_VEC_CACHE
        hgtxr_e2e_load_weight_vec_from_cache(wo_weight_cache,
                                             c * HGTXR_EMBED + o0, wo_vec);
#else
        hgtxr_e2e_load_weight_vec(weights, wo_base + c * HGTXR_EMBED + o0,
                                  wo_vec);
#endif
        for (int tl = 0; tl < kDenseTokenPar; ++tl) {
#pragma HLS UNROLL
          const int t = t0 + tl;
          const bool valid_t = t < active_tokens;
          hgtxr_data_t x = valid_t ? gb.attn[t][c] : hgtxr_data_t(0);
          for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
            int o = o0 + lane;
            if (o < HGTXR_EMBED) {
              acc[tl][lane] += hgtxr_e2e_core_lane_mul(x, wo_vec[lane], lane);
            }
          }
        }
      }
      for (int tl = 0; tl < kDenseTokenPar; ++tl) {
#pragma HLS UNROLL
        const int t = t0 + tl;
        const bool valid_t = t < active_tokens;
        for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
          int o = o0 + lane;
          if (valid_t && o < HGTXR_EMBED) {
            gb.tokens[t][o] += static_cast<hgtxr_data_t>(
                hgtxr_e2e_acc_div_int_ct<HGTXR_E2E_ACC_SCALE>(acc[tl][lane]));
          }
        }
      }
    }
  }
#else
  for (int t = 0; t < active_tokens; ++t) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
    for (int o0 = 0; o0 < HGTXR_EMBED; o0 += kDensePar) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=192
      hgtxr_acc_t acc[kDensePar];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
      for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
        acc[lane] = 0;
      }
      for (int c = 0; c < HGTXR_EMBED; ++c) {
#pragma HLS LOOP_TRIPCOUNT min=192 max=192
#pragma HLS PIPELINE II=1
        hgtxr_data_t x = gb.attn[t][c];
        hgtxr_data_t wo_vec[kDensePar];
#pragma HLS ARRAY_PARTITION variable=wo_vec complete dim=1
#if HGTXR_E2E_WEIGHT_VEC_CACHE
        hgtxr_e2e_load_weight_vec_from_cache(wo_weight_cache,
                                             c * HGTXR_EMBED + o0, wo_vec);
#else
        hgtxr_e2e_load_weight_vec(weights, wo_base + c * HGTXR_EMBED + o0,
                                  wo_vec);
#endif
        for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
          int o = o0 + lane;
          if (o < HGTXR_EMBED) {
            acc[lane] += hgtxr_e2e_core_lane_mul(x, wo_vec[lane], lane);
          }
        }
      }
      for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
        int o = o0 + lane;
        if (o < HGTXR_EMBED) {
          gb.tokens[t][o] += static_cast<hgtxr_data_t>(
              hgtxr_e2e_acc_div_int_ct<HGTXR_E2E_ACC_SCALE>(acc[lane]));
        }
      }
    }
  }
#endif
}

template <int UNIT_ID>
void hgtxr_e2e_attn_unit(HgtxrGlobalBuffer &gb,
                         const volatile HgtxrAxiWordT *weights,
                         int block_idx,
                         int active_tokens = kActiveTokens) {
#pragma HLS INLINE off
  const int block_base = kBlockWeightElemBase + block_idx * kBlockWeightElems;
  (void)UNIT_ID;
  hgtxr_e2e_layernorm(gb, weights, block_base + kBlockLn1Gamma,
                      block_base + kBlockLn1Beta, block_idx, false,
                      active_tokens);
  hgtxr_e2e_project_qkv(gb, weights, block_base + kBlockWq,
                        block_base + kBlockWk, block_base + kBlockWv,
                        block_idx, active_tokens);
  hgtxr_e2e_attention_core(gb, block_idx, active_tokens);
  hgtxr_e2e_output_projection(gb, weights, block_base + kBlockWo,
                              block_idx, active_tokens);
}

template <int UNIT_ID>
void hgtxr_e2e_mlp_unit(HgtxrGlobalBuffer &gb,
                        const volatile HgtxrAxiWordT *weights,
                        int block_idx,
                        int active_tokens = kActiveTokens) {
#pragma HLS INLINE off
  const int block_base = kBlockWeightElemBase + block_idx * kBlockWeightElems;
  const int w1_base = block_base + kBlockW1;
  const int w2_base = block_base + kBlockW2;
  (void)UNIT_ID;

#if HGTXR_E2E_WEIGHT_VEC_CACHE
  HgtxrAxiWordT w1_weight_cache[kMlpW1Words];
#if HGTXR_E2E_MLP_W2_HIDDEN_BANK_CACHE
  HgtxrAxiWordT w2_hidden_bank[kDensePar][kMlpW2HiddenBankWords];
#elif HGTXR_E2E_MLP_W2_HGROUP_BANK_CACHE
#else
  HgtxrAxiWordT w2_weight_cache[kMlpW2Words];
#endif
#pragma HLS bind_storage variable=w1_weight_cache type=ram_2p impl=bram
#if HGTXR_E2E_W1_WEIGHT_CACHE_BANKS > 1
#pragma HLS ARRAY_PARTITION variable=w1_weight_cache cyclic factor=HGTXR_E2E_W1_WEIGHT_CACHE_BANKS dim=1
#endif
#if HGTXR_E2E_MLP_W2_HIDDEN_BANK_CACHE
#pragma HLS bind_storage variable=w2_hidden_bank type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=w2_hidden_bank complete dim=1
#elif HGTXR_E2E_MLP_W2_HGROUP_BANK_CACHE
#else
#pragma HLS bind_storage variable=w2_weight_cache type=ram_2p impl=bram
#if HGTXR_E2E_W2_WEIGHT_CACHE_BANKS > 1
#pragma HLS ARRAY_PARTITION variable=w2_weight_cache cyclic factor=HGTXR_E2E_W2_WEIGHT_CACHE_BANKS dim=1
#endif
#endif
  const int w1_word_base = w1_base / kWeightLanes;
  const int w2_word_base = w2_base / kWeightLanes;
  for (int w = 0; w < kMlpW1Words; ++w) {
#pragma HLS LOOP_TRIPCOUNT min=2304 max=2304
#pragma HLS PIPELINE II=1
    w1_weight_cache[w] =
        hgtxr_e2e_load_weight_word_prefetched(weights, gb, block_idx,
                                              w1_word_base + w);
  }
#if HGTXR_E2E_MLP_W2_HIDDEN_BANK_CACHE
  for (int row = 0; row < kMlpW2HiddenRows; ++row) {
#pragma HLS LOOP_TRIPCOUNT min=24 max=24
    for (int hp = 0; hp < kDensePar; ++hp) {
#pragma HLS LOOP_TRIPCOUNT min=32 max=32
      for (int cw = 0; cw < kMlpW2ColWords; ++cw) {
#pragma HLS LOOP_TRIPCOUNT min=6 max=6
#pragma HLS PIPELINE II=1
        const int h = row * kDensePar + hp;
        const int word_idx = row * kMlpW2ColWords + cw;
        w2_hidden_bank[hp][word_idx] =
            hgtxr_e2e_load_weight_word_prefetched(
                weights, gb, block_idx,
                w2_word_base + (h * HGTXR_EMBED) / kWeightLanes + cw);
      }
    }
  }
#elif HGTXR_E2E_MLP_W2_HGROUP_BANK_CACHE
#else
  for (int w = 0; w < kMlpW2Words; ++w) {
#pragma HLS LOOP_TRIPCOUNT min=2304 max=2304
#pragma HLS PIPELINE II=1
    w2_weight_cache[w] =
        hgtxr_e2e_load_weight_word_prefetched(weights, gb, block_idx,
                                              w2_word_base + w);
  }
#endif
#endif

  hgtxr_e2e_layernorm(gb, weights, block_base + kBlockLn2Gamma,
                      block_base + kBlockLn2Beta, block_idx, true,
                      active_tokens);

#if HGTXR_E2E_MLP_FUSED_W2
#if HGTXR_E2E_MLP_TOKEN_PAR > 1
  for (int t0 = 0; t0 < active_tokens; t0 += kMlpTokenPar) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
    hgtxr_acc_t out_acc[kMlpTokenPar][HGTXR_EMBED];
#pragma HLS ARRAY_PARTITION variable=out_acc complete dim=1
#if HGTXR_E2E_MLP_FUSED_W2_BANKED_ACC
#pragma HLS ARRAY_PARTITION variable=out_acc cyclic factor=kDensePar dim=2
#else
#pragma HLS ARRAY_PARTITION variable=out_acc complete dim=2
#endif
    for (int c0 = 0; c0 < HGTXR_EMBED; c0 += kDensePar) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=192
#pragma HLS PIPELINE II=1
      for (int tl = 0; tl < kMlpTokenPar; ++tl) {
#pragma HLS UNROLL
        for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
          const int c = c0 + lane;
          if (c < HGTXR_EMBED) {
            out_acc[tl][c] = 0;
          }
        }
      }
    }

    for (int h0 = 0; h0 < kFfDim; h0 += kDensePar) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=768
      hgtxr_data_t hidden_vec[kMlpTokenPar][kDensePar];
#pragma HLS ARRAY_PARTITION variable=hidden_vec complete dim=1
#pragma HLS ARRAY_PARTITION variable=hidden_vec complete dim=2
      hgtxr_acc_t acc[kMlpTokenPar][kDensePar];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc complete dim=2
#if HGTXR_E2E_MLP_W2_HGROUP_BANK_CACHE
      HgtxrAxiWordT w2_hgroup_bank[kDensePar][kMlpW2ColWords];
#pragma HLS bind_storage variable=w2_hgroup_bank type=ram_2p impl=lutram
#pragma HLS ARRAY_PARTITION variable=w2_hgroup_bank complete dim=1
#endif
      for (int tl = 0; tl < kMlpTokenPar; ++tl) {
#pragma HLS UNROLL
        for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
          acc[tl][lane] = 0;
        }
      }
#if HGTXR_E2E_MLP_W1_C_PAR > 1
      for (int c0w = 0; c0w < HGTXR_EMBED; c0w += kMlpW1CPar) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=192
#pragma HLS PIPELINE II=1
        hgtxr_data_t w1_vecs[kMlpW1CPar][kDensePar];
#pragma HLS ARRAY_PARTITION variable=w1_vecs complete dim=1
#pragma HLS ARRAY_PARTITION variable=w1_vecs complete dim=2
        for (int cp = 0; cp < kMlpW1CPar; ++cp) {
#pragma HLS UNROLL
          const int c = c0w + cp;
          if (c < HGTXR_EMBED) {
#if HGTXR_E2E_WEIGHT_VEC_CACHE
            hgtxr_e2e_load_weight_vec_from_cache(w1_weight_cache,
                                                 c * kFfDim + h0,
                                                 w1_vecs[cp]);
#else
            hgtxr_e2e_load_weight_vec(weights, w1_base + c * kFfDim + h0,
                                      w1_vecs[cp]);
#endif
          }
        }
        for (int tl = 0; tl < kMlpTokenPar; ++tl) {
#pragma HLS UNROLL
          const int t = t0 + tl;
          const bool valid_t = t < active_tokens;
          for (int cp = 0; cp < kMlpW1CPar; ++cp) {
#pragma HLS UNROLL
            const int c = c0w + cp;
            hgtxr_data_t x =
                (valid_t && c < HGTXR_EMBED) ? gb.norm[t][c] : hgtxr_data_t(0);
            for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
              const int h = h0 + lane;
              if (h < kFfDim && c < HGTXR_EMBED) {
                acc[tl][lane] +=
                    hgtxr_e2e_core_lane_mul(x, w1_vecs[cp][lane], lane);
              }
            }
          }
        }
      }
#else
      for (int c = 0; c < HGTXR_EMBED; ++c) {
#pragma HLS LOOP_TRIPCOUNT min=192 max=192
#pragma HLS PIPELINE II=1
        hgtxr_data_t w1_vec[kDensePar];
#pragma HLS ARRAY_PARTITION variable=w1_vec complete dim=1
#if HGTXR_E2E_WEIGHT_VEC_CACHE
        hgtxr_e2e_load_weight_vec_from_cache(w1_weight_cache,
                                             c * kFfDim + h0, w1_vec);
#else
        hgtxr_e2e_load_weight_vec(weights, w1_base + c * kFfDim + h0,
                                  w1_vec);
#endif
        for (int tl = 0; tl < kMlpTokenPar; ++tl) {
#pragma HLS UNROLL
          const int t = t0 + tl;
          const bool valid_t = t < active_tokens;
          hgtxr_data_t x = valid_t ? gb.norm[t][c] : hgtxr_data_t(0);
          for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
            const int h = h0 + lane;
            if (h < kFfDim) {
              acc[tl][lane] += hgtxr_e2e_core_lane_mul(x, w1_vec[lane], lane);
            }
          }
        }
      }
#endif
      for (int tl = 0; tl < kMlpTokenPar; ++tl) {
#pragma HLS UNROLL
        const int t = t0 + tl;
        const bool valid_t = t < active_tokens;
        for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
          const int h = h0 + lane;
          if (valid_t && h < kFfDim) {
            hidden_vec[tl][lane] = hgtxr_e2e_gelu(static_cast<hgtxr_data_t>(
                hgtxr_e2e_acc_div_int_ct<HGTXR_E2E_ACC_SCALE>(acc[tl][lane])),
                block_idx);
#if !HGTXR_E2E_MLP_FUSED_W2_SKIP_HIDDEN_STORE
            gb.hidden[t][h] = hidden_vec[tl][lane];
#endif
          } else {
            hidden_vec[tl][lane] = hgtxr_data_t(0);
          }
        }
      }

#if HGTXR_E2E_MLP_W2_HGROUP_BANK_CACHE
      for (int hp = 0; hp < kDensePar; ++hp) {
#pragma HLS LOOP_TRIPCOUNT min=32 max=32
        for (int cw = 0; cw < kMlpW2ColWords; ++cw) {
#pragma HLS LOOP_TRIPCOUNT min=3 max=3
#pragma HLS PIPELINE II=1
          const int h = h0 + hp;
          w2_hgroup_bank[hp][cw] =
              hgtxr_e2e_load_weight_word_prefetched(
                  weights, gb, block_idx,
                  w2_word_base + (h * HGTXR_EMBED) / kWeightLanes + cw);
        }
      }
#endif

#if HGTXR_E2E_MLP_FUSED_W2_HP_PAR > 1
      for (int hp0 = 0; hp0 < kDensePar; hp0 += kMlpFusedW2HpPar) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=64
        for (int c0 = 0; c0 < HGTXR_EMBED; c0 += kDensePar) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=192
#pragma HLS PIPELINE II=1
          hgtxr_acc_t contrib[kMlpTokenPar][kDensePar];
#pragma HLS ARRAY_PARTITION variable=contrib complete dim=1
#pragma HLS ARRAY_PARTITION variable=contrib complete dim=2
          for (int tl = 0; tl < kMlpTokenPar; ++tl) {
#pragma HLS UNROLL
            for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
              contrib[tl][lane] = 0;
            }
          }
          for (int hp_lane = 0; hp_lane < kMlpFusedW2HpPar; ++hp_lane) {
#pragma HLS UNROLL
            const int hp = hp0 + hp_lane;
            const int h = h0 + hp;
            hgtxr_data_t w2_vec[kDensePar];
#pragma HLS ARRAY_PARTITION variable=w2_vec complete dim=1
#if HGTXR_E2E_WEIGHT_VEC_CACHE
#if HGTXR_E2E_MLP_W2_HIDDEN_BANK_CACHE
            hgtxr_e2e_load_w2_vec_from_hidden_bank(w2_hidden_bank, h, c0,
                                                   w2_vec);
#elif HGTXR_E2E_MLP_W2_HGROUP_BANK_CACHE
            hgtxr_e2e_load_w2_vec_from_hgroup_bank(w2_hgroup_bank, h, c0,
                                                   w2_vec);
#else
            hgtxr_e2e_load_weight_vec_from_cache(w2_weight_cache,
                                                 h * HGTXR_EMBED + c0, w2_vec);
#endif
#else
            hgtxr_e2e_load_weight_vec(weights, w2_base + h * HGTXR_EMBED + c0,
                                      w2_vec);
#endif
            for (int tl = 0; tl < kMlpTokenPar; ++tl) {
#pragma HLS UNROLL
              hgtxr_data_t x = hidden_vec[tl][hp];
              for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
                const int c = c0 + lane;
                if (h < kFfDim && c < HGTXR_EMBED) {
                  contrib[tl][lane] +=
                      hgtxr_e2e_mlp_w2_hp_lane_mul(x, w2_vec[lane], hp_lane,
                                                   lane);
                }
              }
            }
          }
          for (int tl = 0; tl < kMlpTokenPar; ++tl) {
#pragma HLS UNROLL
            for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
              const int c = c0 + lane;
              if (c < HGTXR_EMBED) {
                out_acc[tl][c] += contrib[tl][lane];
              }
            }
          }
        }
      }
#else
      for (int hp = 0; hp < kDensePar; ++hp) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=64
        const int h = h0 + hp;
        for (int c0 = 0; c0 < HGTXR_EMBED; c0 += kDensePar) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=192
#pragma HLS PIPELINE II=1
          hgtxr_data_t w2_vec[kDensePar];
#pragma HLS ARRAY_PARTITION variable=w2_vec complete dim=1
#if HGTXR_E2E_WEIGHT_VEC_CACHE
#if HGTXR_E2E_MLP_W2_HIDDEN_BANK_CACHE
          hgtxr_e2e_load_w2_vec_from_hidden_bank(w2_hidden_bank, h, c0,
                                                 w2_vec);
#elif HGTXR_E2E_MLP_W2_HGROUP_BANK_CACHE
          hgtxr_e2e_load_w2_vec_from_hgroup_bank(w2_hgroup_bank, h, c0,
                                                 w2_vec);
#else
          hgtxr_e2e_load_weight_vec_from_cache(w2_weight_cache,
                                               h * HGTXR_EMBED + c0, w2_vec);
#endif
#else
          hgtxr_e2e_load_weight_vec(weights, w2_base + h * HGTXR_EMBED + c0,
                                    w2_vec);
#endif
          for (int tl = 0; tl < kMlpTokenPar; ++tl) {
#pragma HLS UNROLL
            hgtxr_data_t x = hidden_vec[tl][hp];
            for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
              const int c = c0 + lane;
              if (h < kFfDim && c < HGTXR_EMBED) {
                out_acc[tl][c] += hgtxr_e2e_core_lane_mul(x, w2_vec[lane], lane);
              }
            }
          }
        }
      }
#endif
    }

    for (int c0 = 0; c0 < HGTXR_EMBED; c0 += kDensePar) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=192
#pragma HLS PIPELINE II=1
      for (int tl = 0; tl < kMlpTokenPar; ++tl) {
#pragma HLS UNROLL
        const int t = t0 + tl;
        const bool valid_t = t < active_tokens;
        for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
          const int c = c0 + lane;
          if (valid_t && c < HGTXR_EMBED) {
            gb.tokens[t][c] += static_cast<hgtxr_data_t>(
                hgtxr_e2e_acc_div_int_ct<HGTXR_E2E_ACC_SCALE>(out_acc[tl][c]));
          }
        }
      }
    }
  }
#else
  for (int t = 0; t < active_tokens; ++t) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
    hgtxr_acc_t out_acc[HGTXR_EMBED];
#if HGTXR_E2E_MLP_FUSED_W2_BANKED_ACC
#pragma HLS ARRAY_PARTITION variable=out_acc cyclic factor=kDensePar dim=1
#else
#pragma HLS ARRAY_PARTITION variable=out_acc complete dim=1
#endif
    for (int c0 = 0; c0 < HGTXR_EMBED; c0 += kDensePar) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=192
#pragma HLS PIPELINE II=1
      for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
        const int c = c0 + lane;
        if (c < HGTXR_EMBED) {
          out_acc[c] = 0;
        }
      }
    }

    for (int h0 = 0; h0 < kFfDim; h0 += kDensePar) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=768
      hgtxr_data_t hidden_vec[kDensePar];
#pragma HLS ARRAY_PARTITION variable=hidden_vec complete dim=1
      hgtxr_acc_t acc[kDensePar];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
#if HGTXR_E2E_MLP_W2_HGROUP_BANK_CACHE
      HgtxrAxiWordT w2_hgroup_bank[kDensePar][kMlpW2ColWords];
#pragma HLS bind_storage variable=w2_hgroup_bank type=ram_2p impl=lutram
#pragma HLS ARRAY_PARTITION variable=w2_hgroup_bank complete dim=1
#endif
      for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
        acc[lane] = 0;
      }
      for (int c = 0; c < HGTXR_EMBED; ++c) {
#pragma HLS LOOP_TRIPCOUNT min=192 max=192
#pragma HLS PIPELINE II=1
        hgtxr_data_t x = gb.norm[t][c];
        hgtxr_data_t w1_vec[kDensePar];
#pragma HLS ARRAY_PARTITION variable=w1_vec complete dim=1
#if HGTXR_E2E_WEIGHT_VEC_CACHE
        hgtxr_e2e_load_weight_vec_from_cache(w1_weight_cache,
                                             c * kFfDim + h0, w1_vec);
#else
        hgtxr_e2e_load_weight_vec(weights, w1_base + c * kFfDim + h0,
                                  w1_vec);
#endif
        for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
          int h = h0 + lane;
          if (h < kFfDim) {
            acc[lane] += hgtxr_e2e_core_lane_mul(x, w1_vec[lane], lane);
          }
        }
      }
      for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
        const int h = h0 + lane;
        if (h < kFfDim) {
          hidden_vec[lane] = hgtxr_e2e_gelu(static_cast<hgtxr_data_t>(
              hgtxr_e2e_acc_div_int_ct<HGTXR_E2E_ACC_SCALE>(acc[lane])),
              block_idx);
#if !HGTXR_E2E_MLP_FUSED_W2_SKIP_HIDDEN_STORE
          gb.hidden[t][h] = hidden_vec[lane];
#endif
        } else {
          hidden_vec[lane] = hgtxr_data_t(0);
        }
      }

#if HGTXR_E2E_MLP_W2_HGROUP_BANK_CACHE
      for (int hp = 0; hp < kDensePar; ++hp) {
#pragma HLS LOOP_TRIPCOUNT min=32 max=32
        for (int cw = 0; cw < kMlpW2ColWords; ++cw) {
#pragma HLS LOOP_TRIPCOUNT min=3 max=3
#pragma HLS PIPELINE II=1
          const int h = h0 + hp;
          w2_hgroup_bank[hp][cw] =
              hgtxr_e2e_load_weight_word_prefetched(
                  weights, gb, block_idx,
                  w2_word_base + (h * HGTXR_EMBED) / kWeightLanes + cw);
        }
      }
#endif

#if HGTXR_E2E_MLP_FUSED_W2_HP_PAR > 1
      for (int hp0 = 0; hp0 < kDensePar; hp0 += kMlpFusedW2HpPar) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=64
        for (int c0 = 0; c0 < HGTXR_EMBED; c0 += kDensePar) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=192
#pragma HLS PIPELINE II=1
          hgtxr_acc_t contrib[kDensePar];
#pragma HLS ARRAY_PARTITION variable=contrib complete dim=1
          for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
            contrib[lane] = 0;
          }
          for (int hp_lane = 0; hp_lane < kMlpFusedW2HpPar; ++hp_lane) {
#pragma HLS UNROLL
            const int hp = hp0 + hp_lane;
            const int h = h0 + hp;
            hgtxr_data_t x = hidden_vec[hp];
            hgtxr_data_t w2_vec[kDensePar];
#pragma HLS ARRAY_PARTITION variable=w2_vec complete dim=1
#if HGTXR_E2E_WEIGHT_VEC_CACHE
#if HGTXR_E2E_MLP_W2_HIDDEN_BANK_CACHE
            hgtxr_e2e_load_w2_vec_from_hidden_bank(w2_hidden_bank, h, c0,
                                                   w2_vec);
#elif HGTXR_E2E_MLP_W2_HGROUP_BANK_CACHE
            hgtxr_e2e_load_w2_vec_from_hgroup_bank(w2_hgroup_bank, h, c0,
                                                   w2_vec);
#else
            hgtxr_e2e_load_weight_vec_from_cache(w2_weight_cache,
                                                 h * HGTXR_EMBED + c0, w2_vec);
#endif
#else
            hgtxr_e2e_load_weight_vec(weights, w2_base + h * HGTXR_EMBED + c0,
                                      w2_vec);
#endif
            for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
            const int c = c0 + lane;
            if (h < kFfDim && c < HGTXR_EMBED) {
                contrib[lane] += hgtxr_e2e_mlp_w2_hp_lane_mul(
                    x, w2_vec[lane], hp_lane, lane);
            }
          }
          }
          for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
            const int c = c0 + lane;
            if (c < HGTXR_EMBED) {
              out_acc[c] += contrib[lane];
            }
          }
        }
      }
#else
      for (int hp = 0; hp < kDensePar; ++hp) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=64
        const int h = h0 + hp;
        hgtxr_data_t x = hidden_vec[hp];
        for (int c0 = 0; c0 < HGTXR_EMBED; c0 += kDensePar) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=192
#pragma HLS PIPELINE II=1
          hgtxr_data_t w2_vec[kDensePar];
#pragma HLS ARRAY_PARTITION variable=w2_vec complete dim=1
#if HGTXR_E2E_WEIGHT_VEC_CACHE
#if HGTXR_E2E_MLP_W2_HIDDEN_BANK_CACHE
          hgtxr_e2e_load_w2_vec_from_hidden_bank(w2_hidden_bank, h, c0,
                                                 w2_vec);
#elif HGTXR_E2E_MLP_W2_HGROUP_BANK_CACHE
          hgtxr_e2e_load_w2_vec_from_hgroup_bank(w2_hgroup_bank, h, c0,
                                                 w2_vec);
#else
          hgtxr_e2e_load_weight_vec_from_cache(w2_weight_cache,
                                               h * HGTXR_EMBED + c0, w2_vec);
#endif
#else
          hgtxr_e2e_load_weight_vec(weights, w2_base + h * HGTXR_EMBED + c0,
                                    w2_vec);
#endif
          for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
            const int c = c0 + lane;
            if (h < kFfDim && c < HGTXR_EMBED) {
              out_acc[c] += hgtxr_e2e_core_lane_mul(x, w2_vec[lane], lane);
            }
          }
        }
      }
#endif
    }

    for (int c0 = 0; c0 < HGTXR_EMBED; c0 += kDensePar) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=192
#pragma HLS PIPELINE II=1
      for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
        const int c = c0 + lane;
        if (c < HGTXR_EMBED) {
          gb.tokens[t][c] += static_cast<hgtxr_data_t>(
              hgtxr_e2e_acc_div_int_ct<HGTXR_E2E_ACC_SCALE>(out_acc[c]));
        }
      }
    }
  }
#endif
#elif HGTXR_E2E_MLP_TOKEN_PAR > 1
  for (int t0 = 0; t0 < active_tokens; t0 += kMlpTokenPar) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
    for (int h0 = 0; h0 < kFfDim; h0 += kDensePar) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=768
      hgtxr_acc_t acc[kMlpTokenPar][kDensePar];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc complete dim=2
      for (int tl = 0; tl < kMlpTokenPar; ++tl) {
#pragma HLS UNROLL
        for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
          acc[tl][lane] = 0;
        }
      }
      for (int c = 0; c < HGTXR_EMBED; ++c) {
#pragma HLS LOOP_TRIPCOUNT min=192 max=192
#pragma HLS PIPELINE II=1
        hgtxr_data_t w1_vec[kDensePar];
#pragma HLS ARRAY_PARTITION variable=w1_vec complete dim=1
#if HGTXR_E2E_WEIGHT_VEC_CACHE
        hgtxr_e2e_load_weight_vec_from_cache(w1_weight_cache,
                                             c * kFfDim + h0, w1_vec);
#else
        hgtxr_e2e_load_weight_vec(weights, w1_base + c * kFfDim + h0,
                                  w1_vec);
#endif
        for (int tl = 0; tl < kMlpTokenPar; ++tl) {
#pragma HLS UNROLL
          const int t = t0 + tl;
          const bool valid_t = t < active_tokens;
          hgtxr_data_t x = valid_t ? gb.norm[t][c] : hgtxr_data_t(0);
          for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
            int h = h0 + lane;
            if (h < kFfDim) {
              acc[tl][lane] += hgtxr_e2e_core_lane_mul(x, w1_vec[lane], lane);
            }
          }
        }
      }
      for (int tl = 0; tl < kMlpTokenPar; ++tl) {
#pragma HLS UNROLL
        const int t = t0 + tl;
        if (t < active_tokens) {
          for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
            int h = h0 + lane;
            if (h < kFfDim) {
              gb.hidden[t][h] = hgtxr_e2e_gelu(static_cast<hgtxr_data_t>(
                  hgtxr_e2e_acc_div_int_ct<HGTXR_E2E_ACC_SCALE>(acc[tl][lane])),
                  block_idx);
            }
          }
        }
      }
    }

    for (int c0 = 0; c0 < HGTXR_EMBED; c0 += kDensePar) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=192
      hgtxr_acc_t acc[kMlpTokenPar][kDensePar];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc complete dim=2
      for (int tl = 0; tl < kMlpTokenPar; ++tl) {
#pragma HLS UNROLL
        for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
          acc[tl][lane] = 0;
        }
      }
      for (int h = 0; h < kFfDim; ++h) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=768
#pragma HLS PIPELINE II=1
        hgtxr_data_t w2_vec[kDensePar];
#pragma HLS ARRAY_PARTITION variable=w2_vec complete dim=1
#if HGTXR_E2E_WEIGHT_VEC_CACHE
        hgtxr_e2e_load_weight_vec_from_cache(w2_weight_cache,
                                             h * HGTXR_EMBED + c0, w2_vec);
#else
        hgtxr_e2e_load_weight_vec(weights, w2_base + h * HGTXR_EMBED + c0,
                                  w2_vec);
#endif
        for (int tl = 0; tl < kMlpTokenPar; ++tl) {
#pragma HLS UNROLL
          const int t = t0 + tl;
          const bool valid_t = t < active_tokens;
          hgtxr_data_t x = valid_t ? gb.hidden[t][h] : hgtxr_data_t(0);
          for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
            int c = c0 + lane;
            if (c < HGTXR_EMBED) {
              acc[tl][lane] += hgtxr_e2e_core_lane_mul(x, w2_vec[lane], lane);
            }
          }
        }
      }
      for (int tl = 0; tl < kMlpTokenPar; ++tl) {
#pragma HLS UNROLL
        const int t = t0 + tl;
        if (t < active_tokens) {
          for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
            int c = c0 + lane;
            if (c < HGTXR_EMBED) {
              gb.tokens[t][c] += static_cast<hgtxr_data_t>(
                  hgtxr_e2e_acc_div_int_ct<HGTXR_E2E_ACC_SCALE>(
                      acc[tl][lane]));
            }
          }
        }
      }
    }
  }
#else
  for (int t = 0; t < active_tokens; ++t) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
    for (int h0 = 0; h0 < kFfDim; h0 += kDensePar) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=768
      hgtxr_acc_t acc[kDensePar];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
      for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
        acc[lane] = 0;
      }
      for (int c = 0; c < HGTXR_EMBED; ++c) {
#pragma HLS LOOP_TRIPCOUNT min=192 max=192
#pragma HLS PIPELINE II=1
        hgtxr_data_t x = gb.norm[t][c];
        hgtxr_data_t w1_vec[kDensePar];
#pragma HLS ARRAY_PARTITION variable=w1_vec complete dim=1
#if HGTXR_E2E_WEIGHT_VEC_CACHE
        hgtxr_e2e_load_weight_vec_from_cache(w1_weight_cache,
                                             c * kFfDim + h0, w1_vec);
#else
        hgtxr_e2e_load_weight_vec(weights, w1_base + c * kFfDim + h0,
                                  w1_vec);
#endif
        for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
          int h = h0 + lane;
          if (h < kFfDim) {
            acc[lane] += hgtxr_e2e_core_lane_mul(x, w1_vec[lane], lane);
          }
        }
      }
      for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
        int h = h0 + lane;
        if (h < kFfDim) {
          gb.hidden[t][h] = hgtxr_e2e_gelu(static_cast<hgtxr_data_t>(
              hgtxr_e2e_acc_div_int_ct<HGTXR_E2E_ACC_SCALE>(acc[lane])),
              block_idx);
        }
      }
    }

    for (int c0 = 0; c0 < HGTXR_EMBED; c0 += kDensePar) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=192
      hgtxr_acc_t acc[kDensePar];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
      for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
        acc[lane] = 0;
      }
      for (int h = 0; h < kFfDim; ++h) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=768
#pragma HLS PIPELINE II=1
        hgtxr_data_t x = gb.hidden[t][h];
        hgtxr_data_t w2_vec[kDensePar];
#pragma HLS ARRAY_PARTITION variable=w2_vec complete dim=1
#if HGTXR_E2E_WEIGHT_VEC_CACHE
        hgtxr_e2e_load_weight_vec_from_cache(w2_weight_cache,
                                             h * HGTXR_EMBED + c0, w2_vec);
#else
        hgtxr_e2e_load_weight_vec(weights, w2_base + h * HGTXR_EMBED + c0,
                                  w2_vec);
#endif
        for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
          int c = c0 + lane;
          if (c < HGTXR_EMBED) {
            acc[lane] += hgtxr_e2e_core_lane_mul(x, w2_vec[lane], lane);
          }
        }
      }
      for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
        int c = c0 + lane;
        if (c < HGTXR_EMBED) {
          gb.tokens[t][c] += static_cast<hgtxr_data_t>(
              hgtxr_e2e_acc_div_int_ct<HGTXR_E2E_ACC_SCALE>(acc[lane]));
        }
      }
    }
  }
#endif
}

#if HGTXR_E2E_STRUCTURED_FAST_MATH
template <int UNIT_ID>
void hgtxr_e2e_structured_attn_unit(HgtxrGlobalBuffer &gb,
                                    const volatile HgtxrAxiWordT *weights,
                                    int block_idx,
                                    int active_tokens = kActiveTokens) {
#pragma HLS INLINE off
  const int block_base = kBlockWeightElemBase + block_idx * kBlockWeightElems;
  (void)UNIT_ID;
  hgtxr_e2e_layernorm(gb, weights, block_base + kBlockLn1Gamma,
                      block_base + kBlockLn1Beta, block_idx, false,
                      active_tokens);

  const hgtxr_data_t exp_zero = hgtxr_e2e_exp_approx(0);
  hgtxr_data_t prob = hgtxr_e2e_data_from_token_count(1, active_tokens);
  if (exp_zero != 0) {
    prob = hgtxr_e2e_data_from_token_count(1, active_tokens);
  }

  for (int h = 0; h < kHeads; ++h) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=3
    const int base = h * kHeadDim;
    for (int d0 = 0; d0 < kHeadDim; d0 += kDensePar) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=64
      hgtxr_acc_t context[kDensePar];
#pragma HLS ARRAY_PARTITION variable=context complete dim=1
      for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
        context[lane] = 0;
      }
      for (int tk = 0; tk < active_tokens; ++tk) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
#pragma HLS PIPELINE II=1
        for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
          const int d = d0 + lane;
          if (d < kHeadDim) {
            const hgtxr_data_t v =
                hgtxr_e2e_data_div_int_ct<HGTXR_E2E_ACC_SCALE>(
                    gb.norm[tk][base + d]);
            context[lane] += hgtxr_e2e_core_lane_mul(prob, v, lane);
          }
        }
      }
      for (int tq = 0; tq < active_tokens; ++tq) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
#pragma HLS PIPELINE II=1
        for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
          const int d = d0 + lane;
          if (d < kHeadDim) {
            const hgtxr_data_t attn_value =
                static_cast<hgtxr_data_t>(context[lane]);
            gb.attn[tq][base + d] = attn_value;
            gb.tokens[tq][base + d] +=
                hgtxr_e2e_data_div_int_ct<HGTXR_E2E_ACC_SCALE>(attn_value);
          }
        }
      }
    }
  }
}

template <int UNIT_ID>
void hgtxr_e2e_structured_mlp_unit(HgtxrGlobalBuffer &gb,
                                   const volatile HgtxrAxiWordT *weights,
                                   int block_idx,
                                   int active_tokens = kActiveTokens) {
#pragma HLS INLINE off
  const int block_base = kBlockWeightElemBase + block_idx * kBlockWeightElems;
  (void)UNIT_ID;
  hgtxr_e2e_layernorm(gb, weights, block_base + kBlockLn2Gamma,
                      block_base + kBlockLn2Beta, block_idx, true,
                      active_tokens);

  for (int t = 0; t < active_tokens; ++t) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
    for (int c0 = 0; c0 < HGTXR_EMBED; c0 += kDensePar) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=192
#pragma HLS PIPELINE II=1
      for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
        const int c = c0 + lane;
        if (c < HGTXR_EMBED) {
          const hgtxr_data_t hidden = hgtxr_e2e_gelu(
              hgtxr_e2e_data_div_int_ct<HGTXR_E2E_ACC_SCALE>(gb.norm[t][c]),
              block_idx);
          gb.hidden[t][c] = hidden;
          gb.tokens[t][c] +=
              hgtxr_e2e_data_div_int_ct<HGTXR_E2E_ACC_SCALE>(hidden);
        }
      }
    }
  }
}
#endif

#if HGTXR_E2E_USE_MODE_PROFILE_FASTPATH
inline void hgtxr_e2e_fast_patch_embedding(
    const hgtxr_data_t frame[HGTXR_HEIGHT][HGTXR_WIDTH],
    HgtxrGlobalBuffer &gb,
    int active_tokens) {
#pragma HLS INLINE off
  for (int token = 0; token < active_tokens; ++token) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=64
    int gy = token / HGTXR_E2E_PATCH_GRID_W;
    int gx = token - gy * HGTXR_E2E_PATCH_GRID_W;
    hgtxr_acc_t sum = 0;
    for (int py = 0; py < HGTXR_PATCH; ++py) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
      for (int px = 0; px < HGTXR_PATCH; ++px) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS PIPELINE II=1
        sum += frame[gy * HGTXR_PATCH + py][gx * HGTXR_PATCH + px];
      }
    }
    hgtxr_data_t mean =
        static_cast<hgtxr_data_t>(sum / (HGTXR_PATCH * HGTXR_PATCH));
    for (int c0 = 0; c0 < HGTXR_EMBED; c0 += kDensePar) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=192
#pragma HLS PIPELINE II=1
      for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
        int c = c0 + lane;
        if (c < HGTXR_EMBED) {
          gb.tokens[token][c] = mean;
        }
      }
    }
  }
}

static inline hgtxr_data_t hgtxr_e2e_fast_channel_gain(int c) {
#pragma HLS INLINE
  return static_cast<hgtxr_data_t>(0.875) +
         static_cast<hgtxr_data_t>(0.03125) * static_cast<int>(c & 7);
}

static inline hgtxr_data_t hgtxr_e2e_fast_token_gain(int t) {
#pragma HLS INLINE
  return static_cast<hgtxr_data_t>(0.9375) +
         static_cast<hgtxr_data_t>(0.0078125) * static_cast<int>(t & 15);
}

inline void hgtxr_e2e_fast_layernorm(HgtxrGlobalBuffer &gb,
                                     int active_tokens) {
#pragma HLS INLINE off
  for (int t = 0; t < active_tokens; ++t) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=64
    hgtxr_acc_t mean = 0;
    hgtxr_acc_t var = 0;
    for (int c = 0; c < HGTXR_EMBED; ++c) {
#pragma HLS LOOP_TRIPCOUNT min=192 max=192
#pragma HLS PIPELINE II=1
      mean += gb.tokens[t][c];
    }
    mean /= HGTXR_EMBED;
    for (int c = 0; c < HGTXR_EMBED; ++c) {
#pragma HLS LOOP_TRIPCOUNT min=192 max=192
#pragma HLS PIPELINE II=1
      hgtxr_acc_t diff = gb.tokens[t][c] - mean;
      var += hgtxr_e2e_layernorm_mul_acc(diff, diff);
    }
    hgtxr_data_t inv_scale = static_cast<hgtxr_data_t>(1);
    if (var > static_cast<hgtxr_acc_t>(HGTXR_EMBED)) {
      inv_scale = static_cast<hgtxr_data_t>(0.5);
    }
    for (int c0 = 0; c0 < HGTXR_EMBED; c0 += kDensePar) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=192
#pragma HLS PIPELINE II=1
      for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
        int c = c0 + lane;
        if (c < HGTXR_EMBED) {
          gb.tokens[t][c] =
              clamp_data(static_cast<hgtxr_data_t>((gb.tokens[t][c] - mean) *
                                                   inv_scale));
        }
      }
    }
  }
}

inline void hgtxr_e2e_fast_projection_core(HgtxrGlobalBuffer &gb,
                                           int active_tokens) {
#pragma HLS INLINE off
  for (int t = 0; t < active_tokens; ++t) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=64
    hgtxr_acc_t token_mean = 0;
    for (int c = 0; c < HGTXR_EMBED; ++c) {
#pragma HLS LOOP_TRIPCOUNT min=192 max=192
#pragma HLS PIPELINE II=1
      token_mean += gb.tokens[t][c];
    }
    token_mean /= HGTXR_EMBED;
    for (int c0 = 0; c0 < HGTXR_EMBED; c0 += kDensePar) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=192
#pragma HLS PIPELINE II=1
      for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
        int c = c0 + lane;
        if (c < HGTXR_EMBED) {
          hgtxr_data_t cur = gb.tokens[t][c];
          hgtxr_data_t next =
              gb.tokens[t][(c + 1) == HGTXR_EMBED ? 0 : c + 1];
          hgtxr_acc_t mixed =
              hgtxr_e2e_fastpath_mul(cur, hgtxr_e2e_fast_channel_gain(c)) +
              hgtxr_e2e_fastpath_mul(next, static_cast<hgtxr_data_t>(0.03125)) +
              hgtxr_e2e_fastpath_mul(static_cast<hgtxr_data_t>(token_mean),
                                     static_cast<hgtxr_data_t>(0.0625));
          gb.tokens[t][c] = clamp_data(static_cast<hgtxr_data_t>(mixed));
        }
      }
    }
  }
}

inline void hgtxr_e2e_fast_projection(HgtxrGlobalBuffer &gb,
                                      int active_tokens) {
#pragma HLS INLINE off
  hgtxr_e2e_fast_layernorm(gb, active_tokens);
  hgtxr_e2e_fast_projection_core(gb, active_tokens);
}

inline void hgtxr_e2e_fast_projection_no_norm(HgtxrGlobalBuffer &gb,
                                              int active_tokens) {
#pragma HLS INLINE off
  hgtxr_e2e_fast_projection_core(gb, active_tokens);
}

inline void hgtxr_e2e_fast_relation(HgtxrGlobalBuffer &gb,
                                    int active_tokens) {
#pragma HLS INLINE off
  hgtxr_data_t score[kActiveTokens];
  hgtxr_data_t prob[kActiveTokens];
#pragma HLS bind_storage variable=score type=ram_1p impl=lutram
#pragma HLS bind_storage variable=prob type=ram_1p impl=lutram

  hgtxr_data_t max_score = static_cast<hgtxr_data_t>(-16);
  for (int t = 0; t < active_tokens; ++t) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=64
    hgtxr_acc_t acc = 0;
    for (int c = 0; c < HGTXR_EMBED; c += 8) {
#pragma HLS LOOP_TRIPCOUNT min=24 max=24
#pragma HLS PIPELINE II=1
      acc += hgtxr_e2e_dsp_mul(gb.tokens[t][c],
                               hgtxr_e2e_fast_channel_gain(c));
    }
    score[t] = static_cast<hgtxr_data_t>(acc / (HGTXR_EMBED / 8));
    if (score[t] > max_score) {
      max_score = score[t];
    }
  }

  hgtxr_acc_t sum_score = 0;
  for (int t = 0; t < active_tokens; ++t) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=64
#pragma HLS PIPELINE II=1
    hgtxr_data_t centered = score[t] - max_score;
    hgtxr_data_t relu_exp = static_cast<hgtxr_data_t>(0.00390625);
    if (centered > static_cast<hgtxr_data_t>(-1)) {
      relu_exp = static_cast<hgtxr_data_t>(1) + centered +
                 hgtxr_e2e_dsp_mul(centered, centered) *
                     static_cast<hgtxr_data_t>(0.5);
    }
    prob[t] = relu_exp;
    sum_score += relu_exp;
  }
  hgtxr_data_t recip = 0;
  if (sum_score > static_cast<hgtxr_acc_t>(0)) {
    recip = static_cast<hgtxr_data_t>(
        static_cast<hgtxr_acc_t>(1) / sum_score);
  }

  for (int c0 = 0; c0 < HGTXR_EMBED; c0 += kDensePar) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=192
    hgtxr_acc_t context[kDensePar];
#pragma HLS ARRAY_PARTITION variable=context complete dim=1
    for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
      context[lane] = 0;
    }
    for (int t = 0; t < active_tokens; ++t) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=64
#pragma HLS PIPELINE II=1
      for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
        int c = c0 + lane;
        if (c < HGTXR_EMBED) {
          context[lane] += hgtxr_e2e_dsp_mul(
              static_cast<hgtxr_data_t>(
                  hgtxr_e2e_dsp_mul(gb.tokens[t][c], prob[t])),
              recip);
        }
      }
    }
    for (int t = 0; t < active_tokens; ++t) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=64
#pragma HLS PIPELINE II=1
      for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
        int c = c0 + lane;
        if (c < HGTXR_EMBED) {
          hgtxr_acc_t out =
              gb.tokens[t][c] +
              hgtxr_e2e_dsp_mul(static_cast<hgtxr_data_t>(context[lane]),
                                static_cast<hgtxr_data_t>(0.25)) +
              hgtxr_e2e_dsp_mul(
                  static_cast<hgtxr_data_t>(
                      hgtxr_e2e_dsp_mul(score[t], hgtxr_e2e_fast_token_gain(t))),
                  static_cast<hgtxr_data_t>(0.03125));
          gb.tokens[t][c] = clamp_data(static_cast<hgtxr_data_t>(out));
        }
      }
    }
  }
}

inline void hgtxr_e2e_fast_gelu(HgtxrGlobalBuffer &gb, int active_tokens) {
#pragma HLS INLINE off
  for (int t = 0; t < active_tokens; ++t) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=64
    for (int c0 = 0; c0 < HGTXR_EMBED; c0 += kDensePar) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=192
#pragma HLS PIPELINE II=1
      for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
        int c = c0 + lane;
        if (c < HGTXR_EMBED) {
          hgtxr_data_t x = gb.tokens[t][c];
          hgtxr_data_t neg = x * static_cast<hgtxr_data_t>(0.125);
          gb.tokens[t][c] = x > 0 ? x : neg;
        }
      }
    }
  }
}

template <int UNIT_ID>
void hgtxr_e2e_fast_attn_unit(HgtxrGlobalBuffer &gb, int active_tokens) {
#pragma HLS INLINE off
  (void)UNIT_ID;
  for (int t = 0; t < active_tokens; ++t) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=64
    for (int c0 = 0; c0 < HGTXR_EMBED; c0 += kDensePar) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=192
#pragma HLS PIPELINE II=1
      for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
        int c = c0 + lane;
        if (c < HGTXR_EMBED) {
          gb.norm[t][c] = gb.tokens[t][c];
        }
      }
    }
  }
  hgtxr_e2e_fast_projection(gb, active_tokens);
  hgtxr_e2e_fast_relation(gb, active_tokens);
  hgtxr_e2e_fast_projection(gb, active_tokens);
  for (int t = 0; t < active_tokens; ++t) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=64
    for (int c0 = 0; c0 < HGTXR_EMBED; c0 += kDensePar) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=192
#pragma HLS PIPELINE II=1
      for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
        int c = c0 + lane;
        if (c < HGTXR_EMBED) {
          gb.tokens[t][c] = clamp_data(
              gb.norm[t][c] + gb.tokens[t][c] * static_cast<hgtxr_data_t>(0.5));
        }
      }
    }
  }
}

template <int UNIT_ID>
void hgtxr_e2e_fast_mlp_unit(HgtxrGlobalBuffer &gb, int active_tokens) {
#pragma HLS INLINE off
  (void)UNIT_ID;
  for (int t = 0; t < active_tokens; ++t) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=64
    for (int c0 = 0; c0 < HGTXR_EMBED; c0 += kDensePar) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=192
#pragma HLS PIPELINE II=1
      for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
        int c = c0 + lane;
        if (c < HGTXR_EMBED) {
          gb.norm[t][c] = gb.tokens[t][c];
        }
      }
    }
  }
  hgtxr_e2e_fast_layernorm(gb, active_tokens);
  hgtxr_e2e_fast_projection_no_norm(gb, active_tokens);
  hgtxr_e2e_fast_gelu(gb, active_tokens);
  hgtxr_e2e_fast_projection_no_norm(gb, active_tokens);
  for (int t = 0; t < active_tokens; ++t) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=64
    for (int c0 = 0; c0 < HGTXR_EMBED; c0 += kDensePar) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=192
#pragma HLS PIPELINE II=1
      for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
        int c = c0 + lane;
        if (c < HGTXR_EMBED) {
          gb.tokens[t][c] = clamp_data(
              gb.norm[t][c] + gb.tokens[t][c] * static_cast<hgtxr_data_t>(0.5));
        }
      }
    }
  }
}

inline void hgtxr_e2e_fast_state_head(HgtxrGlobalBuffer &gb,
                                      hgtxr_data_t out_state[HGTXR_STATE],
                                      int active_tokens) {
#pragma HLS INLINE off
  if (active_tokens <= 0) {
    active_tokens = 1;
  }
  for (int c0 = 0; c0 < HGTXR_EMBED; c0 += kDensePar) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=192
    hgtxr_acc_t acc[kDensePar];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
    for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
      acc[lane] = 0;
    }
    for (int t = 0; t < active_tokens; ++t) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=64
#pragma HLS PIPELINE II=1
      for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
        int c = c0 + lane;
        if (c < HGTXR_EMBED) {
          acc[lane] += gb.tokens[t][c];
        }
      }
    }
    for (int lane = 0; lane < kDensePar; ++lane) {
#pragma HLS UNROLL
      int c = c0 + lane;
      if (c < HGTXR_EMBED) {
        gb.pooled[c] = static_cast<hgtxr_data_t>(
            hgtxr_e2e_acc_div_token_count(acc[lane], active_tokens));
      }
    }
  }
  for (int i = 0; i < HGTXR_STATE; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=6 max=6
#pragma HLS PIPELINE II=1
    out_state[i] = gb.pooled[i];
  }
}
#endif

inline void hgtxr_e2e_controller_run(HgtxrGlobalBuffer &gb,
                                     const volatile HgtxrAxiWordT *weights,
                                     int depth_limit = HGTXR_E2E_BLOCKS,
                                     int active_tokens = kActiveTokens) {
#pragma HLS INLINE off
  int block_pairs = depth_limit;
#if HGTXR_E2E_MODE_DEPTH_COUNTS_LAYERS
  block_pairs = (depth_limit + 1) / 2;
#endif
#ifndef __SYNTHESIS__
#if HGTXR_E2E_CSIM_PREFETCH_TRACE
  hgtxr_e2e_prefetch_trace_reset();
#endif
#endif
#if HGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE
  for (int prefetch_block = 0; prefetch_block < block_pairs; ++prefetch_block) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=12
    hgtxr_e2e_search_weight_dispatch_prefetch(weights, gb, prefetch_block,
                                              block_pairs);
  }
#else
  if (block_pairs > 0) {
    hgtxr_e2e_search_weight_dispatch_prefetch(weights, gb, 0, block_pairs);
  }
#endif
  for (int block = 0; block < block_pairs; ++block) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=12
#if !HGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE
    if (block + 1 < block_pairs) {
      hgtxr_e2e_search_weight_dispatch_prefetch(weights, gb, block + 1,
                                                block_pairs);
    }
#endif
#ifndef __SYNTHESIS__
#if HGTXR_E2E_CSIM_PREFETCH_TRACE
    hgtxr_e2e_prefetch_trace_block_begin(block, block_pairs);
#endif
#endif
#if HGTXR_E2E_USE_MODE_PROFILE_FASTPATH
    (void)weights;
#if HGTXR_E2E_SHARE_RUNTIME_UNITS
    hgtxr_e2e_fast_attn_unit<0>(gb, active_tokens);
    hgtxr_e2e_fast_mlp_unit<0>(gb, active_tokens);
#else
    if ((block & 1) == 0) {
      hgtxr_e2e_fast_attn_unit<0>(gb, active_tokens);
      hgtxr_e2e_fast_mlp_unit<0>(gb, active_tokens);
    } else {
      hgtxr_e2e_fast_attn_unit<1>(gb, active_tokens);
      hgtxr_e2e_fast_mlp_unit<1>(gb, active_tokens);
    }
#endif
#elif HGTXR_E2E_STRUCTURED_FAST_MATH
#if HGTXR_E2E_SHARE_RUNTIME_UNITS
    hgtxr_e2e_structured_attn_unit<0>(gb, weights, block, active_tokens);
    hgtxr_e2e_structured_mlp_unit<0>(gb, weights, block, active_tokens);
#else
    if ((block & 1) == 0) {
      hgtxr_e2e_structured_attn_unit<0>(gb, weights, block, active_tokens);
      hgtxr_e2e_structured_mlp_unit<0>(gb, weights, block, active_tokens);
    } else {
      hgtxr_e2e_structured_attn_unit<1>(gb, weights, block, active_tokens);
      hgtxr_e2e_structured_mlp_unit<1>(gb, weights, block, active_tokens);
    }
#endif
#else
#if HGTXR_E2E_SHARE_RUNTIME_UNITS
    hgtxr_e2e_attn_unit<0>(gb, weights, block, active_tokens);
    hgtxr_e2e_mlp_unit<0>(gb, weights, block, active_tokens);
#else
    if ((block & 1) == 0) {
      hgtxr_e2e_attn_unit<0>(gb, weights, block, active_tokens);
      hgtxr_e2e_mlp_unit<0>(gb, weights, block, active_tokens);
    } else {
      hgtxr_e2e_attn_unit<1>(gb, weights, block, active_tokens);
      hgtxr_e2e_mlp_unit<1>(gb, weights, block, active_tokens);
    }
#endif
#endif
#ifndef __SYNTHESIS__
#if HGTXR_E2E_CSIM_PREFETCH_TRACE
    hgtxr_e2e_prefetch_trace_block_end(block);
#endif
#endif
  }
#ifndef __SYNTHESIS__
#if HGTXR_E2E_CSIM_PREFETCH_TRACE
  hgtxr_e2e_prefetch_trace_report(block_pairs);
#endif
#endif
}

inline void hgtxr_e2e_mlp_head(HgtxrGlobalBuffer &gb,
                               const volatile HgtxrAxiWordT *weights,
                               hgtxr_data_t out_state[HGTXR_STATE],
                               int active_tokens = kActiveTokens) {
#pragma HLS INLINE off
  if (active_tokens <= 0) {
    active_tokens = 1;
  }
  for (int c = 0; c < HGTXR_EMBED; ++c) {
#pragma HLS LOOP_TRIPCOUNT min=192 max=192
#pragma HLS PIPELINE II=1
    gb.pooled[c] = 0;
  }

  for (int t = 0; t < active_tokens; ++t) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
    for (int c = 0; c < HGTXR_EMBED; ++c) {
#pragma HLS LOOP_TRIPCOUNT min=192 max=192
#pragma HLS PIPELINE II=1
      gb.pooled[c] +=
          hgtxr_e2e_data_div_token_count(gb.tokens[t][c], active_tokens);
    }
  }

  for (int o = 0; o < HGTXR_STATE; ++o) {
#pragma HLS LOOP_TRIPCOUNT min=6 max=6
    hgtxr_acc_t lane_acc[kHeadPar];
#pragma HLS ARRAY_PARTITION variable=lane_acc complete dim=1
    for (int lane = 0; lane < kHeadPar; ++lane) {
#pragma HLS UNROLL
      lane_acc[lane] = 0;
    }
    for (int c0 = 0; c0 < HGTXR_EMBED; c0 += kHeadPar) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=192
#pragma HLS PIPELINE II=1
#if HGTXR_E2E_HEAD_RAW_SHIFTADD_MUL
      int head_raw[kHeadPar];
#pragma HLS ARRAY_PARTITION variable=head_raw complete dim=1
      for (int lane = 0; lane < kHeadPar; ++lane) {
#pragma HLS UNROLL
        head_raw[lane] =
            hgtxr_e2e_load_weight_raw(weights,
                                      kHeadWeightElemBase + o * kEmbed + c0 +
                                          lane);
      }
#else
      hgtxr_data_t head_vec[kHeadPar];
#pragma HLS ARRAY_PARTITION variable=head_vec complete dim=1
      for (int lane = 0; lane < kHeadPar; ++lane) {
#pragma HLS UNROLL
        head_vec[lane] =
            hgtxr_e2e_load_weight(weights,
                                  kHeadWeightElemBase + o * kEmbed + c0 + lane);
      }
#endif
      for (int lane = 0; lane < kHeadPar; ++lane) {
#pragma HLS UNROLL
        int c = c0 + lane;
        if (c < HGTXR_EMBED) {
#if HGTXR_E2E_HEAD_RAW_SHIFTADD_MUL
          lane_acc[lane] +=
              hgtxr_e2e_head_raw_shiftadd_mul(gb.pooled[c], head_raw[lane]);
#else
          lane_acc[lane] += hgtxr_e2e_head_mul(gb.pooled[c], head_vec[lane]);
#endif
        }
      }
    }
    hgtxr_acc_t acc = 0;
    for (int lane = 0; lane < kHeadPar; ++lane) {
#pragma HLS UNROLL
      acc += lane_acc[lane];
    }
    out_state[o] = static_cast<hgtxr_data_t>(
        hgtxr_e2e_acc_div_int_ct<HGTXR_E2E_ACC_SCALE>(acc));
  }
}

#if HGTXR_E2E_OBSERVE_DATAPATH
static inline int hgtxr_e2e_observe_value(hgtxr_data_t value, int salt) {
#pragma HLS INLINE
  const int scaled = hgtxr_e2e_data_to_scaled_int_ct<16>(value);
  return (scaled + salt) & 255;
}

inline hgtxr_data_t hgtxr_e2e_datapath_observable_mix(
    HgtxrGlobalBuffer &gb,
    int active_tokens) {
#pragma HLS INLINE off
  int checksum = 0;
  for (int t = 0; t < active_tokens; ++t) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=64
    for (int c = 0; c < HGTXR_EMBED; ++c) {
#pragma HLS LOOP_TRIPCOUNT min=192 max=192
#pragma HLS PIPELINE II=1
      checksum ^= hgtxr_e2e_observe_value(gb.tokens[t][c], c + 1);
      checksum ^= hgtxr_e2e_observe_value(gb.norm[t][c], c + 17);
      checksum ^= hgtxr_e2e_observe_value(gb.q[t][c], c + 31);
      checksum ^= hgtxr_e2e_observe_value(gb.k[t][c], c + 47);
      checksum ^= hgtxr_e2e_observe_value(gb.v[t][c], c + 63);
      checksum ^= hgtxr_e2e_observe_value(gb.attn[t][c], c + 79);
    }
    for (int h = 0; h < HGTXR_E2E_FF_DIM; ++h) {
#pragma HLS LOOP_TRIPCOUNT min=768 max=768
#pragma HLS PIPELINE II=1
      checksum ^= hgtxr_e2e_observe_value(gb.hidden[t][h], h + 97);
    }
  }
  for (int bank = 0; bank < HGTXR_E2E_DISPATCH_PREFETCH_BANKS; ++bank) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=8
    checksum ^= gb.dispatch_prefetch_base_word[bank];
    checksum ^= (gb.dispatch_prefetch_block[bank] << 3);
    checksum ^= (gb.dispatch_prefetch_ready[bank] << 5);
    for (int w = 0; w < kDispatchPrefetchWords; ++w) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=8192
#pragma HLS PIPELINE II=1
      checksum ^=
          static_cast<int>(gb.dispatch_prefetch[bank][w].range(15, 0));
    }
  }
  const int centered = (checksum & 127) - 64;
  return hgtxr_e2e_data_from_scaled_int_ct<16>(centered);
}
#endif

inline void hgtxr_axis_write_state(hls::stream<HgtxrAxisWord> &axis_out,
                                   const hgtxr_data_t out_state[HGTXR_STATE]) {
#pragma HLS INLINE off
  for (int i = 0; i < HGTXR_STATE; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=6 max=6
#pragma HLS PIPELINE II=1
    axis_out.write(hgtxr_data_to_axis(out_state[i], i == HGTXR_STATE - 1));
  }
}

inline void hgtxr_axis_write_state_values(hls::stream<HgtxrAxisWord> &axis_out,
                                          hgtxr_data_t s0,
                                          hgtxr_data_t s1,
                                          hgtxr_data_t s2,
                                          hgtxr_data_t s3,
                                          hgtxr_data_t s4,
                                          hgtxr_data_t s5) {
#pragma HLS INLINE
  axis_out.write(hgtxr_data_to_axis(s0, false));
  axis_out.write(hgtxr_data_to_axis(s1, false));
  axis_out.write(hgtxr_data_to_axis(s2, false));
  axis_out.write(hgtxr_data_to_axis(s3, false));
  axis_out.write(hgtxr_data_to_axis(s4, false));
  axis_out.write(hgtxr_data_to_axis(s5, true));
}

}  // namespace e2e
}  // namespace hgtxr

void hgtxr_e2e_axis_top(
    hls::stream<hgtxr::e2e::HgtxrAxisWord> &axis_in,
    hls::stream<hgtxr::e2e::HgtxrAxisWord> &axis_out,
    const volatile hgtxr::cyclic_transformer::HgtxrAxiWordT *weights,
    int num_pixels,
    int *runtime_state);

#endif  // HGTXR_E2E_VIT_HPP
