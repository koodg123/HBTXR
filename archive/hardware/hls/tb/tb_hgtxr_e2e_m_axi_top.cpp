#include "../include/hgtxr_e2e_vit.hpp"

#if defined(HGTXR_E2E_USE_ACTIVE8_GOLDEN)
#include "e2e_axis_vector_active8_golden.hpp"
#elif defined(HGTXR_E2E_USE_ACTIVE16_GOLDEN)
#include "e2e_axis_vector_active16_golden.hpp"
#elif defined(HGTXR_E2E_USE_ACTIVE16_FF64_GOLDEN)
#include "e2e_axis_vector_active16_ff64_golden.hpp"
#elif defined(HGTXR_E2E_USE_ACTIVE16_FF128_GOLDEN)
#include "e2e_axis_vector_active16_ff128_golden.hpp"
#elif defined(HGTXR_E2E_USE_ACTIVE16_FF256_GOLDEN)
#include "e2e_axis_vector_active16_ff256_golden.hpp"
#elif defined(HGTXR_E2E_USE_ACTIVE16_FF768_GOLDEN)
#include "e2e_axis_vector_active16_ff768_golden.hpp"
#elif defined(HGTXR_E2E_USE_ACTIVE32_FF768_GOLDEN)
#include "e2e_axis_vector_active32_ff768_golden.hpp"
#elif defined(HGTXR_E2E_USE_ACTIVE64_FF768_GOLDEN)
#include "e2e_axis_vector_active64_ff768_golden.hpp"
#elif defined(HGTXR_E2E_USE_ACTIVE128_FF768_GOLDEN)
#include "e2e_axis_vector_active128_ff768_golden.hpp"
#elif defined(HGTXR_E2E_USE_ACTIVE196_FF768_GOLDEN)
#include "e2e_axis_vector_active196_ff768_golden.hpp"
#elif defined(HGTXR_E2E_USE_ACTIVE196_B6_FF768_GOLDEN)
#include "e2e_axis_vector_active196_b6_ff768_golden.hpp"
#elif defined(HGTXR_E2E_USE_HGPIPE_MATH_GOLDEN)
#include "e2e_axis_vector_hgpipe_math_golden.hpp"
#elif defined(HGTXR_E2E_USE_HGPIPE_MATH_LNQ_GOLDEN)
#include "e2e_axis_vector_hgpipe_math_lnq_golden.hpp"
#elif defined(HGTXR_E2E_USE_HGPIPE_MATH_LNQ_ACTIVE8_GOLDEN)
#include "e2e_axis_vector_hgpipe_math_lnq_active8_golden.hpp"
#elif defined(HGTXR_E2E_USE_HGPIPE_MATH_LNQ_ACTIVE16_GOLDEN)
#include "e2e_axis_vector_hgpipe_math_lnq_active16_golden.hpp"
#elif defined(HGTXR_E2E_USE_VREF_P0_HGPIPE_LNQ_ACTIVE16_SOFTMAX_INPUT_X2_GOLDEN)
#include "e2e_axis_vector_vref_p0_hgpipe_lnq_active16_softmax_input_x2_golden.hpp"
#else
#ifndef HGTXR_E2E_GOLDEN_HEADER
#define HGTXR_E2E_GOLDEN_HEADER "e2e_axis_vector_golden.hpp"
#endif
#include HGTXR_E2E_GOLDEN_HEADER
#endif

#include <cstdio>

void hgtxr_e2e_m_axi_top(
    const hgtxr_data_t frame[HGTXR_HEIGHT][HGTXR_WIDTH],
    const volatile hgtxr::cyclic_transformer::HgtxrAxiWordT *weights,
    hgtxr_data_t out_state[HGTXR_STATE],
    int *runtime_state);

#ifndef HGTXR_E2E_STRICT_GOLDEN
#define HGTXR_E2E_STRICT_GOLDEN 1
#endif

#if HGTXR_E2E_STRICT_GOLDEN
static_assert(kExpectedBlocks == HGTXR_E2E_BLOCKS,
              "E2E golden block count must match HGTXR_E2E_BLOCKS");
static_assert(kExpectedActiveTokens == HGTXR_E2E_ACTIVE_TOKENS,
              "E2E golden active token count must match HGTXR_E2E_ACTIVE_TOKENS");
static_assert(kExpectedPatchGridH == HGTXR_E2E_PATCH_GRID_H,
              "E2E golden patch_grid_h must match HGTXR_E2E_PATCH_GRID_H");
static_assert(kExpectedPatchGridW == HGTXR_E2E_PATCH_GRID_W,
              "E2E golden patch_grid_w must match HGTXR_E2E_PATCH_GRID_W");
static_assert(kExpectedFfDim == HGTXR_E2E_FF_DIM,
              "E2E golden FF dim must match HGTXR_E2E_FF_DIM");
#endif

namespace {

void set_weight(volatile hgtxr::cyclic_transformer::HgtxrAxiWordT *weights,
                int elem_offset, int value) {
  const int word_idx = elem_offset / hgtxr::e2e::kWeightLanes;
  const int lane_idx = elem_offset - word_idx * hgtxr::e2e::kWeightLanes;
  hgtxr::cyclic_transformer::HgtxrAxiWordT word = weights[word_idx];
  ap_int<HGTXR_WEIGHT_BIT_WIDTH> packed = value;
  word.range((lane_idx + 1) * HGTXR_WEIGHT_BIT_WIDTH - 1,
             lane_idx * HGTXR_WEIGHT_BIT_WIDTH) =
      packed.range(HGTXR_WEIGHT_BIT_WIDTH - 1, 0);
  weights[word_idx] = word;
}

void set_live_weight_bit(
    volatile hgtxr::cyclic_transformer::HgtxrAxiWordT *weights) {
  hgtxr::cyclic_transformer::HgtxrAxiWordT word = weights[0];
  word.range(0, 0) = kExpectedRuntimeState - 1;
  weights[0] = word;
}

void init_q4_vector_weights(
    volatile hgtxr::cyclic_transformer::HgtxrAxiWordT *weights) {
  for (int i = 0; i < HGTXR_E2E_WEIGHT_DEPTH; ++i) {
    weights[i] = 0;
  }

  for (int c = 0; c < kGroupChannels; ++c) {
    for (int p = 0; p < hgtxr::e2e::kPatchElems; ++p) {
      set_weight(weights,
                 hgtxr::e2e::kPatchWeightElemBase +
                     c * hgtxr::e2e::kPatchElems + p,
                 kPatchRaw[c]);
    }
    set_weight(weights,
               hgtxr::e2e::kHeadWeightElemBase + c * hgtxr::e2e::kEmbed + c,
               kHeadRaw[c]);
  }
  for (int i = 0; i < kPatchExtraEntriesCount; ++i) {
    for (int p = 0; p < hgtxr::e2e::kPatchElems; ++p) {
      set_weight(weights,
                 hgtxr::e2e::kPatchWeightElemBase +
                     kPatchExtraEntries[i].channel * hgtxr::e2e::kPatchElems + p,
                 kPatchExtraEntries[i].raw);
    }
  }
  for (int i = 0; i < kHeadExtraEntriesCount; ++i) {
    set_weight(weights,
               hgtxr::e2e::kHeadWeightElemBase +
                   kHeadExtraEntries[i].row * hgtxr::e2e::kEmbed +
                   kHeadExtraEntries[i].col,
               kHeadExtraEntries[i].raw);
  }

  for (int block = 0; block < HGTXR_E2E_BLOCKS; ++block) {
    const int block_base =
        hgtxr::e2e::kBlockWeightElemBase +
        block * hgtxr::e2e::kBlockWeightElems;
    for (int c = 0; c < hgtxr::e2e::kEmbed; ++c) {
      const int group = c % kGroupChannels;
#if defined(HGTXR_E2E_USE_HGPIPE_LNQ_GAMMA_RAW) || defined(HGTXR_E2E_USE_HGPIPE_MATH_LNQ_GOLDEN) || defined(HGTXR_E2E_USE_HGPIPE_MATH_LNQ_ACTIVE8_GOLDEN) || defined(HGTXR_E2E_USE_HGPIPE_MATH_LNQ_ACTIVE16_GOLDEN)
      const int ln_gamma_raw = kLnGammaRaw;
#else
      const int ln_gamma_raw = 0;
#endif
      set_weight(weights, block_base + hgtxr::e2e::kBlockLn1Gamma + c,
                 ln_gamma_raw);
      set_weight(weights, block_base + hgtxr::e2e::kBlockLn1Beta + c,
                 kLnBetaRaw);
      set_weight(weights, block_base + hgtxr::e2e::kBlockLn2Gamma + c,
                 ln_gamma_raw);
      set_weight(weights, block_base + hgtxr::e2e::kBlockLn2Beta + c,
                 kLnBetaRaw);
      set_weight(weights,
                 block_base + hgtxr::e2e::kBlockWq +
                     c * hgtxr::e2e::kEmbed + c,
                 kQkvDiagRaw);
      set_weight(weights,
                 block_base + hgtxr::e2e::kBlockWk +
                     c * hgtxr::e2e::kEmbed + c,
                 kQkvDiagRaw);
      set_weight(weights,
                 block_base + hgtxr::e2e::kBlockWv +
                     c * hgtxr::e2e::kEmbed + c,
                 kQkvDiagRaw);
      set_weight(weights,
                 block_base + hgtxr::e2e::kBlockWo +
                     c * hgtxr::e2e::kEmbed + group,
                 kWoGroupRaw);
      set_weight(weights,
                 block_base + hgtxr::e2e::kBlockW1 +
                     c * hgtxr::e2e::kFfDim + group,
                 kMlpW1GroupRaw);
    }
    for (int h = 0; h < kGroupChannels; ++h) {
      set_weight(weights,
                 block_base + hgtxr::e2e::kBlockW2 +
                     h * hgtxr::e2e::kEmbed + h,
                 kMlpW2DiagRaw);
    }
    for (int i = 0; i < kQkvExtraEntriesCount; ++i) {
      const HgtxrQkvEntry &entry = kQkvExtraEntries[i];
      set_weight(weights,
                 block_base + hgtxr::e2e::kBlockWq +
                     entry.input * hgtxr::e2e::kEmbed + entry.output,
                 entry.q);
      set_weight(weights,
                 block_base + hgtxr::e2e::kBlockWk +
                     entry.input * hgtxr::e2e::kEmbed + entry.output,
                 entry.k);
      set_weight(weights,
                 block_base + hgtxr::e2e::kBlockWv +
                     entry.input * hgtxr::e2e::kEmbed + entry.output,
                 entry.v);
    }
    for (int i = 0; i < kWoExtraEntriesCount; ++i) {
      const HgtxrMatrixEntry &entry = kWoExtraEntries[i];
      set_weight(weights,
                 block_base + hgtxr::e2e::kBlockWo +
                     entry.row * hgtxr::e2e::kEmbed + entry.col,
                 entry.raw);
    }
    for (int i = 0; i < kMlpW1ExtraEntriesCount; ++i) {
      const HgtxrMatrixEntry &entry = kMlpW1ExtraEntries[i];
      set_weight(weights,
                 block_base + hgtxr::e2e::kBlockW1 +
                     entry.row * hgtxr::e2e::kFfDim + entry.col,
                 entry.raw);
    }
    for (int i = 0; i < kMlpW2ExtraEntriesCount; ++i) {
      const HgtxrMatrixEntry &entry = kMlpW2ExtraEntries[i];
      set_weight(weights,
                 block_base + hgtxr::e2e::kBlockW2 +
                     entry.row * hgtxr::e2e::kEmbed + entry.col,
                 entry.raw);
    }
  }
  set_live_weight_bit(weights);
}

void init_frame(hgtxr_data_t frame[HGTXR_HEIGHT][HGTXR_WIDTH]) {
  for (int y = 0; y < HGTXR_HEIGHT; ++y) {
    for (int x = 0; x < HGTXR_WIDTH; ++x) {
      frame[y][x] = static_cast<hgtxr_data_t>((x + y) & 0xff) /
                    static_cast<hgtxr_data_t>(128);
    }
  }
}

}  // namespace

int main() {
  static hgtxr_data_t frame[HGTXR_HEIGHT][HGTXR_WIDTH];
  static volatile hgtxr::cyclic_transformer::HgtxrAxiWordT
      weights[HGTXR_E2E_WEIGHT_DEPTH];
  hgtxr_data_t out_state[HGTXR_STATE];
  int runtime_state = 0;

  init_q4_vector_weights(weights);
  init_frame(frame);

  hgtxr_e2e_m_axi_top(frame, weights, out_state, &runtime_state);

  int failures = 0;
  for (int i = 0; i < HGTXR_STATE; ++i) {
    const int actual = static_cast<int>(
        out_state[i] * static_cast<hgtxr_data_t>(16));
#if HGTXR_E2E_STRICT_GOLDEN
    if (actual != kExpectedRaw[i]) {
      std::printf("FAIL: e2e_m_axi output[%d] actual=%d expected=%d\n", i,
                  actual, kExpectedRaw[i]);
      ++failures;
    }
#endif
    std::printf("e2e_m_axi_out[%d]=%d\n", i, actual);
  }

  std::printf("runtime_state=%d failures=%d\n", runtime_state, failures);
  if (runtime_state != kExpectedRuntimeState) {
    return 1;
  }
#if HGTXR_E2E_STRICT_GOLDEN
  if (failures != 0) {
    return 1;
  }
#endif
  std::printf("E2E m_axi vector comparison passed\n");
  return 0;
}
