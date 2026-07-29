#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>

#ifndef HGTXR_ENABLE_CYCLIC_TRANSFORMER_TOP
#define HGTXR_ENABLE_CYCLIC_TRANSFORMER_TOP 1
#endif

#include "../src/hgtxr_top.cpp"

namespace {

static constexpr int kTokens = 2;
static constexpr int kLayers = 6;
static constexpr int kValuesPerLayer = kTokens * HGTXR_EMBED;
static constexpr double kTolerance = 0.063;
static constexpr const char *kWeightPath =
    "hardware/refs/weights/cyclic_weights_s2_block_software_initial_q4_head64.bin";
static constexpr const char *kExpectedPath =
    "docs/resources/s2_block_full_q4w8a_head64_outputs_2026_06_10.f32bin";

static hgtxr::cyclic_transformer::HgtxrAxiWordT
    weights[hgtxr::cyclic_transformer::HGTXR_CYCLIC_WEIGHT_TOTAL_WORDS];
static float expected[kLayers][kTokens][HGTXR_EMBED];


bool open_input_file(std::ifstream &in, const char *path) {
  in.open(path, std::ios::binary);
  if (in) {
    return true;
  }
  in.close();
  const std::string csim_relative = std::string("../../../../../..") + "/" + path;
  in.open(csim_relative.c_str(), std::ios::binary);
  if (in) {
    return true;
  }
  in.close();
  const std::string workspace_absolute =
      std::string("/home/user/project/PRJXR/HGTXR/") + path;
  in.open(workspace_absolute.c_str(), std::ios::binary);
  return static_cast<bool>(in);
}

bool load_weights() {
  std::ifstream in;
  if (!open_input_file(in, kWeightPath)) {
    std::printf("FAIL: open weights %s\n", kWeightPath);
    return false;
  }
  for (int word_idx = 0;
       word_idx < hgtxr::cyclic_transformer::HGTXR_CYCLIC_WEIGHT_TOTAL_WORDS;
       ++word_idx) {
    unsigned char bytes[HGTXR_BUS_WIDTH / 8] = {0};
    in.read(reinterpret_cast<char *>(bytes), sizeof(bytes));
    if (in.gcount() != static_cast<std::streamsize>(sizeof(bytes))) {
      std::printf("FAIL: short weight read at word %d\n", word_idx);
      return false;
    }
    hgtxr::cyclic_transformer::HgtxrAxiWordT word = 0;
    for (int byte_idx = 0; byte_idx < static_cast<int>(sizeof(bytes)); ++byte_idx) {
      word.range(8 * byte_idx + 7, 8 * byte_idx) = bytes[byte_idx];
    }
    weights[word_idx] = word;
  }
  return true;
}

bool load_expected() {
  std::ifstream in;
  if (!open_input_file(in, kExpectedPath)) {
    std::printf("FAIL: open expected %s\n", kExpectedPath);
    return false;
  }
  in.read(reinterpret_cast<char *>(expected), sizeof(expected));
  if (in.gcount() != static_cast<std::streamsize>(sizeof(expected))) {
    std::printf("FAIL: short expected read\n");
    return false;
  }
  return true;
}

void init_tokens(token_buffer_t tokens) {
  const double start = -1.5;
  const double stop = 1.75;
  const int count = kTokens * HGTXR_EMBED;
  for (int token = 0; token < HGTXR_TOKENS; ++token) {
    for (int channel = 0; channel < HGTXR_EMBED; ++channel) {
      tokens[token][channel] = 0;
    }
  }
  for (int token = 0; token < kTokens; ++token) {
    for (int channel = 0; channel < HGTXR_EMBED; ++channel) {
      const int idx = token * HGTXR_EMBED + channel;
      const double value = start + (stop - start) * idx / (count - 1);
      tokens[token][channel] = static_cast<hgtxr_data_t>(value);
    }
  }
}

double compare_layer(int layer, const token_buffer_t tokens) {
  double max_abs = 0.0;
  double sum_abs = 0.0;
  int max_token = 0;
  int max_channel = 0;
  double max_actual = 0.0;
  double max_ref = 0.0;
  for (int token = 0; token < kTokens; ++token) {
    for (int channel = 0; channel < HGTXR_EMBED; ++channel) {
      const double actual = static_cast<double>(tokens[token][channel]);
      const double ref = static_cast<double>(expected[layer][token][channel]);
      const double diff = std::fabs(actual - ref);
      if (diff > max_abs) {
        max_abs = diff;
        max_token = token;
        max_channel = channel;
        max_actual = actual;
        max_ref = ref;
      }
      sum_abs += diff;
    }
  }
  std::printf(
      "layer=%d max_abs_diff=%.8f mean_abs_diff=%.8f max_token=%d "
      "max_channel=%d max_actual=%.8f max_ref=%.8f sample_actual=%.8f "
      "sample_ref=%.8f\n",
      layer, max_abs, sum_abs / kValuesPerLayer, max_token, max_channel,
      max_actual, max_ref, static_cast<double>(tokens[0][0]),
      static_cast<double>(expected[layer][0][0]));
  return max_abs;
}

}  // namespace

int hgtxr_s2_block_vector_debug() {
  static_assert(HGTXR_CYCLIC_HEADS == 3, "expected heads=3");
  static_assert(HGTXR_CYCLIC_HEAD_DIM == 64, "expected head_dim=64");
  static_assert(HGTXR_TILE_CHANNELS == 64, "expected head-aligned tile");
  static_assert(HGTXR_WEIGHT_WIDTH == 4, "expected Q4 weights");
  if (!load_weights() || !load_expected()) {
    return 1;
  }

  token_buffer_t tokens;
  double global_max = 0.0;
  for (int layer = 0; layer < kLayers; ++layer) {
    init_tokens(tokens);
    hgtxr_cyclic_transformer_stage<kTokens>(tokens, weights, layer);
    const double layer_max = compare_layer(layer, tokens);
    if (layer_max > global_max) {
      global_max = layer_max;
    }
  }
  std::printf("s2_block_vector_global_max_abs_diff=%.8f\n", global_max);
  if (global_max > kTolerance) {
    std::printf("FAIL: S2 block vector mismatch\n");
    return 1;
  }
  std::printf("S2 block vector comparison passed\n");
  return 0;
}

#ifndef HGTXR_S2_BLOCK_VECTOR_NO_MAIN
int main() { return hgtxr_s2_block_vector_debug(); }
#endif

void frame_patch_embed(const hgtxr_data_t[HGTXR_HEIGHT][HGTXR_WIDTH],
                       token_buffer_t) {}
void event_patch_embed(const hgtxr_data_t[HGTXR_HEIGHT][HGTXR_WIDTH],
                       const hgtxr_data_t[HGTXR_HEIGHT][HGTXR_WIDTH],
                       token_buffer_t) {}
void pool_tokens(const token_buffer_t, pooled_buffer_t) {}
void matmul(const hgtxr_data_t *, const hgtxr_data_t *, hgtxr_data_t *, int, int,
            int) {}
void attention_stage(token_buffer_t) {}
void mlp_stage(token_buffer_t) {}
void fusion(const pooled_buffer_t, const pooled_buffer_t, const hgtxr_data_t[HGTXR_STATE],
            hgtxr_data_t[HGTXR_EMBED * 2]) {}
void search_head(const pooled_buffer_t, hgtxr_data_t[HGTXR_SEARCH_LOGITS]) {}
void track_head(const hgtxr_data_t[HGTXR_EMBED * 2],
                const hgtxr_data_t[HGTXR_STATE], hgtxr_data_t[HGTXR_STATE]) {}
HGTXRDecision runtime_fsm(hgtxr_data_t, hgtxr_data_t, hgtxr_data_t,
                          hgtxr_data_t, hgtxr_data_t, bool) {
  HGTXRDecision d;
  d.state = 0;
  d.reason = 0;
  return d;
}
void global_buffer_write_search(const token_buffer_t, token_buffer_t) {}
void global_buffer_write_track(const token_buffer_t, token_buffer_t) {}
void noc_route_tokens(const token_buffer_t, token_buffer_t, int) {}
void weight_prefetcher(int, int, int *prefetch_valid) { *prefetch_valid = 1; }
int controller_active_tokens(int) { return HGTXR_SEARCH_TOKENS; }
int controller_depth_limit(int) { return HGTXR_SEARCH_DEPTH; }
void layernorm_stage(token_buffer_t) {}
void softmax_stage(token_buffer_t) {}
void gelu_stage(token_buffer_t) {}
void rmu_projection_stage(token_buffer_t) {}
void smu_relation_stage(token_buffer_t) {}
