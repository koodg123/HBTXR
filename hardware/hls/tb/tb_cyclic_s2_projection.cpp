#include "hgtxr_cyclic_s2_projection.hpp"

#include <cstdio>

using namespace hgtxr::cyclic_transformer;

static int check(bool cond, const char *name) {
  if (!cond) {
    std::printf("FAIL: %s\n", name);
    return 1;
  }
  std::printf("PASS: %s\n", name);
  return 0;
}

static HgtxrAxiWordT pack_lane_word(int lane, HgtxrDataT value) {
  HgtxrDataIntT bits = 0;
  bits.range(HGTXR_DATA_WIDTH - 1, 0) = value.range(HGTXR_DATA_WIDTH - 1, 0);
  HgtxrAxiWordT word = (HgtxrAxiWordT)((ap_uint<HGTXR_DATA_WIDTH>)bits);
  return word << (lane * HGTXR_DATA_WIDTH);
}

static void store_weight_elem(HgtxrAxiWordT *words, int elem_offset,
                              HgtxrDataT value) {
  const int word_idx = elem_offset / HGTXR_AXI_WEIGHT_LANES;
  const int lane_idx = elem_offset % HGTXR_AXI_WEIGHT_LANES;
  words[word_idx] = words[word_idx] | pack_lane_word(lane_idx, value);
}

int main() {
  int failures = 0;
  failures += check(hgtxr_s2_channel_pair_block_index(0, 0, 0, 6) == 0,
                    "s2 index first");
  failures += check(hgtxr_s2_channel_pair_block_index(0, 1, 0, 6) == 6,
                    "s2 index output stride");
  failures += check(hgtxr_s2_channel_pair_block_index(5, 5, 5, 6) == 215,
                    "s2 index last");
  failures += check(hgtxr_s2_channel_pair_word_base(1, 0, 0, 6) ==
                        36 * HGTXR_CYCLIC_WEIGHT_LAYER_STRIDE_WORDS,
                    "s2 word base layer stride");

  HgtxrAxiWordT words[2];
  words[0] = pack_lane_word(0, (HgtxrDataT)1) | pack_lane_word(1, (HgtxrDataT)-2);
  words[1] = pack_lane_word(0, (HgtxrDataT)3);
  failures += check(hgtxr_load_weight_elem(words, 0) == (HgtxrDataT)1,
                    "load elem lane0");
  failures += check(hgtxr_load_weight_elem(words, 1) == (HgtxrDataT)-2,
                    "load elem lane1");
  failures += check(hgtxr_load_weight_elem(words, HGTXR_AXI_WEIGHT_LANES) ==
                        (HgtxrDataT)3,
                    "load elem next word");

  HgtxrDataT x0[2][2] = {{1, 2}, {3, 4}};
  HgtxrDataT w0[2][2] = {{1, 0}, {0, 1}};
  HgtxrDataT x1[2][2] = {{5, 6}, {7, 8}};
  HgtxrDataT w1[2][2] = {{1, 1}, {1, 1}};
  HgtxrAccumT acc[2][2];
  hgtxr_s2_project_pair_accum<2, 2, 2>(x0, w0, acc, true);
  hgtxr_s2_project_pair_accum<2, 2, 2>(x1, w1, acc, false);
  failures += check(acc[0][0] == (HgtxrAccumT)12, "s2 accum 0,0");
  failures += check(acc[0][1] == (HgtxrAccumT)13, "s2 accum 0,1");
  failures += check(acc[1][0] == (HgtxrAccumT)18, "s2 accum 1,0");
  failures += check(acc[1][1] == (HgtxrAccumT)19, "s2 accum 1,1");

  HgtxrAxiWordT packed_weights[2 * HGTXR_CYCLIC_WEIGHT_LAYER_STRIDE_WORDS] = {};
  const int base0 = hgtxr_s2_channel_pair_word_base(0, 0, 0, 2);
  const int base1 = hgtxr_s2_channel_pair_word_base(0, 0, 1, 2);
  store_weight_elem(packed_weights + base0, HGTXR_CYCLIC_WEIGHT_WO_OFFSET + 0,
                    (HgtxrDataT)1);
  store_weight_elem(packed_weights + base0, HGTXR_CYCLIC_WEIGHT_WO_OFFSET + 1,
                    (HgtxrDataT)0);
  store_weight_elem(packed_weights + base0, HGTXR_CYCLIC_WEIGHT_WO_OFFSET + 2,
                    (HgtxrDataT)0);
  store_weight_elem(packed_weights + base0, HGTXR_CYCLIC_WEIGHT_WO_OFFSET + 3,
                    (HgtxrDataT)1);
  store_weight_elem(packed_weights + base1, HGTXR_CYCLIC_WEIGHT_WO_OFFSET + 0,
                    (HgtxrDataT)1);
  store_weight_elem(packed_weights + base1, HGTXR_CYCLIC_WEIGHT_WO_OFFSET + 1,
                    (HgtxrDataT)1);
  store_weight_elem(packed_weights + base1, HGTXR_CYCLIC_WEIGHT_WO_OFFSET + 2,
                    (HgtxrDataT)1);
  store_weight_elem(packed_weights + base1, HGTXR_CYCLIC_WEIGHT_WO_OFFSET + 3,
                    (HgtxrDataT)1);

  HgtxrAccumT packed_acc[2][2];
  hgtxr_s2_project_pair_accum<2, 2, 2>(
      packed_weights, 0, 0, 0, 2, HGTXR_CYCLIC_WEIGHT_WO_OFFSET, x0,
      packed_acc, true);
  hgtxr_s2_project_pair_accum<2, 2, 2>(
      packed_weights, 0, 0, 1, 2, HGTXR_CYCLIC_WEIGHT_WO_OFFSET, x1,
      packed_acc, false);
  failures += check(packed_acc[0][0] == (HgtxrAccumT)12,
                    "s2 packed wrapper 0,0");
  failures += check(packed_acc[0][1] == (HgtxrAccumT)13,
                    "s2 packed wrapper 0,1");
  failures += check(packed_acc[1][0] == (HgtxrAccumT)18,
                    "s2 packed wrapper 1,0");
  failures += check(packed_acc[1][1] == (HgtxrAccumT)19,
                    "s2 packed wrapper 1,1");

  HgtxrAxiWordT qkv_weights[2 * HGTXR_CYCLIC_WEIGHT_LAYER_STRIDE_WORDS] = {};
  const int qkv_base0 = hgtxr_s2_channel_pair_word_base(0, 0, 0, 2);
  const int qkv_base1 = hgtxr_s2_channel_pair_word_base(0, 0, 1, 2);
  store_weight_elem(qkv_weights + qkv_base0, HGTXR_CYCLIC_WEIGHT_WQ_OFFSET + 0,
                    (HgtxrDataT)1);
  store_weight_elem(qkv_weights + qkv_base0, HGTXR_CYCLIC_WEIGHT_WQ_OFFSET + 3,
                    (HgtxrDataT)1);
  store_weight_elem(qkv_weights + qkv_base1, HGTXR_CYCLIC_WEIGHT_WQ_OFFSET + 0,
                    (HgtxrDataT)1);
  store_weight_elem(qkv_weights + qkv_base1, HGTXR_CYCLIC_WEIGHT_WQ_OFFSET + 1,
                    (HgtxrDataT)1);
  store_weight_elem(qkv_weights + qkv_base1, HGTXR_CYCLIC_WEIGHT_WQ_OFFSET + 2,
                    (HgtxrDataT)1);
  store_weight_elem(qkv_weights + qkv_base1, HGTXR_CYCLIC_WEIGHT_WQ_OFFSET + 3,
                    (HgtxrDataT)1);

  store_weight_elem(qkv_weights + qkv_base0, HGTXR_CYCLIC_WEIGHT_WK_OFFSET + 0,
                    (HgtxrDataT)2);
  store_weight_elem(qkv_weights + qkv_base0, HGTXR_CYCLIC_WEIGHT_WK_OFFSET + 3,
                    (HgtxrDataT)2);
  store_weight_elem(qkv_weights + qkv_base1, HGTXR_CYCLIC_WEIGHT_WK_OFFSET + 0,
                    (HgtxrDataT)2);
  store_weight_elem(qkv_weights + qkv_base1, HGTXR_CYCLIC_WEIGHT_WK_OFFSET + 1,
                    (HgtxrDataT)2);
  store_weight_elem(qkv_weights + qkv_base1, HGTXR_CYCLIC_WEIGHT_WK_OFFSET + 2,
                    (HgtxrDataT)2);
  store_weight_elem(qkv_weights + qkv_base1, HGTXR_CYCLIC_WEIGHT_WK_OFFSET + 3,
                    (HgtxrDataT)2);

  store_weight_elem(qkv_weights + qkv_base0, HGTXR_CYCLIC_WEIGHT_WV_OFFSET + 0,
                    (HgtxrDataT)3);
  store_weight_elem(qkv_weights + qkv_base0, HGTXR_CYCLIC_WEIGHT_WV_OFFSET + 3,
                    (HgtxrDataT)3);
  store_weight_elem(qkv_weights + qkv_base1, HGTXR_CYCLIC_WEIGHT_WV_OFFSET + 0,
                    (HgtxrDataT)3);
  store_weight_elem(qkv_weights + qkv_base1, HGTXR_CYCLIC_WEIGHT_WV_OFFSET + 1,
                    (HgtxrDataT)3);
  store_weight_elem(qkv_weights + qkv_base1, HGTXR_CYCLIC_WEIGHT_WV_OFFSET + 2,
                    (HgtxrDataT)3);
  store_weight_elem(qkv_weights + qkv_base1, HGTXR_CYCLIC_WEIGHT_WV_OFFSET + 3,
                    (HgtxrDataT)3);

  HgtxrAccumT q_acc[2][2];
  HgtxrAccumT k_acc[2][2];
  HgtxrAccumT v_acc[2][2];
  hgtxr_s2_qkv_project_pair_accum<2, 2, 2>(
      qkv_weights, 0, 0, 0, 2, x0, q_acc, k_acc, v_acc, true);
  hgtxr_s2_qkv_project_pair_accum<2, 2, 2>(
      qkv_weights, 0, 0, 1, 2, x1, q_acc, k_acc, v_acc, false);
  failures += check(q_acc[0][0] == (HgtxrAccumT)12, "s2 q_acc 0,0");
  failures += check(q_acc[1][1] == (HgtxrAccumT)19, "s2 q_acc 1,1");
  failures += check(k_acc[0][0] == (HgtxrAccumT)24, "s2 k_acc 0,0");
  failures += check(k_acc[1][1] == (HgtxrAccumT)38, "s2 k_acc 1,1");
  failures += check(v_acc[0][0] == (HgtxrAccumT)36, "s2 v_acc 0,0");
  failures += check(v_acc[1][1] == (HgtxrAccumT)57, "s2 v_acc 1,1");


  failures += check(hgtxr_s2_mlp_layer_block_stride(2, 2) == 8,
                    "s2 mlp layer stride");
  failures += check(hgtxr_s2_mlp_w1_block_index(0, 1, 0, 2, 2) == 2,
                    "s2 mlp w1 index");
  failures += check(hgtxr_s2_mlp_w2_block_index(0, 1, 1, 2, 2) == 7,
                    "s2 mlp w2 index");

  HgtxrAxiWordT mlp_weights[8 * HGTXR_CYCLIC_WEIGHT_LAYER_STRIDE_WORDS] = {};
  const int mlp_w1_base0 = hgtxr_s2_mlp_w1_word_base(0, 0, 0, 2, 2);
  const int mlp_w1_base1 = hgtxr_s2_mlp_w1_word_base(0, 0, 1, 2, 2);
  store_weight_elem(mlp_weights + mlp_w1_base0, HGTXR_CYCLIC_WEIGHT_W1_OFFSET + 0,
                    (HgtxrDataT)1);
  store_weight_elem(mlp_weights + mlp_w1_base0, HGTXR_CYCLIC_WEIGHT_W1_OFFSET + 3,
                    (HgtxrDataT)1);
  store_weight_elem(mlp_weights + mlp_w1_base1, HGTXR_CYCLIC_WEIGHT_W1_OFFSET + 0,
                    (HgtxrDataT)1);
  store_weight_elem(mlp_weights + mlp_w1_base1, HGTXR_CYCLIC_WEIGHT_W1_OFFSET + 1,
                    (HgtxrDataT)1);
  store_weight_elem(mlp_weights + mlp_w1_base1, HGTXR_CYCLIC_WEIGHT_W1_OFFSET + 2,
                    (HgtxrDataT)1);
  store_weight_elem(mlp_weights + mlp_w1_base1, HGTXR_CYCLIC_WEIGHT_W1_OFFSET + 3,
                    (HgtxrDataT)1);

  HgtxrAccumT hidden_acc[2][2];
  hgtxr_s2_mlp_w1_pair_accum<2, 2, 2>(
      mlp_weights, 0, 0, 0, 2, 2, x0, hidden_acc, true);
  hgtxr_s2_mlp_w1_pair_accum<2, 2, 2>(
      mlp_weights, 0, 0, 1, 2, 2, x1, hidden_acc, false);
  failures += check(hidden_acc[0][0] == (HgtxrAccumT)12,
                    "s2 mlp w1 hidden 0,0");
  failures += check(hidden_acc[1][1] == (HgtxrAccumT)19,
                    "s2 mlp w1 hidden 1,1");

  const int mlp_w2_base0 = hgtxr_s2_mlp_w2_word_base(0, 0, 0, 2, 2);
  const int mlp_w2_base1 = hgtxr_s2_mlp_w2_word_base(0, 0, 1, 2, 2);
  store_weight_elem(mlp_weights + mlp_w2_base0, HGTXR_CYCLIC_WEIGHT_W2_OFFSET + 0,
                    (HgtxrDataT)1);
  store_weight_elem(mlp_weights + mlp_w2_base0, HGTXR_CYCLIC_WEIGHT_W2_OFFSET + 3,
                    (HgtxrDataT)1);
  store_weight_elem(mlp_weights + mlp_w2_base1, HGTXR_CYCLIC_WEIGHT_W2_OFFSET + 0,
                    (HgtxrDataT)1);
  store_weight_elem(mlp_weights + mlp_w2_base1, HGTXR_CYCLIC_WEIGHT_W2_OFFSET + 1,
                    (HgtxrDataT)1);
  store_weight_elem(mlp_weights + mlp_w2_base1, HGTXR_CYCLIC_WEIGHT_W2_OFFSET + 2,
                    (HgtxrDataT)1);
  store_weight_elem(mlp_weights + mlp_w2_base1, HGTXR_CYCLIC_WEIGHT_W2_OFFSET + 3,
                    (HgtxrDataT)1);

  HgtxrAccumT mlp_out_acc[2][2];
  hgtxr_s2_mlp_w2_pair_accum<2, 2, 2>(
      mlp_weights, 0, 0, 0, 2, 2, x0, mlp_out_acc, true);
  hgtxr_s2_mlp_w2_pair_accum<2, 2, 2>(
      mlp_weights, 0, 0, 1, 2, 2, x1, mlp_out_acc, false);
  failures += check(mlp_out_acc[0][0] == (HgtxrAccumT)12,
                    "s2 mlp w2 out 0,0");
  failures += check(mlp_out_acc[1][1] == (HgtxrAccumT)19,
                    "s2 mlp w2 out 1,1");

  if (failures != 0) {
    std::printf("cyclic s2 projection smoke failed: %d\n", failures);
    return 1;
  }
  std::printf("cyclic s2 projection smoke passed\n");
  return 0;
}
