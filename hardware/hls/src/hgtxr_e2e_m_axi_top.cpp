#include "../include/hgtxr_e2e_vit.hpp"

void hgtxr_e2e_m_axi_top(
    const hgtxr_data_t frame[HGTXR_HEIGHT][HGTXR_WIDTH],
    const volatile hgtxr::cyclic_transformer::HgtxrAxiWordT *weights,
    hgtxr_data_t out_state[HGTXR_STATE],
    int *runtime_state) {
#pragma HLS INTERFACE m_axi port=frame offset=slave bundle=gmem_frame depth=65536 max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi port=weights offset=slave bundle=gmem_e2e_weights depth=HGTXR_E2E_WEIGHT_DEPTH max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi port=out_state offset=slave bundle=gmem_out_state depth=6 max_write_burst_length=16 num_write_outstanding=2
#pragma HLS INTERFACE m_axi port=runtime_state offset=slave bundle=gmem_runtime_state depth=1 max_write_burst_length=2 num_write_outstanding=2
#pragma HLS INTERFACE s_axilite port=frame bundle=control
#pragma HLS INTERFACE s_axilite port=weights bundle=control
#pragma HLS INTERFACE s_axilite port=out_state bundle=control
#pragma HLS INTERFACE s_axilite port=runtime_state bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  static token_buffer_t tokens;
  static hgtxr::e2e::HgtxrGlobalBuffer gb;
  hgtxr_data_t local_out[HGTXR_STATE];
  hgtxr::cyclic_transformer::HgtxrAxiWordT live_weight_word = weights[0];
  ap_uint<1> live_weight_bit = live_weight_word.range(0, 0);
#if HGTXR_E2E_SMALL_MEM_LUTRAM
#pragma HLS bind_storage variable=gb.pooled type=ram_2p impl=lutram
#else
#pragma HLS bind_storage variable=gb.pooled type=ram_2p impl=bram
#endif
#if HGTXR_E2E_URAM_FRAME_TOKENS
#pragma HLS bind_storage variable=tokens type=ram_2p impl=uram
#else
#pragma HLS bind_storage variable=tokens type=ram_2p impl=bram
#endif
#if HGTXR_E2E_URAM_GB_TOKENS
#pragma HLS bind_storage variable=gb.tokens type=ram_2p impl=uram
#else
#pragma HLS bind_storage variable=gb.tokens type=ram_2p impl=bram
#endif
#if HGTXR_E2E_URAM_NORM
#pragma HLS bind_storage variable=gb.norm type=ram_2p impl=uram
#else
#pragma HLS bind_storage variable=gb.norm type=ram_2p impl=bram
#endif
#if HGTXR_E2E_URAM_Q
#pragma HLS bind_storage variable=gb.q type=ram_2p impl=uram
#else
#pragma HLS bind_storage variable=gb.q type=ram_2p impl=bram
#endif
#if HGTXR_E2E_URAM_K
#pragma HLS bind_storage variable=gb.k type=ram_2p impl=uram
#else
#pragma HLS bind_storage variable=gb.k type=ram_2p impl=bram
#endif
#if HGTXR_E2E_URAM_V
#pragma HLS bind_storage variable=gb.v type=ram_2p impl=uram
#else
#pragma HLS bind_storage variable=gb.v type=ram_2p impl=bram
#endif
#if HGTXR_E2E_URAM_ATTN
#pragma HLS bind_storage variable=gb.attn type=ram_2p impl=uram
#else
#pragma HLS bind_storage variable=gb.attn type=ram_2p impl=bram
#endif
#if HGTXR_E2E_LUTRAM_HIDDEN
#pragma HLS bind_storage variable=gb.hidden type=ram_2p impl=lutram
#elif HGTXR_E2E_URAM_HIDDEN
#pragma HLS bind_storage variable=gb.hidden type=ram_2p impl=uram
#else
#pragma HLS bind_storage variable=gb.hidden type=ram_2p impl=bram
#endif
#pragma HLS ARRAY_PARTITION variable=gb.tokens cyclic factor=HGTXR_E2E_DENSE_PAR dim=2
#pragma HLS ARRAY_PARTITION variable=gb.norm cyclic factor=HGTXR_E2E_DENSE_PAR dim=2
#pragma HLS ARRAY_PARTITION variable=gb.q cyclic factor=HGTXR_E2E_DENSE_PAR dim=2
#pragma HLS ARRAY_PARTITION variable=gb.k cyclic factor=HGTXR_E2E_DENSE_PAR dim=2
#pragma HLS ARRAY_PARTITION variable=gb.v cyclic factor=HGTXR_E2E_DENSE_PAR dim=2
#pragma HLS ARRAY_PARTITION variable=gb.attn cyclic factor=HGTXR_E2E_DENSE_PAR dim=2
#pragma HLS ARRAY_PARTITION variable=gb.hidden cyclic factor=HGTXR_E2E_DENSE_PAR dim=2
#pragma HLS ARRAY_PARTITION variable=tokens cyclic factor=HGTXR_E2E_DENSE_PAR dim=2
#pragma HLS ARRAY_PARTITION variable=local_out complete dim=1

  hgtxr::e2e::hgtxr_conv_patch_embedding(frame, weights, tokens);
  hgtxr::e2e::hgtxr_global_buffer_load(tokens, gb);
  hgtxr::e2e::hgtxr_e2e_controller_run(gb, weights);
  hgtxr::e2e::hgtxr_e2e_mlp_head(gb, weights, local_out);
  local_out[0] = local_out[0] + static_cast<hgtxr_data_t>(live_weight_bit);

  for (int i = 0; i < HGTXR_STATE; ++i) {
#pragma HLS PIPELINE II=1
    out_state[i] = local_out[i];
  }
  *runtime_state = 1 + static_cast<int>(live_weight_bit);
}
