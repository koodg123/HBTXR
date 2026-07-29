#include "../include/hgtxr_e2e_vit.hpp"

void hgtxr_e2e_axis_top(
    hls::stream<hgtxr::e2e::HgtxrAxisWord> &axis_in,
    hls::stream<hgtxr::e2e::HgtxrAxisWord> &axis_out,
    const volatile hgtxr::cyclic_transformer::HgtxrAxiWordT *weights,
    int num_pixels,
    int *runtime_state) {
#pragma HLS INTERFACE axis port=axis_in depth=HGTXR_STREAM_FIFO_DEPTH
#pragma HLS INTERFACE axis port=axis_out depth=HGTXR_STREAM_FIFO_DEPTH
#if !HGTXR_E2E_OMIT_WEIGHT_AXI
#if HGTXR_E2E_LOW_FANOUT_WEIGHT_AXI
#pragma HLS INTERFACE m_axi port=weights offset=slave bundle=gmem_e2e_weights depth=HGTXR_E2E_WEIGHT_DEPTH max_read_burst_length=16 num_read_outstanding=2
#else
#pragma HLS INTERFACE m_axi port=weights offset=slave bundle=gmem_e2e_weights depth=HGTXR_E2E_WEIGHT_DEPTH max_read_burst_length=64 num_read_outstanding=8
#endif
#endif
#pragma HLS INTERFACE m_axi port=runtime_state offset=slave bundle=gmem_e2e_runtime depth=1 max_write_burst_length=2 num_write_outstanding=2
#pragma HLS INTERFACE s_axilite port=weights bundle=control
#pragma HLS INTERFACE s_axilite port=num_pixels bundle=control
#pragma HLS INTERFACE s_axilite port=runtime_state bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  static hgtxr_data_t frame[HGTXR_HEIGHT][HGTXR_WIDTH];
  static token_buffer_t tokens;
  static hgtxr::e2e::HgtxrGlobalBuffer gb;
  hgtxr_data_t out_state[HGTXR_STATE];
#if HGTXR_E2E_OMIT_WEIGHT_AXI
  (void)weights;
  ap_uint<1> live_weight_bit = 0;
#else
  hgtxr::cyclic_transformer::HgtxrAxiWordT live_weight_word = weights[0];
  ap_uint<1> live_weight_bit = live_weight_word.range(0, 0);
#endif
#pragma HLS bind_storage variable=frame type=ram_2p impl=bram
#if HGTXR_E2E_DISPATCH_PREFETCH_FULL_BLOCK || HGTXR_E2E_URAM_DISPATCH_PREFETCH
#pragma HLS bind_storage variable=gb.dispatch_prefetch type=ram_2p impl=uram
#endif
#if HGTXR_E2E_SMALL_MEM_LUTRAM
#pragma HLS bind_storage variable=gb.pooled type=ram_2p impl=lutram
#if !(HGTXR_E2E_DISPATCH_PREFETCH_FULL_BLOCK || HGTXR_E2E_URAM_DISPATCH_PREFETCH)
#pragma HLS bind_storage variable=gb.dispatch_prefetch type=ram_2p impl=lutram
#endif
#else
#pragma HLS bind_storage variable=gb.pooled type=ram_2p impl=bram
#if !(HGTXR_E2E_DISPATCH_PREFETCH_FULL_BLOCK || HGTXR_E2E_URAM_DISPATCH_PREFETCH)
#pragma HLS bind_storage variable=gb.dispatch_prefetch type=ram_2p impl=bram
#endif
#endif
#if HGTXR_E2E_LUTRAM_FRAME_TOKENS
#pragma HLS bind_storage variable=tokens type=ram_2p impl=lutram
#elif HGTXR_E2E_URAM_FRAME_TOKENS
#pragma HLS bind_storage variable=tokens type=ram_2p impl=uram
#else
#pragma HLS bind_storage variable=tokens type=ram_2p impl=bram
#endif
#if HGTXR_E2E_LUTRAM_GB_TOKENS
#pragma HLS bind_storage variable=gb.tokens type=ram_2p impl=lutram
#elif HGTXR_E2E_URAM_GB_TOKENS
#pragma HLS bind_storage variable=gb.tokens type=ram_2p impl=uram
#else
#pragma HLS bind_storage variable=gb.tokens type=ram_2p impl=bram
#endif
#if HGTXR_E2E_LUTRAM_NORM
#pragma HLS bind_storage variable=gb.norm type=ram_2p impl=lutram
#elif HGTXR_E2E_URAM_NORM
#pragma HLS bind_storage variable=gb.norm type=ram_2p impl=uram
#else
#pragma HLS bind_storage variable=gb.norm type=ram_2p impl=bram
#endif
#if HGTXR_E2E_LUTRAM_Q
#pragma HLS bind_storage variable=gb.q type=ram_2p impl=lutram
#elif HGTXR_E2E_URAM_Q
#pragma HLS bind_storage variable=gb.q type=ram_2p impl=uram
#else
#pragma HLS bind_storage variable=gb.q type=ram_2p impl=bram
#endif
#if HGTXR_E2E_LUTRAM_K
#pragma HLS bind_storage variable=gb.k type=ram_2p impl=lutram
#elif HGTXR_E2E_URAM_K
#pragma HLS bind_storage variable=gb.k type=ram_2p impl=uram
#else
#pragma HLS bind_storage variable=gb.k type=ram_2p impl=bram
#endif
#if HGTXR_E2E_LUTRAM_V
#pragma HLS bind_storage variable=gb.v type=ram_2p impl=lutram
#elif HGTXR_E2E_URAM_V
#pragma HLS bind_storage variable=gb.v type=ram_2p impl=uram
#else
#pragma HLS bind_storage variable=gb.v type=ram_2p impl=bram
#endif
#if HGTXR_E2E_LUTRAM_ATTN
#pragma HLS bind_storage variable=gb.attn type=ram_2p impl=lutram
#elif HGTXR_E2E_URAM_ATTN
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
#pragma HLS ARRAY_PARTITION variable=gb.tokens cyclic factor=HGTXR_E2E_MEM_BANK_PAR dim=2
#pragma HLS ARRAY_PARTITION variable=gb.norm cyclic factor=HGTXR_E2E_MEM_BANK_PAR dim=2
#pragma HLS ARRAY_PARTITION variable=gb.q cyclic factor=HGTXR_E2E_MEM_BANK_PAR dim=2
#pragma HLS ARRAY_PARTITION variable=gb.k cyclic factor=HGTXR_E2E_MEM_BANK_PAR dim=2
#pragma HLS ARRAY_PARTITION variable=gb.v cyclic factor=HGTXR_E2E_MEM_BANK_PAR dim=2
#pragma HLS ARRAY_PARTITION variable=gb.attn cyclic factor=HGTXR_E2E_MEM_BANK_PAR dim=2
#pragma HLS ARRAY_PARTITION variable=gb.hidden cyclic factor=HGTXR_E2E_MEM_BANK_PAR dim=2
#pragma HLS ARRAY_PARTITION variable=tokens cyclic factor=HGTXR_E2E_MEM_BANK_PAR dim=2
#if HGTXR_E2E_TOKEN_BANK_PAR > 1
#pragma HLS ARRAY_PARTITION variable=gb.tokens cyclic factor=HGTXR_E2E_TOKEN_BANK_PAR dim=1
#pragma HLS ARRAY_PARTITION variable=gb.norm cyclic factor=HGTXR_E2E_TOKEN_BANK_PAR dim=1
#pragma HLS ARRAY_PARTITION variable=gb.q cyclic factor=HGTXR_E2E_TOKEN_BANK_PAR dim=1
#pragma HLS ARRAY_PARTITION variable=gb.k cyclic factor=HGTXR_E2E_TOKEN_BANK_PAR dim=1
#pragma HLS ARRAY_PARTITION variable=gb.v cyclic factor=HGTXR_E2E_TOKEN_BANK_PAR dim=1
#pragma HLS ARRAY_PARTITION variable=gb.attn cyclic factor=HGTXR_E2E_TOKEN_BANK_PAR dim=1
#pragma HLS ARRAY_PARTITION variable=gb.hidden cyclic factor=HGTXR_E2E_TOKEN_BANK_PAR dim=1
#pragma HLS ARRAY_PARTITION variable=tokens cyclic factor=HGTXR_E2E_TOKEN_BANK_PAR dim=1
#endif
#pragma HLS ARRAY_PARTITION variable=out_state complete dim=1

  const hgtxr::e2e::HgtxrRuntimeSchedule schedule =
      hgtxr::e2e::hgtxr_e2e_make_runtime_schedule(num_pixels);
  const int mode = schedule.mode;
  const int active_tokens = schedule.active_tokens;
  const int depth_limit = schedule.depth_limit;
  const int read_h = mode == HGTXR_MODE_TRACK ? HGTXR_TRACK_H : HGTXR_SEARCH_H;
  const int read_w = mode == HGTXR_MODE_TRACK ? HGTXR_TRACK_W : HGTXR_SEARCH_W;
  hgtxr::e2e::hgtxr_axis_read_frame(axis_in, frame, read_h, read_w);
#if HGTXR_E2E_USE_MODE_PROFILE_FASTPATH
  hgtxr::e2e::hgtxr_e2e_fast_patch_embedding(frame, gb, active_tokens);
  hgtxr::e2e::hgtxr_e2e_controller_run(gb, weights, depth_limit,
                                       active_tokens);
  hgtxr::e2e::hgtxr_e2e_fast_state_head(gb, out_state, active_tokens);
#else
  if (mode == HGTXR_MODE_TRACK) {
    hgtxr::e2e::hgtxr_event_conv_patch_embedding(frame, weights, tokens,
                                                 active_tokens);
  } else {
    hgtxr::e2e::hgtxr_conv_patch_embedding(frame, weights, tokens, active_tokens);
  }
  hgtxr::e2e::hgtxr_global_buffer_load(tokens, gb, active_tokens);
  hgtxr::e2e::hgtxr_e2e_controller_run(gb, weights, depth_limit,
                                       active_tokens);
  hgtxr::e2e::hgtxr_e2e_mlp_head(gb, weights, out_state, active_tokens);
#endif
#if HGTXR_E2E_OBSERVE_DATAPATH
  out_state[HGTXR_STATE - 1] =
      out_state[HGTXR_STATE - 1] +
      hgtxr::e2e::hgtxr_e2e_datapath_observable_mix(gb, active_tokens);
#endif
  out_state[0] = out_state[0] + static_cast<hgtxr_data_t>(live_weight_bit);
  hgtxr::e2e::hgtxr_axis_write_state_values(
      axis_out, out_state[0], out_state[1], out_state[2], out_state[3],
      out_state[4], out_state[5]);
#if HGTXR_E2E_REPORT_MODE_STATE
  *runtime_state = mode;
#else
  *runtime_state = 1 + static_cast<int>(live_weight_bit);
#endif
}
