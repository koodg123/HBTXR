#include "../include/common.h"

#ifndef HGTXR_PARALLELISM_FACTOR
#define HGTXR_PARALLELISM_FACTOR 8
#endif

#ifndef HGTXR_MODE_PROFILE_PAR
#define HGTXR_MODE_PROFILE_PAR HGTXR_PARALLELISM_FACTOR
#endif

namespace {

template <int ACTIVE_TOKENS>
using active_token_buffer_t = hgtxr_data_t[ACTIVE_TOKENS][HGTXR_EMBED];

hgtxr_acc_t mode_profile_dsp_mul(hgtxr_data_t lhs, hgtxr_data_t rhs) {
#pragma HLS INLINE
  hgtxr_acc_t product;
#pragma HLS bind_op variable=product op=mul impl=dsp
  product = static_cast<hgtxr_acc_t>(lhs) * static_cast<hgtxr_acc_t>(rhs);
  return product;
}

hgtxr_data_t mode_channel_gain(int c) {
#pragma HLS INLINE
  int bucket = c & 7;
  return static_cast<hgtxr_data_t>(0.875) +
         static_cast<hgtxr_data_t>(0.03125) * bucket;
}

hgtxr_data_t mode_token_gain(int t) {
#pragma HLS INLINE
  int bucket = t & 15;
  return static_cast<hgtxr_data_t>(0.9375) +
         static_cast<hgtxr_data_t>(0.0078125) * bucket;
}

template <int ACTIVE_TOKENS>
void mode_frame_patch_embed_active(
    const hgtxr_data_t frame[HGTXR_HEIGHT][HGTXR_WIDTH],
    active_token_buffer_t<ACTIVE_TOKENS> tokens) {
#pragma HLS INLINE off
  for (int token = 0; token < ACTIVE_TOKENS; ++token) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_SEARCH_TOKENS
    int gy = token / HGTXR_GRID_W;
    int gx = token % HGTXR_GRID_W;
    hgtxr_acc_t sum = 0;
    for (int py = 0; py < HGTXR_PATCH; ++py) {
#pragma HLS LOOP_TRIPCOUNT min=HGTXR_PATCH max=HGTXR_PATCH
      for (int px = 0; px < HGTXR_PATCH; ++px) {
#pragma HLS LOOP_TRIPCOUNT min=HGTXR_PATCH max=HGTXR_PATCH
#pragma HLS PIPELINE II=1
        sum += frame[gy * HGTXR_PATCH + py][gx * HGTXR_PATCH + px];
      }
    }
    hgtxr_data_t mean =
        static_cast<hgtxr_data_t>(sum / (HGTXR_PATCH * HGTXR_PATCH));
    for (int c_base = 0; c_base < HGTXR_EMBED;
         c_base += HGTXR_MODE_PROFILE_PAR) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_EMBED
#pragma HLS PIPELINE II=1
      for (int lane = 0; lane < HGTXR_MODE_PROFILE_PAR; ++lane) {
#pragma HLS UNROLL
        int c = c_base + lane;
        if (c < HGTXR_EMBED) {
          tokens[token][c] = mean;
        }
      }
    }
  }
}

template <int ACTIVE_TOKENS>
void mode_event_patch_embed_active(
    const hgtxr_data_t event_pos[HGTXR_HEIGHT][HGTXR_WIDTH],
    const hgtxr_data_t event_neg[HGTXR_HEIGHT][HGTXR_WIDTH],
    active_token_buffer_t<ACTIVE_TOKENS> tokens) {
#pragma HLS INLINE off
  for (int token = 0; token < ACTIVE_TOKENS; ++token) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_TRACK_TOKENS
    int gy = token / HGTXR_GRID_W;
    int gx = token % HGTXR_GRID_W;
    hgtxr_acc_t sum = 0;
    for (int py = 0; py < HGTXR_PATCH; ++py) {
#pragma HLS LOOP_TRIPCOUNT min=HGTXR_PATCH max=HGTXR_PATCH
      for (int px = 0; px < HGTXR_PATCH; ++px) {
#pragma HLS LOOP_TRIPCOUNT min=HGTXR_PATCH max=HGTXR_PATCH
#pragma HLS PIPELINE II=1
        int y = gy * HGTXR_PATCH + py;
        int x = gx * HGTXR_PATCH + px;
        sum += event_pos[y][x] - event_neg[y][x];
      }
    }
    hgtxr_data_t mean =
        static_cast<hgtxr_data_t>(sum / (HGTXR_PATCH * HGTXR_PATCH));
    for (int c_base = 0; c_base < HGTXR_EMBED;
         c_base += HGTXR_MODE_PROFILE_PAR) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_EMBED
#pragma HLS PIPELINE II=1
      for (int lane = 0; lane < HGTXR_MODE_PROFILE_PAR; ++lane) {
#pragma HLS UNROLL
        int c = c_base + lane;
        if (c < HGTXR_EMBED) {
          tokens[token][c] = mean;
        }
      }
    }
  }
}

template <int ACTIVE_TOKENS>
void mode_layernorm_active(active_token_buffer_t<ACTIVE_TOKENS> tokens) {
#pragma HLS INLINE off
  for (int t = 0; t < ACTIVE_TOKENS; ++t) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_SEARCH_TOKENS
    hgtxr_acc_t mean = 0;
    hgtxr_acc_t var = 0;
    for (int c = 0; c < HGTXR_EMBED; ++c) {
#pragma HLS LOOP_TRIPCOUNT min=HGTXR_EMBED max=HGTXR_EMBED
#pragma HLS PIPELINE II=1
      mean += tokens[t][c];
    }
    mean /= HGTXR_EMBED;
    for (int c = 0; c < HGTXR_EMBED; ++c) {
#pragma HLS LOOP_TRIPCOUNT min=HGTXR_EMBED max=HGTXR_EMBED
#pragma HLS PIPELINE II=1
      hgtxr_acc_t diff = tokens[t][c] - mean;
      var += mode_profile_dsp_mul(static_cast<hgtxr_data_t>(diff),
                                  static_cast<hgtxr_data_t>(diff));
    }
    hgtxr_data_t inv_scale = static_cast<hgtxr_data_t>(1);
    if (var > static_cast<hgtxr_acc_t>(HGTXR_EMBED)) {
      inv_scale = static_cast<hgtxr_data_t>(0.5);
    }
    for (int c_base = 0; c_base < HGTXR_EMBED;
         c_base += HGTXR_MODE_PROFILE_PAR) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_EMBED
#pragma HLS PIPELINE II=1
      for (int lane = 0; lane < HGTXR_MODE_PROFILE_PAR; ++lane) {
#pragma HLS UNROLL
        int c = c_base + lane;
        if (c < HGTXR_EMBED) {
          tokens[t][c] = clamp_data((tokens[t][c] - mean) * inv_scale);
        }
      }
    }
  }
}

template <int ACTIVE_TOKENS>
void mode_projection_active(active_token_buffer_t<ACTIVE_TOKENS> tokens) {
#pragma HLS INLINE off
  mode_layernorm_active<ACTIVE_TOKENS>(tokens);
  for (int t = 0; t < ACTIVE_TOKENS; ++t) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_SEARCH_TOKENS
    hgtxr_acc_t token_mean = 0;
    for (int c = 0; c < HGTXR_EMBED; ++c) {
#pragma HLS LOOP_TRIPCOUNT min=HGTXR_EMBED max=HGTXR_EMBED
#pragma HLS PIPELINE II=1
      token_mean += tokens[t][c];
    }
    token_mean /= HGTXR_EMBED;
    for (int c_base = 0; c_base < HGTXR_EMBED;
         c_base += HGTXR_MODE_PROFILE_PAR) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_EMBED
#pragma HLS PIPELINE II=1
      for (int lane = 0; lane < HGTXR_MODE_PROFILE_PAR; ++lane) {
#pragma HLS UNROLL
        int c = c_base + lane;
        if (c < HGTXR_EMBED) {
          hgtxr_data_t cur = tokens[t][c];
          hgtxr_data_t next =
              tokens[t][(c + 1) == HGTXR_EMBED ? 0 : c + 1];
          hgtxr_acc_t mixed =
              mode_profile_dsp_mul(cur, mode_channel_gain(c)) +
              mode_profile_dsp_mul(next, static_cast<hgtxr_data_t>(0.03125)) +
              mode_profile_dsp_mul(static_cast<hgtxr_data_t>(token_mean),
                                   static_cast<hgtxr_data_t>(0.0625));
          tokens[t][c] = clamp_data(mixed);
        }
      }
    }
  }
}

template <int ACTIVE_TOKENS>
void mode_relation_active(active_token_buffer_t<ACTIVE_TOKENS> tokens) {
#pragma HLS INLINE off
  hgtxr_data_t score[ACTIVE_TOKENS];
  hgtxr_data_t prob[ACTIVE_TOKENS];
#pragma HLS bind_storage variable=score type=ram_1p impl=lutram
#pragma HLS bind_storage variable=prob type=ram_1p impl=lutram

  hgtxr_data_t max_score = static_cast<hgtxr_data_t>(-32768);
  for (int t = 0; t < ACTIVE_TOKENS; ++t) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_SEARCH_TOKENS
    hgtxr_acc_t acc = 0;
    for (int c = 0; c < HGTXR_EMBED; c += 8) {
#pragma HLS LOOP_TRIPCOUNT min=24 max=24
#pragma HLS PIPELINE II=1
      acc += mode_profile_dsp_mul(tokens[t][c], mode_channel_gain(c));
    }
    score[t] = static_cast<hgtxr_data_t>(acc / (HGTXR_EMBED / 8));
    if (score[t] > max_score) {
      max_score = score[t];
    }
  }

  hgtxr_acc_t sum_score = 0;
  for (int t = 0; t < ACTIVE_TOKENS; ++t) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_SEARCH_TOKENS
#pragma HLS PIPELINE II=1
    hgtxr_data_t centered = score[t] - max_score;
    hgtxr_data_t relu_exp = static_cast<hgtxr_data_t>(0.00390625);
    if (centered > static_cast<hgtxr_data_t>(-1)) {
      relu_exp = static_cast<hgtxr_data_t>(1) + centered +
                 mode_profile_dsp_mul(centered, centered) *
                     static_cast<hgtxr_data_t>(0.5);
    }
    prob[t] = relu_exp;
    sum_score += relu_exp;
  }
  hgtxr_data_t recip = 0;
  if (sum_score > static_cast<hgtxr_acc_t>(0)) {
    recip = static_cast<hgtxr_data_t>(static_cast<hgtxr_data_t>(1) /
                                      static_cast<hgtxr_data_t>(sum_score));
  }

  for (int c_base = 0; c_base < HGTXR_EMBED;
       c_base += HGTXR_MODE_PROFILE_PAR) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_EMBED
    hgtxr_acc_t context[HGTXR_MODE_PROFILE_PAR];
#pragma HLS ARRAY_PARTITION variable=context complete dim=1
    for (int lane = 0; lane < HGTXR_MODE_PROFILE_PAR; ++lane) {
#pragma HLS UNROLL
      context[lane] = 0;
    }
    for (int t = 0; t < ACTIVE_TOKENS; ++t) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_SEARCH_TOKENS
#pragma HLS PIPELINE II=1
      for (int lane = 0; lane < HGTXR_MODE_PROFILE_PAR; ++lane) {
#pragma HLS UNROLL
        int c = c_base + lane;
        if (c < HGTXR_EMBED) {
          context[lane] += mode_profile_dsp_mul(
              static_cast<hgtxr_data_t>(mode_profile_dsp_mul(tokens[t][c],
                                                             prob[t])),
              recip);
        }
      }
    }
    for (int t = 0; t < ACTIVE_TOKENS; ++t) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_SEARCH_TOKENS
#pragma HLS PIPELINE II=1
      for (int lane = 0; lane < HGTXR_MODE_PROFILE_PAR; ++lane) {
#pragma HLS UNROLL
        int c = c_base + lane;
        if (c < HGTXR_EMBED) {
          hgtxr_acc_t out =
              static_cast<hgtxr_acc_t>(tokens[t][c]) +
              mode_profile_dsp_mul(static_cast<hgtxr_data_t>(context[lane]),
                                   static_cast<hgtxr_data_t>(0.25)) +
              mode_profile_dsp_mul(
                  static_cast<hgtxr_data_t>(
                      mode_profile_dsp_mul(score[t], mode_token_gain(t))),
                  static_cast<hgtxr_data_t>(0.03125));
          tokens[t][c] = clamp_data(out);
        }
      }
    }
  }
}

template <int ACTIVE_TOKENS>
void mode_gelu_active(active_token_buffer_t<ACTIVE_TOKENS> tokens) {
#pragma HLS INLINE off
  for (int t = 0; t < ACTIVE_TOKENS; ++t) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_SEARCH_TOKENS
    for (int c_base = 0; c_base < HGTXR_EMBED;
         c_base += HGTXR_MODE_PROFILE_PAR) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_EMBED
#pragma HLS PIPELINE II=1
      for (int lane = 0; lane < HGTXR_MODE_PROFILE_PAR; ++lane) {
#pragma HLS UNROLL
        int c = c_base + lane;
        if (c < HGTXR_EMBED) {
          hgtxr_data_t x = tokens[t][c];
          hgtxr_data_t neg = static_cast<hgtxr_data_t>(
              x * static_cast<hgtxr_data_t>(0.125));
          tokens[t][c] = x > 0 ? x : neg;
        }
      }
    }
  }
}

template <int ACTIVE_TOKENS>
void mode_attention_active(active_token_buffer_t<ACTIVE_TOKENS> tokens) {
#pragma HLS INLINE off
  hgtxr_data_t residual[ACTIVE_TOKENS][HGTXR_EMBED];
#if HGTXR_MODE_PROFILE_PAR >= 16
#pragma HLS bind_storage variable=residual type=ram_2p impl=uram
#else
#pragma HLS bind_storage variable=residual type=ram_2p impl=bram
#endif
#pragma HLS ARRAY_PARTITION variable=tokens cyclic factor=HGTXR_MODE_PROFILE_PAR dim=2
#pragma HLS ARRAY_PARTITION variable=residual cyclic factor=HGTXR_MODE_PROFILE_PAR dim=2

  for (int t = 0; t < ACTIVE_TOKENS; ++t) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_SEARCH_TOKENS
    for (int c_base = 0; c_base < HGTXR_EMBED;
         c_base += HGTXR_MODE_PROFILE_PAR) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_EMBED
#pragma HLS PIPELINE II=1
      for (int lane = 0; lane < HGTXR_MODE_PROFILE_PAR; ++lane) {
#pragma HLS UNROLL
        int c = c_base + lane;
        if (c < HGTXR_EMBED) {
          residual[t][c] = tokens[t][c];
        }
      }
    }
  }

  mode_projection_active<ACTIVE_TOKENS>(tokens);
  mode_relation_active<ACTIVE_TOKENS>(tokens);
  mode_projection_active<ACTIVE_TOKENS>(tokens);

  for (int t = 0; t < ACTIVE_TOKENS; ++t) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_SEARCH_TOKENS
    for (int c_base = 0; c_base < HGTXR_EMBED;
         c_base += HGTXR_MODE_PROFILE_PAR) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_EMBED
#pragma HLS PIPELINE II=1
      for (int lane = 0; lane < HGTXR_MODE_PROFILE_PAR; ++lane) {
#pragma HLS UNROLL
        int c = c_base + lane;
        if (c < HGTXR_EMBED) {
          tokens[t][c] = clamp_data(residual[t][c] +
                                    tokens[t][c] *
                                        static_cast<hgtxr_data_t>(0.5));
        }
      }
    }
  }
}

template <int ACTIVE_TOKENS>
void mode_mlp_active(active_token_buffer_t<ACTIVE_TOKENS> tokens) {
#pragma HLS INLINE off
  hgtxr_data_t residual[ACTIVE_TOKENS][HGTXR_EMBED];
#if HGTXR_MODE_PROFILE_PAR >= 16
#pragma HLS bind_storage variable=residual type=ram_2p impl=uram
#else
#pragma HLS bind_storage variable=residual type=ram_2p impl=bram
#endif
#pragma HLS ARRAY_PARTITION variable=tokens cyclic factor=HGTXR_MODE_PROFILE_PAR dim=2
#pragma HLS ARRAY_PARTITION variable=residual cyclic factor=HGTXR_MODE_PROFILE_PAR dim=2

  for (int t = 0; t < ACTIVE_TOKENS; ++t) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_SEARCH_TOKENS
    for (int c_base = 0; c_base < HGTXR_EMBED;
         c_base += HGTXR_MODE_PROFILE_PAR) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_EMBED
#pragma HLS PIPELINE II=1
      for (int lane = 0; lane < HGTXR_MODE_PROFILE_PAR; ++lane) {
#pragma HLS UNROLL
        int c = c_base + lane;
        if (c < HGTXR_EMBED) {
          residual[t][c] = tokens[t][c];
        }
      }
    }
  }

  mode_layernorm_active<ACTIVE_TOKENS>(tokens);
  mode_projection_active<ACTIVE_TOKENS>(tokens);
  mode_gelu_active<ACTIVE_TOKENS>(tokens);
  mode_projection_active<ACTIVE_TOKENS>(tokens);

  for (int t = 0; t < ACTIVE_TOKENS; ++t) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_SEARCH_TOKENS
    for (int c_base = 0; c_base < HGTXR_EMBED;
         c_base += HGTXR_MODE_PROFILE_PAR) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_EMBED
#pragma HLS PIPELINE II=1
      for (int lane = 0; lane < HGTXR_MODE_PROFILE_PAR; ++lane) {
#pragma HLS UNROLL
        int c = c_base + lane;
        if (c < HGTXR_EMBED) {
          tokens[t][c] = clamp_data(residual[t][c] +
                                    tokens[t][c] *
                                        static_cast<hgtxr_data_t>(0.5));
        }
      }
    }
  }
}

template <int ACTIVE_TOKENS>
void mode_pool_active(const active_token_buffer_t<ACTIVE_TOKENS> tokens,
                      pooled_buffer_t pooled) {
#pragma HLS INLINE off
  for (int c_base = 0; c_base < HGTXR_EMBED;
       c_base += HGTXR_MODE_PROFILE_PAR) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_EMBED
    hgtxr_acc_t acc[HGTXR_MODE_PROFILE_PAR];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
    for (int lane = 0; lane < HGTXR_MODE_PROFILE_PAR; ++lane) {
#pragma HLS UNROLL
      acc[lane] = 0;
    }
    for (int t = 0; t < ACTIVE_TOKENS; ++t) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_SEARCH_TOKENS
#pragma HLS PIPELINE II=1
      for (int lane = 0; lane < HGTXR_MODE_PROFILE_PAR; ++lane) {
#pragma HLS UNROLL
        int c = c_base + lane;
        if (c < HGTXR_EMBED) {
          acc[lane] += tokens[t][c];
        }
      }
    }
    for (int lane = 0; lane < HGTXR_MODE_PROFILE_PAR; ++lane) {
#pragma HLS UNROLL
      int c = c_base + lane;
      if (c < HGTXR_EMBED) {
        pooled[c] = static_cast<hgtxr_data_t>(acc[lane] / ACTIVE_TOKENS);
      }
    }
  }
}

template <int ACTIVE_TOKENS, int DEPTH_LIMIT>
void mode_transformer_profile(active_token_buffer_t<ACTIVE_TOKENS> tokens) {
#pragma HLS INLINE off
  for (int i = 0; i < DEPTH_LIMIT / 2; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HGTXR_SEARCH_DEPTH
    mode_attention_active<ACTIVE_TOKENS>(tokens);
    mode_mlp_active<ACTIVE_TOKENS>(tokens);
  }
}

}  // namespace

void hgtxr_search_profile_top(
    const hgtxr_data_t frame[HGTXR_HEIGHT][HGTXR_WIDTH],
    hgtxr_data_t out_logits[HGTXR_SEARCH_LOGITS],
    int *runtime_state) {
#pragma HLS INTERFACE m_axi port=frame offset=slave bundle=gmem_frame depth=65536 max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi port=out_logits offset=slave bundle=gmem_search_out depth=HGTXR_SEARCH_LOGITS max_write_burst_length=16 num_write_outstanding=2
#pragma HLS INTERFACE m_axi port=runtime_state offset=slave bundle=gmem_runtime depth=1 max_write_burst_length=2 num_write_outstanding=2
#pragma HLS INTERFACE s_axilite port=frame bundle=control
#pragma HLS INTERFACE s_axilite port=out_logits bundle=control
#pragma HLS INTERFACE s_axilite port=runtime_state bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  hgtxr_data_t tokens[HGTXR_SEARCH_TOKENS][HGTXR_EMBED];
  pooled_buffer_t pooled;
#pragma HLS bind_storage variable=tokens type=ram_2p impl=uram
#pragma HLS ARRAY_PARTITION variable=tokens cyclic factor=HGTXR_MODE_PROFILE_PAR dim=2
#pragma HLS ARRAY_PARTITION variable=pooled cyclic factor=HGTXR_MODE_PROFILE_PAR dim=1

  mode_frame_patch_embed_active<HGTXR_SEARCH_TOKENS>(frame, tokens);
  mode_transformer_profile<HGTXR_SEARCH_TOKENS, HGTXR_SEARCH_DEPTH>(tokens);
  mode_pool_active<HGTXR_SEARCH_TOKENS>(tokens, pooled);
  search_head(pooled, out_logits);
  *runtime_state = HGTXR_MODE_SEARCH;
}

void hgtxr_track_profile_top(
    const hgtxr_data_t event_pos[HGTXR_HEIGHT][HGTXR_WIDTH],
    const hgtxr_data_t event_neg[HGTXR_HEIGHT][HGTXR_WIDTH],
    const hgtxr_data_t frame_pooled[HGTXR_EMBED],
    const hgtxr_data_t prev_state[HGTXR_STATE],
    hgtxr_data_t out_state[HGTXR_STATE],
    int *runtime_state) {
#pragma HLS INTERFACE m_axi port=event_pos offset=slave bundle=gmem_event_pos depth=65536 max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi port=event_neg offset=slave bundle=gmem_event_neg depth=65536 max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi port=frame_pooled offset=slave bundle=gmem_frame_pooled depth=HGTXR_EMBED max_read_burst_length=32 num_read_outstanding=4
#pragma HLS INTERFACE m_axi port=prev_state offset=slave bundle=gmem_prev_state depth=HGTXR_STATE max_read_burst_length=16 num_read_outstanding=2
#pragma HLS INTERFACE m_axi port=out_state offset=slave bundle=gmem_track_out depth=HGTXR_STATE max_write_burst_length=16 num_write_outstanding=2
#pragma HLS INTERFACE m_axi port=runtime_state offset=slave bundle=gmem_runtime depth=1 max_write_burst_length=2 num_write_outstanding=2
#pragma HLS INTERFACE s_axilite port=event_pos bundle=control
#pragma HLS INTERFACE s_axilite port=event_neg bundle=control
#pragma HLS INTERFACE s_axilite port=frame_pooled bundle=control
#pragma HLS INTERFACE s_axilite port=prev_state bundle=control
#pragma HLS INTERFACE s_axilite port=out_state bundle=control
#pragma HLS INTERFACE s_axilite port=runtime_state bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  hgtxr_data_t tokens[HGTXR_TRACK_TOKENS][HGTXR_EMBED];
  pooled_buffer_t event_pooled;
  hgtxr_data_t fused[HGTXR_EMBED * 2];
#pragma HLS bind_storage variable=tokens type=ram_2p impl=bram
#pragma HLS bind_storage variable=fused type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=tokens cyclic factor=HGTXR_MODE_PROFILE_PAR dim=2
#pragma HLS ARRAY_PARTITION variable=event_pooled cyclic factor=HGTXR_MODE_PROFILE_PAR dim=1

  mode_event_patch_embed_active<HGTXR_TRACK_TOKENS>(event_pos, event_neg,
                                                    tokens);
  mode_transformer_profile<HGTXR_TRACK_TOKENS, HGTXR_TRACK_CUT_DEPTH>(tokens);
  mode_pool_active<HGTXR_TRACK_TOKENS>(tokens, event_pooled);
  fusion(frame_pooled, event_pooled, prev_state, fused);
  track_head(fused, prev_state, out_state);
  *runtime_state = HGTXR_MODE_TRACK;
}
