# No-Board HLS Report Collection

Evidence level: existing Vitis HLS `csynth.xml` reports only. Board inference is excluded.

## Top Variants

| Variant | Module | Latency cycles | Latency ms @ target | Mode target ms | Target met | LUT | LUT % | DSP | DSP % | BRAM_18K | BRAM % | URAM | URAM % | XML |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| hgtxr_e2e_axis_hls | hgtxr_e2e_axis_top | 71316968 | 356.58 | n/a | n/a | 81168 | 35.23 | 332 | 19.21 | 298 | 47.76 | 64 | 66.67 | `generated/hgtxr_e2e_axis_hls/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.xml` |
| hgtxr_e2e_axis_par16_c3_mem8 | hgtxr_e2e_axis_top | 48598922 | 242.99 | n/a | n/a | 113124 | 49.10 | 604 | 34.95 | 292 | 46.79 | 64 | 66.67 | `generated/hgtxr_e2e_axis_par16_c3_mem8/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.xml` |
| hgtxr_e2e_axis_par16_c3b_mem16 | hgtxr_e2e_axis_top | 37508072 | 187.54 | n/a | n/a | 126506 | 54.91 | 604 | 34.95 | 332 | 53.21 | 64 | 66.67 | `generated/hgtxr_e2e_axis_par16_c3b_mem16/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.xml` |
| hgtxr_e2e_axis_par16_c3b_mem16_hls | hgtxr_e2e_axis_top | 37508072 | 187.54 | n/a | n/a | 126506 | 54.91 | 604 | 34.95 | 332 | 53.21 | 64 | 66.67 | `generated/hgtxr_e2e_axis_par16_c3b_mem16_hls/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.xml` |
| hgtxr_e2e_axis_par16_hls | hgtxr_e2e_axis_top | 37508072 | 187.54 | n/a | n/a | 127916 | 55.52 | 604 | 34.95 | 338 | 54.17 | 64 | 66.67 | `generated/hgtxr_e2e_axis_par16_hls/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.xml` |
| hgtxr_e2e_axis_par32_dsp_mixed_mem16_no_board | hgtxr_e2e_axis_top | 26139644 | 130.70 | n/a | n/a | 193546 | 84.00 | 1148 | 66.44 | 204 | 32.69 | 112 | 116.67 | `generated/hgtxr_e2e_axis_par32_dsp_mixed_mem16_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.xml` |
| hgtxr_e2e_axis_par32_dsp_mixed_stream_mem16_no_board | hgtxr_e2e_axis_top | 26139644 | 130.70 | n/a | n/a | 193546 | 84.00 | 1148 | 66.44 | 332 | 53.21 | 64 | 66.67 | `generated/hgtxr_e2e_axis_par32_dsp_mixed_stream_mem16_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.xml` |
| hgtxr_e2e_axis_par32_dsp_uram_mem16_no_board | hgtxr_e2e_axis_top | 26139644 | 130.70 | n/a | n/a | 193546 | 84.00 | 1148 | 66.44 | 124 | 19.87 | 160 | 166.67 | `generated/hgtxr_e2e_axis_par32_dsp_uram_mem16_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.xml` |
| hgtxr_e2e_axis_vref_p0_softmax_input_x2_csynth | hgtxr_e2e_axis_top | 498485 | 2.49 | n/a | n/a | 43236 | 18.77 | 128 | 7.41 | 64 | 10.26 | 88 | 91.67 | `generated/hgtxr_e2e_axis_vref_p0_softmax_input_x2_csynth/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.xml` |
| hgtxr_e2e_axis_vref_p0_softmax_input_x2_dsp_mixed_stream_csynth | hgtxr_e2e_axis_top | 498485 | 2.49 | n/a | n/a | 43236 | 18.77 | 128 | 7.41 | 144 | 23.08 | 32 | 33.33 | `generated/hgtxr_e2e_axis_vref_p0_softmax_input_x2_dsp_mixed_stream_csynth/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.xml` |
| hgtxr_e2e_axis_vref_p0_softmax_input_x2_dsp_mixed_stream_ip | hgtxr_e2e_axis_top | 498485 | 2.49 | n/a | n/a | 43236 | 18.77 | 128 | 7.41 | 144 | 23.08 | 32 | 33.33 | `generated/hgtxr_e2e_axis_vref_p0_softmax_input_x2_dsp_mixed_stream_ip/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.xml` |
| hgtxr_e2e_axis_vref_p0_softmax_input_x2_qkv_uram_csynth | hgtxr_e2e_axis_top | 498485 | 2.49 | n/a | n/a | 43236 | 18.77 | 128 | 7.41 | 114 | 18.27 | 40 | 41.67 | `generated/hgtxr_e2e_axis_vref_p0_softmax_input_x2_qkv_uram_csynth/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.xml` |
| hgtxr_e2e_axis_vref_p0_softmax_input_x2_qkv_uram_ip | hgtxr_e2e_axis_top | 498485 | 2.49 | n/a | n/a | 43236 | 18.77 | 128 | 7.41 | 114 | 18.27 | 40 | 41.67 | `generated/hgtxr_e2e_axis_vref_p0_softmax_input_x2_qkv_uram_ip/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.xml` |
| hgtxr_e2e_hls | hgtxr_e2e_axis_top | 37508072 | 187.54 | n/a | n/a | 127916 | 55.52 | 604 | 34.95 | 338 | 54.17 | 64 | 66.67 | `generated/hgtxr_e2e_hls/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.xml` |
| hgtxr_mode_search_par32_no_board | hgtxr_search_profile_top | 772268 | 3.86 | 4.00 | True | 65557 | 28.45 | 294 | 17.01 | 8 | 1.28 | 96 | 100.00 | `generated/hgtxr_mode_search_par32_no_board/solution_mode_q4w8a/syn/report/hgtxr_search_profile_top_csynth.xml` |
| hgtxr_mode_track_par32_no_board | hgtxr_track_profile_top | 99449 | 0.50 | 1.00 | True | 69728 | 30.26 | 302 | 17.48 | 51 | 8.17 | 64 | 66.67 | `generated/hgtxr_mode_track_par32_no_board/solution_mode_q4w8a/syn/report/hgtxr_track_profile_top_csynth.xml` |

## C3b Block Rows

| Module | Latency cycles | LUT | DSP | BRAM_18K | URAM | XML |
|---|---:|---:|---:|---:|---:|---|
| hgtxr_e2e_attention_core | 1186585 | 7094 | 64 | 0 | 0 | `generated/hgtxr_e2e_axis_par16_c3b_mem16/solution_e2e_q4w8a/syn/report/hgtxr_e2e_attention_core_csynth.xml` |
| hgtxr_e2e_attn_unit_0_s | 2247180 | 34434 | 207 | 45 | 0 | `generated/hgtxr_e2e_axis_par16_c3b_mem16/solution_e2e_q4w8a/syn/report/hgtxr_e2e_attn_unit_0_s_csynth.xml` |
| hgtxr_e2e_attn_unit_1_s | 2247180 | 34434 | 207 | 45 | 0 | `generated/hgtxr_e2e_axis_par16_c3b_mem16/solution_e2e_q4w8a/syn/report/hgtxr_e2e_attn_unit_1_s_csynth.xml` |
| hgtxr_e2e_axis_top | 37508072 | 126506 | 604 | 332 | 64 | `generated/hgtxr_e2e_axis_par16_c3b_mem16/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.xml` |
| hgtxr_e2e_controller_run | 37296061 | 114988 | 572 | 250 | 64 | `generated/hgtxr_e2e_axis_par16_c3b_mem16/solution_e2e_q4w8a/syn/report/hgtxr_e2e_controller_run_csynth.xml` |
| hgtxr_e2e_mlp_head | 75633 | 5584 | 32 | 0 | 0 | `generated/hgtxr_e2e_axis_par16_c3b_mem16/solution_e2e_q4w8a/syn/report/hgtxr_e2e_mlp_head_csynth.xml` |
| hgtxr_e2e_mlp_unit_0_s | 3968826 | 16998 | 79 | 0 | 0 | `generated/hgtxr_e2e_axis_par16_c3b_mem16/solution_e2e_q4w8a/syn/report/hgtxr_e2e_mlp_unit_0_s_csynth.xml` |
| hgtxr_e2e_mlp_unit_1_s | 3968826 | 16998 | 79 | 0 | 0 | `generated/hgtxr_e2e_axis_par16_c3b_mem16/solution_e2e_q4w8a/syn/report/hgtxr_e2e_mlp_unit_1_s_csynth.xml` |
| hgtxr_e2e_project_qkv | 453359 | 12715 | 96 | 45 | 0 | `generated/hgtxr_e2e_axis_par16_c3b_mem16/solution_e2e_q4w8a/syn/report/hgtxr_e2e_project_qkv_csynth.xml` |

## Notes

- `latency_worst_ms_at_target` is derived from HLS target clock period and worst-case cycles.
- p95/p99 and DMA bandwidth are not emitted here because they require runtime or board instrumentation.
- Use this table as the baseline for `NB-P0-01` parallelism and `NB-P0-03` memory-binding DSE.
