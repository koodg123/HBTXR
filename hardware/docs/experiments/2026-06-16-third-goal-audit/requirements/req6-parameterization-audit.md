# Req6 Parameterization Audit

- status: `pass`
- root: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR`
- checks: `71/71`
- requirement: Req6 parameterize tiling, parallelism, bus width, bit width, buffer size, FIFO depth

## Knobs

| Macro | Value |
|---|---:|
| `HGTXR_TILING_FACTOR` | `1` |
| `HGTXR_PARALLELISM_FACTOR` | `8` |
| `HGTXR_BUS_WIDTH` | `256` |
| `HGTXR_BIT_WIDTH` | `8` |
| `HGTXR_WEIGHT_BIT_WIDTH` | `4` |
| `HGTXR_BUFFER_SIZE` | `256` |
| `HGTXR_FIFO_DEPTH` | `128` |

## Coverage

- config: top-level ZCU104 Q4/W8A defaults are override-safe #ifndef macros
- derived_params: core tile, parallel, bus, bit, local-buffer, and FIFO constants derive from Req6 macros
- hls_pragmas: AXIS FIFO depth and E2E memory partition factors consume derived macros
- tool_overrides: CSim/CSynth Tcl can override E2E parallelism for PAR16/C3b-style builds and PAR32 exploration
- legality: bus/bit widths, head dimensions, dense parallelism, packed weight lanes, buffer size, and FIFO depth are checked for HLS-legal relationships
- static_asserts: E2E header has compile-time guards for active tokens, heads, dimensions, dense parallelism, packed weight lanes, and AXIS width
- sweep: machine-readable sweep matrix covers all requested knobs
- parallelism_extensions: C3b PAR16 is the validated resource-matrix path and PAR32 remains exploratory until fresh csynth, routing, resource-fit, and no-overwrite evidence exist

## Checks

| Check | Status | Detail |
|---|---|---|
| config_defines_HGTXR_TILING_FACTOR | `pass` | 1 |
| config_defines_HGTXR_PARALLELISM_FACTOR | `pass` | 8 |
| config_defines_HGTXR_BUS_WIDTH | `pass` | 256 |
| config_defines_HGTXR_BIT_WIDTH | `pass` | 8 |
| config_defines_HGTXR_WEIGHT_BIT_WIDTH | `pass` | 4 |
| config_defines_HGTXR_BUFFER_SIZE | `pass` | 256 |
| config_defines_HGTXR_FIFO_DEPTH | `pass` | 128 |
| zcu104_q4w8a_default_HGTXR_TILING_FACTOR | `pass` | 1 |
| zcu104_q4w8a_default_HGTXR_PARALLELISM_FACTOR | `pass` | 8 |
| zcu104_q4w8a_default_HGTXR_BUS_WIDTH | `pass` | 256 |
| zcu104_q4w8a_default_HGTXR_BIT_WIDTH | `pass` | 8 |
| zcu104_q4w8a_default_HGTXR_WEIGHT_BIT_WIDTH | `pass` | 4 |
| zcu104_q4w8a_default_HGTXR_BUFFER_SIZE | `pass` | 256 |
| zcu104_q4w8a_default_HGTXR_FIFO_DEPTH | `pass` | 128 |
| tiling_to_tile_tokens | `pass` | hls/include/hgtxr_cyclic_transformer_params.hpp:68 |
| tiling_to_tile_channels | `pass` | hls/include/hgtxr_cyclic_transformer_params.hpp:71 |
| parallelism_to_head_par | `pass` | hls/include/hgtxr_cyclic_transformer_params.hpp:84 |
| parallelism_to_pe_par | `pass` | hls/include/hgtxr_cyclic_transformer_params.hpp:87 |
| bus_width_to_bus_bits | `pass` | hls/include/hgtxr_cyclic_transformer_params.hpp:96 |
| bit_width_to_data_w | `pass` | hls/include/hgtxr_cyclic_transformer_params.hpp:99 |
| buffer_size_to_local_depth | `pass` | hls/include/hgtxr_cyclic_transformer_params.hpp:122 |
| fifo_depth_to_stream_fifo | `pass` | hls/include/hgtxr_cyclic_transformer_params.hpp:125 |
| axis_word_uses_bus_width | `pass` | hls/include/hgtxr_e2e_vit.hpp:129 |
| dense_par_defaults_to_parallelism | `pass` | hls/include/hgtxr_e2e_vit.hpp:33 |
| mem_bank_par_defaults_to_dense | `pass` | hls/include/hgtxr_e2e_vit.hpp:36 |
| q4_extract_specialized_for_bus | `pass` | hls/include/hgtxr_e2e_vit.hpp:219 |
| e2e_static_assert_active_tokens_fit | `pass` | hls/include/hgtxr_e2e_vit.hpp:140 |
| e2e_static_assert_heads_positive | `pass` | hls/include/hgtxr_e2e_vit.hpp:142 |
| e2e_static_assert_head_dim_covers_embed | `pass` | hls/include/hgtxr_e2e_vit.hpp:143 |
| e2e_static_assert_dense_par_positive | `pass` | hls/include/hgtxr_e2e_vit.hpp:145 |
| e2e_static_assert_dense_par_divides_embed | `pass` | hls/include/hgtxr_e2e_vit.hpp:146 |
| e2e_static_assert_dense_par_divides_ff_dim | `pass` | hls/include/hgtxr_e2e_vit.hpp:148 |
| e2e_static_assert_weight_lanes_positive | `pass` | hls/include/hgtxr_e2e_vit.hpp:150 |
| e2e_static_assert_dense_par_divides_weight_lanes | `pass` | hls/include/hgtxr_e2e_vit.hpp:151 |
| e2e_static_assert_axis_width_matches_cyclic_axi | `pass` | hls/include/hgtxr_e2e_vit.hpp:153 |
| axis_ports_use_fifo_depth | `pass` | hls/src/hgtxr_e2e_axis_top.cpp:9<br>hls/src/hgtxr_e2e_axis_top.cpp:10 |
| axis_partitions_use_mem_bank_par | `pass` | hls/src/hgtxr_e2e_axis_top.cpp:70<br>hls/src/hgtxr_e2e_axis_top.cpp:71<br>hls/src/hgtxr_e2e_axis_top.cpp:72<br>hls/src/hgtxr_e2e_axis_top.cpp:73<br>hls/src/hgtxr_e2e_axis_top.cpp:74<br>hls/src/hgtxr_e2e_axis_top.cpp:75<br>hls/src/hgtxr_e2e_axis_top.cpp:76<br>hls/src/hgtxr_e2e_axis_top.cpp:77 |
| m_axi_partitions_use_dense_par | `pass` | hls/src/hgtxr_e2e_m_axi_top.cpp:68<br>hls/src/hgtxr_e2e_m_axi_top.cpp:69<br>hls/src/hgtxr_e2e_m_axi_top.cpp:70<br>hls/src/hgtxr_e2e_m_axi_top.cpp:71<br>hls/src/hgtxr_e2e_m_axi_top.cpp:72<br>hls/src/hgtxr_e2e_m_axi_top.cpp:73<br>hls/src/hgtxr_e2e_m_axi_top.cpp:74<br>hls/src/hgtxr_e2e_m_axi_top.cpp:75 |
| csim_tcl_includes_q4w8a_config | `pass` | vivado/scripts/run_e2e_q4w8a_csim.tcl |
| csim_tcl_has_parallelism_override | `pass` | vivado/scripts/run_e2e_q4w8a_csim.tcl |
| csim_tcl_supports_par16_par32 | `pass` | vivado/scripts/run_e2e_q4w8a_csim.tcl |
| csynth_tcl_includes_q4w8a_config | `pass` | vivado/scripts/run_e2e_q4w8a_csynth.tcl |
| csynth_tcl_has_parallelism_override | `pass` | vivado/scripts/run_e2e_q4w8a_csynth.tcl |
| csynth_tcl_supports_par16_par32 | `pass` | vivado/scripts/run_e2e_q4w8a_csynth.tcl |
| legal_bus_width_byte_aligned | `pass` | {'bus_width': 256} |
| legal_bus_width_data_divisible | `pass` | {'bus_width': 256, 'bit_width': 8} |
| legal_bus_width_weight_divisible | `pass` | {'bus_width': 256, 'weight_bit_width': 4} |
| legal_weight_width_lte_data_width | `pass` | {'weight_bit_width': 4, 'bit_width': 8} |
| legal_data_width_lt_acc_width | `pass` | {'bit_width': 8, 'acc_width': 32} |
| legal_model_dim_divides_heads | `pass` | {'model_dim': 192, 'heads': 3} |
| legal_head_dim_matches_model_heads | `pass` | {'head_dim': 64, 'expected': 64} |
| legal_ff_dim_matches_mlp_ratio | `pass` | {'ff_dim': 768, 'expected': 768} |
| legal_dense_parallelism_matches_req6_parallelism | `pass` | {'dense_par': 8, 'parallelism': 8} |
| legal_dense_parallelism_divides_embed | `pass` | {'model_dim': 192, 'dense_par': 8} |
| legal_dense_parallelism_divides_ff_dim | `pass` | {'ff_dim': 768, 'dense_par': 8} |
| legal_weight_lanes_divide_dense_parallelism | `pass` | {'weight_lanes': 64, 'dense_par': 8} |
| legal_buffer_size_positive | `pass` | {'buffer_size': 256} |
| legal_fifo_depth_positive | `pass` | {'fifo_depth': 128} |
| sweep_covers_tiling_factor | `pass` | 1<br>2<br>4<br>8 |
| sweep_covers_parallelism_factor | `pass` | 1<br>2<br>4<br>8<br>16<br>32 |
| sweep_covers_bus_width | `pass` | 64<br>128<br>256<br>512 |
| sweep_covers_bit_width | `pass` | 8<br>16 |
| sweep_covers_buffer_size | `pass` | 64<br>128<br>256<br>512 |
| sweep_covers_fifo_depth | `pass` | 16<br>32<br>64<br>128 |
| sweep_yaml_parsed | `pass` | configs/sweeps/zcu104_cyclic_transformer_sweep.yaml |
| parallelism_extension_c3b_par16_validated | `pass` | validated_resource_matrix |
| parallelism_extension_c3b_par16_evidence_resource_matrix | `pass` | docs/resources/e2e_resource_matrix_2026_06_10.json:C3b |
| parallelism_extension_c3b_par16_expected_result | `pass` | DSP 604, LUT 126506, URAM 64, latency 37508072 cycles, routed WNS positive. |
| parallelism_extension_par32_exploratory_not_default | `pass` | exploratory_not_default |
| parallelism_extension_par32_requires_fresh_reports | `pass` | Requires fresh csynth, routed timing, resource-fit audit, and no C3b artifact overwrite. |
| parallelism_extension_par32_records_risk | `pass` | Potential latency reduction with higher DSP/partition pressure and timing risk. |
