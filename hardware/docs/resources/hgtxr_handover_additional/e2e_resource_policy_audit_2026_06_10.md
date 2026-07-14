# HGTXR E2E Resource Policy Audit

- status: `pass`
- root: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR`
- checks: `36/36`
- fail_count: `0`

## Observed

- bind_op_dsp_count: `2`
- rmu_smu_bind_op_dsp_count: `2`
- rmu_smu_dsp_helper_count: `6`
- rmu_smu_dsp_acc_helper_count: `6`
- bind_storage_uram_count: `3`
- bind_storage_lutram_count: `3`
- cyclic_weight_uram_count: `6`
- cyclic_large_temp_uram_count: `4`
- cyclic_small_tile_lutram_count: `9`
- c3b: `{'parallelism': 16, 'memory_banks': 16, 'dsp': 604, 'lut': 126506, 'uram': 64, 'latency_cycles': 37508072}`
- c3b_csynth: `{'path': '/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/generated/hgtxr_e2e_axis_par16_c3b_mem16_hls/solution_e2e_q4w8a/syn/report/csynth.xml', 'latency_cycles': 37508072, 'resources': {'bram_18k': 332, 'dsp': 604, 'ff': 59505, 'lut': 126506, 'uram': 64}}`

## Checks

| Check | Status | Detail |
|---|---|---|
| force_dsp_macro_default | `pass` | 1 |
| force_uram_macro_default | `pass` | 1 |
| small_mem_lutram_macro_default | `pass` | 1 |
| cyclic_weight_tiles_uram_macro_default | `pass` | 1 |
| cyclic_large_temps_uram_macro_default | `pass` | 1 |
| cyclic_small_tile_lutram_macro_default | `pass` | 1 |
| dsp_bind_op_present | `pass` | 2 |
| rmu_smu_dsp_bind_op_present | `pass` | 2 |
| rmu_smu_dsp_helper_defined | `pass` | rmu_smu_dsp_mul |
| rmu_smu_dsp_acc_helper_defined | `pass` | rmu_smu_dsp_mul_acc |
| rmu_projection_uses_dsp_helper | `pass` | RMU projection helper calls |
| smu_relation_uses_dsp_helper | `pass` | SMU relation helper calls |
| uram_bind_storage_present | `pass` | 3 |
| lutram_bind_storage_present | `pass` | 3 |
| cyclic_weight_tiles_uram_pragmas | `pass` | 6 |
| cyclic_large_temps_uram_pragmas | `pass` | 4 |
| cyclic_small_tile_lutram_pragmas | `pass` | 9 |
| zcu104_parallelism_default | `pass` | 8 |
| zcu104_dense_parallelism_default | `pass` | 8 |
| zcu104_fifo_depth_default | `pass` | 128 |
| resource_matrix_pass | `pass` | pass |
| resource_matrix_recommends_c3b | `pass` | {'best_latency_variants': ['C1', 'C3b'], 'highest_dsp_variants': ['C1', 'C3b'], 'lowest_lut_c_variant': 'C3b', 'reason': 'C3b keeps PAR16 DSP/latency gain while reducing LUT versus C1 and retaining positive routed WNS.', 'recommended_board_smoke_variant': 'C3b', 'variant_count': 4} |
| c3b_parallelism_16 | `pass` | 16 |
| c3b_memory_banks_16 | `pass` | 16 |
| c3b_dsp_increased_vs_a1 | `pass` | C3b=604 A1=332 |
| c3b_lut_lower_than_c1 | `pass` | C3b=126506 C1=127916 |
| c3b_uram_positive | `pass` | 64 |
| c3b_latency_not_worse_than_c1 | `pass` | C3b=37508072 C1=37508072 |
| c3b_csynth_xml_exists | `pass` | /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/generated/hgtxr_e2e_axis_par16_c3b_mem16_hls/solution_e2e_q4w8a/syn/report/csynth.xml |
| c3b_csynth_resources_match_matrix | `pass` | csynth={'bram_18k': 332, 'dsp': 604, 'ff': 59505, 'lut': 126506, 'uram': 64} matrix={'bram_18k': 332, 'dsp': 604, 'ff': 59505, 'lut': 126506, 'uram': 64} |
| c3b_csynth_latency_matches_matrix | `pass` | csynth=37508072 matrix=37508072 |
| c3b_csynth_dsp_lte_threshold | `pass` | 604 <= 604 |
| c3b_csynth_uram_lte_threshold | `pass` | 64 <= 64 |
| c3b_csynth_lut_lte_threshold | `pass` | 126506 <= 126506 |
| c3b_csynth_latency_lte_threshold | `pass` | 37508072 <= 37508072 |
| c3b_routed_wns_gte_threshold | `pass` | 4.415 >= 4.415 |

## Safety

- executes_hls: `False`
- executes_vivado: `False`
- writes_hls_source: `False`
- writes_canonical_inputs: `False`
