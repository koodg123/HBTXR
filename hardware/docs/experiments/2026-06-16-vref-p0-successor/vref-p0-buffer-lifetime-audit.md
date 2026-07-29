# VREF-P0-02 Buffer Lifetime And Placement Audit

- status: `pass`
- root: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR`
- checks: `51/51`
- fail_count: `0`

## Policy

- vref: `VREF-P0-02`
- source: `ME-ViT`
- scope: `single-load/on-chip retention and resource placement audit; no HLS source mutation`
- large_buffers: `tokens, norm, Q/K/V, attention, hidden stay URAM candidates`
- small_buffers: `pooled and attention-row scratch stay LUTRAM/BRAM candidates`
- promotion_gate: `C3b path unchanged; csynth/routed evidence required for any successor variant`

## Observed C3b

- parallelism: `16`
- memory_banks: `16`
- dsp: `604`
- lut: `126506`
- uram: `64`
- latency_cycles: `37508072`

## Macro Snapshot

| Macro | Value |
|---|---:|
| `HGTXR_E2E_FORCE_URAM_BUFFERS` | `1` |
| `HGTXR_E2E_SMALL_MEM_LUTRAM` | `1` |
| `HGTXR_E2E_QKV_WEIGHT_CACHE` | `1` |
| `HGTXR_E2E_URAM_QKV_WEIGHT_CACHE` | `0` |
| `HGTXR_E2E_DENSE_PAR` | `8` |
| `HGTXR_PARALLELISM_FACTOR` | `8` |
| `HGTXR_BUFFER_SIZE` | `256` |
| `HGTXR_FIFO_DEPTH` | `128` |

## Checks

| Check | Status | Detail |
|---|---|---|
| force_uram_buffers_enabled | `pass` | 1 |
| small_mem_lutram_enabled | `pass` | 1 |
| qkv_weight_cache_enabled | `pass` | 1 |
| qkv_weight_cache_not_forced_to_uram | `pass` | 0 |
| axis_large_tokens_uram | `pass` | /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/hls/src/hgtxr_e2e_axis_top.cpp:31 |
| maxis_large_tokens_uram | `pass` | /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/hls/src/hgtxr_e2e_m_axi_top.cpp:29 |
| axis_large_gb.tokens_uram | `pass` | /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/hls/src/hgtxr_e2e_axis_top.cpp:36 |
| maxis_large_gb.tokens_uram | `pass` | /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/hls/src/hgtxr_e2e_m_axi_top.cpp:34 |
| axis_large_gb.norm_uram | `pass` | /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/hls/src/hgtxr_e2e_axis_top.cpp:41 |
| maxis_large_gb.norm_uram | `pass` | /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/hls/src/hgtxr_e2e_m_axi_top.cpp:39 |
| axis_large_gb.q_uram | `pass` | /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/hls/src/hgtxr_e2e_axis_top.cpp:46 |
| maxis_large_gb.q_uram | `pass` | /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/hls/src/hgtxr_e2e_m_axi_top.cpp:44 |
| axis_large_gb.k_uram | `pass` | /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/hls/src/hgtxr_e2e_axis_top.cpp:51 |
| maxis_large_gb.k_uram | `pass` | /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/hls/src/hgtxr_e2e_m_axi_top.cpp:49 |
| axis_large_gb.v_uram | `pass` | /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/hls/src/hgtxr_e2e_axis_top.cpp:56 |
| maxis_large_gb.v_uram | `pass` | /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/hls/src/hgtxr_e2e_m_axi_top.cpp:54 |
| axis_large_gb.attn_uram | `pass` | /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/hls/src/hgtxr_e2e_axis_top.cpp:61 |
| maxis_large_gb.attn_uram | `pass` | /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/hls/src/hgtxr_e2e_m_axi_top.cpp:59 |
| axis_large_gb.hidden_uram | `pass` | /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/hls/src/hgtxr_e2e_axis_top.cpp:66 |
| maxis_large_gb.hidden_uram | `pass` | /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/hls/src/hgtxr_e2e_m_axi_top.cpp:64 |
| axis_pooled_lutram | `pass` | /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/hls/src/hgtxr_e2e_axis_top.cpp:26 |
| maxis_pooled_lutram | `pass` | /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/hls/src/hgtxr_e2e_m_axi_top.cpp:24 |
| attention_small_score_lutram | `pass` | /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/hls/include/hgtxr_e2e_vit.hpp:950 |
| attention_small_prob_lutram | `pass` | /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/hls/include/hgtxr_e2e_vit.hpp:951 |
| attention_small_exp_raw_lutram | `pass` | /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/hls/include/hgtxr_e2e_vit.hpp:952 |
| rmu_smu_small_score_lutram | `pass` | /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/hls/src/rmu_smu.cpp:61 |
| rmu_smu_small_prob_lutram | `pass` | /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/hls/src/rmu_smu.cpp:62 |
| qkv_cache_q_weight_cache_bram_default | `pass` | /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/hls/include/hgtxr_e2e_vit.hpp:856 |
| qkv_cache_k_weight_cache_bram_default | `pass` | /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/hls/include/hgtxr_e2e_vit.hpp:857 |
| qkv_cache_v_weight_cache_bram_default | `pass` | /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/hls/include/hgtxr_e2e_vit.hpp:858 |
| qkv_cache_q_weight_cache_uram_successor_branch | `pass` | /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/hls/include/hgtxr_e2e_vit.hpp:852 |
| qkv_cache_k_weight_cache_uram_successor_branch | `pass` | /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/hls/include/hgtxr_e2e_vit.hpp:853 |
| qkv_cache_v_weight_cache_uram_successor_branch | `pass` | /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/hls/include/hgtxr_e2e_vit.hpp:854 |
| qkv_successor_file_present | `pass` | /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/generated/signoff/vref_p0_qkv_uram_cache_successor_2026_06_16.json |
| qkv_successor_status_pass | `pass` | pass |
| qkv_successor_macro_uram_enabled | `pass` | HGTXR_E2E_URAM_QKV_WEIGHT_CACHE=1 |
| qkv_successor_csim_pass | `pass` | pass |
| qkv_successor_csynth_pass | `pass` | pass |
| qkv_successor_overlay_routed | `pass` | pass |
| qkv_successor_uram_increased_vs_dsp_mixed_stream | `pass` | 8 |
| qkv_successor_bram_reduced_vs_dsp_mixed_stream | `pass` | -30 |
| qkv_successor_uram_positive | `pass` | 40 |
| qkv_successor_physical_smoke_pending_only | `pass` | {'note': 'Successor-only evidence; C3b remains protected baseline.', 'remaining': ['physical_smoke_json'], 'status': 'hls-resource-pass-routed-pending'} |
| axis_mem_bank_partition_present | `pass` | 7 |
| maxis_dense_partition_present | `pass` | 7 |
| c3b_parallelism_16 | `pass` | 16 |
| c3b_memory_banks_16 | `pass` | 16 |
| c3b_uram_positive | `pass` | 64 |
| c3b_dsp_increased_vs_a1 | `pass` | C3b=604 A1=332 |
| c3b_lut_lower_than_c1 | `pass` | C3b=126506 C1=127916 |
| c3b_latency_not_worse_than_c1 | `pass` | C3b=37508072 C1=37508072 |

## Safety

- executes_hls: `False`
- executes_vivado: `False`
- writes_hls_source: `False`
- overwrites_c3b_artifacts: `False`
