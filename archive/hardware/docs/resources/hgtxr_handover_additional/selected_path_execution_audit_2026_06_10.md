# HGTXR Selected Path Execution Audit

- status: `pass`
- root: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR`
- Path 1: `A2 then A1`
- Path 2: `C, with C3b as board-smoke candidate`
- E: `pending`
- checks: `22/22`

## Observed

| Variant | PAR | Mem Banks | Latency | DSP | LUT | URAM | Board |
|---|---:|---:|---:|---:|---:|---:|---|
| A2 | 8 | 8 | 81514836 | 334 | 84220 | 64 | `bit-hwh-ready` |
| A1 | 8 | 8 | 71316968 | 332 | 81168 | 64 | `bit-hwh-ready` |
| C3B | 16 | 16 | 37508072 | 604 | 126506 | 64 | `ready-for-board-smoke` |

## Checks

| Check | Status | Detail |
|---|---|---|
| resource_matrix_pass | `pass` | pass |
| a2_present | `pass` | A1,A2,C1,C3b |
| a1_present | `pass` | A1,A2,C1,C3b |
| c1_present | `pass` | A1,A2,C1,C3b |
| c3b_present | `pass` | A1,A2,C1,C3b |
| e_pending_policy | `pass` | True |
| path1_a2_selection_first | `pass` | Path 1, first |
| path1_a1_after_a2 | `pass` | Path 1, after A2 |
| path1_a2_artifacts_ready | `pass` | {'bit': {'exists': True, 'path': '/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/pynq/hgtxr/hgtxr_e2e_m_axi.bit'}, 'hwh': {'exists': True, 'path': '/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/pynq/hgtxr/hgtxr_e2e_m_axi.hwh'}} |
| path1_a1_artifacts_ready | `pass` | {'bit': {'exists': True, 'path': '/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/pynq/hgtxr/hgtxr_e2e_axis_dma.bit'}, 'hwh': {'exists': True, 'path': '/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/pynq/hgtxr/hgtxr_e2e_axis_dma.hwh'}} |
| path1_a2_parallelism_8 | `pass` | 8 |
| path1_a1_parallelism_8 | `pass` | 8 |
| path1_a2_uram_positive | `pass` | 64 |
| path1_a1_dsp_positive | `pass` | 332 |
| path2_c1_par16 | `pass` | 16 |
| path2_c3b_par16 | `pass` | 16 |
| path2_c3b_mem16 | `pass` | 16 |
| path2_c3b_artifacts_ready | `pass` | {'bit': {'exists': True, 'path': '/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/pynq/hgtxr/hgtxr_e2e_axis_dma_c3b_mem16.bit'}, 'hwh': {'exists': True, 'path': '/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/pynq/hgtxr/hgtxr_e2e_axis_dma_c3b_mem16.hwh'}} |
| path2_c_dsp_gain_vs_a1 | `pass` | 604 > 332 |
| path2_c_latency_gain_vs_a1 | `pass` | 37508072 < 71316968 |
| path2_c3b_lut_below_c1 | `pass` | 126506 < 127916 |
| path2_c3b_recommended | `pass` | C3b |

## Safety

- Does not run HLS or Vivado.
- Does not run board smoke.
- Does not create board result JSON.
- Does not create XR-VITs replacement policy.
