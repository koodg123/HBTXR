# HGTXR E2E Resource Matrix

- status: `pass`
- root: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR`
- resource authority: `HLS csynth.xml`
- Vivado authority: `routed timing and power reports`
- E pending: `True`

## Summary

- best latency variants: `C1, C3b`
- highest DSP variants: `C1, C3b`
- lowest LUT C variant: `C3b`
- recommended board smoke variant: `C3b`
- reason: C3b keeps PAR16 DSP/latency gain while reducing LUT versus C1 and retaining positive routed WNS.

## Matrix

| ID | Interface | PAR | Mem Banks | Latency Cycles | HLS Clock ns | DSP | LUT | FF | BRAM18K | URAM | WNS ns | Power W | Board |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A2 | m_axi + AXI-Lite | 8 | 8 | 81,514,836 | 3.744 | 334 | 84,220 | 46,502 | 326 | 64 | 1.926 | 4.060 | bit-hwh-ready |
| A1 | AXI-Stream + AXI DMA + AXI-Lite | 8 | 8 | 71,316,968 | 3.744 | 332 | 81,168 | 43,804 | 298 | 64 | 4.120 | 3.476 | bit-hwh-ready |
| C1 | AXI-Stream + AXI DMA + AXI-Lite | 16 | 8 | 37,508,072 | 3.953 | 604 | 127,916 | 59,507 | 338 | 64 | 4.723 | 3.475 | bit-hwh-ready |
| C3b | AXI-Stream + AXI DMA + AXI-Lite | 16 | 16 | 37,508,072 | 3.953 | 604 | 126,506 | 59,505 | 332 | 64 | 4.415 | 3.476 | ready-for-board-smoke |

## Deltas Vs A1

| ID | Latency % | DSP % | LUT % | URAM % |
|---|---:|---:|---:|---:|
| A2 | 14.300 | 0.600 | 3.760 | 0.000 |
| A1 | 0.000 | 0.000 | 0.000 | 0.000 |
| C1 | -47.410 | 81.930 | 57.590 | 0.000 |
| C3b | -47.410 | 81.930 | 55.860 | 0.000 |

## Sources

- `A2` HLS: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/generated/hgtxr_e2e_m_axi_hls/solution_e2e_q4w8a/syn/report/csynth.xml`
- `A2` timing: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/generated/build/vivado/hgtxr_e2e_m_axi_overlay/hgtxr_e2e_m_axi_overlay.runs/impl_1/hgtxr_e2e_m_axi_system_wrapper_timing_summary_routed.rpt`
- `A2` power: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/generated/build/vivado/hgtxr_e2e_m_axi_overlay/hgtxr_e2e_m_axi_overlay.runs/impl_1/hgtxr_e2e_m_axi_system_wrapper_power_routed.rpt`
- `A1` HLS: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/generated/hgtxr_e2e_axis_hls/solution_e2e_q4w8a/syn/report/csynth.xml`
- `A1` timing: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/generated/build/vivado/hgtxr_e2e_axis_dma_overlay/hgtxr_e2e_axis_dma_overlay.runs/impl_1/hgtxr_e2e_axis_dma_system_wrapper_timing_summary_routed.rpt`
- `A1` power: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/generated/build/vivado/hgtxr_e2e_axis_dma_overlay/hgtxr_e2e_axis_dma_overlay.runs/impl_1/hgtxr_e2e_axis_dma_system_wrapper_power_routed.rpt`
- `C1` HLS: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/generated/hgtxr_e2e_axis_par16_hls/solution_e2e_q4w8a/syn/report/csynth.xml`
- `C1` timing: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/generated/build/vivado/hgtxr_e2e_axis_dma_par16_overlay/hgtxr_e2e_axis_dma_par16_overlay.runs/impl_1/hgtxr_e2e_axis_dma_par16_system_wrapper_timing_summary_routed.rpt`
- `C1` power: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/generated/build/vivado/hgtxr_e2e_axis_dma_par16_overlay/hgtxr_e2e_axis_dma_par16_overlay.runs/impl_1/hgtxr_e2e_axis_dma_par16_system_wrapper_power_routed.rpt`
- `C3b` HLS: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/generated/hgtxr_e2e_axis_par16_c3b_mem16_hls/solution_e2e_q4w8a/syn/report/csynth.xml`
- `C3b` timing: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/generated/build/vivado/hgtxr_e2e_axis_dma_c3b_mem16_overlay/hgtxr_e2e_axis_dma_c3b_mem16_overlay.runs/impl_1/hgtxr_e2e_axis_dma_c3b_mem16_system_wrapper_timing_summary_routed.rpt`
- `C3b` power: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/generated/build/vivado/hgtxr_e2e_axis_dma_c3b_mem16_overlay/hgtxr_e2e_axis_dma_c3b_mem16_overlay.runs/impl_1/hgtxr_e2e_axis_dma_c3b_mem16_system_wrapper_power_routed.rpt`
