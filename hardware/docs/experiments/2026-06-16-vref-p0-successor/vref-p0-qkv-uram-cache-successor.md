# VREF-P0-02 QKV Weight Cache URAM Successor

- status: `pass`
- candidate: `VREF-P0-02-qkv-weight-cache-uram`
- macro: `HGTXR_E2E_URAM_QKV_WEIGHT_CACHE=1`
- resource_policy: `dsp_mixed_stream`
- CSim: `pass`
- CSynth: `pass`
- latency_cycles: `498485`
- estimated_clock_ns: `4.069`
- resources: BRAM_18K `114`, DSP `128`, FF `19664`, LUT `43236`, URAM `40`
- delta_vs_dsp_mixed_stream: BRAM_18K `-30`, URAM `8`, DSP `0`, LUT `0`
- ip_package: `pass`
- overlay: `pass`
- physical_smoke: `missing`
- pynq_plumbing: `ready-for-artifacts`
- promotion: `hls-resource-pass-routed-pending`

## Checks

| Check | Status | Detail | Threshold |
|---|---|---|---|
| csim_pass | `pass` | [] |  |
| csynth_pass | `pass` | [] |  |
| latency_lte_c3b | `pass` | 498485 | 37508072 |
| estimated_clock_lte_target | `pass` | 4.069 | 5.0 |
| dsp_lte_c3b | `pass` | 128 | 604 |
| lut_lte_c3b | `pass` | 43236 | 126506 |
| uram_lte_c3b | `pass` | 40 | 64 |
| bram_reduced_vs_dsp_mixed_stream | `pass` | {'qkv_uram': 114, 'dsp_mixed_stream': 144} |  |
| uram_increased_vs_dsp_mixed_stream | `pass` | {'qkv_uram': 40, 'dsp_mixed_stream': 32} |  |
| uram_reduced_vs_default | `pass` | {'qkv_uram': 40, 'default': 88} |  |

## Optional Promotion Evidence

- ip_component_xml: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/generated/hgtxr_e2e_axis_vref_p0_softmax_input_x2_qkv_uram_ip/solution_e2e_q4w8a/impl/ip/component.xml`
- ip_export_zip: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/generated/hgtxr_e2e_axis_vref_p0_softmax_input_x2_qkv_uram_ip/solution_e2e_q4w8a/impl/export.zip`
- ip_command: `LD_LIBRARY_PATH=/tools/Xilinx/Vitis_HLS/2023.2/lib/lnx64.o/Rhel/9:$LD_LIBRARY_PATH HGTXR_E2E_PROJECT_NAME=hgtxr_e2e_axis_vref_p0_softmax_input_x2_qkv_uram_ip HGTXR_E2E_SCALE=custom HGTXR_E2E_RESOURCE_POLICY=dsp_mixed_stream HGTXR_E2E_CUSTOM_SCALE_FLAGS='-DHGTXR_E2E_BLOCKS=2 -DHGTXR_E2E_ACTIVE_TOKENS=16 -DHGTXR_E2E_PATCH_GRID_H=1 -DHGTXR_E2E_PATCH_GRID_W=16 -DHGTXR_E2E_FF_DIM=32 -DHGTXR_E2E_STRICT_GOLDEN=1 -DHGTXR_E2E_USE_VREF_P0_HGPIPE_LNQ_ACTIVE16_SOFTMAX_INPUT_X2_GOLDEN=1 -DHGTXR_E2E_USE_HGPIPE_INT_GELUQ=1 -DHGTXR_E2E_HGPIPE_GELUQ_INPUT_SCALE=16 -DHGTXR_E2E_HGPIPE_GELUQ_OUTPUT_SCALE=4 -DHGTXR_E2E_USE_HGPIPE_INT_SOFTMAXQ=1 -DHGTXR_E2E_HGPIPE_SOFTMAX_INPUT_SCALE=32 -DHGTXR_E2E_HGPIPE_SOFTMAX_PROB_SCALE=4 -DHGTXR_E2E_USE_HGPIPE_INT_LAYERNORMQ=1 -DHGTXR_E2E_USE_HGPIPE_LNQ_GAMMA_RAW=1 -DHGTXR_E2E_HGPIPE_LAYERNORM_INPUT_SCALE=16 -DHGTXR_E2E_HGPIPE_LAYERNORM_OUTPUT_SCALE=4 -DHGTXR_E2E_HGPIPE_LAYERNORM_BIAS_SHIFT=33 -DHGTXR_E2E_URAM_QKV_WEIGHT_CACHE=1' /tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/vivado/scripts/package_e2e_axis_ip.tcl`
- overlay_bit: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/generated/build/vivado/overlay/hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_overlay/hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram.bit`
- overlay_hwh: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/generated/build/vivado/overlay/hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_overlay/hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram.hwh`
- pynq_bit: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/pynq/hgtxr/hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram.bit`
- pynq_hwh: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/pynq/hgtxr/hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram.hwh`
- timing_report: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/generated/build/vivado/hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_overlay/hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_overlay.runs/impl_1/hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_system_wrapper_timing_summary_routed.rpt`
- route_status_report: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/generated/build/vivado/hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_overlay/hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_overlay.runs/impl_1/hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_system_wrapper_route_status.rpt`
- overlay_command: `LD_LIBRARY_PATH=/tools/Xilinx/Vivado/2023.2/lib/lnx64.o/Rhel/9:$LD_LIBRARY_PATH /tools/Xilinx/Vivado/2023.2/bin/vivado -mode batch -source /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/vivado/scripts/build_e2e_axis_dma_bitstream.tcl -tclargs -project_name hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_overlay -bd_name hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_system -artifact_name hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram -hls_ip_repo /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/generated/hgtxr_e2e_axis_vref_p0_softmax_input_x2_qkv_uram_ip/solution_e2e_q4w8a/impl/ip`
- physical_smoke_checked_paths: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/pynq/hgtxr/e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json, /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/generated/pynq/e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json`
- pynq_variant: `vref-p0-softmax-input-x2-qkv-uram`
- pynq_preset: `axis-vref-p0-softmax-input-x2-qkv-uram`
- pynq_package_command: `python3 tools/package_e2e_axis_dma_pynq_bundle.py --variant vref-p0-softmax-input-x2-qkv-uram --out-dir /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/generated/pynq/e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_smoke_bundle --tar /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/generated/pynq/e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_smoke_bundle.tar.gz`
- pynq_session_command: `python3 tools/prepare_zcu104_smoke_session.py --variant vref-p0-softmax-input-x2-qkv-uram --preset axis-vref-p0-softmax-input-x2-qkv-uram --bundle-dir /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/generated/pynq/e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_smoke_bundle --tar /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/generated/pynq/e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_smoke_bundle.tar.gz --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR`
- pynq_remote_dry_run_command: `python3 tools/run_zcu104_c3b_smoke_remote.py --profile vref-p0-softmax-input-x2-qkv-uram --host <zcu104-ip-or-host> --user xilinx`
- pynq_dry_run_import_command: `python3 tools/import_pynq_smoke_result.py /path/to/e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json --preset axis-vref-p0-softmax-input-x2-qkv-uram --dry-run`

## Remaining

- physical_smoke_json

## Safety

- executes_hls: `False`
- executes_vivado: `False`
- writes_hls_source: `False`
- overwrites_c3b_artifacts: `False`
- creates_board_result: `False`
