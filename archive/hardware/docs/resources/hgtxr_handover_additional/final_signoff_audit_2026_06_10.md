# HGTXR Final Signoff Audit

- status: `blocked`
- source_mode: `final-signoff`
- blockers: `2`
- warnings: `5`

## Blockers

- `requested XR-VITs sibling` (reference-input): missing; no approved replacement policy; XR_Accel candidate exists and candidate audit recommends it, approval still required: /home/kjm26/project/PRJXR/XR-VIT/XR_Accel; see generated/signoff/xr_vits_unblock_packet_2026_06_10.md; dry-run policy command: python3 tools/create_xr_vits_replacement_policy.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --approve --approved-by <approved-by> --reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff" --dry-run
  - path: `/home/kjm26/project/PRJXR/XR-VITs`
  - action: Restore the XR-VITs sibling checkout or record an approved replacement source.
- `C3b AXIS/DMA physical smoke result` (physical-board-smoke): not captured at canonical path yet; see generated/signoff/final_unblock_commands_2026_06_10.md; dry-run import: python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --import-c3b-smoke-json /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --dry-run-import-c3b-smoke --allow-blocked
  - path: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json`
  - action: Run the C3b bundle on ZCU104 PYNQ, validate it on-board, then import the JSON on host.

## Next Commands

```sh
tar -xzf e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz
cd e2e_axis_dma_c3b_mem16_smoke_bundle
./run_e2e_axis_dma_c3b_mem16_file_smoke.sh
./validate_e2e_axis_dma_c3b_mem16_file_smoke.sh
python3 tools/import_pynq_smoke_result.py /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --preset axis-c3b-mem16
```

## Warnings

- `tool spec-kit`: not found on PATH (`n/a`)
- `tool specify`: not found on PATH (`n/a`)
- `legacy PYNQ hwh stale old-flow IP`: hgtxr_top_0 found; ignored because E2E m_axi artifacts are checked separately (`/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/pynq/hgtxr/hgtxr.hwh`)
- `VREF-P0 successor physical smoke result`: not captured yet; successor promotion remains gated by valid ZCU104 smoke JSON (`/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/pynq/hgtxr/e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.json`)
- `VREF-P0-02 QKV URAM physical smoke result`: not captured yet; QKV URAM successor promotion remains gated by valid ZCU104 smoke JSON (`/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/pynq/hgtxr/e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json`)
