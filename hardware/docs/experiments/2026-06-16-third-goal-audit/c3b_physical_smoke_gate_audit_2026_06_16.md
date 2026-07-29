# C3b Physical Smoke Gate Audit

- status: `blocked_missing_canonical_physical_smoke_result`
- preset: `axis-c3b-mem16`
- variant: `c3b-mem16`
- ready_for_board: `True`
- physical_smoke_pass: `False`
- canonical_result_exists: `False`
- canonical_result_status: `missing`
- canonical_validation_exists: `False`
- canonical_validation_status: `missing`
- bundle_status: `pass`
- session_status: `pass`
- expected_out_raw: `[32, -13, 26, -6, 14, -11]`
- expected_runtime_state: `2`

## Reference Artifacts
- `generated/pynq/e2e_axis_dma_c3b_mem16_smoke_bundle/BUNDLE_MANIFEST.json`: `True`
- `generated/pynq/e2e_axis_dma_c3b_mem16_smoke_session.json`: `True`
- `generated/signoff/zcu104_c3b_smoke_remote_run_2026_06_10.json`: `True`

## Remaining Blockers
- `C3b AXIS/DMA physical smoke result`

## Next Actions
- Run generated/pynq/e2e_axis_dma_c3b_mem16_smoke_session.md on ZCU104.
- Copy e2e_axis_dma_c3b_mem16_file_smoke.json back to pynq/hgtxr/.
- Validate with tools/validate_pynq_smoke_result.py --preset axis-c3b-mem16.

## Safety
- Does not run board smoke.
- Does not create board result JSON.
- Does not write canonical unblock inputs.
