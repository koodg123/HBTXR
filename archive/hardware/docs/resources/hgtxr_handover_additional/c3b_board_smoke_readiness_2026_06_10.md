# HGTXR C3b Board Smoke Readiness

- status: `ready-for-board`
- ready: `True`
- target: `ZCU104 PYNQ`
- variant: `c3b-mem16`
- tar_sha256: `3de505a6627409b4717ea95e5e76d74f0667c61b601fd96db1e495319adda712`
- physical_result: `missing`

## Inputs

- transfer_manifest: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/docs/resources/c3b_smoke_transfer_manifest_2026_06_10.json`
- session_json: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/generated/pynq/e2e_axis_dma_c3b_mem16_smoke_session.json`
- bundle_dir: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/generated/pynq/e2e_axis_dma_c3b_mem16_smoke_bundle`
- sha256_file: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/docs/resources/e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz.sha256`

## Board Commands

```sh
sha256sum -c e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz.sha256
tar -xzf e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz
cd e2e_axis_dma_c3b_mem16_smoke_bundle
./run_e2e_axis_dma_c3b_mem16_file_smoke.sh
./validate_e2e_axis_dma_c3b_mem16_file_smoke.sh
```

## Host Import

```sh
python3 tools/import_pynq_smoke_result.py /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --preset axis-c3b-mem16 --json-out /tmp/hgtxr_c3b_smoke_import.json --validation-out /tmp/hgtxr_c3b_smoke_validation.json
```

## Errors

- None.

## Warnings

- `Physical C3b board smoke result is not present yet.`

## Next Steps

- Run C3b smoke on ZCU104 and copy back e2e_axis_dma_c3b_mem16_file_smoke.json.
- Import result with tools/import_pynq_smoke_result.py --preset axis-c3b-mem16.
