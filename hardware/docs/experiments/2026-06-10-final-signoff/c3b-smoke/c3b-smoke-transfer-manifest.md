# HGTXR C3b Smoke Transfer Manifest

- status: `pass`
- target: `ZCU104 PYNQ`
- variant: `c3b-mem16`
- tar: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/generated/pynq/e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz`
- tar_sha256: `3de505a6627409b4717ea95e5e76d74f0667c61b601fd96db1e495319adda712`

## Transfer Files

- `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/generated/pynq/e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz`
- `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/generated/signoff/e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz.sha256`

## Board Verify

```sh
sha256sum -c e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz.sha256
```

## Board Run

```sh
tar -xzf e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz
cd e2e_axis_dma_c3b_mem16_smoke_bundle
./run_e2e_axis_dma_c3b_mem16_file_smoke.sh
./validate_e2e_axis_dma_c3b_mem16_file_smoke.sh
```

## Host Copy-Back

```sh
python3 tools/import_pynq_smoke_result.py /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --preset axis-c3b-mem16 --json-out /tmp/hgtxr_c3b_smoke_import.json --validation-out /tmp/hgtxr_c3b_smoke_validation.json
python3 tools/validate_pynq_smoke_result.py hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json --preset axis-c3b-mem16
python3 tools/check_third_goal_preflight.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --mode final-signoff
```

## Expected Output

- result_json: `e2e_axis_dma_c3b_mem16_file_smoke.json`
- validation_json: `e2e_axis_dma_c3b_mem16_file_smoke_validation.json`
- expected_runtime_state: `2`
- expected_out_raw: `[32, -13, 26, -6, 14, -11]`

## Bundle Validation Errors

- None.
