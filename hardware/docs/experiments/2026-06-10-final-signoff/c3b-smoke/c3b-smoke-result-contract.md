# HGTXR C3b Board Smoke Result Contract

- status: `pass`
- preset: `axis-c3b-mem16`
- variant: `c3b-mem16`
- canonical_result_path: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json`

## Required Fields

| Field | Expected |
|---|---|
| `status` | `pass` |
| `variant` | `c3b-mem16` |
| `weights_mode` | `file` |
| `expected_runtime_state` | `2` |
| `runtime_state` | `2` |
| `runtime_match` | `True` |
| `expected_out_raw` | `[32, -13, 26, -6, 14, -11]` |
| `out_raw` | `[32, -13, 26, -6, 14, -11]` |
| `output_match` | `True` |
| `dma_in_name` | `non-empty` |
| `dma_out_name` | `non-empty` |
| `bitfile` | `non-empty` |
| `hwhfile` | `non-empty` |
| `out_state` | `list length 6` |

## Commands

### Validate

```sh
python3 tools/validate_pynq_smoke_result.py /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --preset axis-c3b-mem16
```

### Dry-Run Import

```sh
python3 tools/import_pynq_smoke_result.py /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --preset axis-c3b-mem16 --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --dry-run --json-out /tmp/c3b_smoke_import_2026_06_10.json --validation-out /tmp/c3b_smoke_import_validation_2026_06_10.json
```

### Active Import

```sh
python3 tools/import_pynq_smoke_result.py /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --preset axis-c3b-mem16 --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --json-out /tmp/c3b_smoke_import_2026_06_10.json --validation-out /tmp/c3b_smoke_import_validation_2026_06_10.json
```

## Safety

- executes_commands: `False`
- creates_board_result: `False`
- creates_xr_vits_policy: `False`
- writes_canonical_inputs: `False`
