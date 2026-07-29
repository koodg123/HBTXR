# HGTXR Final Unblock Intake

- status: `blocked`
- root: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR`
- next_action: Supply inputs for: C3b AXIS/DMA physical smoke result, requested XR-VITs sibling

## Candidate Audit

- status: `blocked`
- would_clear_all: `False`
- remaining_blockers: `['C3b AXIS/DMA physical smoke result', 'requested XR-VITs sibling']`

## Blocker Status

| Blocker | Current | Candidate | Ready | Next Input |
|---|---:|---:|---:|---|
| `C3b AXIS/DMA physical smoke result` | `missing` | `not-supplied` | `False` | `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json` |
| `requested XR-VITs sibling` | `missing` | `fail` | `False` | `/home/kjm26/project/PRJXR/XR-VITs OR /home/kjm26/project/PRJXR/XR-VIT/HGTXR/docs/resources/xr_vits_replacement_policy.json` |

## Next Inputs

- `C3b AXIS/DMA physical smoke result`: capture ZCU104 C3b smoke, dry-run import it, then import to canonical path -> `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json`
- `requested XR-VITs sibling`: restore exact XR-VITs source or approve fingerprint-bound XR_Accel replacement policy -> `/home/kjm26/project/PRJXR/XR-VITs OR /home/kjm26/project/PRJXR/XR-VIT/HGTXR/docs/resources/xr_vits_replacement_policy.json`

## Closure Readiness

- status: `blocked`
- current_ready: `False`
- candidate_ready: `False`

## Dry-Run Final Runner

```sh
# Not ready.
```

## Active Final Runner

```sh
# Not ready.
```

## Safety

- executes_commands: `False`
- creates_board_result: `False`
- creates_xr_vits_policy: `False`
- writes_canonical_inputs: `False`
