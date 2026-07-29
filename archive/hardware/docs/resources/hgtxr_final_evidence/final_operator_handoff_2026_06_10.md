# HGTXR Final Operator Handoff

- status: `pending-operator-actions`
- root: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR`

## Remaining Blockers

- `requested XR-VITs sibling`
- `C3b AXIS/DMA physical smoke result`

## Board Smoke

- status: `ready-for-board`
- ready: `True`
- variant/preset: `c3b-mem16` / `axis-c3b-mem16`
- bundle_tar: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/generated/pynq/e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz`
- bundle_sha256: `3de505a6627409b4717ea95e5e76d74f0667c61b601fd96db1e495319adda712`
- physical_result: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json`
- expected_runtime_state: `2`
- expected_out_raw: `[32, -13, 26, -6, 14, -11]`

```sh
python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --zcu104-host <zcu104-ip-or-host> --zcu104-user xilinx --execute-zcu104-smoke --allow-blocked
python3 tools/run_zcu104_c3b_smoke_remote.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --host <zcu104-ip-or-host> --user xilinx --execute
python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --import-c3b-smoke-json /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --allow-blocked
python3 tools/validate_pynq_smoke_result.py /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --preset axis-c3b-mem16
python3 tools/import_pynq_smoke_result.py /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --preset axis-c3b-mem16 --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --dry-run --json-out /tmp/c3b_smoke_import_2026_06_10.json --validation-out /tmp/c3b_smoke_import_validation_2026_06_10.json
python3 tools/import_pynq_smoke_result.py /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --preset axis-c3b-mem16 --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --json-out /tmp/c3b_smoke_import_2026_06_10.json --validation-out /tmp/c3b_smoke_import_validation_2026_06_10.json
```

## XR-VITs Gate

- candidate_status: `fail`
- candidate_mode: `exact`
- requested_path: `/home/kjm26/project/PRJXR/XR-VITs`
- replacement_path: ``
- reference_resolution_status: `candidate-ready-needs-approval`
- reference_resolution_ready: `False`
- reference_approval_required: `True`
- reference_candidate_score: `99`

### Exact Restore

```sh
test -d /home/kjm26/project/PRJXR/XR-VITs
python3 tools/check_third_goal_preflight.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --mode final-signoff
```

### Replacement Approval

```sh
python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --approve-xr-vits-replacement --dry-run-xr-vits-replacement --xr-vits-approved-by <approved-by> --xr-vits-replacement-reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff" --allow-blocked
python3 tools/create_xr_vits_replacement_policy.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --approve --approved-by <approved-by> --reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff" --dry-run
python3 tools/create_xr_vits_replacement_policy.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --approve --approved-by <approved-by> --reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff"
python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --approve-xr-vits-replacement --xr-vits-approved-by <approved-by> --xr-vits-replacement-reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff" --allow-blocked
```

### Reference Resolution Commands

```sh
python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked --approve-xr-vits-replacement --dry-run-xr-vits-replacement --xr-vits-approved-by <approved-by> --xr-vits-replacement-reason "Approve XR_Accel as XR-VITs replacement" --xr-vits-replacement-path /home/kjm26/project/PRJXR/XR-VIT/XR_Accel
python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked --approve-xr-vits-replacement --xr-vits-approved-by <approved-by> --xr-vits-replacement-reason "Approve XR_Accel as XR-VITs replacement" --xr-vits-replacement-path /home/kjm26/project/PRJXR/XR-VIT/XR_Accel
```

### Policy Integrity

- required: `True`
- policy_exists: `False`
- validation_status: `pending-policy-creation`
- generator: `tools/create_xr_vits_replacement_policy.py`
- validator: `tools/check_final_blocker_closure_readiness.py`
- required_policy_fields: `['candidate_audit_fingerprint', 'candidate_audit_recommendation_snapshot', 'candidate_audit_meta', 'approval_event', 'policy_fingerprint']`
- candidate_audit_fingerprint: ``
- policy_fingerprint: ``

## Final Evidence Manifest Contract

- status: `pass`
- source: `manifest-json`
- path: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/docs/resources/final_evidence_manifest_2026_06_10.json`
- required: `86/86`
- consistency_checks: `347`
- failed_consistency_checks: `[]`

## Combined One-Shot

### Exact Restore + C3b Import

```sh
test -d /home/kjm26/project/PRJXR/XR-VITs
python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --import-c3b-smoke-json /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json
```

### Replacement Approval + C3b Import

```sh
python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --import-c3b-smoke-json /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --dry-run-import-c3b-smoke --approve-xr-vits-replacement --dry-run-xr-vits-replacement --xr-vits-approved-by <approved-by> --xr-vits-replacement-reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff" --allow-blocked
python3 tools/create_xr_vits_replacement_policy.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --approve --approved-by <approved-by> --reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff" --dry-run
python3 tools/create_xr_vits_replacement_policy.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --approve --approved-by <approved-by> --reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff"
python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --import-c3b-smoke-json /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --approve-xr-vits-replacement --xr-vits-approved-by <approved-by> --xr-vits-replacement-reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff"
```

## Final Signoff

- current_status: `blocked`
- expected_result: `status=pass, final_preflight_failures=0`

```sh
python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR
```

## Safety

- This handoff does not create board smoke results.
- This handoff does not create XR-VITs replacement policy.
- This handoff does not execute network commands.
- Commands with placeholders require operator replacement before execution.
