# HGTXR Third Goal Unblock Checklist

- status: `pending-unblock`
- root: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR`
- final_blockers: `2`
- steps: `3`

## Final Blockers

- `C3b AXIS/DMA physical smoke result`
- `requested XR-VITs sibling`

## Steps

### B1 Resolve XR-VITs reference gate
- status: `pending-user-choice`
- why: Final signoff requires the exact XR-VITs sibling or an explicitly approved replacement policy.
- option `B1a`: Restore exact XR-VITs checkout
```sh
test -d /home/kjm26/project/PRJXR/XR-VITs
python3 tools/check_third_goal_preflight.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --mode final-signoff
```
- option `B1b`: Approve XR_Accel as replacement
```sh
python3 tools/create_xr_vits_replacement_policy.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --approve --approved-by <name> --reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff" --dry-run
python3 tools/create_xr_vits_replacement_policy.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --approve --approved-by <name> --reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff"
python3 tools/check_third_goal_preflight.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --mode final-signoff
```

### B2 Run and import C3b ZCU104 physical smoke
- status: `pending-board-run`
- why: Final signoff requires a real board-produced C3b smoke JSON.
- board commands:
```sh
tar -xzf e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz
cd e2e_axis_dma_c3b_mem16_smoke_bundle
./run_e2e_axis_dma_c3b_mem16_file_smoke.sh
./validate_e2e_axis_dma_c3b_mem16_file_smoke.sh
```
- host commands:
```sh
python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --import-c3b-smoke-json /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --dry-run-import-c3b-smoke --allow-blocked
python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --import-c3b-smoke-json /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --allow-blocked
python3 tools/validate_pynq_smoke_result.py hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json --preset axis-c3b-mem16
python3 tools/check_third_goal_preflight.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --mode final-signoff
```

### B3 Run final host signoff after blockers are cleared
- status: `pending`
```sh
python3 tools/check_third_goal_preflight.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --mode final-signoff --json-out generated/signoff/third_goal_final_signoff_2026_06_10.json
python3 tools/write_final_signoff_audit.py --preflight-json generated/signoff/third_goal_final_signoff_2026_06_10.json --json-out generated/signoff/final_signoff_audit_2026_06_10.json --markdown-out generated/signoff/final_signoff_audit_2026_06_10.md
python3 tools/write_third_goal_completion_audit.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --json-out generated/signoff/third_goal_completion_audit_2026_06_10.json --markdown-out generated/signoff/third_goal_completion_audit_2026_06_10.md
```
- expected: final-signoff fail=0 and completion audit status=pass

