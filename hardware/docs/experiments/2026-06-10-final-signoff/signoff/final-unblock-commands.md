# HGTXR Final Unblock Commands

- status: `pending-unblock`
- root: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR`
- readiness_status: `ready-for-board`
- xr_vits_packet_status: `pending-user-choice`
- c3b_smoke_contract_status: `pass`
- xr_vits_policy_integrity_required: `True`

## Safety

- This card does not create board smoke results.
- This card does not create an XR-VITs replacement policy.
- This card does not run network commands by itself.
- Replace placeholders before executing commands.
- XR-VITs replacement policy must include candidate audit fingerprint, recommendation snapshot, candidate metadata, and policy fingerprint.

## Remaining Blockers

- `C3b AXIS/DMA physical smoke result`
- `requested XR-VITs sibling`

## Commands

### U0 Dry-run final blocker closure readiness
- status: `advisory-no-side-effects`
- why: Check whether supplied C3b smoke and XR-VITs choices would clear final blockers before writing canonical unblock inputs.
```sh
python3 tools/check_final_blocker_closure_readiness.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --c3b-no-require-paths --json-out /tmp/hgtxr_final_blocker_readiness_current.json --markdown-out /tmp/hgtxr_final_blocker_readiness_current.md
python3 tools/check_final_blocker_closure_readiness.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --c3b-smoke-json /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --xr-vits-mode exact --json-out /tmp/hgtxr_final_blocker_readiness_exact.json --markdown-out /tmp/hgtxr_final_blocker_readiness_exact.md
python3 tools/check_final_blocker_closure_readiness.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --c3b-smoke-json /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --xr-vits-mode replacement --approved-by <approved-by> --reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff" --json-out /tmp/hgtxr_final_blocker_readiness_replacement.json --markdown-out /tmp/hgtxr_final_blocker_readiness_replacement.md
```
- expected: status=would-clear-with-candidates before running side-effectful final unblock command.

### U1 Run C3b ZCU104 smoke through SSH/SCP
- status: `pending-board-run`
- why: Final signoff needs a real board-produced C3b smoke JSON.
- precheck:
```sh
python3 tools/check_c3b_board_smoke_readiness.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR
```
```sh
python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --zcu104-host <zcu104-ip-or-host> --zcu104-user xilinx --execute-zcu104-smoke --allow-blocked
python3 tools/run_zcu104_c3b_smoke_remote.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --host <zcu104-ip-or-host> --user xilinx --execute
python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --import-c3b-smoke-json /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --allow-blocked
python3 tools/validate_pynq_smoke_result.py /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --preset axis-c3b-mem16
python3 tools/import_pynq_smoke_result.py /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --preset axis-c3b-mem16 --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --dry-run --json-out /tmp/c3b_smoke_import_2026_06_10.json --validation-out /tmp/c3b_smoke_import_validation_2026_06_10.json
python3 tools/import_pynq_smoke_result.py /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --preset axis-c3b-mem16 --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --json-out /tmp/c3b_smoke_import_2026_06_10.json --validation-out /tmp/c3b_smoke_import_validation_2026_06_10.json
```
- C3b contract:
  - status: `pass`
  - preset: `axis-c3b-mem16`
  - canonical_result_path: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json`
  - expected_out_raw: `[32, -13, 26, -6, 14, -11]`

### U2 Resolve XR-VITs reference gate
- status: `pending-user-choice`
- why: Final signoff requires the exact XR-VITs sibling or explicit replacement approval.
- option `U2a`: Restore exact XR-VITs sibling
```sh
test -d /home/kjm26/project/PRJXR/XR-VITs
python3 tools/check_third_goal_preflight.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --mode final-signoff
```
- option `U2b`: Approve XR_Accel replacement policy
```sh
python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --approve-xr-vits-replacement --dry-run-xr-vits-replacement --xr-vits-approved-by <approved-by> --xr-vits-replacement-reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff" --allow-blocked
python3 tools/create_xr_vits_replacement_policy.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --approve --approved-by <approved-by> --reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff" --dry-run
python3 tools/create_xr_vits_replacement_policy.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --approve --approved-by <approved-by> --reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff"
python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --approve-xr-vits-replacement --xr-vits-approved-by <approved-by> --xr-vits-replacement-reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff" --allow-blocked
```

### U4 Combined one-shot unblock
- status: `pending-user-choice`
- why: Use after a real C3b smoke JSON is copied back and the XR-VITs decision is known.
- option `U4a`: Import C3b result with restored XR-VITs
```sh
test -d /home/kjm26/project/PRJXR/XR-VITs
python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --import-c3b-smoke-json /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json
```
- option `U4b`: Import C3b result and approve XR_Accel
```sh
python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --import-c3b-smoke-json /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --dry-run-import-c3b-smoke --approve-xr-vits-replacement --dry-run-xr-vits-replacement --xr-vits-approved-by <approved-by> --xr-vits-replacement-reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff" --allow-blocked
python3 tools/create_xr_vits_replacement_policy.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --approve --approved-by <approved-by> --reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff" --dry-run
python3 tools/create_xr_vits_replacement_policy.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --approve --approved-by <approved-by> --reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff"
python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --import-c3b-smoke-json /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --approve-xr-vits-replacement --xr-vits-approved-by <approved-by> --xr-vits-replacement-reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff"
```

### U5 Optional QKV URAM successor smoke and promotion check
- status: `optional-ready-for-board-run`
- why: QKV URAM is not required for default final signoff, but this path validates the successor before any promotion claim.
```sh
python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --zcu104-host <zcu104-ip-or-host> --zcu104-user xilinx --require-qkv-uram-physical-smoke --allow-blocked
python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --zcu104-host <zcu104-ip-or-host> --zcu104-user xilinx --execute-qkv-uram-smoke --allow-blocked
python3 tools/validate_pynq_smoke_result.py /path/to/e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json --preset axis-vref-p0-softmax-input-x2-qkv-uram
python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --import-qkv-uram-smoke-json /path/to/e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json --dry-run-import-qkv-uram-smoke --allow-blocked
python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --import-qkv-uram-smoke-json /path/to/e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json --require-qkv-uram-physical-smoke --allow-blocked
```
- expected: QKV physical smoke validates only optional successor promotion; default final blockers remain C3b smoke and XR-VITs gate.

### U3 Run final signoff
- status: `pending`
- why: Run after U1/U2 are cleared.
```sh
python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR
```
- expected: status=pass, final_preflight_failures=0
