# HGTXR Final Unblock Closeout Packet Validation

- status: `pass`
- checks: `49/49`
- fail_count: `0`

## Checks

| Check | Status | Detail |
|---|---|---|
| packet_status_ready | `pass` | ready-for-operator-unblock |
| remaining_blockers_expected | `pass` | ['C3b AXIS/DMA physical smoke result', 'requested XR-VITs sibling'] |
| required_artifact_count | `pass` | 12 |
| required_artifacts_all_pass | `pass` | ['pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass'] |
| required_artifact_hashes_present | `pass` | sha256 on all required artifacts |
| missing_required_artifacts_empty | `pass` | [] |
| board_package_ready | `pass` | ready-for-board |
| board_package_preset | `pass` | axis-c3b-mem16 |
| board_package_sha | `pass` | 3de505a6627409b4717ea95e5e76d74f0667c61b601fd96db1e495319adda712 |
| board_expected_runtime | `pass` | 2 |
| board_expected_output | `pass` | [32, -13, 26, -6, 14, -11] |
| c3b_contract_status | `pass` | pass |
| c3b_contract_preset | `pass` | axis-c3b-mem16 |
| c3b_contract_variant | `pass` | c3b-mem16 |
| c3b_contract_canonical_path | `pass` | /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json |
| c3b_contract_runtime | `pass` | 2 |
| c3b_contract_output | `pass` | [32, -13, 26, -6, 14, -11] |
| c3b_contract_commands_present | `pass` | {'active_import_command': 'python3 tools/import_pynq_smoke_result.py /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --preset axis-c3b-mem16 --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --json-out /tmp/c3b_smoke_import_2026_06_10.json --validation-out /tmp/c3b_smoke_import_validation_2026_06_10.json', 'canonical_result_path': '/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json', 'dry_run_import_command': 'python3 tools/import_pynq_smoke_result.py /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --preset axis-c3b-mem16 --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --dry-run --json-out /tmp/c3b_smoke_import_2026_06_10.json --validation-out /tmp/c3b_smoke_import_validation_2026_06_10.json', 'expected_out_raw': [32, -13, 26, -6, 14, -11], 'expected_runtime_state': 2, 'preset': 'axis-c3b-mem16', 'status': 'pass', 'validate_command': 'python3 tools/validate_pynq_smoke_result.py /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --preset axis-c3b-mem16', 'variant': 'c3b-mem16'} |
| xr_resolution_known | `pass` | candidate-ready-needs-approval |
| xr_candidate_score | `pass` | 99 |
| xr_candidate_path | `pass` | /home/kjm26/project/PRJXR/XR-VIT/XR_Accel |
| xr_policy_integrity_required | `pass` | True |
| xr_policy_integrity_fields | `pass` | ['candidate_audit_fingerprint', 'candidate_audit_recommendation_snapshot', 'candidate_audit_meta', 'approval_event', 'policy_fingerprint'] |
| xr_policy_integrity_generator | `pass` | tools/create_xr_vits_replacement_policy.py |
| xr_policy_integrity_validator | `pass` | tools/check_final_blocker_closure_readiness.py |
| xr_legacy_policy_not_accepted | `pass` | False |
| xr_policy_validation_status | `pass` | stored=pending-policy-creation live=pending-policy-creation |
| xr_policy_validation_not_legacy | `pass` | pending-policy-creation |
| xr_policy_validation_embedded_matches_live | `pass` | stored=pending-policy-creation live=pending-policy-creation |
| closure_status_blocked | `pass` | blocked |
| closure_current_not_ready | `pass` | False |
| vref_successor_not_final_blocker | `pass` | required=False policy=False |
| vref_successor_gate_status_known | `pass` | ready-for-physical-smoke |
| vref_successor_required_blocker_consistent | `pass` | ['C3b AXIS/DMA physical smoke result', 'requested XR-VITs sibling'] |
| vref_successor_variant_selected | `pass` | dsp_mixed_stream |
| vref_successor_physical_path | `pass` | /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/pynq/hgtxr/e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.json |
| dry_run_commands_present | `pass` | 3 |
| dry_run_uses_closure_checker | `pass` | ['python3 tools/check_final_blocker_closure_readiness.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --c3b-no-require-paths --json-out /tmp/hgtxr_final_blocker_readiness_current.json --markdown-out /tmp/hgtxr_final_blocker_readiness_current.md', 'python3 tools/check_final_blocker_closure_readiness.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --c3b-smoke-json /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --xr-vits-mode exact --json-out /tmp/hgtxr_final_blocker_readiness_exact.json --markdown-out /tmp/hgtxr_final_blocker_readiness_exact.md', 'python3 tools/check_final_blocker_closure_readiness.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --c3b-smoke-json /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --xr-vits-mode replacement --approved-by <approved-by> --reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff" --json-out /tmp/hgtxr_final_blocker_readiness_replacement.json --markdown-out /tmp/hgtxr_final_blocker_readiness_replacement.md'] |
| combined_unblock_options_present | `pass` | ['U4a', 'U4b'] |
| combined_replacement_has_dry_run | `pass` | ['python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --import-c3b-smoke-json /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --dry-run-import-c3b-smoke --approve-xr-vits-replacement --dry-run-xr-vits-replacement --xr-vits-approved-by <approved-by> --xr-vits-replacement-reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff" --allow-blocked', 'python3 tools/create_xr_vits_replacement_policy.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --approve --approved-by <approved-by> --reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff" --dry-run', 'python3 tools/create_xr_vits_replacement_policy.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --approve --approved-by <approved-by> --reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff"', 'python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --import-c3b-smoke-json /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --approve-xr-vits-replacement --xr-vits-approved-by <approved-by> --xr-vits-replacement-reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff"'] |
| combined_replacement_has_active_command | `pass` | ['python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --import-c3b-smoke-json /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --dry-run-import-c3b-smoke --approve-xr-vits-replacement --dry-run-xr-vits-replacement --xr-vits-approved-by <approved-by> --xr-vits-replacement-reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff" --allow-blocked', 'python3 tools/create_xr_vits_replacement_policy.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --approve --approved-by <approved-by> --reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff" --dry-run', 'python3 tools/create_xr_vits_replacement_policy.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --approve --approved-by <approved-by> --reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff"', 'python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --import-c3b-smoke-json /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --approve-xr-vits-replacement --xr-vits-approved-by <approved-by> --xr-vits-replacement-reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff"'] |
| combined_replacement_uses_policy_generator | `pass` | ['python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --import-c3b-smoke-json /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --dry-run-import-c3b-smoke --approve-xr-vits-replacement --dry-run-xr-vits-replacement --xr-vits-approved-by <approved-by> --xr-vits-replacement-reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff" --allow-blocked', 'python3 tools/create_xr_vits_replacement_policy.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --approve --approved-by <approved-by> --reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff" --dry-run', 'python3 tools/create_xr_vits_replacement_policy.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --approve --approved-by <approved-by> --reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff"', 'python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --import-c3b-smoke-json /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --approve-xr-vits-replacement --xr-vits-approved-by <approved-by> --xr-vits-replacement-reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff"'] |
| qkv_uram_successor_not_final_blocker | `pass` | False |
| qkv_uram_successor_gate_status_known | `pass` | ready-for-physical-smoke |
| qkv_uram_successor_csynth_pass | `pass` | pass |
| qkv_uram_successor_resources_present | `pass` | {'bram_18k': 114, 'dsp': 128, 'ff': 19664, 'lut': 43236, 'uram': 40} |
| qkv_uram_successor_physical_path | `pass` | /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/pynq/hgtxr/e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json |
| qkv_uram_successor_command_group_present | `pass` | ['python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --zcu104-host <zcu104-ip-or-host> --zcu104-user xilinx --require-qkv-uram-physical-smoke --allow-blocked', 'python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --zcu104-host <zcu104-ip-or-host> --zcu104-user xilinx --execute-qkv-uram-smoke --allow-blocked', 'python3 tools/validate_pynq_smoke_result.py /path/to/e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json --preset axis-vref-p0-softmax-input-x2-qkv-uram', 'python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --import-qkv-uram-smoke-json /path/to/e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json --dry-run-import-qkv-uram-smoke --allow-blocked', 'python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --import-qkv-uram-smoke-json /path/to/e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json --require-qkv-uram-physical-smoke --allow-blocked'] |
| no_side_effect_safety | `pass` | {'executes_commands': False, 'executes_network': False, 'creates_board_result': False, 'creates_xr_vits_policy': False, 'writes_canonical_inputs': False} |
