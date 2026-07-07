# HGTXR Final Operator Handoff Validation

- status: `pass`
- pass: `37`
- fail: `0`

## Checks

| Check | Status | Detail |
|---|---|---|
| status | `pass` | pending-operator-actions |
| root | `pass` | /home/kjm26/project/PRJXR/XR-VIT/HGTXR |
| remaining_blockers | `pass` | remaining_blockers is list |
| board_ready | `pass` | True |
| board_preset | `pass` | axis-c3b-mem16 |
| board_bundle | `pass` | /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/generated/pynq/e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz |
| board_expected_runtime | `pass` | 2 |
| board_expected_output | `pass` | [32, -13, 26, -6, 14, -11] |
| board_execute_command | `pass` | board execute command present |
| board_import_command | `pass` | board import command present |
| board_validate_command | `pass` | board validate command present |
| xr_requested_path | `pass` | /home/kjm26/project/PRJXR/XR-VITs |
| xr_exact_command | `pass` | exact restore check present |
| xr_replacement_command | `pass` | replacement approval command present |
| xr_resolution_present | `pass` | xr_vits.reference_resolution present |
| xr_resolution_status_known | `pass` | candidate-ready-needs-approval |
| xr_resolution_matches_blocker | `pass` | blocked=True resolution_ready=False |
| xr_resolution_candidate_score | `pass` | candidate_status=pass score=99 |
| xr_resolution_dry_run_command | `pass` | resolution dry-run approval command present |
| xr_resolution_approve_command | `pass` | resolution approval command present |
| xr_resolution_no_policy_write | `pass` | False |
| xr_resolution_no_canonical_write | `pass` | False |
| evidence_manifest_contract_present | `pass` | final_evidence_manifest_contract present |
| evidence_manifest_contract_pass | `pass` | pass |
| evidence_manifest_required_complete | `pass` | 50/50 |
| evidence_manifest_consistency_count | `pass` | 32 |
| evidence_manifest_failed_consistency_zero | `pass` | [] |
| evidence_manifest_no_policy_write | `pass` | False |
| evidence_manifest_no_board_create | `pass` | False |
| evidence_manifest_no_canonical_write | `pass` | False |
| combined_exact | `pass` | combined exact import command present |
| combined_replacement | `pass` | combined replacement dry-run and approval command present |
| final_signoff_command | `pass` | final signoff command present |
| safety_no_board_create | `pass` | False |
| safety_no_policy_create | `pass` | False |
| safety_no_network_exec | `pass` | False |
| operator_required | `pass` | True |

## Safety

- Does not execute commands.
- Does not create board smoke results.
- Does not create XR-VITs replacement policy.
- Does not write canonical unblock inputs.
