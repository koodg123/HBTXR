# HGTXR Final Unblock Closeout Packet

- status: `ready-for-operator-unblock`
- root: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR`

## Remaining Blockers

- `requested XR-VITs sibling`
- `C3b AXIS/DMA physical smoke result`

## Required Artifacts

| Artifact | Status | SHA256 |
|---|---|---|
| `docs/resources/final_unblock_commands_2026_06_10.json` | `pass` | `101ab0e4e406b4b0050c05f547e46567eb10406dc815b1d52b8caf01705bc4cd` |
| `docs/resources/final_unblock_commands_2026_06_10.md` | `pass` | `a28e526761519ab912b58639d59ad6fdcacdd590cabc636dcf257db394433d85` |
| `docs/resources/c3b_smoke_result_contract_2026_06_10.json` | `pass` | `06b459926d8719bc709d9eb6e1e96ca26fc4ed5bd7a6874bd095e5194ac27df8` |
| `docs/resources/c3b_smoke_result_contract_2026_06_10.md` | `pass` | `1c2a6e1af4034d1e0a3596194d0d73df9dcbf70cec5049e0a0ece6b49f88133a` |
| `docs/resources/c3b_board_smoke_readiness_2026_06_10.json` | `pass` | `742814452ee0d4cae1ae9c511ee5da42ad96b1939c9484b45f65cfb29425d004` |
| `docs/resources/c3b_smoke_transfer_manifest_2026_06_10.json` | `pass` | `d5790083c30815929b619f137898e5b1153adeaf497e4967d6d23fbf2f2e88a2` |
| `docs/resources/e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz.sha256` | `pass` | `1f082bf9ec7e71c10a0caa475a8a3ac0e333e1c175829e4119832c9b53f86687` |
| `docs/resources/xr_vits_reference_resolution_2026_06_10.json` | `pass` | `173ff8591e11024e8e9b29dd21dcdd1cfd832f3878acaba7ff91a0d067f9e3fc` |
| `docs/resources/final_blocker_closure_readiness_2026_06_10.json` | `pass` | `25ff68310de4c3ddd42eb2b0e89660f1173e1cb743d37a559052217cf45e5494` |
| `docs/resources/final_unblock_intake_2026_06_10.json` | `pass` | `fa84e40c89fc7a90a16d9e8d7b64d2ecfbe19fed465cd9df4b467a8223dcdeac` |
| `docs/resources/final_unblock_intake_2026_06_10.md` | `pass` | `bfdaf19964593b42c9b726a721bb6e859bd11009a68f946eae55054ec42c0085` |
| `docs/resources/final_signoff_audit_2026_06_10.json` | `pass` | `bb922efcfd86266cc2c04cfb959590dfe7a8746645b61285e4b54c678af5a99b` |

## Dry-Run Readiness

```sh
python3 tools/check_final_blocker_closure_readiness.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --c3b-no-require-paths --json-out /tmp/hgtxr_final_blocker_readiness_current.json --markdown-out /tmp/hgtxr_final_blocker_readiness_current.md
python3 tools/check_final_blocker_closure_readiness.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --c3b-smoke-json /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --xr-vits-mode exact --json-out /tmp/hgtxr_final_blocker_readiness_exact.json --markdown-out /tmp/hgtxr_final_blocker_readiness_exact.md
python3 tools/check_final_blocker_closure_readiness.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --c3b-smoke-json /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --xr-vits-mode replacement --approved-by <approved-by> --reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff" --json-out /tmp/hgtxr_final_blocker_readiness_replacement.json --markdown-out /tmp/hgtxr_final_blocker_readiness_replacement.md
```

## Combined Unblock

### U4a

```sh
test -d /home/kjm26/project/PRJXR/XR-VITs
python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --import-c3b-smoke-json /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json
```

### U4b

```sh
python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --import-c3b-smoke-json /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --dry-run-import-c3b-smoke --approve-xr-vits-replacement --dry-run-xr-vits-replacement --xr-vits-approved-by <approved-by> --xr-vits-replacement-reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff" --allow-blocked
python3 tools/create_xr_vits_replacement_policy.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --approve --approved-by <approved-by> --reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff" --dry-run
python3 tools/create_xr_vits_replacement_policy.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --approve --approved-by <approved-by> --reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff"
python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --import-c3b-smoke-json /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --approve-xr-vits-replacement --xr-vits-approved-by <approved-by> --xr-vits-replacement-reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff"
```

## XR-VITs Policy Integrity

- required: `True`
- generator: `tools/create_xr_vits_replacement_policy.py`
- validator: `tools/check_final_blocker_closure_readiness.py`
- legacy_policy_clears_final_signoff: `False`
- required_policy_fields: `['candidate_audit_fingerprint', 'candidate_audit_recommendation_snapshot', 'candidate_audit_meta', 'approval_event', 'policy_fingerprint']`
- policy_exists: `False`
- validation_status: `pending-policy-creation`
- candidate_audit_fingerprint: ``
- policy_fingerprint: ``

## VREF Successor Gate

- status: `ready-for-physical-smoke`
- required_for_final_signoff: `False`
- require_physical_smoke_for_final_signoff: `False`
- recommended_resource_variant: `dsp_mixed_stream`
- projection_status: `not_promotable`
- physical_smoke_status: `not_captured`
- expected_result_json: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/pynq/hgtxr/e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.json`

## QKV URAM Successor Gate

- status: `ready-for-physical-smoke`
- required_for_final_signoff: `False`
- csynth_status: `pass`
- latency_cycles: `498485`
- resources: `{'bram_18k': 114, 'dsp': 128, 'ff': 19664, 'lut': 43236, 'uram': 40}`
- routed_wns_ns: `4.517`
- route_errors: `0`
- physical_smoke_status: `not_captured`
- expected_result_json: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/pynq/hgtxr/e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json`

### QKV URAM Operator Commands

```sh
python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --zcu104-host <zcu104-ip-or-host> --zcu104-user xilinx --require-qkv-uram-physical-smoke --allow-blocked
python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --zcu104-host <zcu104-ip-or-host> --zcu104-user xilinx --execute-qkv-uram-smoke --allow-blocked
python3 tools/validate_pynq_smoke_result.py /path/to/e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json --preset axis-vref-p0-softmax-input-x2-qkv-uram
python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --import-qkv-uram-smoke-json /path/to/e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json --dry-run-import-qkv-uram-smoke --allow-blocked
python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --import-qkv-uram-smoke-json /path/to/e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json --require-qkv-uram-physical-smoke --allow-blocked
```

## Safety

- executes_commands: `False`
- executes_network: `False`
- creates_board_result: `False`
- creates_xr_vits_policy: `False`
- writes_canonical_inputs: `False`
