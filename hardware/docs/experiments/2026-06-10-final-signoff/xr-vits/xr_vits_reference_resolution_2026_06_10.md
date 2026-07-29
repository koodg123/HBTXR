# HGTXR XR-VITs Reference Resolution

- status: `candidate-ready-needs-approval`
- root: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR`
- requested_path: `/home/kjm26/project/PRJXR/XR-VITs`
- resolution_ready: `False`
- approval_required: `True`

## Evidence

- exact: `missing`
- active_policy: `missing`
- candidate: `pass`
- candidate_role: `candidate-xr-accel`
- candidate_path: `/home/kjm26/project/PRJXR/XR-VIT/XR_Accel`

## Commands

```sh
python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked --approve-xr-vits-replacement --dry-run-xr-vits-replacement --xr-vits-approved-by <approved-by> --xr-vits-replacement-reason "Approve XR_Accel as XR-VITs replacement" --xr-vits-replacement-path /home/kjm26/project/PRJXR/XR-VIT/XR_Accel
python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked --approve-xr-vits-replacement --xr-vits-approved-by <approved-by> --xr-vits-replacement-reason "Approve XR_Accel as XR-VITs replacement" --xr-vits-replacement-path /home/kjm26/project/PRJXR/XR-VIT/XR_Accel
```

## Safety

- Does not execute commands.
- Does not create board smoke results.
- Does not create XR-VITs replacement policy.
- Does not write canonical unblock inputs.
