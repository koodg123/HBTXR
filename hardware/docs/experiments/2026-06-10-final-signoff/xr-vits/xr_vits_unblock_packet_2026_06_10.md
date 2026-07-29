# HGTXR XR-VITs Unblock Packet

- status: `pending-user-choice`
- requested_path: `/home/kjm26/project/PRJXR/XR-VITs`
- requested_path_exists: `False`
- active_policy_exists: `False`
- active_policy_approved: `False`
- next_required_action: Restore exact XR-VITs or explicitly approve XR_Accel replacement.

## Recommendation

- role: `candidate-xr-accel`
- path: `/home/kjm26/project/PRJXR/XR-VIT/XR_Accel`
- score: `99`
- reason: highest score for ZCU104/cyclic/DeiT/HLS evidence

## Options

### restore-exact-xr-vits
- title: Restore exact XR-VITs checkout
- effect: Final signoff can use the requested source path directly.
```sh
test -d /home/kjm26/project/PRJXR/XR-VITs
python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked
```

### approve-xr-accel-replacement
- title: Explicitly approve XR_Accel as XR-VITs replacement
- effect: Creates active replacement policy only after explicit approval metadata is supplied.
```sh
python3 tools/create_xr_vits_replacement_policy.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --replacement-path /home/kjm26/project/PRJXR/XR-VIT/XR_Accel --approve --approved-by <approved-by> --reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff" --dry-run
python3 tools/create_xr_vits_replacement_policy.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --replacement-path /home/kjm26/project/PRJXR/XR-VIT/XR_Accel --approve --approved-by <approved-by> --reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff"
python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked
```

## Safety

- This packet does not create docs/resources/xr_vits_replacement_policy.json.
- Active replacement policy requires explicit user approval.
- Exact XR-VITs restore remains the strictest path.
