# Execution Plan

This plan describes the next steps after the current merge and validation state.

## Remote Repository

- Remote target: `https://github.com/koodg123/ViT_Accel`
- Current local repository target: `ViT_Accel`

## Immediate Plan

### 1. Close the active VCK190 OOC validation

The most important active job is the host-identity retry of:

```bash
workspace/flow/scripts/targets/vck190/readme.sh step4-ooc
```

Expected output:

- either a clean OOC completion
- or a later, more meaningful Vivado failure point than the original license-checkout crash

Reason:

- this run already progressed past the old `realloc()` failure point
- it is the strongest indicator that the Docker host-identity mitigation is working

### 2. Re-run ZCU102 with the same host-identity profile

After the active VCK190 OOC run finishes, retry:

```bash
workspace/flow/scripts/targets/zynqmp/vivado.sh step4-bitstream zcu102
```

with the same Docker profile:

- host network
- fixed hostname
- host `udev` visibility
- host `machine-id` visibility

Reason:

- the old `zcu102` failure also matched the same host-info / WebTalk / license neighborhood
- the VCK190 improvement should be checked against the ZynqMP path as well

### 3. Finish the real Spinal `.dat` inventory

Continue the case-by-case HLS replay for the remaining missing bundles:

- `ATTN3..9`
- `MLP1..11`

Command surface:

```bash
workspace/flow/scripts/ops/spinal_dat_recover.sh
workspace/flow/scripts/run/spinal_dat_recover_backend.sh
```

Target condition:

- strict exported Spinal bundle has no missing `.dat` files

### 4. Re-validate the Spinal-backed ZynqMP bitstream path without placeholders

After the real `.dat` inventory is complete:

```bash
workspace/flow/scripts/targets/zynqmp/spinal.sh step4-prepare zcu102
workspace/flow/scripts/targets/zynqmp/spinal.sh step4-bitstream zcu102
```

Goal:

- remove the temporary placeholder-assisted smoke-test dependency

## Medium-Term Plan

### 5. Freeze the validated execution profiles

Once `vck190` and `zcu102` are both stable:

- write the final recommended Docker launch profile into the runbooks
- separate host prerequisites from optional workarounds
- keep one recommended command per common scenario

### 6. Consolidate legacy and config-driven entrypoints

The merged tree currently keeps both:

- README-style wrappers
- config-driven `hgpipe` commands

Next cleanup target:

- keep wrappers for operator convenience
- make `automation/cli/hgpipe.py` the canonical internal API surface

### 7. Publish the repository

After validation and documentation are stable:

1. ensure the local git history is clean
2. verify the configured `origin`
3. push to `https://github.com/koodg123/ViT_Accel`

## Decision Criteria

The repository can be considered ready for publication when all of the following are true:

- `vck190` reaches a confirmed `step4-ooc` result
- `zcu102` reaches a confirmed `step4-bitstream` result
- the Spinal export no longer depends on placeholder `.dat` files
- the main runbooks match the real validated commands
