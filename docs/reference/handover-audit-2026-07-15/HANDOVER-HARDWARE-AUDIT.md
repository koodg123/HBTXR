# HANDOVER hardware semantic audit

## Decision summary

The current HBTXR hardware tree remains authoritative. HANDOVER hardware is a
source of regression vectors, board/configuration knowledge, and selected
automation patterns; it is not a replacement source tree. No RTL, HLS, driver,
or board image is promoted by this audit.

## Evidence and classification

| Source | Observed evidence | Class | Decision |
| --- | --- | --- | --- |
| `HANDOVER/HGTXR/hardware` | 701 paths; 394 common with current HBTXR, only 45 identical and 349 changed | `ADAPT` | Audit modules and tests semantically; never overwrite the common tree. |
| HG-PIPE Quantization | 125 unique blobs already present; implementation overlap 33, with 29 identical and 4 changed | `NO-OP` / targeted `REFERENCE_ONLY` | Keep archive; examine four differences only if a current quantization defect or provenance question requires it. |
| ICCAD24 HG-PIPE | 928/964 unique blobs represented; absent set dominated by SPINAL 33 and cache 3 | `REFERENCE_ONLY` | Preserve compact headers/docs. Do not import caches or full generated case tree. |
| `HANDOVER/XR_Accel` | 824/1,044 unique blobs represented; absent material includes workspace, docs, automation, and configs | `ADAPT` | Prioritize cyclic ZCU104 automation/config and reusable report logic. Exclude generated workspace. |
| `HANDOVER/ViT_Accel` | 827/1,532 unique blobs represented; large absent workspace component | `REFERENCE_ONLY` / future `ADAPT` | Reuse board-independent orchestration/report ideas only after XR work; do not import the full DeiT workspace. |

## HGTXR hardware review

The 349 changed files among 394 common paths show that the current tree has
evolved while retaining recognizable module names. A filename-level merge
would be unsafe because interface widths, fixed-point types, scheduling,
buffers, quantization, target clocks, and generated-tool assumptions may have
changed together.

Recommended extraction order:

1. Mine HGTXR-only unit/regression tests and golden vectors that describe
   operator behavior without depending on generated projects.
2. Compare the matching current module's input/output shape, AXI protocol,
   quantization scale/rounding/saturation, state reset, and latency contract.
3. Recreate useful assertions in the current test harness.
4. Change an active module only if a focused failing test demonstrates a gap.

Golden-vector candidates include LayerNorm, GELU, Softmax, quantized matrix
multiplication, QKV/attention, reshape/packing, and end-to-end search/track
mode boundaries. Each vector needs source revision, dtype/scale, tolerance,
seed, and expected-output hash.

## XR accelerator candidates

### High-priority selective adaptation

- `automation/cyclic_top.py`: extract graph/topology generation and validation
  rules into the current automation framework; remove source-tree and tool-path
  assumptions.
- `automation/experiments.py`: adapt experiment enumeration, immutable run IDs,
  command rendering, and result collection; do not inherit shell injection or
  absolute-path behavior.
- `configs/.../hbtxr_cyclic_zcu104.json` and
  `hbtxr_streaming_zcu104.json`: translate through a versioned current schema.
- ZCU104 option A/B/C experiment configs: retain as design-space inputs, not as
  validated performance claims.
- Cyclic ZCU104 C++ case modules: inspect scheduling and buffering patterns,
  then implement only the portions supported by current unit/golden tests.
- Status/runbook/report parsing: reuse board-independent field extraction and
  failure classification after tool-version fixtures pass.

The C3 evidence committed compact XR configs and documentation under
`hardware/configs/xr_accel` and `hardware/docs/**/xr_accel`. These are evidence
and future schema-normalization candidates; they are not consumed by the active
build by virtue of being present.

### Required XR gates

- schema version and rejection of unknown/ambiguous fields;
- ZCU104 part/board, clock, AXI width, memory topology, and tool-version lock;
- generated command dry-run from a clean checkout;
- C-simulation and current golden-vector equivalence;
- synthesis resource ceiling and timing-report parser fixtures;
- post-route timing/resource/power evidence before any performance statement;
- on-board smoke with bitstream/HWH hash, input/output hashes, host/board logs,
  and explicit failure state.

## ViT accelerator candidates

ViT_Accel is lower priority because 705 unique blobs are not represented and
much of the gap is a generated/full-model workspace. Potentially reusable,
board-independent pieces are:

- multi-board experiment orchestration and target capability metadata;
- report parsing/normalization for HLS, synthesis, implementation, and power;
- run manifests, status aggregation, and reproducibility checklists;
- portable validation utilities that accept current paths and schemas.

The full DeiT workspace, copied IP, generated RTL/project state, board outputs,
and cached reports are `EXCLUDED`. A future need for VCK190/ZCU102/ZU15EG
support should trigger a separate design proposal rather than enlarge the
current ZCU104 integration implicitly.

## Artifact and path exclusions

The following remain outside active source and Git integration:

- `.Xil`, Vivado/Vitis projects, `generated`, build instances, caches, and logs;
- bitstreams, HWH/XSA/DCP/IP archives, checkpoints, datasets, and large bundles;
- machine absolute paths, tool-install paths, remote credentials, and host-only
  launcher state;
- Python cache files and ICCAD24 cached/generated case outputs.

External hardware artifacts require a manifest containing source commit,
tool/board versions, build/config hash, artifact hash, command, and storage URI.

## Hardware compatibility gate

Before an `ADAPT` item can change active HBTXR hardware:

1. **Interface:** shapes, bit widths, AXI handshakes, register map, reset, and
   framing are documented and asserted.
2. **Numerics:** fixed-point type, scale, rounding, saturation, tolerance, and
   golden-vector provenance are fixed.
3. **Simulation:** unit C/C++ test, C-sim, RTL/co-sim where applicable, and
   end-to-end mode tests pass.
4. **Synthesis:** tool/part/clock are pinned; resource and latency reports are
   parsed with fixtures and compared to an approved ceiling.
5. **Implementation:** post-route timing, utilization, power, and design-rule
   evidence are recorded; estimates alone cannot clear promotion.
6. **Board:** transfer manifest, hashes, runner version, dry-run, physical smoke,
   and rollback procedure exist.
7. **Software match:** current software quantization, packing, tensor layout,
   and metric fixtures match the hardware contract.

## Ordered backlog

1. `P0` — import/adapt compatible HGTXR golden vectors and regression assertions.
2. `P1` — prototype XR cyclic automation/config translation in isolation.
3. `P1` — validate one ZCU104 cyclic case through simulation and report parsing;
   require separate approval before board execution or active-source changes.
4. `P2` — reconcile the four changed quantization implementation files only on
   a demonstrated behavioral need.
5. `P3` — consider ViT multi-board orchestration if multi-board support becomes
   a declared HBTXR requirement.
6. `P3` — consider ICCAD24 SPINAL material only if an active dependency appears.

## Conclusion

The most defensible hardware reuse is test-first: port golden behavior and
portable automation, then make the smallest current-tree change needed to pass
explicit gates. Generated workspaces and historic implementation output provide
provenance but must not be treated as maintainable or board-validated source.
