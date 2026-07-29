# HANDOVER integration matrix

## Classification rules

- `NO-OP`: current HBTXR already contains equivalent inspected content.
- `REFERENCE_ONLY`: retain for provenance or design context; do not execute.
- `ADAPT`: re-express behavior through current interfaces with focused tests.
- `ACTIVE_CANDIDATE`: adaptation may be proposed after prerequisite gates.
- `EXCLUDED`: generated, large, machine-specific, sensitive, or non-maintainable.
- `LICENSE_BLOCKED`: technical review may continue, but promotion is forbidden
  until provenance and license evidence are complete.

## Decision matrix

| ID | HANDOVER source / concept | Target surface | Class | Priority | Required work and acceptance evidence |
| --- | --- | --- | --- | --- | --- |
| S-00 | HBTXR analysis, 320/320 identical | `algorithm/analysis` | `NO-OP` | P0 | No copy. Recompare only when source revision changes. |
| S-01 | Common HBTXR third-party files, 71/71 identical | current archive/reference | `NO-OP` | P0 | No copy; retain current history. |
| S-02 | FECET/SWIFT imported ports | `algorithm/archive/imports` | `REFERENCE_ONLY` | P0 | Already archived in C1. Do not import from active code; license/API review required for reuse. |
| S-03 | EV-Eye/EX-Gaze analysis scripts/results | `algorithm/analysis` | `REFERENCE_ONLY` → `ADAPT` | P1 | C2 evidence exists. Replace absolute paths and adapt to current hybrid APIs; data/metric/checkpoint gates. |
| S-04 | HGTXR Stage-1 recovery/ensemble | current search interface | `ADAPT` | P1 | Feature-flagged strategy, deterministic fixtures, ranking/recovery and latency regression. |
| S-05 | HGTXR Stage-2 support/state behavior | current tracking interface | `ADAPT` | P1 | Explicit state machine, reset/sequence tests, disabled-path equivalence. |
| S-06 | XR64–XR68 teacher targets/failure buckets | current experiment configs/reports | `ACTIVE_CANDIDATE` | P2 | Recreate with current trainer, fixed split/seeds/metrics, ablation and failure taxonomy. |
| S-07 | Checkpoint interpolation/averaging | isolated current tool | `ACTIVE_CANDIDATE` | P1 | Key/shape/dtype/buffer audit, state policy, manifest, load/numerical tests. |
| S-08 | Metric/paper-target bridges | current evaluation layer | `ACTIVE_CANDIDATE` | P1 | Task and coordinate schemas, golden conversions, aggregation/invalid-sample policy. |
| S-09 | ERVT/Retina non-identical scripts/configs | legacy reference → current adapters | `LICENSE_BLOCKED` | P2 | Verify upstream/license first; then data/API regression and current-style rewrite. |
| S-10 | Legacy experiment/readiness/complexity reports and FECET README | `references` / analysis docs | `REFERENCE_ONLY` | P2 | Preserve provenance; reconcile claims only against reproduced current results. |
| H-00 | Standalone HG-PIPE quantization blobs already represented | current quantization/archive | `NO-OP` | P0 | No bulk import; four changed implementation paths reviewed only on demonstrated need. |
| H-01 | HGTXR operator golden vectors/regressions | current hardware tests | `ADAPT` | P0 | Provenance, scale/dtype/tolerance, unit/C-sim and software-match gates. |
| H-02 | HGTXR common active hardware sources, 349/394 changed | current hardware source | `REFERENCE_ONLY` | P0 | Never overwrite. Use only to explain or test a focused current defect. |
| H-03 | XR cyclic topology/experiment automation | current automation/tools | `ADAPT` | P1 | Remove absolute/tool paths; schema, dry-run, parser fixtures, command safety tests. |
| H-04 | XR ZCU104 cyclic/streaming and option A/B/C configs | versioned current hardware schema | `ACTIVE_CANDIDATE` | P1 | Schema translation, board/part/clock/memory validation, unknown-field rejection. |
| H-05 | XR cyclic ZCU104 C++ case patterns | current HLS case | `ACTIVE_CANDIDATE` | P1 | Small design proposal, current interfaces, golden/C-sim/synthesis gates; separate approval. |
| H-06 | ViT board-independent orchestration/report parsing | current automation/docs | `ADAPT` | P3 | Consider only after XR; portable parser fixtures and declared multi-board requirement. |
| H-07 | ViT full DeiT workspace and generated board projects | none | `EXCLUDED` | P0 | Keep external; do not add to Git. |
| H-08 | ICCAD24 compact headers/docs | `references/legacy-codebase` | `REFERENCE_ONLY` | P3 | C3 evidence exists. No compile/runtime claim. |
| H-09 | ICCAD24 SPINAL-only gap | none unless dependency declared | `REFERENCE_ONLY` | P3 | Import only if an active dependency and compatible license are proven. |
| A-00 | Checkpoints, datasets, HDF5, runs, logs | external artifact/data store | `EXCLUDED` | P0 | Hash manifest, access terms, schema/config, URI; no normal Git import. |
| A-01 | Vivado/Vitis generated trees, `.Xil`, bitstreams, HWH/XSA/DCP/IP | governed hardware artifact store | `EXCLUDED` | P0 | Source/config/tool/board hashes, validation and rollback; no source-tree import. |
| L-00 | ERVT, TENNs-Eye, TDTracker, BRAT | any active target | `LICENSE_BLOCKED` | P0 | Complete file/behavior-level upstream and license mapping before promotion. |

The summary above is descriptive. The following register is the normative S8
execution contract for items proposed for future work. Target files marked
“proposed” do not yet exist and are not authorized by this analysis commit.

## Normative actionable candidate register

| ID | Exact source (repository revision and path) | Exact proposed HBTXR target | Class / license | Expected benefit and required adaptation | Executable validation command after implementation | Material risk | Dependencies / priority | Rollback |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S-04 | HGTXR `70c9718117ca`: `software/src/hbtxr/models/tracker/search_branch.py` | `algorithm/hybrid/src/models/tracker/search_branch.py`; proposed `algorithm/hybrid/tests/handover/test_search_recovery.py` | `ADAPT`; source license must be verified | Preserve recovery/ensemble behavior behind the current search interface and feature flag. | `python -m pytest -q algorithm/hybrid/tests/handover/test_search_recovery.py` | Ranking drift, latency increase, hidden source-config coupling. | G1–G6, E1; P1 | Remove adapter/flag and restore current default; retain tests as reference. |
| S-05 | HGTXR `70c9718117ca`: `software/src/hbtxr/models/tracker/track_branch.py`, `software/src/hbtxr/models/tracker/head_factory.py`, `software/src/hbtxr/models/hybrid_tracker.py` | `algorithm/hybrid/src/models/tracker/track_branch.py`; proposed `algorithm/hybrid/tests/handover/test_track_state_adapter.py` | `ADAPT`; source license must be verified | Re-express support-adaptive/track-state behavior as explicit per-sequence state. | `python -m pytest -q algorithm/hybrid/tests/handover/test_track_state_adapter.py` | Cross-sequence state leakage, reset regressions, numerical drift. | S-04 contract vocabulary, G1–G6; P1 | Disable/remove adapter and state fields; current track path remains default. |
| S-06 | HGTXR `70c9718117ca`: `software/scripts/v3/run_xr64_teacher_target_construction.sh`, `software/scripts/v3/build_xr64_teacher_target_overrides.py`, `software/scripts/v3/build_failure_bucket_manifest.py`, `software/scripts/v3/run_xr65_p10p5_recovery_from_xr64c.sh`, `software/scripts/v3/run_xr66_lowsim_targeted_recovery.sh`, `software/scripts/v3/run_xr67_fullmanifest_lowsim_weighted_recovery.sh`, `software/scripts/v3/run_xr68_fixedlr_lowsim_weighted_recovery.sh` | proposed `algorithm/hybrid/configs/handover/xr64_xr68/`; proposed `algorithm/hybrid/tests/handover/test_xr64_xr68_contract.py` | `ACTIVE_CANDIDATE`; code/data terms must be verified | Recreate teacher-target and failure-bucket experiments using current trainer/config schemas; do not copy logs/data. | `python -m pytest -q algorithm/hybrid/tests/handover/test_xr64_xr68_contract.py` | Split leakage, cherry-picked buckets, unreproducible external artifacts. | E1, S-04/S-05 evidence, G1–G6/G10; P2 | Delete proposed configs/manifests; no current trainer change is required. |
| S-07 | HGTXR `70c9718117ca`: `software/scripts/v3/average_hbtxr_checkpoints.py`, `software/scripts/v3/interpolate_hbtxr_checkpoints.py`, `software/modules/interpolation.py` | proposed `algorithm/hybrid/tools/checkpoint_blend.py`; proposed `algorithm/hybrid/tests/handover/test_checkpoint_blend.py` | `ACTIVE_CANDIDATE`; source license must be verified | Implement format-aware inspection/average/interpolation with a provenance manifest and explicit state policy. | `python -m pytest -q algorithm/hybrid/tests/handover/test_checkpoint_blend.py` | Silent key/buffer loss, incompatible architecture, optimizer-state corruption. | E1 checkpoint contract, G1/G5/G6; P1 | Remove tool and generated manifests; source checkpoints are read-only and unchanged. |
| S-08 | HGTXR `70c9718117ca`: `software/metrics.py`, `software/scripts/v3/check_metric_coordinate_sanity.py`, `software/scripts/v3/check_metric_protocol_bridge.py`, `software/scripts/v3/check_paper_target_bridge_evaluator_schema_contract.py` | proposed `algorithm/hybrid/src/evaluation/handover_bridge.py`; proposed `algorithm/hybrid/tests/handover/test_evaluation_bridge.py` | `ACTIVE_CANDIDATE`; source license must be verified | Normalize task/coordinate records and compute explicitly named center/bbox/ellipse metrics. | `python -m pytest -q algorithm/hybrid/tests/handover/test_evaluation_bridge.py` | Invalid cross-task comparisons, axis scaling error, aggregation drift. | E1 data contract, G1–G6; P1 | Remove bridge and keep current evaluators; generated comparison reports are disposable. |
| H-01 | HGTXR `70c9718117ca`: `hardware/tools/export_s2_pytorch_golden.py`, `hardware/tools/validate_s2_block_full.py`, `hardware/tools/validate_hgpipe_lut_math.py` | proposed `hardware/refs/handover/hgtxr/`; proposed `hardware/tests/handover/test_hgtxr_golden_vectors.py` | `ADAPT`; code/reference license must be verified | Recreate compact operator vectors with scale/dtype/tolerance/source hashes against current software/hardware. | `python -m pytest -q hardware/tests/handover/test_hgtxr_golden_vectors.py` | False equivalence from mismatched quantization or tolerance. | G1/G7 and current vector exporter; P0 | Remove new vectors/tests; active HLS/RTL remains untouched. |
| H-03 | XR_Accel `5041401aec53`: `automation/cyclic_top.py`, `automation/experiments.py` | proposed `hardware/tools/xr_cyclic_plan.py`; proposed `hardware/tests/test_xr_cyclic_plan.py` | `ADAPT`; source license must be verified | Port topology/experiment planning, safe command rendering, immutable run IDs, and result parsing without absolute paths. | `python -m pytest -q hardware/tests/test_xr_cyclic_plan.py` | Unsafe command construction, tool/path coupling, stale result association. | H-04 schema, G1/G7/G8; P1 | Delete new tool/test and any dry-run manifests; no build output is retained. |
| H-04 | XR_Accel `5041401aec53`: `configs/designs/hbtxr_cyclic_zcu104.json`, `configs/designs/hbtxr_streaming_zcu104.json`, `configs/experiments/zcu104_hbtxr_option_a_cyclic.json`, `configs/experiments/zcu104_hbtxr_option_a_streaming.json`, `configs/experiments/zcu104_hbtxr_option_b_cyclic.json`, `configs/experiments/zcu104_hbtxr_option_b_streaming.json`, `configs/experiments/zcu104_hbtxr_option_c_cyclic.json`, `configs/experiments/zcu104_hbtxr_option_c_streaming.json`, `configs/models/hbtxr_option_a.json`, `configs/models/hbtxr_option_b.json`, `configs/models/hbtxr_option_c.json` | C3 references under `hardware/configs/xr_accel/`; proposed `hardware/configs/xr_accel/schema.json` and `hardware/tests/test_xr_accel_config_schema.py` | `ACTIVE_CANDIDATE`; config provenance/license must be verified | Add versioned translation/validation; reject unknown board, part, clock, memory, or interface fields. | `python -m pytest -q hardware/tests/test_xr_accel_config_schema.py` | Treating historic configs as board-validated; implicit defaults. | G1/G7, before H-03/H-05; P1 | Remove schema/translator; keep C3 files explicitly reference-only. |
| H-05 | XR_Accel `5041401aec53`: `workspace/hardware/case_cyclic_zcu104/CYCLIC_VIT_TOP.cpp`, `workspace/hardware/case_cyclic_zcu104/MHA_CORE_EVEN.cpp`, `workspace/hardware/case_cyclic_zcu104/MHA_CORE_ODD.cpp`, `workspace/hardware/case_cyclic_zcu104/MLP_CORE_EVEN.cpp`, `workspace/hardware/case_cyclic_zcu104/MLP_CORE_ODD.cpp`, `workspace/hardware/case_cyclic_zcu104/SEARCH_HEAD.cpp`, `workspace/hardware/case_cyclic_zcu104/TRACK_HEAD.cpp` | proposed `hardware/src/hls/xr_cyclic/`; proposed `hardware/tests/hls/test_xr_cyclic.cpp` | `ACTIVE_CANDIDATE`; source license must be verified | Reimplement one bounded cyclic case through current interfaces after vector/schema gates; never import the workspace. | `bash hardware/scripts/run_hls_csim.sh --case xr_cyclic` | Interface/quantization mismatch, resource/timing failure, generated-workspace dependency. | H-01/H-03/H-04, G1/G7–G9, separate approval; P1 | Revert the isolated `xr_cyclic` directory/config; preserve current HLS top and artifact hashes. |
| H-06 | ViT_Accel `90a82ddecd32`: `automation/experiments.py`, `automation/metadata.py`, `automation/reference_export.py`, `automation/utils/artifacts.py`, `automation/vivado.py` | proposed `hardware/tools/vit_multiboard_report.py`; proposed `hardware/tests/test_vit_multiboard_report.py` | `ADAPT`; source license must be verified | Extract board-independent orchestration/report normalization only if multi-board support is declared. | `python -m pytest -q hardware/tests/test_vit_multiboard_report.py` | Scope expansion, tool-version parsing drift, accidental workspace import. | H-03 stable, explicit multi-board requirement, G1/G8; P3 | Remove tool/tests and external report manifests; ZCU104 flow is unchanged. |

## HANDOVER/HBTXR ten-path trace

This table enumerates the nine byte-different and one target-missing paths from
the inspected `main..etri-server` delta. The first four target files exist and
differ. The five reports exist under current `algorithm/analysis/report` and
differ. The FECET README has no current target counterpart.

| Exact HANDOVER source at `7d1b0cace624` | Exact current/proposed HBTXR target | Disposition |
| --- | --- | --- |
| `references/codebase/software/ais2024/ERVT/configs/hbtxr_subject_independent_img64.json` | `references/legacy-codebase/software/ais2024/ERVT/configs/hbtxr_subject_independent_img64.json` | `LICENSE_BLOCKED`; compare schema only after ERVT clearance. |
| `references/codebase/software/ais2024/ERVT/train_hbtxr_subject_independent.py` | `references/legacy-codebase/software/ais2024/ERVT/train_hbtxr_subject_independent.py` | `LICENSE_BLOCKED`; do not replace current reference. |
| `references/codebase/software/retina/configs/hbtxr_subject_independent_img64_patch4.yaml` | `references/legacy-codebase/software/retina/configs/hbtxr_subject_independent_img64_patch4.yaml` | `REFERENCE_ONLY`; mine contract differences, no overwrite. |
| `references/codebase/software/retina/scripts/train_hbtxr_subject_independent.py` | `references/legacy-codebase/software/retina/scripts/train_hbtxr_subject_independent.py` | `REFERENCE_ONLY`; mine behavior, no overwrite. |
| `references/report/HBTXR_experiment_plan_2026-06-30.md` | `algorithm/analysis/report/HBTXR_experiment_plan_2026-06-30.md` | `REFERENCE_ONLY`; reconcile only with reproduced current evidence. |
| `references/report/HBTXR_img64_subject_independent_target_porting_overview_2026-06-29.md` | `algorithm/analysis/report/HBTXR_img64_subject_independent_target_porting_overview_2026-06-29.md` | `REFERENCE_ONLY`. |
| `references/report/HBTXR_target_model_complexity_2026-06-30.csv` | `algorithm/analysis/report/HBTXR_target_model_complexity_2026-06-30.csv` | `REFERENCE_ONLY`; recompute before updating. |
| `references/report/HBTXR_target_model_complexity_2026-06-30.md` | `algorithm/analysis/report/HBTXR_target_model_complexity_2026-06-30.md` | `REFERENCE_ONLY`; recompute before updating. |
| `references/report/HBTXR_target_training_readiness_2026-06-29.md` | `algorithm/analysis/report/HBTXR_target_training_readiness_2026-06-29.md` | `REFERENCE_ONLY`; current readiness is authoritative. |
| `references/report/FECET/README.md` | proposed `references/report/FECET/README.md` (absent) | `REFERENCE_ONLY`; add only with provenance/license need, otherwise leave absent. |

For these ten paths the validation command is a future file-level comparison,
`git diff --no-index -- <HANDOVER-source> <HBTXR-target>`, followed by the
relevant contract tests. Risk is provenance/claim regression; rollback is to
leave the current target unchanged or remove a newly added reference. Priority
is P2 except license clearance, which is a P0 gate.

## Dependency order

```text
provenance/license clearance
  -> data, metric, checkpoint and hardware interface contracts
  -> golden/unit tests
  -> isolated adapters behind defaults/feature flags
  -> same-split software ablation or hardware simulation/synthesis
  -> explicit active-integration approval
  -> board or release validation
```

No downstream step can waive an upstream gate. In particular, a favorable
accuracy report does not waive license/data-contract checks, and a successful
synthesis estimate does not waive post-route or board validation.

## Execution waves

### Wave 0 — preserve and stop duplication

Close `S-00`, `S-01`, and `H-00` as no-op. Keep C1–C3 material in its documented
archive/evidence/reference scope. Do not treat presence in HBTXR as activation.

### Wave 1 — contracts and regression evidence

Implement the shared data/evaluation/checkpoint contracts, adapt HGTXR software
and hardware golden tests, and define the XR config schema. This wave produces
tests and interfaces, not a board release.

### Wave 2 — isolated behavioral adapters

Prototype `S-04`, `S-05`, `S-07`, `S-08`, and `H-03` behind explicit entrypoints
or feature flags. Preserve current defaults and collect same-condition evidence.

### Wave 3 — experiments and hardware candidate

Recreate XR64–XR68 experiments and select at most one XR ZCU104 cyclic case for
a separately approved simulation/synthesis path. Board execution and active
hardware changes require new authorization.

### Wave 4 — optional future breadth

Evaluate ViT multi-board utilities and ICCAD24 SPINAL only when the product or
research scope explicitly needs them.

## Stop conditions

Stop promotion and return the item to reference-only when any of these occurs:

- upstream or license cannot be established;
- dataset/privacy permission or subject split cannot be verified;
- input, coordinate, metric, checkpoint, register, or numeric contracts remain
  ambiguous;
- the adapter requires replacing current modules wholesale;
- tests pass only with source-specific absolute paths or generated state;
- claimed improvement is not reproduced under the same split/metric/budget;
- hardware evidence stops at estimate when implementation/board proof is claimed.

## Matrix conclusion

The matrix authorizes analysis and a future test-first adaptation backlog only.
It does not authorize active-source integration, external artifact import,
board execution, push, or release. The immediate value lies in contracts,
golden tests, and two narrow adapter families: HGTXR search/tracking behavior
and XR cyclic automation/configuration.
