# HBTXR Follow-up Implementation Plan

## 한국어 실행 요약

이 계획은 HANDOVER 코드를 일괄 복사하지 않고, 현재 HBTXR 인터페이스에
필요한 계약·검증·가속기 자동화만 순차적으로 재구성한다. 모든 단계는
테스트와 롤백을 먼저 정의하며, 의존성 설치·실험 학습·HLS C-sim·합성·
보드 실행·CRLF/LFS 변경·Spec-Kit 초기화·push는 각각 별도 승인을 받는다.
계획 작성 자체는 구현이나 커밋을 승인하지 않는다.

## Active execution filter — code/file integration only

The user narrowed execution on 2026-07-15 to integration work that requires no
dependency installation, reproduction run, experiment, synthesis, board, tool
initialization, remote mutation, or release action.

**Selected sequential tasks:** T-010, T-020, T-100, T-110, and T-210.

**Excluded from this execution:**

- T-000: dependency installation.
- T-120: checkpoint blending requires unavailable PyTorch runtime verification.
- T-200: reproduction/golden evidence generation.
- T-220: XR prototype/dry-run automation remains experiment-adjacent.
- T-300–T-320: behavioral characterization and promotion experiments.
- T-400–T-410: XR64–XR68 reproduction contract/run.
- T-490–T-520: launcher/C-sim/synthesis/board experiment surfaces.
- T-600–T-630: audit/policy normalization, Spec-Kit, and push/release operations.

Selected tasks may use only already available local tools. A task whose required
runtime verification depends on unavailable NumPy, PyTorch, or pytest must stop
at a clearly reported partial/blocked boundary rather than install packages or
infer a pass. Local task commits remain allowed; push remains excluded.

**Execution-specific dependency replacement:** T-010 and T-020 use the current
Python 3.12 standard library and available PyYAML instead of T-000. T-100 is
implemented as a dependency-light immutable contract owner and is verified with
stdlib `unittest`; it may use NumPy types only behind `TYPE_CHECKING`. T-110 is
a pure coordinate/task contract layer verified without importing the existing
Torch transform stack. T-210 validates only the already versioned XR JSON graph
and replaces its T-200 dependency with those current config files; it does not
consume or generate golden vectors and does not invoke T-220.

**Active-slice verification commands:**

```bash
python3 -m unittest discover -s tests/provenance -p 'test_source_registry.py'
python3 -m unittest discover -s tests/provenance -p 'test_artifact_manifest.py'
PYTHONPATH=algorithm/hybrid/src python3 -m unittest discover -s algorithm/hybrid/tests/handover -p 'test_data_contract.py'
PYTHONPATH=algorithm/hybrid/src python3 -m unittest discover -s algorithm/hybrid/tests/handover -p 'test_evaluation_bridge.py'
python3 -m unittest discover -s hardware/tests -p 'test_xr_accel_config_schema.py'
python3 -m compileall -q tools/provenance algorithm/hybrid/src/data algorithm/hybrid/src/evaluation hardware/tools
```

Each task also runs its task-specific JSON/YAML/validator and `git diff --check`
commands. No `.venv`, package manager, GPU, Vitis, Vivado, or board command is
allowed in this slice.

## Goal

Convert the HANDOVER recommendations into reversible, testable HBTXR changes:
establish provenance and artifact governance; define data, evaluation, and
checkpoint contracts; characterize HGTXR search/tracking behavior; add hardware
golden-vector and XR config/dry-run validation; and prepare a blocked-by-default
XR64–XR68 reproduction contract.

## Architecture

Current HBTXR remains the only active owner. HANDOVER is read-only evidence.
Promotion follows this sequence:

```text
environment + provenance
  -> contracts and fail-closed validators
  -> current-behavior characterization
  -> optional feature-flagged adapters
  -> reproducibility dry-run
  -> separately approved training or C-sim
  -> separately approved synthesis
  -> separately approved board execution
  -> separately approved push/release
```

## Tech Stack

- Python 3.12, dataclasses, NumPy, PyTorch, PyYAML, pytest
- JSON/Markdown manifests with SHA-256 provenance
- existing HBTXR `hardware/tools`, HLS scripts, and C++17 smoke fallback
- semantic Git commits with staged allowlists

## Baseline/Authority Refs

- `docs/aegis/baseline/2026-07-15-follow-up-baseline.md`
- `docs/Spec.md`; `docs/track/ADR.md` ADR-001 through ADR-004
- `docs/analysis/HANDOVER-INTEGRATION-MATRIX.md`
- `docs/analysis/HANDOVER-RECOMMENDATIONS.md`
- HGTXR `70c9718117ca`, XR_Accel `5041401aec53`,
  ViT_Accel `90a82ddecd32`, HANDOVER/HBTXR `7d1b0cace624`

## Compatibility Boundary

- Default algorithm and hardware behavior remains unchanged until a dedicated
  adapter task passes feature-off equivalence and receives approval.
- Existing tracker output names and checkpoint payload keys remain accepted.
- Existing XR JSON remains reference evidence and is not rewritten.
- Current HLS/RTL tops and board artifacts remain untouched through dry-run work.
- Existing evaluator/load paths remain until a future migration proves callers.

## Verification

Every task requires a focused command, related regression command, staged
allowlist, and rollback. Before every commit:

```bash
git diff --check
git status --short --branch
git diff --cached --name-status
```

Commands using `.venv` require T-000 dependency-install approval. Vitis,
Vivado, GPU training, and board commands remain separate gates.

## Prompt Brief

- Goal: reuse only evidence-backed HANDOVER behavior through current interfaces.
- Inputs: six analysis artifacts, current source/tests, pinned repositories,
  current tool availability, and the user's request for a concrete plan.
- Assumptions: compact synthetic tests/config/manifests are sufficient for early gates.
- Unknowns: license compatibility, install authority, datasets/checkpoints,
  FPGA tools/board access, storage policy.
- Safety: no bulk copy, secret/data import, generated workspace, network install,
  training, synthesis, board action, LFS initialization, or push without approval.
- Output: concern-sized commits, exact Task Cards, evidence, and stop decisions.
- Acceptance: traceable provenance, fail-closed contracts, current-default
  equivalence, zero prohibited artifacts, bounded residual risks.

## Best-output criteria

| Criterion | Acceptance |
|---|---|
| Completeness | E1–E6, governance, CRLF, and approval gates mapped |
| Evidence | Source SHA/path, current owner, test, and stop state per candidate |
| Executability | Exact files, commands, expected result, dependency, commit |
| Consistency | Current contracts and classification vocabulary reused |
| Safety | License/data/artifact/tool/board/push gates fail closed |
| Maintainability | Minimal owners; adapters conditional; defaults retained |

## Plan Basis

### Facts

- Current HBTXR already owns data contracts, tracker branches/runtime,
  checkpoint load/save, hardware tools, HLS scripts, and reference directories.
- A dedicated evaluation package and HANDOVER-focused test suite do not exist.
- The default Python cannot run planned tests without a local environment.
- XR configs are reference evidence without an active schema owner.
- HGTXR and current tracker paths are related but mostly byte-different.

### Assumptions and unknowns

- An ignored `.venv` is the least-invasive test environment if approved.
- Synthetic fixtures avoid private datasets and generated hardware projects.
- License, GPU/data/checkpoint, Vitis/Vivado, board, and remote storage remain unknown.

## BaselineUsageDraft

- Required: baseline snapshot, Spec, ADR-001–004, six analysis documents.
- Acknowledged and cited: all required refs plus pinned source SHAs.
- Missing: approved license matrix, artifact store policy, runtime toolchain.
- Decision: continue planning; fail closed at affected execution gates.

## Requirement Ready Check

- Sources: user request, Spec, recommendations, integration matrix.
- Scope: E1–E6 plus governance decision gates.
- Acceptance: G0–G10 and task commands below.
- Open blockers: dependency install, license release, training, synthesis, board,
  LFS/external storage, push.
- Decision: ready for plan; execution is not approved by plan creation.

## Change Necessity

- Need: make selected HANDOVER knowledge testable in current HBTXR.
- No-change option: reference-only content cannot validate contracts or behavior.
- Minimum code boundary: current data/checkpoint/tracker owners, one evaluation
  package, HANDOVER tests, provenance validators, XR schema/dry-run tools.
- Decision: code-change after explicit execution approval.

## Existence Check

| Proposed surface | Reuse candidate | Decision |
|---|---|---|
| `data/handover_adapter.py` | `data/contracts.py`, `dataset.py` | Create only if explicit translators do not fit the contract owner. |
| `src/evaluation/` | no canonical package | Add: central task/coordinate/aggregation contract is needed. |
| `checkpoint_blend.py` | `training/checkpoints.py` | Add pure blending owner; reuse payload inspection. |
| `tests/handover/` | `tests/external_pipeline/` | Add focused evidence suite; keep existing regressions. |
| `tools/provenance/` | no validator | Add fail-closed source/artifact validators. |
| `xr_accel_config.py` | raw reference JSON | Add schema/loader before automation. |
| `xr_cyclic_plan.py` | hardware tool patterns | Add dry-run-only planner after schema. |
| Stage strategy modules | current branch hooks | Defer until a failing characterization proves need. |

## Architecture Integrity Lens

- Invariant: HBTXR owns runtime behavior; HANDOVER is never a parallel owner.
- Canonical owners: data contracts, tracker/runtime, checkpoint training package,
  hardware tools/HLS scripts, provenance registry.
- Simplification: contracts and schemas precede adapters and runs.
- Falsifier: no adapter when current behavior already satisfies the contract.
- Retirement: remove/never add adapters on duplication, license block, failed
  equivalence, or no measurable benefit.
- Verdict: contract-first, conditional-adapter plan.

## Plan-Time Complexity Check

| Target | Lines | Pressure | Boundary |
|---|---:|---|---|
| `data/contracts.py` | 48 | low | add contract types; isolate translators if needed |
| `training/checkpoints.py` | 139 | medium | inspection here, blending in pure module |
| `search_branch.py` | 251 | medium/high | characterization first; conditional strategy file |
| `track_branch.py` | 178 | medium | reuse optional hooks; conditional strategy file |
| `runtime/tracker.py` | 79 | low | sequence reset only on proven failure |
| hardware validators | 231–587 | high in largest file | add focused tools/tests; do not extend 587-line validator |

No fallback or adapter logic is added before a failing characterization proves
the gap. New modules must have one owner and a removal trigger.

## Execution Readiness View

- Intent Lock: selective, test-backed adaptation only.
- Scope Fence: HBTXR listed paths; HANDOVER read-only; no external mutation.
- Baseline Lock: branch/HEAD/tools/licenses checked before each wave.
- Approved Behavior: planning only; execution needs user approval.
- Compatibility: current defaults, output ABI, checkpoint load, HLS top, XR refs stable.
- Retirement: conditional adapters removed or never added when unnecessary.
- Execution: sequential waves; implementer → spec reviewer → quality reviewer.
- Drift: stop on owner/source/license/tool/split/board drift; revise plan.
- Evidence: commands, staged manifest, reviews, commit SHA, uncovered risks.
- Advisory boundary: not completion or release authority.

## Execution DAG

```text
T-000 environment
  -> T-010 provenance registry -> T-020 artifact policy
  -> T-100 data -> T-110 evaluation -> T-120 checkpoint
  -> T-200 hardware goldens -> T-210 XR schema -> T-220 XR dry-run
  -> T-300 Stage-1 -> T-310 Stage-2 -> T-320 conditional promotion
  -> T-400 XR64-XR68 static reproduction
  -> [T-410 training: separate approval]
  -> [T-490 active HLS launcher remediation]
  -> [T-500 C-sim] -> [T-510 synthesis] -> [T-520 board]
  -> T-600 CRLF/tooling governance
  -> [T-610 line-ending/LFS policy: separate approval]
  -> [T-620 Spec-Kit scaffold: separate approval]
  -> [T-630 push: separate approval]
```

## Approval gates

| Gate | Permitted work | Explicitly excluded |
|---|---|---|
| A | T-000 dependency manifest | network install until approved |
| B | T-010–T-400 source/tests/docs/dry-run | training, HLS active case, external mutation |
| C | T-410 reproduction run | dataset/checkpoint access and GPU run until approved |
| D | T-490 launcher remediation, then T-500 C-sim | synthesis and board |
| E | T-510 synthesis | board transfer/run |
| F | T-520 board | push/release |
| G | T-610 line-ending/LFS policy | archive mutation until exact allowlist is approved |
| H | T-620 Spec-Kit initialization | repo scaffold mutation until preview is approved |
| I | T-630 push/PR | remote mutation until separate user approval |

## Epic-to-task and decision-gate map

| Epic / decision | Tasks | Start gate | Exit evidence |
|---|---|---|---|
| E1 Contract foundation | T-000, T-100, T-110 | dependency install approval | contract/rejection and existing regression tests |
| E2 HGTXR behavioral regression | T-300, T-310, T-320 | T-010 license status and E1 | characterization, feature-off equivalence, benchmark decision |
| E3 Checkpoint tooling | T-120 | E1 environment/data vocabulary | compatibility tests and hash manifest |
| E4 Hardware golden suite | T-200 | T-010/T-020 | compact vectors, tolerance/hash verification |
| E5 XR cyclic prototype | T-210, T-220, T-490, T-500, T-510, T-520 | config/dry-run approval; launcher/tool/board approvals | schema, external-only launcher, C-sim, synthesis, board evidence by gate |
| E6 Reproducible experiments | T-400, T-410 | static contract then separate data/GPU approval | blocked dry-run or same-condition result manifest |
| License registry | T-010 | planning approval | complete registry; unresolved stays blocked |
| Artifact store/LFS | T-020, T-610 | policy then separate attributes/LFS approval | schema and explicit ADR/attributes decision |
| CRLF archive | T-600, optional T-610 | audit; normalization separately approved | inventory and optional hash-mapped normalization |
| Spec-Kit | T-620 | separate scaffold approval | reviewed scaffold and Spec ownership ADR |
| Push | T-630 | separate push approval | remote ref verification; no implicit release |

## Sequential task plan

### T-000 — Establish the validation environment

**Files:** create `algorithm/requirements-followup.txt`; never commit `.venv/`.

**Why:** the default interpreter lacks pytest, NumPy, and PyTorch. Evidence from
planned Python tests is impossible until a reproducible local environment exists.

**Steps**

1. Add `pytest>=8,<9`, `numpy>=1.26,<3`, `PyYAML>=6,<7`, `torch>=2.2,<3`.
2. Obtain approval before network/dependency installation.
3. After approval:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r algorithm/requirements-followup.txt
.venv/bin/python -c 'import numpy, pytest, torch, yaml; print(numpy.__version__, pytest.__version__, torch.__version__, yaml.__version__)'
.venv/bin/python -m pytest -q algorithm/hybrid/tests/external_pipeline/test_dataset.py
```

**Expected:** imports and baseline dataset test exit 0.

**Commit:** `chore(test): Define follow-up validation dependencies`

**Rollback/stop:** delete `.venv` and revert the requirements commit. Stop on
resolver conflict, unsupported PyTorch/Python, unexpected GPU dependency, or
baseline test failure before new source edits.

### T-010 — Add the source and license registry

**Files**

- `docs/provenance/handover-source-registry.json`
- `docs/provenance/HANDOVER-LICENSE-MATRIX.md`
- `docs/provenance/HANDOVER-NOTICE-PLAN.md`
- `tools/provenance/validate_source_registry.py`
- `tests/provenance/test_source_registry.py`

**Steps**

1. Require candidate ID/class, repo/branch/SHA/path/blob SHA-256, license path
   and scope, notice, code/model/data rights, destination, adaptation summary,
   owner/reviewer/date, validation, ambiguity, and promotion decision.
2. Populate ERVT → TENNs-Eye → TDTracker → BRAT → FECET → Retina →
   HG-PIPE/ICCAD24 → HGTXR → XR/ViT.
3. Reject missing fields, unknown class, hash mismatch, credential-like URI,
   and `ACTIVE_CANDIDATE` with unresolved license.
4. Verify:

```bash
python3 -m json.tool docs/provenance/handover-source-registry.json >/dev/null
python3 tools/provenance/validate_source_registry.py docs/provenance/handover-source-registry.json
python3 -m unittest discover -s tests/provenance -p 'test_source_registry.py'
git diff --check
```

**Expected:** every candidate is traceable; uncertain items remain
`LICENSE_BLOCKED`; all commands exit 0.

**Commit:** `docs(provenance): Add HANDOVER source registry`

**Rollback/stop:** revert the commit. Stop on unknown upstream, license-scope
conflict, code/model/data term conflict, or blob mismatch. Only an authorized
reviewer may clear `LICENSE_BLOCKED`.

### T-020 — Define external artifact policy

**Files**

- `docs/Artifact-Policy.md`
- `docs/provenance/artifact-manifest.schema.json`
- `docs/provenance/examples/external-artifact-manifest.example.json`
- `tools/provenance/validate_artifact_manifest.py`
- `tests/provenance/test_artifact_manifest.py`
- `.gitignore` only if a tested gap exists
- `.gitattributes` only after a separate LFS/line-ending decision

**Steps**

1. Require artifact type/status, SHA-256, byte count, source/config/split hashes,
   architecture/tool/board versions, non-secret URI, access/privacy/license,
   generation command/status, validation owner/date, and replacement hash.
2. Default checkpoints, HDF5/data, raw predictions, runs/logs, `.Xil`, builds,
   bit/HWH/XSA/DCP/IP to external storage.
3. Reject missing hashes, sensitive URI/subject IDs, or unsupported validation claims.
4. Verify:

```bash
python3 -m json.tool docs/provenance/artifact-manifest.schema.json >/dev/null
python3 tools/provenance/validate_artifact_manifest.py docs/provenance/examples/external-artifact-manifest.example.json
python3 -m unittest discover -s tests/provenance -p 'test_artifact_manifest.py'
git diff --cached --name-only --diff-filter=ACMR | rg '(^|/)(runs|\.Xil|build|__pycache__)(/|$)|\.(pt|pth|ckpt|h5|hdf5|bit|hwh|xsa|dcp)$' && exit 1 || true
```

**Expected:** schema/example/tests pass; prohibited staged scan is empty.

**Commit:** `docs(artifacts): Define external artifact manifest policy`

**Rollback/stop:** revert files. LFS tracking, uploads, moves, and deletions need
separate approval.

### T-100 — Add the canonical sample contract

**Files:** modify `algorithm/hybrid/src/data/contracts.py`; create
`algorithm/hybrid/src/data/handover_adapter.py` only if translation does not fit
the contract owner; create `algorithm/hybrid/tests/handover/test_data_contract.py`;
do not modify `dataset.py` in this task.

**Steps**

1. Define an immutable record for events, optional frame, target, subject, eye,
   sequence, timestamp window, and source metadata.
2. Add explicit translators for `2x64x64`, `50x2x64x64`, `30x3x64x64`, and
   `100x2x64x64`, including rank, axes, dtype, polarity, and temporal meaning.
3. Reject silent reshape/truncation/channel coercion, ambiguous polarity,
   missing split provenance, or lost sequence/eye identity.
4. Verify:

```bash
.venv/bin/python -m pytest -q algorithm/hybrid/tests/handover/test_data_contract.py
.venv/bin/python -m pytest -q algorithm/hybrid/tests/external_pipeline/test_dataset.py
```

**Expected:** accepted fixtures normalize deterministically; ambiguous fixtures
fail with named contract errors; existing test passes.

**Commit:** `feat(data): Add HANDOVER sample contract`

**Rollback/retirement:** revert contract/adapter/tests. Do not connect the
dataset until a later manifest-backed test passes.

### T-110 — Add task- and coordinate-aware evaluation

**Files**

- `algorithm/hybrid/src/evaluation/__init__.py`
- `algorithm/hybrid/src/evaluation/contracts.py`
- `algorithm/hybrid/src/evaluation/handover_bridge.py`
- `algorithm/hybrid/tests/handover/test_evaluation_bridge.py`
- reuse `data/transform.py` and `data/utils.py`

**Steps**

1. Define separate center, bbox, and ellipse records plus sensor, ROI, and
   post-transform domains.
2. Implement explicit per-axis conversion among `80x60`, `640x480`, and `64x64`.
3. Define invalid-sample, subject weighting, macro/micro, and P1/P5/P10 policies.
4. Reject task/domain mismatch instead of comparing silently.
5. Verify:

```bash
.venv/bin/python -m pytest -q algorithm/hybrid/tests/handover/test_evaluation_bridge.py
.venv/bin/python -m pytest -q algorithm/hybrid/tests/external_pipeline/test_dataset.py
```

**Expected:** corner/center goldens and P1/P5/P10 pass; ambiguous/cross-task
inputs fail closed; existing regression passes.

**Commit:** `feat(eval): Add task-aware HANDOVER evaluation bridge`

**Rollback/retirement:** remove the new package/tests. Existing evaluators stay
canonical until a future caller migration proves equivalence.

### T-120 — Add checkpoint compatibility and blending

**Files:** modify `algorithm/hybrid/src/training/checkpoints.py`; create
`algorithm/hybrid/src/training/checkpoint_blend.py`,
`algorithm/hybrid/tools/checkpoint_blend.py`, and
`algorithm/hybrid/tests/handover/test_checkpoint_blend.py`.

**Steps**

1. Inspect model keys, shapes, dtypes, parameters, buffers, architecture/config,
   and payload state classes before output.
2. Average/interpolate compatible model state only.
3. Record source/output hashes, weights/alpha, and included/excluded state policy.
4. For blending, reject shape slicing and silent optimizer/scheduler/scaler loss.
5. Verify:

```bash
.venv/bin/python -m pytest -q algorithm/hybrid/tests/handover/test_checkpoint_blend.py
.venv/bin/python -m pytest -q algorithm/hybrid/tests/external_pipeline/test_pretrained_loader.py
.venv/bin/python -m pytest -q algorithm/hybrid/tests/external_pipeline/test_runtime_e2e.py
```

**Expected:** endpoints/equal average match goldens; mismatch writes no output;
existing load/runtime tests pass.

**Commit:** `feat(checkpoint): Add compatibility-aware checkpoint blending`

**Rollback/retirement:** remove tool/pure module and revert inspection additions;
input checkpoints and existing load path remain unchanged.

### T-200 — Establish HGTXR hardware golden contracts

**Prerequisite:** T-010 permits adaptation or records an independent
current-HBTXR recreation path; otherwise remain `LICENSE_BLOCKED`.

**Files:** create `hardware/refs/handover/hgtxr/manifest.json`, compact synthetic
vectors in that directory, and
`hardware/tests/handover/test_hgtxr_golden_vectors.py`. Reuse current HBTXR
export/validation tools without overwriting them.

**Steps**

1. Cover LayerNorm, GELU, Softmax, quantized matmul, QKV/attention, S2 block.
2. Record seed, dtype, scale, rounding, saturation, tolerance, source revision,
   hashes, and commands.
3. Verify:

```bash
.venv/bin/python -m pytest -q hardware/tests/handover/test_hgtxr_golden_vectors.py
.venv/bin/python hardware/tools/validate_hgpipe_lut_math.py
```

**Expected:** shape/dtype/hash pass; samples outside tolerance = 0.

**Commit:** `test(hardware): Add HGTXR golden-vector contract`

**Rollback/stop:** remove vectors/tests/manifest. Stop on arbitrary tolerance
expansion, quantization ambiguity, license block, or external artifact need.

### T-210 — Add XR configuration schema

**Files:** create `hardware/configs/xr_accel/schema.json`,
`hardware/tools/xr_accel_config.py`, and
`hardware/tests/test_xr_accel_config_schema.py`; do not modify the existing 12
reference JSON files.

**Steps**

1. Require schema version, board/part, clock, AXI width, memory topology, model,
   design, experiment links, and source provenance.
2. Reject unknown fields, implicit defaults, absolute paths/path escape,
   unsupported board/part, and unresolved links.
3. Translate references to an in-memory normalized manifest.
4. Verify:

```bash
.venv/bin/python -m pytest -q hardware/tests/test_xr_accel_config_schema.py
python3 -m json.tool hardware/configs/xr_accel/designs/hbtxr_cyclic_zcu104.json >/dev/null
python3 -m json.tool hardware/configs/xr_accel/targets/zcu104.json >/dev/null
git diff -- hardware/configs/xr_accel/designs hardware/configs/xr_accel/experiments hardware/configs/xr_accel/models hardware/configs/xr_accel/targets
```

**Expected:** all references translate; invalid fixtures fail; final diff is empty.

**Commit:** `feat(hardware): Validate XR accelerator configurations`

**Rollback:** remove schema/loader/tests; reference evidence stays unchanged.

### T-220 — Add XR cyclic dry-run planner

**Files:** create `hardware/tools/xr_cyclic_plan.py`,
`hardware/tests/test_xr_cyclic_plan.py`, and fixtures
`hardware/tests/fixtures/xr_cyclic/{csynth_pass.xml,timing_pass.rpt,timing_fail.rpt}`;
reuse `hardware/tools/extract_resource_metrics.py`.

**Steps**

1. Consume only T-210 normalized configs.
2. Render argv arrays, never shell strings; derive run ID from canonical config hash.
3. Keep execution impossible; parse only synthetic reports; reject stale run IDs,
   path escape, and shell metacharacters.
4. Verify:

```bash
.venv/bin/python -m pytest -q hardware/tests/test_xr_cyclic_plan.py
.venv/bin/python hardware/tools/xr_cyclic_plan.py --config hardware/configs/xr_accel/experiments/zcu104_hbtxr_option_a_cyclic.json --dry-run --output /tmp/hbtxr-xr-plan.json
python3 -m json.tool /tmp/hbtxr-xr-plan.json >/dev/null
```

**Expected:** deterministic run ID/argv; no Vivado/Vitis process; unsafe/stale
fixtures rejected.

**Commit:** `feat(hardware): Add XR cyclic dry-run planner`

**Rollback:** remove planner/tests/fixtures and `/tmp` output; no active hardware
source or build artifact exists.

### T-300 — Characterize Stage-1 recovery and ensemble

**Files initially:** create
`algorithm/hybrid/tests/handover/test_search_recovery.py`; inspect but do not
initially modify `models/tracker/search_branch.py` or
`training/ensemble_teacher.py`.

If an adapter is proposed, also create
`algorithm/hybrid/tools/benchmark_handover_strategies.py` for a deterministic
synthetic benchmark; the tool is not needed for tests-only closure.

**Steps**

1. Test candidate ranking, ties, confidence, recovery/no-recovery, empty and
   out-of-frame candidates, determinism, output ABI, and bounded runtime.
2. Verify:

```bash
.venv/bin/python -m pytest -q algorithm/hybrid/tests/handover/test_search_recovery.py
.venv/bin/python -m pytest -q algorithm/hybrid/tests/external_pipeline/test_model.py
```

3. If current behavior passes, commit tests only and close S-04.
4. Only if a gap is proven and T-010 clears adaptation: create
   `models/tracker/search_strategy.py`; modify tracker config/wiring; keep the
   feature flag false; then run:

```bash
.venv/bin/python -m pytest -q algorithm/hybrid/tests/external_pipeline/test_runtime_e2e.py
.venv/bin/python algorithm/hybrid/tools/benchmark_handover_strategies.py --strategy search --warmup 20 --repeat 200 --json-out /tmp/hbtxr-stage1-benchmark.json
```

**Expected:** current-default equivalence and deterministic behavior; any
conditional adapter passes all tests. Promotion additionally requires median
overhead ≤ 5% and p95 overhead ≤ 10% versus the current path on the same 20
warmups/200 repeats. The user or delegated architecture reviewer must approve
these thresholds before the adapter commit; otherwise characterization closes
tests-only and the adapter remains blocked.

**Commit:** `test(tracker): Characterize search recovery behavior`; any adapter
is a separate `feat(tracker)` commit.

**Rollback/retirement:** tests may remain. Remove the adapter/flag on equivalence,
latency, license, or maintainability failure.

### T-310 — Characterize Stage-2 state and sequence isolation

**Files initially:** create
`algorithm/hybrid/tests/handover/test_track_state_adapter.py`; inspect first
`track_branch.py`, `runtime/tracker.py`, and `models/hybrid_tracker.py`.

**Steps**

1. Test bootstrap → track → lost → search, reset, sequence A/B isolation,
   closed-eye cache invalidation, and all-options-off equivalence.
2. Verify:

```bash
.venv/bin/python -m pytest -q algorithm/hybrid/tests/handover/test_track_state_adapter.py
.venv/bin/python -m pytest -q algorithm/hybrid/tests/external_pipeline/test_model.py
.venv/bin/python -m pytest -q algorithm/hybrid/tests/external_pipeline/test_runtime_e2e.py
```

3. Only on a proven gap: add optional sequence identity/reset to
   `runtime/tracker.py`; add `track_strategy.py` only when current hooks cannot
   express cleared behavior. `None`/disabled must be identical to current output.

**Commit:** `test(tracker): Characterize tracking state isolation`; conditional
adapter is separate.

**Rollback/retirement:** remove optional strategy/reset wiring when unnecessary
or non-equivalent; preserve valid state tests.

### T-320 — Combined regression and promotion decision

**Files:** create `docs/validation/HANDOVER-SOFTWARE-REGRESSION.md`; no maintained
source may be changed in this task. Raw command output may use `/tmp` only.

**Approval state:** included in Gate B for validation and documentation. It does
not authorize an adapter that remained blocked in T-300/T-310.

**Commands**

```bash
.venv/bin/python -m pytest -q algorithm/hybrid/tests/handover
.venv/bin/python -m pytest -q algorithm/hybrid/tests/external_pipeline/test_dataset.py
.venv/bin/python -m pytest -q algorithm/hybrid/tests/external_pipeline/test_model.py
.venv/bin/python -m pytest -q algorithm/hybrid/tests/external_pipeline/test_train_pipeline.py
.venv/bin/python -m pytest -q algorithm/hybrid/tests/external_pipeline/test_runtime_e2e.py
```

**Expected:** all exit 0; feature-off equivalence; no source-specific absolute
path/artifact dependency; no approved latency threshold exceeded.

**Commit:** `docs(validation): Record HANDOVER software regression decision`

**Rollback/stop:** revert only the evidence commit. Return failures to the owner
task; never relax a threshold to promote a candidate. If no adapter was approved,
record `tests-only` rather than treating missing benchmark evidence as pass.

### T-400 — Add blocked-by-default XR64–XR68 reproduction contract

**Files**

- `algorithm/hybrid/hardware_reference/configs/experiments/xr64_xr68_reproduction.yaml`
- `algorithm/hybrid/hardware_reference/docs/resources/xr64_xr68_reproduction_manifest.json`
- `algorithm/hybrid/hardware_reference/docs/resources/xr64_xr68_reproduction_report.md`
- `algorithm/hybrid/hardware_reference/scripts/external/validate_xr64_xr68_reproduction.py`
- `algorithm/hybrid/hardware_reference/tests/test_xr64_xr68_reproduction_contract.py`

**Steps**

1. Record split counts `5929/844/2238`, seeds, evaluator, center/P5/P10 gates,
   and source report/config/checkpoint/data references.
2. Separate `historical`, `available`, `reproduced`; preserve XR65/XR68
   no-promotion conclusions.
3. Prohibit test-derived target overrides; require train/validation provenance.
4. Record missing 8 evaluation rows, 6 overrides, checkpoint/data URI as blockers.
5. Dry-run returns `BLOCKED` without an executable command while inputs or
   approvals are missing.
6. Verify:

```bash
python3 -m json.tool algorithm/hybrid/hardware_reference/docs/resources/xr64_xr68_reproduction_manifest.json >/dev/null
.venv/bin/python -m pytest -q algorithm/hybrid/hardware_reference/tests/test_xr64_xr68_reproduction_contract.py
.venv/bin/python algorithm/hybrid/hardware_reference/scripts/external/validate_xr64_xr68_reproduction.py --manifest algorithm/hybrid/hardware_reference/docs/resources/xr64_xr68_reproduction_manifest.json --dry-run
```

**Expected:** parse/tests pass; validator reports `BLOCKED` with named inputs;
test override count is 0; no GPU or download process starts.

**Commit:** `docs(experiment): Add XR64-XR68 reproduction contract`

**Rollback/stop:** remove contract files. T-410 training, data/checkpoint access,
and result upload require separate approval.

### T-410 — Execute XR64–XR68 reproduction after separate approval

**Approval:** not included in plan creation or Gate B. It requires explicit
authorization for dataset/checkpoint access, dependency/GPU use, run duration,
and the configured external artifact root.

**Files**

- Create after approval:
  `algorithm/hybrid/hardware_reference/scripts/external/run_xr64_xr68_reproduction.py`
- Create compact result documents only:
  `algorithm/hybrid/hardware_reference/docs/resources/xr64_xr68_reproduction_result.json`
  and `.md`
- Do not track checkpoints, datasets, predictions, runs, logs, or raw row tables.

**Steps**

1. Require `HBTXR_ARTIFACT_ROOT` and validate every input URI/hash against T-020.
2. Re-run T-400 validation; refuse execution while any blocker remains.
3. Execute current training/evaluation owners, never HANDOVER entrypoints:

```bash
test -n "${HBTXR_ARTIFACT_ROOT:-}"
.venv/bin/python algorithm/hybrid/hardware_reference/scripts/external/validate_xr64_xr68_reproduction.py --manifest algorithm/hybrid/hardware_reference/docs/resources/xr64_xr68_reproduction_manifest.json --require-ready
.venv/bin/python algorithm/hybrid/hardware_reference/scripts/external/run_xr64_xr68_reproduction.py --manifest algorithm/hybrid/hardware_reference/docs/resources/xr64_xr68_reproduction_manifest.json --artifact-root "$HBTXR_ARTIFACT_ROOT" --execute
```

4. Validate the compact result against split `5929/844/2238`, seeds, evaluator,
   input/output hashes, center/P5/P10, failure buckets, and test override count 0.
5. Preserve a `no-promotion` result whenever G10 is not satisfied.

**Expected:** either fail closed before GPU execution with named blockers, or
produce hash-linked external artifacts plus compact result JSON/Markdown; no
prohibited artifact appears in Git.

**Commit:** `docs(experiment): Record XR64-XR68 reproduction evidence`

**Rollback/stop:** stop on split/evaluator/hash/privacy mismatch or runtime
failure. Revert compact docs only. Do not delete or upload external outputs
without separate authority; retain or quarantine them per artifact policy.

### T-490 — Repair active HLS launchers and enforce external projects

**Approval:** this active hardware-script change is the first part of Gate D and
must pass before T-500. It does not authorize Vitis execution or change HLS
kernel behavior.

**Baseline defect:** the active shell/Tcl launch files use CRLF, so Bash sources
`_common.sh\r` and fails. The current Tcl also opens
`hardware/generated/hgtxr_hls`, which violates T-020 for future generated work.

**Files**

- Modify `hardware/scripts/_common.sh`.
- Modify `hardware/scripts/run_hls_csim.sh`.
- Modify `hardware/scripts/run_hls_csynth.sh`.
- Modify `hardware/vivado/scripts/create_hls_project.tcl`.
- Modify `hardware/vivado/scripts/run_csim.tcl`.
- Modify `hardware/vivado/scripts/run_csynth.tcl`.
- Create `hardware/tests/test_hls_launcher_contract.py`.
- Create `hardware/docs/validation/xr_accel/HLS-LAUNCHER-BASELINE.md`.

**Steps**

1. Record SHA-256 and CRLF counts for the six launch files; normalize only these
   active files to LF and prove the normalization is otherwise byte-equivalent.
2. Add a canonical `--artifact-root` shell option and Tcl `-project_root`
   argument. Resolve the path, reject empty/root/repository-contained paths,
   and pass an argv element rather than constructing an eval string.
3. Preserve the current case and local `g++` no-argument fallback. When Vitis is
   detected, fail closed unless an external artifact root is supplied. Synthesis
   always requires an external root.
4. Replace the hard-coded `hardware/generated/hgtxr_hls` project with the
   validated external project root.
5. Verify without invoking Vitis:

```bash
bash -n hardware/scripts/_common.sh hardware/scripts/run_hls_csim.sh hardware/scripts/run_hls_csynth.sh
.venv/bin/python -m pytest -q hardware/tests/test_hls_launcher_contract.py
bash hardware/scripts/run_hls_csim.sh
test ! -e hardware/generated/hgtxr_hls
git diff --check -- hardware/scripts hardware/vivado/scripts
```

**Expected:** shell syntax and launcher tests pass; the current no-argument local
smoke exits 0; missing/inside-repo artifact roots are rejected; generated HLS
projects can resolve only below an explicitly supplied external root; the active
six-file change is LF-only except for the reviewed argument/path safety logic.

**Commit:** `fix(hardware): Make HLS launchers portable and external-only`

**Rollback/stop:** stop on any kernel/source list/top/part/clock change, local
smoke regression, unallowlisted normalization, or inside-repo output. Revert the
dedicated launcher commit; do not run the restored generator until a new safe
launcher decision is approved.

### T-500 — Add an isolated XR cyclic C-simulation case

**Approval:** separate implementation approval is required. This task does not
authorize synthesis, board access, replacement of an existing HLS top, or storage
of generated projects in Git.

**Files**

- Create `hardware/src/hls/xr_cyclic/xr_cyclic.hpp`.
- Create `hardware/src/hls/xr_cyclic/top.cpp`.
- Create `hardware/src/hls/xr_cyclic/attention.cpp`.
- Create `hardware/src/hls/xr_cyclic/mlp.cpp`.
- Create `hardware/src/hls/xr_cyclic/heads.cpp`.
- Create `hardware/tests/hls/test_xr_cyclic.cpp`.
- Modify `hardware/scripts/run_hls_csim.sh`.
- Create `hardware/vivado/scripts/run_xr_cyclic_csim.tcl`.

**Steps**

1. Implement the XR case under its own top name and directory; do not modify the
   current default HLS implementation.
2. Add `--case xr_cyclic` routing while preserving no-argument behavior.
3. Compare every C-sim output with T-200 compact golden vectors.
4. Require the T-490 external-root contract and run the existing and new routes:

```bash
test -n "${HBTXR_ARTIFACT_ROOT:-}"
bash hardware/scripts/run_hls_csim.sh --case current --artifact-root "$HBTXR_ARTIFACT_ROOT/xr_cyclic/baseline-csim"
bash hardware/scripts/run_hls_csim.sh --case xr_cyclic --artifact-root "$HBTXR_ARTIFACT_ROOT/xr_cyclic/csim"
test ! -e hardware/generated/hgtxr_hls
test ! -e hardware/generated/xr_cyclic
```

**Expected:** the existing current-case regression remains unchanged; the new
case produces zero golden mismatches and no in-repo project. If `vitis_hls` is unavailable, the script
must either use its documented local `g++` route or report the tool gate as
unavailable—never infer an HLS-tool pass from a different compiler.

**Commit:** `feat(hardware): Add isolated XR cyclic C-sim case`

**Rollback/stop:** stop on a golden mismatch, interface ambiguity, or existing
route regression. Revert only the isolated case, routing branch, test, and TCL;
preserve the current default HLS top and externally stored evidence.

### T-510 — Add the XR cyclic synthesis route and resource gate

**Approval:** separate approval is required after T-490 and T-500 pass. The approved
tool version, FPGA part, clock target, resource ceilings, and artifact root must
be recorded before running synthesis.

**Files**

- Modify `hardware/scripts/run_hls_csynth.sh`.
- Create `hardware/vivado/scripts/run_xr_cyclic_csynth.tcl`.
- Create `hardware/configs/xr_accel/xr_cyclic_synthesis_limits.json`.
- Create `hardware/tools/extract_xr_cyclic_hls_metrics.py`.
- Create `hardware/tools/validate_xr_cyclic_synthesis.py`.
- Create `hardware/tests/test_xr_cyclic_synthesis_route.py`.
- Create `hardware/docs/validation/xr_accel/XR-CYCLIC-SYNTHESIS-GATE.md`.

**Steps**

1. Add an explicit `--case xr_cyclic` synthesis route without changing the
   current no-argument route.
2. Pin and validate tool, part, clock, solution name, top name, and an external
   output directory governed by T-490; reject missing or repository-contained roots.
3. Predeclare ZCU104 protective ceilings in the limits JSON: LUT `<=207360`,
   FF `<=414720`, DSP `<=1555`, BRAM_18K `<=561` (90% device ceilings), HLS
   clock period `<=5.0 ns`, and WNS `>=0.0 ns` whenever a routed WNS metric is
   present. Require LUT, FF, DSP, BRAM_18K, clock period, and latency rows.
4. Implement a dedicated XML extractor for the pinned
   `xr_cyclic_solution/syn/report/xr_cyclic_top_csynth.xml`. Read standard Vitis
   HLS `AreaEstimates/Resources` LUT, FF, DSP, and BRAM_18K elements;
   `PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod`; and
   `PerformanceEstimates/SummaryOfOverallLatency/Worst-caseLatency`. Reject a
   missing/ambiguous tag, non-numeric value, unexpected top/solution, and XML
   path outside the approved project root. Unit-test representative pass,
   missing-tag, malformed, and path-escape fixtures.
5. Implement a fail-closed validator that rejects missing/conflicting/non-numeric
   required rows, extraction error/status rows, ceiling violations, negative
   WNS, and source paths outside the approved external project. Identical
   duplicate observations may be collapsed only with all source refs retained.
6. Verify route construction, run the separately approved synthesis, extract
   compact external metrics, and evaluate the declared limits:

```bash
test -n "${HBTXR_ARTIFACT_ROOT:-}"
.venv/bin/python -m pytest -q hardware/tests/test_xr_cyclic_synthesis_route.py
bash hardware/scripts/run_hls_csynth.sh --case xr_cyclic --artifact-root "$HBTXR_ARTIFACT_ROOT/xr_cyclic/csynth/project"
.venv/bin/python hardware/tools/extract_xr_cyclic_hls_metrics.py --csynth-xml "$HBTXR_ARTIFACT_ROOT/xr_cyclic/csynth/project/xr_cyclic_solution/syn/report/xr_cyclic_top_csynth.xml" --project-root "$HBTXR_ARTIFACT_ROOT/xr_cyclic/csynth/project" --json-out "$HBTXR_ARTIFACT_ROOT/xr_cyclic/csynth/metrics.json"
.venv/bin/python hardware/tools/validate_xr_cyclic_synthesis.py --metrics "$HBTXR_ARTIFACT_ROOT/xr_cyclic/csynth/metrics.json" --limits hardware/configs/xr_accel/xr_cyclic_synthesis_limits.json --project-root "$HBTXR_ARTIFACT_ROOT/xr_cyclic/csynth/project" --json-out "$HBTXR_ARTIFACT_ROOT/xr_cyclic/csynth/validation.json"
test ! -e hardware/generated/hgtxr_hls
test ! -e hardware/generated/xr_cyclic
git status --short -- hardware/generated
```

**Expected:** route/XML-extractor tests pass; the current case semantics are
unchanged; the dedicated extractor emits all required rows from the pinned
Vitis XML; validator exits 0 only when timing/resources
meet every predeclared limit. Missing metrics or any breach exits nonzero and is
recorded as failed. The final status command is empty, no generated project
appears under Git, and only the explicitly allowlisted T-510 launcher/TCL,
limits, extractor, validator, test, and compact gate document are tracked.
Generated projects, reports, metrics, and validation JSON remain external.

**Commit:** `feat(hardware): Add XR cyclic synthesis route`

**Rollback/stop:** stop on tool/part/clock drift, missing metrics, timing failure,
ceiling breach, or default-route regression. Revert the route, TCL, limits,
extractor, validator, test, and compact document.
Do not delete external generated outputs without separate authority.

### T-520 — Validate the XR cyclic image on the approved board

**Approval:** separate physical-board approval is required after T-510 and
post-route timing/utilization/power review. Approval must name the board, image,
rollback image, artifact root, operator, and allowed run window.

**Files**

- Create `hardware/tests/pynq/test_xr_cyclic_board_manifest.py`.
- Create `hardware/docs/validation/xr_accel/XR-CYCLIC-BOARD-SMOKE.md`.
- Reuse `hardware/pynq/hgtxr/run_e2e_axis_dma_smoke.py`; create no new runner
  unless a documented interface gap is approved first.

**Steps**

1. Verify bitstream, HWH, source, config, golden input/output, and rollback-image
   hashes before programming the board.
2. Validate the manifest without touching the board.
3. Execute ten measured smoke iterations after one warm-up:

```bash
test -n "${HBTXR_ARTIFACT_ROOT:-}"
.venv/bin/python -m pytest -q hardware/tests/pynq/test_xr_cyclic_board_manifest.py
.venv/bin/python hardware/pynq/hgtxr/run_e2e_axis_dma_smoke.py --bitfile "$HBTXR_ARTIFACT_ROOT/xr_cyclic/xr_cyclic.bit" --hwhfile "$HBTXR_ARTIFACT_ROOT/xr_cyclic/xr_cyclic.hwh" --mode-profile search --weights-mode golden --warmup 1 --repeat 10 --json-out "$HBTXR_ARTIFACT_ROOT/xr_cyclic/board-smoke.json"
```

**Expected:** hashes are checked before load; all ten iterations complete; the
runtime mode and outputs match the approved golden contract; raw logs and images
remain external; the compact Markdown record contains hashes and verdict only.

**Commit:** `docs(hardware): Record XR cyclic board smoke evidence`

**Rollback/stop:** stop on any hash, programming, runtime, or output mismatch.
Restore the approved rollback image, revert only docs/tests, and retain or
quarantine external outputs according to T-020—never delete them automatically.

### T-600 — Audit CRLF archives and record governance decisions

**Files:** create `docs/track/CRLF-ARCHIVE-AUDIT.md`; append decision records to
`docs/track/ADR.md`. This task does not own `.gitattributes` or archive content.

**Steps**

1. Audit without normalizing provenance:

```bash
git grep -Il $'\r' -- algorithm/archive/imports/legacy_hybrid
find algorithm/archive/imports/legacy_hybrid/scripts -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
git diff --check
```

2. Record exact failures, active/reference status, original hashes, and impact.
   Propose but do not apply normalization, LFS, or storage-policy changes.
3. Collect read-only decision evidence:

```bash
specify --help >/dev/null
git lfs version
git status --short --branch
git log --oneline --decorate -10
git remote -v
```

4. Record Spec-Kit, external store/LFS, and push decisions. Do not run
   `specify init`, `git lfs install/track`, upload, or push without approval.

**Expected:** audit distinguishes active from provenance files; ADRs record an
owner and decision/deferral; archive contents and attributes remain unchanged.

**Commit:** `docs(repo): Record archive and tooling governance decisions`

**Rollback:** revert the audit/ADR commit only.

### T-610 — Apply an approved line-ending and large-file policy

**Approval:** this task is conditional on an explicit user decision after T-600.
The decision must identify archive paths, normalization scope, hash mapping, and
whether Git LFS is accepted. No bulk archive rewrite is implied.

**Files**

- Modify `.gitattributes` only for approved path patterns.
- Append the adopted or rejected policy and hash mapping to `docs/track/ADR.md`.
- Modify no archive payload unless its exact allowlist is separately approved.

**Steps**

1. Capture before-hashes and evaluate proposed attributes against named paths.
2. If LFS is approved, add only explicit patterns; otherwise record rejection.
3. If line-ending normalization is approved, stage only the allowlisted files
   in a dedicated commit and verify semantic equality outside line endings.
4. Run:

```bash
git ls-files -z -- algorithm/archive/imports/legacy_hybrid | xargs -0 -r git check-attr -a --
git diff --check
find algorithm/archive/imports/legacy_hybrid/scripts -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
git lfs version
```

If and only if LFS was approved, also run the explicit-pattern form of
`git lfs track <approved-pattern>` and review `.gitattributes` before staging.

**Expected:** only approved paths/patterns change; before/after hashes are
mapped; active source behavior is unchanged; LFS is neither installed nor
enabled implicitly.

**Commit:** `chore(repo): Define approved line-ending and LFS policy`

**Rollback/stop:** stop on unallowlisted churn or semantic change. Revert the
dedicated commit; if LFS tracking was used, untrack only the approved pattern.
Do not rewrite history or automatically remove existing artifacts.

### T-620 — Initialize Spec-Kit after scaffold review

**Approval:** separate approval is required because initialization creates
project files. Existing HBTXR authority documents remain authoritative unless a
reviewed mapping explicitly replaces them.

**Files**

- Preview outside the repo first.
- After approval, create only the reviewed `.specify/**`, `specs/**`, and Codex
  integration files emitted by the accepted scaffold.
- Append the authority/mapping decision to `docs/track/ADR.md`.

**Steps**

1. Generate and inspect an offline preview:

```bash
preview_dir=$(mktemp -d)
specify init "$preview_dir/spec-kit" --integration codex --integration-options="--skills" --script sh
find "$preview_dir/spec-kit" -maxdepth 4 -type f -print | sort
```

2. Record collisions and the mapping to `docs/Master-Plan.md`,
   `docs/Sub-Plan.md`, and `docs/Spec.md`.
3. Only after approval, initialize the reviewed scaffold:

```bash
specify init --here --integration codex --integration-options="--skills" --script sh --force
git status --short
```

**Expected:** the preview is reviewed before repo mutation; initialization does
not silently overwrite authority documents; generated files form one isolated,
reviewable commit.

**Commit:** `chore(spec): Initialize reviewed Spec-Kit scaffold`

**Rollback/stop:** stop on an unreviewed overwrite or authority collision.
Revert the isolated scaffold commit and preserve the pre-init plan/spec files.

### T-630 — Push the reviewed branch

**Approval:** explicit user approval is required after all selected task commits
and local verification. This task performs no file edit or commit.

**Steps**

```bash
git status --short --branch
git fetch origin
git rev-list --left-right --count origin/refactor/hbtxr-structure...HEAD
git log --oneline --decorate origin/refactor/hbtxr-structure..HEAD
git push -u origin refactor/hbtxr-structure
local_sha=$(git rev-parse HEAD)
remote_sha=$(git ls-remote --heads origin refs/heads/refactor/hbtxr-structure | awk '{print $1}')
test "$remote_sha" = "$local_sha"
```

**Expected:** the worktree is clean before push, the ahead/behind state is
reviewed, no force option is used, and the remote branch resolves to local HEAD.

**Commit:** none.

**Rollback/stop:** stop on unexpected divergence, failed verification, or remote
policy failure. A completed push is not rewritten or force-pushed without a new
explicit instruction; report the remote state instead.

## Task Cards

```yaml
task_card:
  task_id: T-000
  sub_agent: codex-native
  role: implementer
  objective: define and after approval create the local validation environment
  file_ownership: [algorithm/requirements-followup.txt]
  assigned_skill: [agyb-essentials:lint-and-validate]
  inputs: [baseline tool report, algorithm imports]
  outputs: [dependency manifest, ignored environment evidence]
  validation: [import smoke, existing dataset test]
  dependencies: []
```

```yaml
task_card:
  task_id: T-010
  sub_agent: codex-native
  role: provenance implementer
  objective: implement a fail-closed source and license registry
  file_ownership: [docs/provenance/handover-source-registry.json, docs/provenance/HANDOVER-LICENSE-MATRIX.md, docs/provenance/HANDOVER-NOTICE-PLAN.md, tools/provenance/validate_source_registry.py, tests/provenance/test_source_registry.py]
  assigned_skill: [artifact-provenance-manager]
  inputs: [HANDOVER source pins, license audit, integration matrix]
  outputs: [JSON source registry, license matrix, notice plan, validator, tests]
  validation: [JSON parse, registry validation, unittest, LICENSE_BLOCKED negative cases]
  dependencies: [T-000]
```

```yaml
task_card:
  task_id: T-020
  sub_agent: codex-native
  role: artifact governance implementer
  objective: define external artifact manifests and prohibited Git payloads
  file_ownership: [docs/Artifact-Policy.md, docs/provenance/artifact-manifest.schema.json, docs/provenance/examples/external-artifact-manifest.example.json, tools/provenance/validate_artifact_manifest.py, tests/provenance/test_artifact_manifest.py, conditional .gitignore]
  assigned_skill: [artifact-provenance-manager]
  inputs: [current tracked artifacts, expected training and FPGA outputs]
  outputs: [artifact policy, schema, validator, ignore rules]
  validation: [unittest, prohibited staged-file scan, hash fixture checks]
  dependencies: [T-000]
```

```yaml
task_card:
  task_id: T-100
  sub_agent: codex-native
  role: software implementer
  objective: implement the canonical dataset and sample contract
  file_ownership: [algorithm/hybrid/src/data/contracts.py, conditional algorithm/hybrid/src/data/handover_adapter.py, algorithm/hybrid/tests/handover/test_data_contract.py]
  assigned_skill: [aegis:test-driven-development]
  inputs: [current data API, HANDOVER field mapping, source registry]
  outputs: [contract adapters, focused tests]
  validation: [focused pytest, existing dataset tests, feature-off equivalence]
  dependencies: [T-010, T-020]
```

```yaml
task_card:
  task_id: T-110
  sub_agent: codex-native
  role: evaluation implementer
  objective: unify evaluation rows and metrics without changing task semantics
  file_ownership: [algorithm/hybrid/src/evaluation/__init__.py, algorithm/hybrid/src/evaluation/contracts.py, algorithm/hybrid/src/evaluation/handover_bridge.py, algorithm/hybrid/tests/handover/test_evaluation_bridge.py]
  assigned_skill: [aegis:test-driven-development]
  inputs: [T-100 contract, current evaluator, HANDOVER metric mapping]
  outputs: [evaluation bridge, regression tests]
  validation: [focused pytest, current evaluator regression, schema negative cases]
  dependencies: [T-100]
```

```yaml
task_card:
  task_id: T-120
  sub_agent: codex-native
  role: checkpoint implementer
  objective: add deterministic checkpoint blend validation and tooling
  file_ownership: [algorithm/hybrid/src/training/checkpoints.py, algorithm/hybrid/src/training/checkpoint_blend.py, algorithm/hybrid/tools/checkpoint_blend.py, algorithm/hybrid/tests/handover/test_checkpoint_blend.py]
  assigned_skill: [aegis:test-driven-development]
  inputs: [current checkpoint loader, source and artifact policies]
  outputs: [blend helper, CLI, validation tests]
  validation: [focused pytest, current pretrained-load tests, hash mismatch failure]
  dependencies: [T-010, T-020, T-100]
```

```yaml
task_card:
  task_id: T-200
  sub_agent: codex-native
  role: hardware implementer
  objective: add compact hardware golden vectors and provenance
  file_ownership: [hardware/refs/handover/hgtxr/manifest.json, compact hardware/refs/handover/hgtxr vectors, hardware/tests/handover/test_hgtxr_golden_vectors.py]
  assigned_skill: [algorithm-hardware-codesign-expert]
  inputs: [cleared HANDOVER vectors, T-010 registry, T-020 policy]
  outputs: [compact goldens, manifest, tests]
  validation: [focused pytest, hash checks, prohibited-payload scan]
  dependencies: [T-010, T-020, T-100]
```

```yaml
task_card:
  task_id: T-210
  sub_agent: codex-native
  role: hardware schema implementer
  objective: define and validate the XR accelerator configuration schema
  file_ownership: [hardware/configs/xr_accel/schema.json, hardware/tools/xr_accel_config.py, hardware/tests/test_xr_accel_config_schema.py]
  assigned_skill: [algorithm-hardware-codesign-expert]
  inputs: [current hardware configuration owners, T-200 goldens]
  outputs: [schema, validator, negative fixtures]
  validation: [focused pytest, schema rejection tests, current config regression]
  dependencies: [T-200]
```

```yaml
task_card:
  task_id: T-220
  sub_agent: codex-native
  role: hardware planning implementer
  objective: add a deterministic no-build XR cyclic planning tool
  file_ownership: [hardware/tools/xr_cyclic_plan.py, hardware/tests/test_xr_cyclic_plan.py, hardware/tests/fixtures/xr_cyclic/csynth_pass.xml, hardware/tests/fixtures/xr_cyclic/timing_pass.rpt, hardware/tests/fixtures/xr_cyclic/timing_fail.rpt]
  assigned_skill: [algorithm-hardware-codesign-expert]
  inputs: [T-200 goldens, T-210 schema, current resource tools]
  outputs: [planner, fixtures, tests]
  validation: [focused pytest, repeatable JSON, no HLS invocation]
  dependencies: [T-200, T-210]
```

```yaml
task_card:
  task_id: T-300
  sub_agent: codex-native
  role: software evaluator
  objective: characterize search strategy behavior and latency before adaptation
  file_ownership: [algorithm/hybrid/tests/handover/test_search_recovery.py, algorithm/hybrid/tools/benchmark_handover_strategies.py, conditional algorithm/hybrid/src/models/tracker/search_strategy.py, conditional tracker config and wiring]
  assigned_skill: [aegis:test-driven-development]
  inputs: [T-100 contract, current search owner, cleared strategy evidence]
  outputs: [characterization tests, benchmark JSON]
  validation: [focused pytest, no source change for tests-only closure, feature-off equivalence and median/p95 gates for adapter closure]
  dependencies: [T-010, T-100, T-110]
```

```yaml
task_card:
  task_id: T-310
  sub_agent: codex-native
  role: software evaluator and conditional implementer
  objective: characterize Stage-2 state isolation and adapt only on a proven gap
  file_ownership: [algorithm/hybrid/tests/handover/test_track_state_adapter.py, conditional algorithm/hybrid/src/runtime/tracker.py, conditional algorithm/hybrid/src/models/tracker/track_strategy.py, conditional algorithm/hybrid/src/models/hybrid_tracker.py]
  assigned_skill: [aegis:test-driven-development]
  inputs: [T-100 contracts, current tracker state owners, cleared source registry]
  outputs: [state characterization tests, optional minimal adapter or no-op verdict]
  validation: [sequence isolation, reset behavior, feature-off equivalence, runtime regression]
  dependencies: [T-010, T-100, T-300]
```

```yaml
task_card:
  task_id: T-320
  sub_agent: codex-native
  role: software regression evaluator
  objective: run full software regressions and record the adoption verdict
  file_ownership: [docs/validation/HANDOVER-SOFTWARE-REGRESSION.md]
  assigned_skill: [aegis:verification-before-completion]
  inputs: [T-300 and T-310 commits or no-op verdicts, current test suites]
  outputs: [compact regression decision]
  validation: [handover, model, training, runtime suites; feature-off equivalence]
  dependencies: [T-300, T-310]
```

```yaml
task_card:
  task_id: T-400
  sub_agent: codex-native
  role: reproducibility analyst
  objective: add a blocked-by-default XR64-XR68 static reproduction contract
  file_ownership: [algorithm/hybrid/hardware_reference/configs/experiments/xr64_xr68_reproduction.yaml, algorithm/hybrid/hardware_reference/docs/resources/xr64_xr68_reproduction_manifest.json, algorithm/hybrid/hardware_reference/docs/resources/xr64_xr68_reproduction_report.md, algorithm/hybrid/hardware_reference/scripts/external/validate_xr64_xr68_reproduction.py, algorithm/hybrid/hardware_reference/tests/test_xr64_xr68_reproduction_contract.py]
  assigned_skill: [ablation-and-experiment-designer]
  inputs: [source reports, contracts, artifact policy]
  outputs: [config, manifest, report, validator, tests]
  validation: [JSON, pytest, BLOCKED dry-run, no execution]
  dependencies: [T-020, T-100, T-110, T-120, T-320]
```

```yaml
task_card:
  task_id: T-410
  sub_agent: codex-native
  role: experiment operator
  objective: execute approved XR64-XR68 training and record compact evidence
  file_ownership: [algorithm/hybrid/hardware_reference/scripts/external/run_xr64_xr68_reproduction.py, algorithm/hybrid/hardware_reference/docs/resources/xr64_xr68_reproduction_result.json, algorithm/hybrid/hardware_reference/docs/resources/xr64_xr68_reproduction_result.md]
  assigned_skill: [ablation-and-experiment-designer]
  inputs: [ready T-400 manifest, approved data checkpoints GPU duration and artifact root]
  outputs: [external artifacts, compact hash-linked result]
  validation: [split and seed checks, evaluator gates, zero test overrides, artifact scan]
  dependencies: [T-400, separate experiment approval]
```

```yaml
task_card:
  task_id: T-490
  sub_agent: codex-native
  role: hardware launcher implementer
  objective: repair active CRLF launchers and enforce external HLS projects
  file_ownership: [hardware/scripts/_common.sh, hardware/scripts/run_hls_csim.sh, hardware/scripts/run_hls_csynth.sh, hardware/vivado/scripts/create_hls_project.tcl, hardware/vivado/scripts/run_csim.tcl, hardware/vivado/scripts/run_csynth.tcl, hardware/tests/test_hls_launcher_contract.py, hardware/docs/validation/xr_accel/HLS-LAUNCHER-BASELINE.md]
  assigned_skill: [bash-defensive-patterns, algorithm-hardware-codesign-expert]
  inputs: [T-020 artifact policy, current CRLF hashes, current launcher behavior]
  outputs: [LF launchers, fail-closed external-root contract, tests, compact baseline]
  validation: [bash syntax, no-argument local smoke, root rejection tests, no in-repo project]
  dependencies: [T-020, T-220, Gate D approval]
```

```yaml
task_card:
  task_id: T-500
  sub_agent: codex-native
  role: HLS implementer
  objective: add an isolated XR cyclic C-simulation case
  file_ownership: [hardware/src/hls/xr_cyclic, hardware/tests/hls/test_xr_cyclic.cpp, hardware/scripts/run_hls_csim.sh, hardware/vivado/scripts/run_xr_cyclic_csim.tcl]
  assigned_skill: [algorithm-hardware-codesign-expert]
  inputs: [T-200 goldens, T-210 schema, T-220 plan, approved tool gate]
  outputs: [isolated HLS case, C-sim test and route]
  validation: [existing no-argument route, XR golden match, explicit tool verdict]
  dependencies: [T-200, T-210, T-220, T-490, separate C-sim approval]
```

```yaml
task_card:
  task_id: T-510
  sub_agent: codex-native
  role: HLS synthesis implementer
  objective: add and execute the approved XR cyclic synthesis route
  file_ownership: [hardware/scripts/run_hls_csynth.sh, hardware/vivado/scripts/run_xr_cyclic_csynth.tcl, hardware/configs/xr_accel/xr_cyclic_synthesis_limits.json, hardware/tools/extract_xr_cyclic_hls_metrics.py, hardware/tools/validate_xr_cyclic_synthesis.py, hardware/tests/test_xr_cyclic_synthesis_route.py, hardware/docs/validation/xr_accel/XR-CYCLIC-SYNTHESIS-GATE.md]
  assigned_skill: [algorithm-hardware-codesign-expert]
  inputs: [T-500 evidence, pinned tool part clock ceilings and artifact root]
  outputs: [synthesis route, predeclared limits, dedicated Vitis XML extractor, fail-closed validator, external metrics and verdict, compact gate record]
  validation: [route and XML fixture pytest, required metric rows, LUT/FF/DSP/BRAM ceilings, clock/WNS target, default-route regression, no in-repo project]
  dependencies: [T-490, T-500, separate synthesis approval]
```

```yaml
task_card:
  task_id: T-520
  sub_agent: codex-native
  role: board evaluator
  objective: validate the approved XR image with a reversible board smoke test
  file_ownership: [hardware/tests/pynq/test_xr_cyclic_board_manifest.py, hardware/docs/validation/xr_accel/XR-CYCLIC-BOARD-SMOKE.md]
  assigned_skill: [algorithm-hardware-codesign-expert]
  inputs: [T-510 and post-route evidence, approved board and rollback image]
  outputs: [manifest test, external smoke JSON, compact board record]
  validation: [hash precheck, ten iterations, golden output, rollback readiness]
  dependencies: [T-510, post-route review, separate board approval]
```

```yaml
task_card:
  task_id: T-600
  sub_agent: codex-native
  role: governance evaluator
  objective: audit CRLF archives and record unresolved governance decisions
  file_ownership: [docs/track/CRLF-ARCHIVE-AUDIT.md, docs/track/ADR.md]
  assigned_skill: [aegis:anti-entropy-governance]
  inputs: [legacy archive, tool checks, current policies]
  outputs: [read-only audit, decision or deferral ADRs]
  validation: [CRLF and bash inventories, diff check, read-only tool checks]
  dependencies: [T-400 or explicit rescheduling]
```

```yaml
task_card:
  task_id: T-610
  sub_agent: codex-native
  role: repository policy implementer
  objective: apply only the approved line-ending and large-file policy
  file_ownership: [.gitattributes, docs/track/ADR.md, explicitly approved archive allowlist]
  assigned_skill: [aegis:anti-entropy-governance]
  inputs: [T-600 audit, explicit normalization and LFS decisions]
  outputs: [targeted attributes and hash mapping or rejection record]
  validation: [git attributes, diff check, bash syntax, semantic equality]
  dependencies: [T-600, separate policy approval]
```

```yaml
task_card:
  task_id: T-620
  sub_agent: codex-native
  role: project tooling implementer
  objective: preview and initialize only the approved Spec-Kit scaffold
  file_ownership: [.specify, specs, reviewed Codex integration files, docs/track/ADR.md]
  assigned_skill: [aegis:writing-plans]
  inputs: [offline scaffold preview, authority mapping, explicit approval]
  outputs: [isolated scaffold commit, authority ADR]
  validation: [preview inventory, collision review, post-init status review]
  dependencies: [T-600, separate Spec-Kit approval]
```

```yaml
task_card:
  task_id: T-630
  sub_agent: codex-native
  role: release operator
  objective: push the verified local branch without force
  file_ownership: []
  assigned_skill: [agyb-essentials:git-pushing]
  inputs: [clean verified worktree, reviewed commits, explicit push approval]
  outputs: [remote branch at local HEAD]
  validation: [status, fetch, ahead-behind review, remote HEAD verification]
  dependencies: [all selected local tasks, separate push approval]
```

## Commit and review policy

- One concern per commit using the messages above.
- Before commit: staged allowlist, `git diff --cached --check`, focused tests,
  prohibited artifact/secret/large-file scan.
- Fresh implementer → fresh spec reviewer → fresh quality reviewer → main integration.
- Never push as part of a task commit.
- Formatting waivers apply only to enumerated provenance, never active source.

## Risks and stop conditions

| Risk | Stop condition | Safe state |
|---|---|---|
| License | source/scope/notice unresolved | `LICENSE_BLOCKED` |
| Dependencies | baseline environment/test cannot reproduce | stop before source edits |
| Data/privacy | rights or split unknown | manifest `BLOCKED`; no access/import |
| Contract | shape/task/coordinate/checkpoint ambiguous | fail closed; no adapter |
| Duplicate owner | current behavior already sufficient | tests only; reject module |
| Runtime | feature-off/output/latency mismatch | remove adapter |
| Hardware | golden/C-sim/resource/timing failure | reference-only |
| Artifacts | prohibited staged file | unstage and remove from Git scope |
| Tool | pytest/Vitis/Vivado/board absent | report unavailable, never infer pass |
| Remote | upload/LFS/push lacks approval | stop and request approval |

## Rollback and retirement

- Revert semantic commits in reverse dependency order; never destructive reset.
- Registries/contracts may remain even when adapters are rejected.
- Conditional strategies retire on duplication, license block, failed equivalence,
  latency regression, or no same-condition benefit.
- Existing evaluators, checkpoint loaders, and HLS tops remain until a separate
  migration inventories callers and proves equivalence.
- Reference archives are never normalized/deleted without hash mapping and approval.
- External artifact replacement is independent of code rollback; manifests carry
  predecessor and replacement hashes.

## Completion evidence

Approved tasks are complete only when they have focused and related tests,
registry coverage or explicit block, zero prohibited staged content, independent
spec/quality reviews, an atomic commit SHA, a clean index/worktree, explicit
uncovered risks, and no unapproved install/training/synthesis/board/upload/push.

## Plan review status

- Specification/executability review: PASS after task/card ownership, T-490
  launcher preflight, and T-510 XML metric-gate remediation.
- Documentation-quality review: PASS with 22 unique sequential tasks and 22
  matching Task Cards.
- Static plan checks: `git diff --check` PASS; every Task Card has all required
  fields; planning changes are documentation-only.
- Authority: reviewed and ready for a user execution decision; no implementation,
  dependency install, experiment, FPGA/board action, commit, or push is implied.

## Execution choice

Recommended: sequential subagent-driven execution with fresh implementation and
two-stage review per task. Inline execution is acceptable only with the same
approval gates, staged allowlists, and review boundaries.
