# Standalone HBTXR Selective Integration Plan

## Goal

Selectively adapt only the useful annotation contracts and dependency-free
metrics from `/mnt/d/dataset/EV_Eye/paper_works/HBTXR` into the current HBTXR
owners. Prove that FECET, HG-PIPE Quantization, and most SWIFT content are
already represented, and keep all unclear-license, data, result, weight,
installation, experiment, and runtime-dependent surfaces blocked.

## Architecture

`HBTXR-Pool/HBTXR` remains the sole runtime and commit owner. The standalone
repository is pinned, read-only evidence. Promotion is contract-first:

```text
source commit + rights
  -> no-op/drift proof
  -> coordinate-frame ADR
  -> pure record contract
  -> pure metrics
  -> pure uncertainty metrics
  -> curated documentation
  -> separately planned runtime candidates, if ever approved
```

Direct branch merge, cherry-pick, whole-directory copy, and archive overwrite
are prohibited.

## Tech Stack

- Git committed-blob comparison and SHA-256 provenance
- Python 3.12 standard library: dataclasses, enum, math, statistics, unittest
- Existing `algorithm/hybrid/src/evaluation` contracts
- JSON/Markdown provenance and Aegis planning artifacts
- No dependency installation, NumPy/OpenCV/PyTorch import, dataset access,
  reproduction, experiment, model execution, synthesis, or board action

## Baseline/Authority Refs

- `docs/Spec.md`
- `docs/track/ADR.md`, especially ADR-001 and ADR-004
- `docs/aegis/baseline/2026-07-15-follow-up-baseline.md`
- `docs/analysis/STANDALONE-HBTXR-SELECTIVE-INTEGRATION-MATRIX.md`
- `docs/analysis/HANDOVER-INTEGRATION-MATRIX.md`
- `docs/provenance/handover-source-registry.json`
- Source `annotation` revision
  `2ff52628bec1eb4edd9227c5c9e720718a9accef`
- Target `refactor/hbtxr-structure` revision at plan time
  `ebe862b11506e819c5bd5abc925299fc5fbb6f1a`

## Compatibility Boundary

- Existing data, evaluation, tracker, preprocess, quantization, archive, and
  checkpoint owners remain canonical.
- Existing `Domain.SENSOR=640x480`, `ROI=80x60`, and
  `POST_TRANSFORM=64x64` behavior cannot change silently.
- The 14 already integrated annotation files are not overwritten.
- FECET/SWIFT archives and active HG-PIPE Quantization code/tests are not
  replaced.
- Default runtime behavior, output ABI, model/checkpoint keys, annotation
  backend selection, and artifact paths remain unchanged.
- New APIs are dependency-free and additive; no CLI or experiment runner is
  added in the first slice.

## TDD Route

- Mode: off
- Decision: skipped
- Strict authority: not applicable
- Test posture: post-change regression using synthetic standard-library fixtures
- Reason: neither the user nor the project requested strict RED/GREEN TDD; the
  plan still requires focused unit evidence before each task commit.
- Verification: task-specific `unittest`, existing evaluation/provenance tests,
  `compileall`, path allowlists, and Git diff checks.

## Verification

No command in this plan installs packages or executes source experiments.
Every implementation task must run its focused command plus:

```bash
if [ -d algorithm/hybrid/tests/evaluation ]; then
  PYTHONPATH=algorithm/hybrid/src python3 -m unittest discover -s algorithm/hybrid/tests/evaluation -p 'test_*.py'
fi
PYTHONPATH=algorithm/hybrid/src python3 -m unittest discover -s algorithm/hybrid/tests/handover -p 'test_*.py'
python3 -m unittest discover -s tests/provenance -p 'test_*.py'
python3 -m compileall -q algorithm/hybrid/src/evaluation tools/provenance
git diff --check
git status --short --branch
git diff --cached --name-status
```

Expected: existing test directories run and pass, absent future-only directories are skipped; compile exits 0; `git diff --check` is silent; the staged
manifest contains only the current task's allowlist.

## 한국어 실행 요약

독립 HBTXR의 2,594개 annotation 파일을 통째로 가져오지 않는다. 이미
대상에 있는 14개는 유지하고, 소스 전용 2,580개 중 데이터와 결과를
제외한 순수 레코드·평가·불확실성 계약만 현재 evaluation 패키지에 맞춰
재작성한다. FECET과 HG-PIPE는 추가 코드 통합이 없고, SWIFT는 anti-blink와
체크포인트 importer만 미래 후보로 남긴다. 단, 독립 저장소 코드에는
루트 라이선스가 없으므로 저작권·사용 권한이 확인되기 전 활성 코드
작업은 시작하지 않는다.

## Aegis Visibility

This plan is required because the source has a separate Git history, unresolved
rights, conflicting coordinate frames, large excluded artifact surfaces, and
potential overlap with existing canonical owners.

## Prompt Brief

- Goal: produce a concrete, reversible, file-level selective integration plan.
- Inputs: two live repositories, pinned HEAD trees, existing evaluation and
  provenance owners, prior HANDOVER matrix, and independent sub-agent audits.
- Assumptions: useful early annotation behavior can be represented as pure
  standard-library contracts and metrics.
- Unknowns: source authorship/license, intended `346x240` target height, need
  for Tobii sync, and future PyTorch/checkpoint authority.
- Constraints: target-only writes; standalone read-only; no bulk merge,
  install, reproduction, experiment, data/weight copy, push, or cleanup.
- Outputs: this plan, a file-level matrix, tracking pointers, exact gates,
  task files, commands, rollback, and stop conditions.
- Acceptance: every candidate is classified; every active task has exact
  source/target paths, compatibility, verification, dependencies, and rollback.

## Best-output criteria

| Criterion | Acceptance |
|---|---|
| Completeness | annotation and all three `references/impl` roots classified |
| Evidence | revisions, file counts, path mappings, no-op and license evidence recorded |
| Executability | exact task files, commands, expected results, commits, and stops |
| Consistency | current evaluation/provenance owners and classification vocabulary reused |
| Safety | rights, coordinates, artifacts, runtime, remote mutation fail closed |
| Maintainability | pure additive owners; no duplicate pipelines or archive replacement |

## Plan Basis

### Facts

- The source and target are separate non-shallow repositories with unrelated
  current branch roots, despite sharing an origin URL.
- Source annotation HEAD has 2,594 tracked files. Target already contains all
  14 early annotation paths: 11 identical and 3 changed.
- Source-only annotation content includes 2,422 sample files; generated data,
  results, figures, tables, weights, and lockfiles dominate the remaining tree.
- `io_schema.py` has absolute `/home/user/...` and `/mnt/e/...` paths and creates
  output directories during import.
- Source metrics use `346x260`; target preprocessing references `346x240`; the
  current evaluation contract maps `SENSOR` to `640x480`.
- FECET and SWIFT trees are already preserved as CRLF-normalized inactive
  archives; HG-PIPE code/tests/configs are already adapted under `quantization`.
- The three `references/impl` roots contain no governing license/notice or
  individual upstream metadata.

### Inferences

- Pure record and metric functions are useful only when they reuse current
  evaluation contracts and avoid source I/O/orchestration.
- Direct copies would create duplicate owners and retain path/data coupling.
- FECET/HG-PIPE additional integration has negative value unless a concrete
  missing behavior is first demonstrated.

### Unknowns

- Whether the source author grants active reuse and under what terms.
- Whether `346x240` is intentional cropping or a defect relative to DAVIS346.
- Whether Tobii sync or SWIFT anti-blink belongs to the current product scope.

## BaselineUsageDraft

- Required baseline refs: Spec, ADR-001/004, follow-up baseline, current
  evaluation contracts, prior HANDOVER matrix, source/target pinned revisions.
- Delivered context refs: live Git/tree comparisons and two independent audits.
- Acknowledged before plan refs: all required refs above.
- Cited in plan refs: all required refs above plus the standalone matrix.
- Missing refs: source license/authorship and an accepted coordinate-frame ADR.
- Decision: continue planning; implementation pauses at G1/G2 until resolved.

## Requirement Ready Check

- Requirement source refs: user's selective-integration request and ADR-001.
- Goals and scope refs: Goal, Compatibility Boundary, matrix.
- User/scenario refs: maintain one integrated HBTXR without bulk merge.
- Requirement items: classify, select, sequence, verify, rollback, and defer.
- Acceptance refs: Best-output criteria and G0-G8 below.
- Open blocker questions: rights and coordinate-frame authority.
- Decision: ready for a gated plan; code tasks are not execution-approved and
  cannot start before the named blockers close.

## Change Necessity

- User-visible need: preserve useful annotation audit semantics in the current
  HBTXR without importing a second pipeline.
- No-change option: adequate for FECET, HG-PIPE, most SWIFT, artifacts, and
  orchestration; these remain no-op/reference/excluded.
- Why code is necessary: current evaluation owns coordinate conversion but not
  annotation record normalization, overlap/boundary metrics, or uncertainty
  contracts identified in ANN-01 through ANN-04.
- Minimum boundary: `algorithm/hybrid/src/evaluation`, focused tests, one
  provenance manifest/validator extension, one curated analysis document.
- Decision: conditional code-change only after G1/G2; otherwise docs-only.

## Existence Check

| Proposed surface | Existing owner/reuse | Decision |
|---|---|---|
| Standalone source registry | HANDOVER registry and validator | Reuse schema; add a separate registry and exact external-root/committed-view support rather than weakening defaults. |
| `annotation_records.py` | evaluation contracts | Add one adapter owner; file I/O remains outside. |
| `annotation_metrics.py` | evaluation policy/bridge | Add pure metric owner; do not create a parallel evaluator pipeline. |
| `annotation_uncertainty.py` | no current equivalent | Add only dependency-free uncertainty contracts after rights clearance. |
| Annotation CLI | scripts and historic orchestration | Reject from first slice. |
| FECET/HG-PIPE active copies | current active/archive owners | Reject; reuse current owners. |
| SWIFT runtime modules | current hybrid runtime plus archive | Defer to a separate plan with runtime authority. |

## Architecture Integrity Lens

- Invariant: the target is the only runtime owner; the source is immutable evidence.
- Canonical owner: evaluation contracts and metrics under
  `algorithm/hybrid/src/evaluation`.
- Responsibility overlap: preprocess remains responsible for data/file I/O;
  evaluation modules accept already-normalized values.
- Higher-level simplification: expand the current coordinate contract before
  introducing any source-specific evaluator.
- Retirement/falsifier: remove or never add a module if current behavior or a
  lower-level helper already satisfies its tests.
- Verdict: three small pure owners, no second pipeline.

## Plan Pressure Test

- Owner/contract/retirement: explicit and additive; no archive becomes active.
- Architecture integrity: current evaluation and provenance owners are reused.
- Verification scope: standard-library unit/static checks only.
- Task executability: exact files, commands, dependencies, and stops follow.
- Pressure result: proceed with planning; execution gated by rights and frames.

## Plan-Time Complexity Check

| Target | Current pressure | Planned boundary | Result |
|---|---|---|---|
| `evaluation/contracts.py` | small, strict size map | one additive domain/frame decision only | within-budget |
| `annotation_records.py` | new single-purpose owner | key parsing and normalized records only | within-budget |
| `annotation_metrics.py` | new single-purpose owner | scalar/point-set metrics only | within-budget |
| `annotation_uncertainty.py` | new single-purpose owner | repeatability/LoA/three-source math only | within-budget |
| provenance validator | security-sensitive | explicit allowlist; defaults unchanged; dedicated tests | at-risk, governed by SI-000 |

No task may absorb file discovery, dataset loading, reporting, plotting, model
inference, or experiment orchestration into these modules.

## Execution Readiness View

- Intent Lock: selective clean adaptation, never bulk copy.
- Scope Fence: target paths listed below; source read-only at pinned commit.
- Baseline Lock: verify target branch/HEAD and source commit before every batch.
- Approved Behavior: plan/document creation only; future execution needs a new
  explicit user instruction.
- Owner constraints: evaluation owns values; preprocess owns I/O; provenance
  owns source identity; archives remain inactive.
- Compatibility: existing domains, transforms, policies, runtimes, archives,
  quantization imports, and outputs remain stable.
- Retirement: revert concern-sized commits; delete additive modules if gates or
  regressions fail; never rewrite source history.
- Task batches: SI-000 -> SI-010 -> SI-020 -> SI-100 -> SI-110 -> SI-120 -> SI-130.
- Test obligations: focused unittest plus current evaluation/provenance suites.
- Review gates: G0-G8, staged allowlist, specification and quality review.
- Drift rule: stop and revise on source SHA, rights, frame, owner, dependency,
  runtime, artifact, or public API drift.
- Evidence: command output, registry hashes, ADR, test logs, staged paths, commit SHA.
- Advisory boundary: this plan is not merge, push, release, or completion authority.

## Execution DAG

```text
SI-000 provenance support
  -> SI-010 rights + no-op freeze
  -> SI-020 coordinate-frame ADR
  -> SI-100 annotation records
  -> SI-110 pure annotation metrics
  -> SI-120 pure uncertainty metrics
  -> SI-130 curated audit documentation

[SI-200 SWIFT anti-blink/checkpoint preflight: separate future plan]
[Tobii, HDF5/MAT, CLI, datasets, results, weights: deferred/excluded]
```

## Approval gates

| Gate | Required evidence | Failure action |
|---|---|---|
| G0 Source lock | target clean; source revision `2ff5262`; committed-view hashes | Stop on drift; refresh matrix, do not copy worktree |
| G1 Rights | author/upstream/license/notice and code/model/data rights | Keep all source-derived code `LICENSE_BLOCKED` |
| G2 Frame contract | ADR resolves 346x260, 346x240, and 640x480 meanings | Do not edit evaluation contracts |
| G3 No-op freeze | normalized comparison for common annotation and three impl roots | Reclassify changed paths before work |
| G4 API | additive signatures, no import-time I/O or optional dependency | Reject or split task |
| G5 Tests | focused and existing suites exit 0 | Revert task commit |
| G6 Artifact/scope | no samples/data/results/weights/lockfiles/archive replacement | Unstage prohibited files and stop |
| G7 Review/commit | spec and quality review; concern-sized local commit | Fix findings before commit |
| G8 Remote | separate explicit push/merge approval | Keep commits local |

## Sequential task plan

### SI-000 — Generalize provenance validation without weakening defaults

**Files**

- Modify `tools/provenance/validate_source_registry.py`.
- Modify `tests/provenance/test_source_registry.py`.
- Create `docs/provenance/standalone-hbtxr-source-registry.json`.

**Why**

The existing validator permits only the target workspace and sibling HANDOVER.
The standalone repo is outside that allowlist, and its CRLF-dirty worktree must
not override committed source evidence.

**Change Necessity**

Markdown alone cannot fail closed on revision, path, and hash drift. The minimum
change is an explicit repeatable `--allow-repo PATH` and a committed-blob source
view; default behavior must remain unchanged.

**Planned interface**

```text
--allow-repo PATH        repeatable exact additional repository root
--source-view MODE       worktree (existing default) | committed
```

`committed` verifies `revision:path` via Git and never hashes the dirty working
file. Relative source paths remain required; the explicit root must resolve
exactly and must not authorize its parent or siblings.

**Steps**

1. Add unit cases proving default rejection, exact-root acceptance, committed
   blob verification, SHA mismatch rejection, traversal rejection, and no
   regression for the existing HANDOVER registry.
2. Add the two CLI options and pass normalized allowed roots/source view into
   validation without changing current defaults.
3. Add registry entries for `STANDALONE_ANNOTATION`, `STANDALONE_FECET`,
   `STANDALONE_HG_PIPE`, and `STANDALONE_SWIFT`, all pinned to source HEAD.
4. Keep unresolved rights `BLOCKED`; no entry may be `ACTIVE_CANDIDATE` while
   its license/rights are unresolved.

**Verification**

```bash
python3 -m unittest discover -s tests/provenance -p 'test_source_registry.py'
python3 tools/provenance/validate_source_registry.py docs/provenance/handover-source-registry.json --workspace-root . --require-local
python3 tools/provenance/validate_source_registry.py docs/provenance/standalone-hbtxr-source-registry.json --workspace-root . --allow-repo ../../HBTXR --source-view committed --require-local
```

Expected: all commands exit 0; four standalone entries validate from committed
blobs; the existing registry command remains unchanged.

**Commit:** `feat(provenance): Pin standalone HBTXR sources`

**Rollback/stop:** revert this commit if the new allowlist broadens beyond exact
roots, reads worktree bytes in committed mode, or weakens any existing rejection.

### SI-010 — Close rights and freeze no-op decisions

**Files**

- Modify `docs/provenance/standalone-hbtxr-source-registry.json`.
- Modify `docs/analysis/STANDALONE-HBTXR-SELECTIVE-INTEGRATION-MATRIX.md` only
  when evidence changes a classification.
- Create `docs/provenance/STANDALONE-HBTXR-SOURCES.md`.

**Why**

The source has no root license, and existing equivalence must be frozen before
any active adaptation.

**Steps**

1. Record author/owner, upstream URL, original revision, license path/text,
   notice obligation, and code/model/data rights separately for annotation,
   FECET, HG-PIPE, and SWIFT.
2. Compare committed source blobs to target paths with CRLF normalization only;
   record counts and exceptions, never copy the source worktree.
3. Lock FECET and HG-PIPE as `NO-OP`; lock SWIFT archive as `NO-OP`; leave
   SWIFT anti-blink/importer as blocked future candidates.
4. If annotation rights remain unresolved, end this workstream after docs and
   do not start SI-020 or code tasks.

**Verification**

```bash
python3 tools/provenance/validate_source_registry.py docs/provenance/standalone-hbtxr-source-registry.json --workspace-root . --allow-repo ../../HBTXR --source-view committed --require-local
rg -n 'UNVERIFIED|PRESENT_UNREVIEWED|BLOCKED|APPROVED' docs/provenance/standalone-hbtxr-source-registry.json docs/provenance/STANDALONE-HBTXR-SOURCES.md
git diff --check
```

Expected: registry validates; every unresolved right is paired with `BLOCKED`;
no source/archive/code file is staged.

**Commit:** `docs(provenance): Classify standalone HBTXR rights and overlap`

**Rollback/stop:** revert documentation if evidence is incorrect. Stop active
promotion on missing author, upstream, license scope, or notice.

### SI-020 — Decide the coordinate-frame contract

**Files**

- Modify `docs/track/ADR.md` with ADR-005.
- Modify the matrix only if the decision changes targets or classifications.

**Why**

The source assumes 346x260, current preprocessing references 346x240, and the
evaluation `SENSOR` domain is 640x480. Treating these as interchangeable would
silently corrupt anisotropic scaling and metric thresholds.

**Steps**

1. Trace all maintained occurrences of 346x260, 346x240, and 640x480 to their
   sensor/crop/post-transform meaning.
2. Record whether 346x240 is an intentional crop and name each distinct frame.
3. Preferred decision, only if evidence confirms it: add a distinct
   `Domain.DAVIS346` mapped to 346x260; do not redefine `Domain.SENSOR`.
4. Record conversion ownership, bounds semantics, angle units, axis convention,
   and the exact rejection policy for unknown frames.
5. If the preferred decision is false, revise SI-100 signatures before code.

**Verification**

```bash
rg -n '346.{0,8}(240|260)|640.{0,8}480|Domain\.' algorithm/hybrid/src algorithm/hybrid/tests docs/track/ADR.md
git diff --check docs/track/ADR.md
```

Expected: ADR-005 has one canonical meaning for every frame and no implicit
single-factor conversion.

**Commit:** `docs(adr): Define annotation coordinate frames`

**Rollback/stop:** revert ADR commit and stop SI-100 if evidence cannot
distinguish sensor, crop, and post-transform frames.

### SI-100 — Add dependency-free annotation record contracts

**Dependencies:** SI-010 rights approved; SI-020 accepted.

**Files**

- Modify `algorithm/hybrid/src/evaluation/contracts.py` only for the accepted
  additive frame/domain.
- Create `algorithm/hybrid/src/evaluation/annotation_records.py`.
- Modify `algorithm/hybrid/src/evaluation/__init__.py`.
- Create `algorithm/hybrid/tests/evaluation/test_annotation_records.py`.

**Why**

Long-form annotation sources need validated, deterministic records before any
metric can be trusted. Current evaluation contracts do not parse annotation
sample keys or source-tagged records.

**Planned public API**

```python
class AnnotationSource(str, Enum): ...
@dataclass(frozen=True)
class SampleKey: motion: str; subject: int; eye: str; session: str
@dataclass(frozen=True)
class AnnotationCenter: key: SampleKey; source: AnnotationSource; center: Center
def parse_sample_key(value: str) -> SampleKey: ...
def normalize_annotation_records(records: Iterable[AnnotationCenter]) -> tuple[AnnotationCenter, ...]: ...
```

Unknown sources, malformed keys, booleans-as-numbers, out-of-frame coordinates,
duplicates, and domain/size mismatch raise `EvaluationContractError`.
Normalization sorts by key then source and performs no file I/O.

**Verification**

```bash
PYTHONPATH=algorithm/hybrid/src python3 -m unittest algorithm.hybrid.tests.evaluation.test_annotation_records
PYTHONPATH=algorithm/hybrid/src python3 -m unittest discover -s algorithm/hybrid/tests/handover -p 'test_*.py'
python3 -m compileall -q algorithm/hybrid/src/evaluation
```

Expected: malformed/bounds/duplicate cases fail closed; valid ordering and
anisotropic conversion are deterministic; existing tests remain green.

**Commit:** `feat(eval): Add annotation record contracts`

**Rollback/stop:** revert commit on existing domain behavior change, new I/O,
optional dependency import, or ambiguous coordinate bounds.

### SI-110 — Add pure annotation metrics

**Dependencies:** SI-100.

**Files**

- Create `algorithm/hybrid/src/evaluation/annotation_metrics.py`.
- Modify `algorithm/hybrid/src/evaluation/__init__.py`.
- Create `algorithm/hybrid/tests/evaluation/test_annotation_metrics.py`.

**Why**

The useful source behavior is metric math, not its hard-coded dataset/report
pipeline. Current evaluation has coordinate contracts but lacks these pure
annotation metrics.

**Planned public API**

```python
class ObservationStatus(str, Enum): ...  # VALID | ABSTAIN | INVALID
@dataclass(frozen=True)
class ErrorObservation:
    subject: str
    motion: str
    error: float | None
    status: ObservationStatus
@dataclass(frozen=True)
class PrecisionSummary:
    coverage: float
    valid_precision: Mapping[int, float]
    all_sample_precision: Mapping[int, float]
    group_precision: Mapping[str, Mapping[int, float]]
def center_error(predicted: Center, expected: Center) -> float: ...
def summarize_precision(observations: Iterable[ErrorObservation], policy: EvaluationPolicy = EvaluationPolicy()) -> PrecisionSummary: ...
def binary_iou(intersection: int, left_area: int, right_area: int) -> float: ...
def binary_dice(intersection: int, left_area: int, right_area: int) -> float: ...
def radius_ratio(detected_area: float, axis_x: float, axis_y: float) -> float: ...
def orientation_delta_degrees(left: float, right: float) -> float: ...
def hausdorff_distance(left: Sequence[tuple[float, float]], right: Sequence[tuple[float, float]]) -> float: ...
def average_symmetric_surface_distance(left: Sequence[tuple[float, float]], right: Sequence[tuple[float, float]]) -> float: ...
```

`valid_precision` uses only `VALID` observations. `all_sample_precision` counts
`ABSTAIN` as a failure; `INVALID` raises under `reject` and is removed from the
eligible denominator under `exclude`. `coverage` is valid/eligible. Subject and
motion identifiers are mandatory so `subject_weighting` and `aggregation` can
produce reviewable macro/micro group scores. No NumPy, OpenCV, Pillow, SciPy,
pandas, filesystem, plotting, or report writing is allowed.

**Verification**

```bash
PYTHONPATH=algorithm/hybrid/src python3 -m unittest algorithm.hybrid.tests.evaluation.test_annotation_metrics
PYTHONPATH=algorithm/hybrid/src python3 -m unittest discover -s algorithm/hybrid/tests/evaluation -p 'test_*.py'
PYTHONPATH=algorithm/hybrid/src python3 -m unittest discover -s algorithm/hybrid/tests/handover -p 'test_*.py'
```

Expected: exact synthetic IoU/Dice/center/orientation/boundary values; symmetry;
identical-set zero distance; invalid and empty inputs rejected. Precision
fixtures must make macro differ from micro and verify valid-only versus
abstain-as-fail totals plus `reject` versus `exclude` behavior.

**Commit:** `feat(eval): Add pure annotation geometry metrics`

**Rollback/stop:** revert if metric conventions cannot be named unambiguously or
if runtime/data dependencies enter the module.

### SI-120 — Add pure repeatability and uncertainty metrics

**Dependencies:** SI-110.

**Files**

- Create `algorithm/hybrid/src/evaluation/annotation_uncertainty.py`.
- Modify `algorithm/hybrid/src/evaluation/__init__.py`.
- Create `algorithm/hybrid/tests/evaluation/test_annotation_uncertainty.py`.

**Why**

Repeatability and source-disagreement estimates are useful, but the source
implementation mixes them with datasets and optimistic train-subject results.

**Planned public API**

```python
@dataclass(frozen=True)
class LimitsOfAgreement: bias: float; lower: float; upper: float
@dataclass(frozen=True)
class RepeatabilitySigma:
    sigma_x: float
    sigma_y: float
    sigma_radial: float
    ddof: int
@dataclass(frozen=True)
class ThreeCorneredHatResult:
    sigmas: Mapping[str, float]
    negative_components: tuple[str, ...]
    independence_declared: bool
    status: str
def repeatability_sigma(points: Sequence[Center], *, ddof: int = 1) -> RepeatabilitySigma: ...
def limits_of_agreement(differences: Iterable[float], *, ddof: int = 1, multiplier: float = 1.96) -> LimitsOfAgreement: ...
def pairwise_rms(left: Sequence[Center], right: Sequence[Center]) -> float: ...
def three_cornered_hat(pairwise_variances: Mapping[tuple[str, str], float], *, independence_declared: bool) -> ThreeCorneredHatResult: ...
```

`repeatability_sigma` uses per-axis sample standard deviations (`ddof=1` by
default) and `sigma_radial = sqrt(sigma_x**2 + sigma_y**2)`. Limits of agreement
use the arithmetic bias and sample standard deviation with `bias ± 1.96*s` by
default. Pairwise RMS requires aligned, equal-length centers in the same domain
and size and computes `sqrt(mean(dx**2 + dy**2))`.

The three-source function requires exactly three named sources and
`independence_declared=True`; it computes each component variance as one half
of the sum of its two pairwise variances minus the remaining pair variance.
Any negative component raises `EvaluationContractError` in the first slice.
Bracket fallback is deferred because pairwise variances alone do not preserve
the bias/radial-RMS inputs needed for the source fallback. A successful result
has empty `negative_components`, `independence_declared=True`, and status `ok`;
returned `sigmas` are square roots of the nonnegative component variances.

**Verification**

```bash
PYTHONPATH=algorithm/hybrid/src python3 -m unittest algorithm.hybrid.tests.evaluation.test_annotation_uncertainty
PYTHONPATH=algorithm/hybrid/src python3 -m unittest discover -s algorithm/hybrid/tests/evaluation -p 'test_*.py'
```

Expected: identical inputs produce zero; symmetry and known hand-calculated
fixtures pass; missing sources, length mismatch, NaN, and negative variance fail
closed. Axis/radial `ddof=1`, LoA multiplier, aligned pairwise RMS, and the
independence declaration are asserted; no bracket result exists in this slice.

**Commit:** `feat(eval): Add annotation uncertainty contracts`

**Rollback/stop:** revert if source independence cannot be represented or a
statistic requires unstated dataset assumptions.

### SI-130 — Curate annotation evidence without importing results

**Dependencies:** SI-100 through SI-120, or docs-only completion after SI-010 if
code remains license-blocked.

**Files**

- Create `docs/analysis/EV-EYE-ANNOTATION-AUDIT.md`.
- Update `docs/analysis/STANDALONE-HBTXR-SELECTIVE-INTEGRATION-MATRIX.md`.
- Update `docs/track/TODO.md`, `docs/track/PROGRESS.md`, and `docs/track/log.md`.

**Why**

Useful source decisions need provenance and caveats, while historical result
numbers, samples, and figures must not become current evidence.

**Steps**

1. Document primary versus derived labels, frame conventions, valid source
   independence, invalid-sample policy, and the users-1-10 train-subject caveat.
2. Link every claim to pinned source paths/revision and the current target owner.
3. State that no source numeric result was reproduced or promoted.
4. Record final class, tests, commit, rollback, and remaining blocked candidates.

**Verification**

```bash
rg -n '2ff52628bec1eb4edd9227c5c9e720718a9accef|346x260|derived|train|LICENSE_BLOCKED' docs/analysis/EV-EYE-ANNOTATION-AUDIT.md
git diff --check
git status --short --branch
```

Expected: all required caveats are present; no sample/result/weight path is
staged as content.

**Commit:** `docs(analysis): Record standalone annotation integration evidence`

**Rollback/stop:** revert only this documentation commit; source and active
runtime remain untouched.

### SI-200 — SWIFT future-lane preflight, not part of the first slice

**Candidate files:** source `references/impl/SWIFT-HBTXR/swift_hbtxr/antiblink.py` and
`tools/import_swift_eye_checkpoint.py`; possible target
`algorithm/hybrid/src/models/antiblink.py` and
`algorithm/hybrid/src/preprocess/swift_eye_checkpoint.py`.

This lane requires a separate plan because it introduces PyTorch/runtime state,
checkpoint compatibility, optional output behavior, model terms, and tests that
cannot be validated under the no-install/no-model boundary. Start it only after:

1. G1 rights are approved for source and model/checkpoint separately.
2. A current caller requires `open_extent`, `should_hold`, or Swift-Eye key import.
3. Runtime dependencies and checkpoint fixtures are explicitly authorized.
4. Feature-off equivalence and sequence-reset obligations are specified.

No file or commit is authorized for SI-200 by this plan.

## File ownership during future execution

| Task | Exclusive write owner | Allowed files |
|---|---|---|
| SI-000 | provenance implementer | validator, provenance test, standalone registry |
| SI-010 | provenance analyst | standalone registry, SOURCES, matrix |
| SI-020 | architecture owner | ADR and matrix only |
| SI-100 | evaluation implementer | contracts, annotation records, init, record test |
| SI-110 | evaluation implementer | annotation metrics, init, metric test |
| SI-120 | evaluation implementer | uncertainty module, init, uncertainty test |
| SI-130 | documentation owner | audit, matrix, tracking docs |

Specification and quality reviewers own no files. Each task is integrated and
committed before the next begins.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Unlicensed internal/vendor code | G1 fail-closed registry; no active port |
| CRLF-dirty source worktree | committed source view only |
| 346x260/346x240/640x480 conflation | ADR before source edits; per-axis conversion tests |
| Duplicate evaluator/preprocess pipeline | current owner reuse; no source I/O/orchestration |
| Historic train-subject claims promoted as current | curated caveat; no numeric result promotion |
| Data/PII/model/checkpoint ingestion | G6 allowlist and permanent exclusions |
| Optional dependency creep | standard-library first slice; compile/import tests |
| Provenance allowlist security regression | exact-root CLI, defaults unchanged, traversal tests |
| SWIFT runtime scope expansion | separate plan and explicit runtime authority |

## Retirement and rollback

- Use concern-sized local commits and `git revert`; never destructive reset.
- Provenance support is retained only if defaults remain fail closed and at
  least one external registry uses it.
- Annotation modules are removed if no maintained caller or contract test uses
  them after the integration review.
- FECET/SWIFT archives remain inactive provenance; HG-PIPE target remains the
  sole active quantization owner.
- No plan step deletes or modifies the standalone repository.
- Push, merge, release, and worktree cleanup remain separately authorized.

## Self-review

- Spec coverage: annotation and all three implementation roots mapped.
- Placeholder scan: no TBD/TODO or unnamed implementation step.
- Type consistency: planned records reuse `Center`, `EvaluationPolicy`, and
  `EvaluationContractError`.
- Compatibility: existing domains/defaults/owners are preserved.
- Change necessity: only ANN-01 through ANN-04 justify code after gates.
- Existence: new files are single-purpose and reuse current evaluation owner.
- Complexity: provenance change is isolated and security-tested; metric modules
  exclude I/O/orchestration.
- Verification: exact commands and expected results exist per task.
- Dual-track: active adaptation and blocked/deferred retirement states are explicit.

## Progress against this plan

- Planning and file-level analysis: complete.
- Code/file implementation: not started and not authorized by plan creation.
- Dependency installation/reproduction/experiments: excluded.
- Push/merge/release: not authorized.
