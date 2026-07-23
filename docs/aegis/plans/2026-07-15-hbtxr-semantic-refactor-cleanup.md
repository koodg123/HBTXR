# 2026-07-21 Algorithm Ownership Supersession Notice

`algorithm` ownership layout와 `src.pools` disposition은
`docs/aegis/plans/2026-07-21-algorithm-modular-ownership-refactor.md`가
후속 권위다. 아래의 `algorithm/common/src/EvEye` 영구 보존, current modality
ownership 고정, "LR scheduler만 이동하고 나머지 pool facade 삭제" 결정은
역사적 계획으로 남지만 실행 권위가 아니다. Reference, hardware, Quantization,
cleanup task 중 새 계획과 충돌하지 않는 부분은 계속 유효하다.

# Goal

메인 HBTXR 저장소의 `algorithm/{common,event,frame,hybrid}` 구조와 비교 실험용 `references/**`를 그대로 보존하면서 correctness와 side-effect 위험을 먼저 제거하고, 실제 필요성이 입증된 package/API/CLI/validation owner만 최소 범위로 정리한다.

성공 조건은 다음과 같다.

- `algorithm/{common,event,frame,hybrid}`와 `algorithm/hybrid/src`의 현재 ownership이 고정되고 이동·rename되지 않는다.
- Reference/history를 제외한 active algorithm 경로·식별자·runtime contract에서 repository-owned `FACET/facet` 이름이 제거된다.
- 제거된 FACET environment controls를 `HBTXR_*` 같은 새 product-prefixed alias로 치환하지 않고 existing config keys 또는 semantic CLI options로 흡수한다.
- `algorithm/hybrid/src/pools`와 `src.pools` 공개 표면이 제거되고, 유일한 실사용 LR scheduler는 `src.optim`으로 이동한다.
- Hybrid bootstrap은 현재 `src` owner를 안정적으로 찾고, Quantization은 독립 package owner로 정리되며, Hybrid namespace migration은 실제 충돌이 재현되고 승인될 때만 수행된다.
- shadowed definition과 import-time file write가 characterization test 아래에서 제거된다.
- 500~1,954라인 monolithic 심볼이 pure rule, I/O, orchestration, rendering owner로 분리된다.
- active/archive/reference/vendor의 권위가 문서·도구·경로에서 일치한다.
- `references/**`는 비교 실험 자산으로 영구 보존되고 삭제·rename·pointer 대체·외부 mirror로의 대체가 금지된다.
- 비-reference binary 제거는 source/license/hash/consumer manifest가 통과한 항목에만 적용된다.
- CLI 이름, JSON key/error code, model/data contract, hardware artifact lookup의 기존 동작을 보존한다.

# Architecture

세 개의 독립 트랙을 사용한다.

1. **Repair Track**: shadowed definition, import-time write, invalid type comment, broad exception처럼 작은 경계에서 안정적으로 고칠 수 있는 문제
2. **Refactor Track**: 현재 algorithm modality 구조를 유지한 bootstrap 정리, active algorithm legacy-name retirement, `src.pools` facade 제거, Quantization package/CLI, hardware validator, data/training owner 분리; Hybrid namespace migration은 조건부
3. **Cleanup Track**: references 영구 보존·비교 catalog, non-reference archive/binary 정리, 문서와 active 경로 migration

모든 트랙은 `maintained → test → analysis → artifact → archive/reference/vendor` 권위 순서를 따른다. Reference/vendor는 active runtime owner가 될 수 없으며, active 코드가 reference를 import하는 새 경로도 만들지 않는다.

# Tech Stack

- Python 3 stdlib AST, unittest/pytest
- setuptools/pyproject packaging
- C++ HLS, Tcl, JSON schema
- Git, SHA-256 manifest, repository-relative provenance
- Markdown/JSON/CSV 분석 산출물

# Baseline/Authority Refs

- Commit: `ebe862b11506e819c5bd5abc925299fc5fbb6f1a`
- `.agents/recon/2026-07-15-semantic-census/codebase-recon.json`
- `.agents/recon/2026-07-15-semantic-census/codebase-recon.md`
- `.agents/recon/2026-07-15-semantic-census/inventory/semantic-summary.json`
- `.agents/recon/2026-07-15-semantic-census/inventory/duplicate-blobs.csv`
- `.agents/recon/2026-07-15-semantic-census/import-scc-evidence.json`
- `docs/analysis/HBTXR-SEMANTIC-CENSUS-2026-07-15.md`
- `docs/analysis/HBTXR-REFACTOR-CLEANUP-MATRIX.md`
- `algorithm/README.md`
- `hardware/docs/architecture/DIRECTORY_LAYOUT.md`
- `references/README.md`
- `docs/provenance/handover-source-registry.json`
- 2026-07-21 user decision: active algorithm에서 reference/history를 제외한 `FACET/facet` 이름과 `algorithm/hybrid/src/pools`를 제거

BaselineUsageDraft:

- Required baseline refs: recon JSON/Markdown, semantic summary, decision matrix, package manifests, directory-layout docs
- Delivered context refs: current tracked tree at `ebe862b`
- Acknowledged before plan refs: all required refs above
- Cited in plan refs: all required refs above
- Missing refs: external import consumers and supported same-interpreter Hybrid/Quantization workflow evidence
- Decision: continue; references는 permanent keep, Hybrid namespace migration은 trigger가 없으면 NO-OP, active-name/`src.pools` retirement는 RC-120/RC-130 consumer baseline 후 실행, non-reference destructive cleanup만 별도 gated

# Compatibility Boundary

다음은 유지해야 한다.

- `train_hbtxr.py` CLI arguments, exit semantics, manifest/checkpoint validation
- `algorithm/{common,event,frame,hybrid}` top-level layout와 `algorithm/common/src/EvEye` 공개 import
- Hybrid의 `algorithm/hybrid/src` owner와 dependency-free `src.evaluation` behavior; RC-100C가 실제 선택·완료된 경우에만 승인된 compatibility plan 적용
- Quantization `hgpipe-quant` command, existing subcommand names, JSON/Markdown schema
- Hardware validation error codes, JSON key names, command text, dry-run behavior
- Dataset record/shape/coordinate contracts
- PYNQ bitstream/HWH lookup behavior until artifact locator migration completes
- `references/**`의 모든 경로·payload hash·source revision·license 상태·internal citation

다음은 2026-07-21 사용자 결정에 따라 의도적으로 호환 종료한다.

- active algorithm의 `common/scripts/facet`, `common/tests/facet`, `facet_main.py` 경로
- `FACET_DISABLE_CUDNN`, `FACET_DEVICES`, `FACET_CKPT_PATH`, `facet_square_direct`, `EventPatchEmbeddingFACET`
- RC-120이 제안했던 모든 `HBTXR_*` replacement environment variables
- `src.pools` import/export와 `algorithm/hybrid/src/pools` package

`references/**`, `algorithm/{analysis,archive}`, `algorithm/hybrid/hardware_reference`, provenance/history, 논문·원본 프로젝트 고유명, 정확한 `references/**/FACET` 입력 경로와 결과 파일명은 이 retirement의 대상이 아니다. `algorithm/hybrid/src/optim/pool.py`, `training.optimizer_pool`, optimizer pool report 계약도 별도 실험 기능이므로 유지한다.

다음은 호환 대상으로 간주하지 않는다.

- 가려져 런타임에서 호출될 수 없는 앞 함수 정의
- import 시 파일을 수정하는 부작용
- machine-local absolute path 기본값
- ignored local cache
- reference 밖의 source/license/hash가 확인되지 않은 불명확 복사본을 active owner로 승격하는 행위

# TDD Route

- Mode: off
- Decision: skipped
- Strict authority: not applicable
- Test posture: characterization baseline followed by post-change regression
- Reason: 사용자가 strict TDD를 요구하지 않았으며 현재 작업은 조사와 계획 작성이다. 실행 시 각 slice는 기존 동작을 먼저 fixture/snapshot으로 고정한다.
- Verification: focused unit/contract tests, import-safety checks, schema snapshots, full component regression

# Verification

계획 실행 시 공통 검증 순서는 다음과 같다.

```bash
python3 .agents/recon/2026-07-15-semantic-census/semantic_scan.py \
  --root . \
  --commit HEAD \
  --out .agents/recon/2026-07-15-semantic-census/inventory-next
python3 -m pytest -q algorithm/hybrid/tests
(cd quantization && python3 -m unittest discover -s tests -p 'test_*.py')
python3 -m unittest discover -s hardware/tests -p 'test_*.py'
python3 -m unittest discover -s tests/provenance -p 'test_*.py'
git diff --check
```

Hardware synthesis, training, dataset generation, board execution은 구조적 리팩터링의 기본 검증에 포함하지 않는다. 해당 동작을 변경하는 별도 계획에서만 실행한다.

## Scope Check

### Aegis Visibility

Package owner, CLI contract, validation owner, permanent reference retention 경계가 동시에 영향을 받으므로 실행 전에 현재 ownership, 조건부 migration trigger, 보존 불변식을 고정해야 한다.

### Plan Basis

- Fact: 전체 tracked tree와 Python 심볼/라인이 전수 인벤토리됐다.
- Fact: shadowed definition, import-time write, monolithic validator가 line evidence로 확인됐다.
- Fact: reference/vendor가 파일·라인·bytes의 대부분을 차지한다.
- Fact: `algorithm/event`와 `algorithm/frame`은 modality별 config/실험 표면이고, 설치되는 algorithm package owner는 `algorithm/common/src/EvEye`다.
- Inference: 두 `src` namespace와 path injection은 동일 interpreter에서 잘못된 import를 유발할 수 있지만 지원 workflow에서 재현되지 않았다.
- User decision: `references/**`는 비교 실험을 위해 영구 보존한다.
- User decision: active algorithm에서는 reference/history를 제외한 `FACET/facet` 이름과 `src.pools` package를 제거하며, 제거된 control을 `HBTXR_*` 이름으로 다시 만들지 않는다.
- Unknown: 외부 consumer가 현재 `src` import와 binary path를 사용하는 범위, Hybrid/Quantization 동시 프로세스 요구 여부

### Requirement Ready Check

- Requirement source refs: 사용자 요청과 semantic census
- Goals and scope refs: 메인 HBTXR만 분석하고 refactor/cleanup 계획 작성
- User/scenario refs: 현재 브랜치를 유지하고 후속 실행 가능한 계획 필요
- Requirement item refs: function/class/line census, refactor plan, cleanup plan
- Acceptance refs: exact evidence, compatibility, verification, retirement gates
- Open blocker questions: Hybrid namespace migration을 요구하는 지원 workflow와 재현 가능한 충돌 증거는 없음; repository 밖 old FACET/`src.pools` consumer는 알 수 없음
- Decision: references 보존 정책과 current algorithm layout은 ready; RC-120/RC-130은 baseline consumer map 후 gated; Quantization migration은 consumer map 후 gated; Hybrid migration은 trigger 없으면 NO-OP; non-reference 삭제만 별도 승인

### Change Necessity

- User-visible need: 안전하게 유지보수하고 저장소 규모·구조 부채를 줄인다.
- No-change/non-code option: algorithm layout과 references는 no-change가 정답이지만 shadowed definition, import side effect, Quantization public `src` package는 남는다.
- Why code change is necessary: 확정된 correctness/side-effect 문제와 Quantization packaging owner를 최소 범위로 수정해야 한다. Hybrid namespace 변경 필요성은 아직 입증되지 않았다.
- Minimum change boundary: RF-001~RF-003, Hybrid bootstrap contract, active legacy-name contracts, `src.pools` facade, Quantization-only package boundary
- Decision: selective code-change; algorithm top-level layout/`EvEye`/reference payload는 no-change, active naming과 `src.pools`는 retirement, Hybrid namespace migration은 conditional

### Existence Check

- Proposed new surface: `algorithm/hybrid/src/optim/lr_schedulers.py`, named Quantization package, hardware shared validation modules, repository/reference asset manifest
- Existing reuse candidate: `models/tracker/head_factory.py`, `loss/stage*.py`, `optim/registry.py`, `optim/pool.py`, `models/controller.py`, `tools/provenance`, component-local tests, current JSON schemas
- Why existing surface is insufficient: LR scheduler만 `src.pools` 아래 유일 owner이고 trainer가 직접 소비한다. 나머지 pool registry는 maintained consumer가 없는 facade이며 이미 domain owner가 있다. Quantization의 installable public package가 일반명 `src`이고 giant validator에는 pure rule owner가 없다.
- Creation proof: LR scheduler를 `optim` domain으로 이동해야 `src.pools`를 제거할 수 있다. 미사용 head/loss/optimizer/runtime registry를 새 package로 재생성할 근거는 없다.
- Entropy/retirement impact: `src.pools`와 old active names에는 permanent shim을 두지 않고, Quantization compatibility shim에만 별도 제거 조건을 둔다.
- Decision: add-with-proof for `optim/lr_schedulers.py`, Quantization package and hardware rule modules; reuse existing domain owners and Hybrid `src` root

### Architecture Integrity Lens

- Invariant: active owner는 reference/vendor를 runtime dependency로 사용하지 않는다.
- Canonical owner: `algorithm/common/src/EvEye`, modality surfaces `event/frame/hybrid`, current `algorithm/hybrid/src`, domain-local model/loss/optim/runtime owners, named Quantization package, hardware HLS/tools, provenance validator
- Responsibility overlap: `src.pools` facade, CLI registration, validation/rendering, data assembly, archive copies
- Higher-level simplification: algorithm layout은 유지하고 active legacy names와 중복 pool facade를 제거하며 bootstrap contract와 Quantization package만 분리한다. Reference size/중복은 삭제 근거가 아니다.
- Retirement falsifier: Hybrid collision/distribution trigger가 없으면 RC-100C는 NO-OP; references는 어떤 trigger에서도 이 workstream의 retirement 대상이 아니다.
- Verdict: repair 먼저, current owner 고정, Quantization migration, 조건부 Hybrid 판단, non-reference cleanup 순서

### Plan Pressure Test

- Owner/contract/retirement: 각 task에 canonical owner와 제거 조건 지정; RC-120/RC-130은 old-name shim 없이 최종 제거
- Architecture integrity: package와 validation 경계에서 해결
- Verification scope: component tests+schema snapshot+recon delta
- Task executability: RC-000/010/020/030과 CL-000만 현재 copy/paste 실행 가능한 task다. RC-100A/100B/100C와 RC-200/210/220/300은 각 readiness·trigger와 별도 slice plan 검토가 필요한 gated/conditional epic이다.
- Pressure result: ready tasks만 proceed; gated epic은 subordinate plan review 전 실행 금지

### Plan-Time Complexity Check

| Artifact class | 대상 | 현재 압력 | 권고 |
|---|---|---|---|
| algorithm owner | `common/event/frame/hybrid` | 이미 명확 | 현재 layout, `EvEye`, Hybrid `src` root 유지 |
| algorithm naming | active `facet` paths/contracts | reference identity와 runtime 이름 혼재 | active-owned 이름은 existing config/semantic option으로 흡수하고 product-prefix 재명명 금지; reference/history allowlist 보존 |
| Hybrid component facade | `src/pools` 6 modules, 331 LOC | maintained consumer는 LR scheduler 한 곳 | scheduler만 `optim`으로 이동, 미사용 facade 제거, `optim/pool.py` 유지 |
| quantization owner | installable `src` | public import ambiguity | `hgpipe_quantization`과 임시 Quantization shim |
| CLI | `quantization/src/cli.py` | 4중 parser/main wrapper | command registration modules |
| validator | hardware 4개 함수 | 908~1,954 LOC | pure rule/collector/render split |
| data/training | EventInputBuilder/TrainingSession | 409~514 LOC | contract/assembly/orchestration split |
| repository | references 1GB+ 및 binary | 비교 자산과 배포 artifact 혼재 | references permanent catalog; binary만 manifest-first externalization |

Projected result: algorithm 전체 namespace rename은 불필요하고 over-budget다. 현재 Hybrid owner를 유지하고 active legacy names와 `src.pools`만 제거하며, LR scheduler·Quantization·validator에만 증명된 새 owner를 만든다.

## Execution Readiness View

- Intent Lock: current algorithm layout와 reference payload 보존, 최소 동작 보존형 리팩터링
- Scope Fence: 메인 HBTXR만 수정; `references/**` destructive change 금지; non-reference external archive/push/delete는 별도 승인
- Baseline Lock: `ebe862b` recon과 current tests
- Approved Behavior: public CLI/schema/data contract 유지
- Owner Constraints: active owner는 maintained tree에만 존재
- Compatibility Boundary: Hybrid current `src` root import 유지; active FACET names와 `src.pools`는 명시적 breaking retirement; Quantization old import/CLI와 binary paths는 측정된 consumer가 남는 동안 shim/locator 유지
- Retirement Boundary: references는 retirement 없음; non-reference old owner만 usage=0, regression pass, manifest verification 후 제거
- Task Batches: A repair → B active-name/`src.pools` retirement → C ownership/Quantization → D conditional Hybrid decision+decomposition → E preservation/non-reference cleanup → F documentation
- Test Obligations: focused characterization, component regression, schema snapshots
- Review Gates: 각 batch 후 code review와 semantic delta
- Drift/Rewind Rules: schema/key/CLI diff 발생 시 해당 slice revert; 다음 batch 진행 금지
- Evidence Required: test logs, diff, consumer inventory, hash manifest, recon delta
- Advisory Boundary: 이 계획은 실행 지침이며 삭제/완료 권한은 아니다.

### Batch readiness

| 상태 | 작업 | 해제 조건 |
|---|---|---|
| READY | RC-000, RC-010, RC-020, RC-030, CL-000 | 사용자 구현 승인 |
| GATED | RC-100A | RC-030 owner map과 bootstrap characterization |
| GATED EPIC | RC-120 | RC-000/RC-030 consumer map, reference/history allowlist, exact slice review |
| GATED EPIC | RC-130 | RC-030/RC-100A consumer map, LR scheduler characterization, exact slice review |
| GATED EPIC | RC-100B, RC-210, RC-300 | exact inventory와 task별 subordinate slice plan 승인 |
| CONDITIONAL | RC-100C | supported co-install/distribution trigger, reproducible collision, 별도 승인; 미충족 시 NO-OP |
| GATED | RC-110, RC-220 | RC-100A/B 완료와 exact owner/test allowlist |
| GATED EPIC | RC-200 | RC-010과 RC-020 완료, golden diagnostics와 rule-cluster slice plan 승인 |
| GATED | CL-100, CL-110, CL-120, CL-130 | CL-000 manifest와 consumer audit 완료; CL-100/120은 references metadata only |
| CANCELLED | CL-101, CL-102, CL-121 | references permanent-retention 사용자 결정으로 실행 금지 |
| BLOCKED | CL-111, CL-112, CL-131 | non-reference binary 전송·삭제 및 active/docs migration 별도 승인 |
| GATED | RC-900 | `docs/track/refactor-cleanup-selected-batch.json`에 선택된 task가 모두 완료 |

## Authoritative dependency table

이 표가 유일한 dependency source다. Task header는 이 표와 같아야 한다.

| Task | Dependencies | Readiness |
|---|---|---|
| RC-000 | — | READY |
| RC-010, RC-020, RC-030, CL-000 | RC-000 | READY after baseline |
| RC-100A | RC-030 | GATED minimal Hybrid bootstrap repair; no rename |
| RC-100B | RC-030, RC-100A | GATED EPIC Quantization-only migration |
| RC-100C | RC-100B + trigger evidence + explicit approval | CONDITIONAL; default NO-OP |
| RC-120 | RC-000, RC-030 | GATED EPIC active algorithm legacy-name retirement |
| RC-130 | RC-030, RC-100A | GATED EPIC `src.pools` retirement; keep `src/optim/pool.py` |
| RC-110 | RC-010, RC-100B | GATED |
| RC-200 | RC-010, RC-020 | GATED EPIC |
| RC-210 | RC-100A, RC-130 | GATED EPIC; current Hybrid paths; serialize `trainer.py` ownership after pools retirement |
| RC-220 | RC-100A, RC-100B | GATED; current Hybrid and named Quantization paths |
| RC-300 | RC-100A, RC-200 | GATED EPIC; current Hybrid owner |
| CL-100, CL-110, CL-120, CL-130 | CL-000 | GATED, non-destructive only |
| CL-101, CL-102 | — | CANCELLED; references stay in repository |
| CL-111 | CL-110 | BLOCKED transfer |
| CL-112 | CL-111 | BLOCKED deletion |
| CL-121 | — | CANCELLED for reference copies |
| CL-131 | RC-100A, RC-100B, CL-110, CL-130; add RC-100C only when selected | BLOCKED active/docs-only migration |
| RC-900 | every task ID in validated `dependency_closure` from `docs/track/refactor-cleanup-selected-batch.json` | GATED per selected batch |

## Task RC-000: Freeze the behavioral and semantic baseline

**Files**

- Verify: `.agents/recon/2026-07-15-semantic-census/**`
- Create during execution: `docs/track/refactor-cleanup-baseline.md`
- Create during execution: `docs/track/refactor-cleanup-task-catalog.json`
- Create during execution: `docs/schemas/refactor-cleanup-selected-batch.schema.json`
- Create during execution: `docs/track/refactor-cleanup-selected-batch.json`
- Create during execution: `tools/validation/validate_refactor_cleanup_batch.py`
- Create during execution: `tests/validation/__init__.py`
- Create during execution: `tests/validation/test_refactor_cleanup_batch.py`

**Selected-batch schema**

- `schema_version`: literal `hbtxr-refactor-cleanup-batch.v1`
- `batch_id`: repository-unique lowercase slug
- `baseline_commit`: exact 40-hex baseline
- `requested_tasks`: unique task IDs requested for this batch
- `dependency_closure`: unique requested tasks plus every transitive dependency
- `tasks`: object whose keys exactly equal `dependency_closure`; each value contains `readiness`, `state`, `gate_refs`, `approval_refs`, and `evidence_paths`
- `readiness`: exact catalog enum `READY`, `GATED`, `GATED_EPIC`, `CONDITIONAL`, `BLOCKED`, or `CANCELLED`
- `disposition`: `execute`, `no-op`, or `cancelled`; CONDITIONAL task는 trigger가 없으면 `no-op`, CANCELLED task는 항상 `cancelled`
- `state`: `planned` at selection or `completed` at RC-900
- `trigger_refs`: CONDITIONAL task를 `execute`할 때 필요한 existing evidence/ADR paths
- `gate_refs`: existing evidence paths required for GATED/GATED_EPIC tasks
- `approval_refs`: existing approval-record paths required for every BLOCKED task
- `evidence_paths`: existing completion evidence paths; required and non-empty in completion phase
- path invariant: validator는 `references/` prefix를 delete, rename, move, external replacement, deduplicate, retirement allowlist에 포함한 batch를 항상 거부한다.

`docs/track/refactor-cleanup-task-catalog.json` mirrors the authoritative dependency table with `task_id`, `dependencies`, and `readiness`. The validator rejects catalog/plan drift through exact expected task IDs and dependency assertions in its unit test.

**Why**

구조 변경과 cleanup의 회귀 여부를 같은 기준으로 판단해야 한다.

**Change Necessity**

Runtime source 변경은 없다. Baseline evidence와 fail-closed batch governance validator만 추가한다.

**Impact/Compatibility**

없음.

**Steps**

1. `git rev-parse HEAD`가 계획의 baseline 또는 승인된 후속 commit인지 기록한다.
2. Dependency table을 task catalog와 JSON schema로 고정한다.
3. Validator negative tests에 unknown ID, missing transitive dependency, readiness mismatch, missing gate ref, unapproved BLOCKED task, selected CANCELLED task, trigger 없는 CONDITIONAL execution, destructive `references/**` path, baseline mismatch, incomplete/missing evidence를 각각 추가한다.
4. Selection manifest의 `dependency_closure`를 계산하고 `--phase selection`으로 검증한다.
5. 네 component test command의 현재 결과와 실패 목록을 기록한다.
6. Quantization CLI help, hardware validator fixture JSON, hybrid import 경로를 snapshot한다.
7. Semantic scanner를 재실행하고 summary hash를 기록한다.
8. Commit: `docs(refactor): freeze semantic cleanup baseline`.

**Verification**

```bash
python3 -m unittest discover -s tests/validation -p 'test_refactor_cleanup_batch.py'
python3 tools/validation/validate_refactor_cleanup_batch.py \
  --phase selection \
  --catalog docs/track/refactor-cleanup-task-catalog.json \
  --schema docs/schemas/refactor-cleanup-selected-batch.schema.json \
  --batch docs/track/refactor-cleanup-selected-batch.json \
  --expected-baseline ebe862b11506e819c5bd5abc925299fc5fbb6f1a
sha256sum .agents/recon/2026-07-15-semantic-census/inventory/*.csv
git diff --check
```

Expected: invalid selection fixtures are non-zero; valid dependency closure passes; every inventory hash와 known baseline failure가 문서에 고정된다.

## Task RC-010: Retire shadowed artifact-patch definitions

**Dependencies**: RC-000

**Files**

- Modify: `quantization/src/artifact_patch.py`
- Modify: `quantization/tests/test_artifact_patch.py`
- Verify: `quantization/tests/test_cli.py`, `test_package_api.py`

**Why**

가려진 구현은 유지보수자가 실제 실행되는 정의를 오판하게 만든다.

**Change Necessity**

문서로는 Python name shadow를 제거할 수 없다. 최소 경계는 중복된 두 함수와 해당 contract tests다.

**Repair Track**

- Canonical owner: 뒤쪽 `:446`, `:663` 구현
- Characterization: 정상/누락/불일치 manifest의 dict key, status, markdown text
- Repair: 앞 정의를 제거하고 공통 helper가 필요하면 뒤 구현 근처에 private 함수로 추출

**Retirement Track**

- 제거 대상: `:240-330`, `:331-445`의 가려진 public 구현
- Keep 조건: 앞 구현의 schema가 외부 문서에서 요구됨이 확인되면 별도 versioned API로 명명
- Delete 조건: tests와 consumer search에서 앞 behavior 요구가 없음

**Steps**

1. 두 정의를 독립 이름으로 임시 로드하는 characterization fixture를 작성한다.
2. output key/status/message 차이를 명시한다.
3. canonical behavior를 현재 runtime export와 일치시키고 앞 정의를 제거한다.
4. package API와 CLI regression을 실행한다.
5. Commit: `refactor(quantization): remove shadowed artifact patch definitions`.

**Verification**

```bash
(cd quantization && python3 -m unittest tests.test_artifact_patch tests.test_package_api tests.test_cli)
rg -n '^def (validate_artifact_patch_matrix_manifest|write_artifact_patch_matrix_manifest_validation_markdown)' quantization/src/artifact_patch.py
```

Expected: 각 public function 정의가 정확히 1개이고 tests가 통과한다.

## Task RC-020: Make dated update scripts import-safe and fix type comments

**Dependencies**: RC-000

**Files**

- Modify: `hardware/tools/update_e2e_full_docs_2026_06_09.py`
- Modify: `hardware/tools/update_e2e_full_vit_2026_06_09.py`
- Create: `hardware/tests/test_update_scripts_import_safety.py`
- Modify: `algorithm/common/src/EvEye/model/DavisEyeEllipse/EPNet/Predict.py`
- Modify: `algorithm/common/src/EvEye/model/DavisEyeEllipse/HBTXR/Predict.py`

**Why**

Import는 파일 시스템을 변경하면 안 되며 invalid type comment는 정적 도구를 방해한다.

**Change Necessity**

최소 변경은 top-level mutation을 `main(argv)`로 이동하고 type comment를 일반 설명으로 바꾸는 것이다.

**Repair Track**

- `build_updates(root) -> list[PlannedWrite]`
- `apply_updates(planned, *, dry_run: bool) -> list[Path]`
- `main(argv: Sequence[str] | None = None) -> int`
- import 시 함수/상수 선언 외 동작 금지

**Retirement Track**

- top-level read/write block 제거
- dated script가 완료된 migration이면 `hardware/archive/migration-logs`로 이동하는 후속 판단

**Steps**

1. 임시 directory를 감시하는 import-only test를 추가한다.
2. 기존 output text와 대상 파일 목록을 fixture로 고정한다.
3. I/O를 `main()` 아래로 이동한다. 직접 실행의 기존 write 동작은 보존하고, 명시적 `--dry-run`에서만 preview를 수행한다.
4. 두 invalid type comment를 `# Convert np.ndarray to torch.Tensor.`로 교체한다.
5. AST type-comment parsing과 hardware tests를 실행한다.
6. Commit: `fix(hardware): make update tools import safe`.

**Verification**

```bash
python3 -m unittest hardware.tests.test_update_scripts_import_safety
python3 - <<'PY'
import ast
from pathlib import Path
for path in Path('algorithm/common/src').rglob('Predict.py'):
    ast.parse(path.read_text(encoding='utf-8'), type_comments=True)
PY
```

Expected: import 전후 파일 목록/hash 동일, invalid type comment 0건.

## Task RC-030: Inventory package and CLI consumers

**Dependencies**: RC-000

**Files**

- Create: `docs/analysis/HBTXR-PACKAGE-CONSUMER-MAP.md`
- Create: `docs/track/ADR-package-namespaces.md`

**Why**

현재 owner를 바꾸기 전에 내부·외부 compatibility boundary와 실제 same-interpreter 요구를 측정해야 한다.

**Change Necessity**

이 task는 docs-only다. 기본 결정은 `PRESERVE_CURRENT_LAYOUT`이며 조사 없이 package move를 승인하지 않는다.

**Steps**

1. tracked Python의 `from src`, `import src`, `PYTHONPATH`, `sys.path`를 목록화한다.
2. pyproject entry points와 shell scripts를 연결한다.
3. 문서/CI/external handoff에서 복사되는 import command를 분리한다.
4. `algorithm/common/src/EvEye`가 설치 owner이고 `event/frame`이 config·실험 surface이며 `hybrid/src`가 현재 Hybrid 구현 owner임을 기록한다.
5. 지원 workflow가 Hybrid와 Quantization을 같은 interpreter에서 요구하는지 `SUPPORTED`, `UNSUPPORTED`, `UNKNOWN`으로 분류한다.
6. Quantization migration은 별도 결정으로 기록하고 Hybrid는 `KEEP_CURRENT`를 기본값으로 둔다.
7. Conditional migration trigger, shim 유지 조건, usage 계측 방법을 ADR에 기록한다.
8. Commit: `docs(architecture): map package namespace consumers`.

**Verification**

```bash
rg -n '(^|\s)(from|import) src|sys\.path|PYTHONPATH' algorithm quantization hardware tools tests
rg -n '\[project\.scripts\]|entry-points|console_scripts' --glob 'pyproject.toml' --glob 'setup.py' .
```

Expected: 모든 current import/entry point가 owner와 supported workflow를 가지며, algorithm layout 결정은 `PRESERVE_CURRENT_LAYOUT`, Hybrid 결정은 `KEEP_CURRENT`, Quantization 결정은 별도 migration gate를 가진다.

## Task RC-100A: Preserve algorithm modality layout and repair only the Hybrid bootstrap contract

**Dependencies**: RC-030

**Status**: GATED minimal repair. `algorithm/{common,event,frame,hybrid}`와 `algorithm/hybrid/src` rename/move는 금지한다.

**Files**

- Modify: `algorithm/hybrid/scripts/external_pipeline/_bootstrap.py`
- Create: `algorithm/hybrid/tests/external_pipeline/test_bootstrap_import_contract.py`
- Modify: `docs/analysis/HBTXR-PACKAGE-CONSUMER-MAP.md`
- Modify: `docs/track/ADR-package-namespaces.md`

**Why**

상위 algorithm 구조와 Hybrid owner는 이미 명확하다. 필요한 최소 변경은 direct script가 current `algorithm/hybrid/src/__init__.py`를 안정적으로 찾는지 고정하는 것이다.

**Change Necessity**

현재 bootstrap은 `src` package의 parent가 아니라 `src` directory 자체를 주입한다. Direct-entry characterization이 실패할 때만 `project_root`를 주입하도록 최소 수정한다. 실패하지 않으면 docs-only NO-OP으로 닫는다.

**Impact/Compatibility**

- `algorithm/common`, `event`, `frame`, `hybrid` 경로 유지
- `EvEye`와 Hybrid `from src...` import 유지
- Hybrid `pyproject.toml`이나 새 `hbtxr_hybrid` package를 만들지 않음

**Steps**

1. Repo-root와 `algorithm/hybrid` working directory에서 bootstrap 후 `src.__file__`을 snapshot한다.
2. Expected path를 `algorithm/hybrid/src/__init__.py`로 고정하고 Quantization `src` 선택 시 실패하게 한다.
3. Characterization이 실패할 때만 `_bootstrap.py`가 `project_root`를 `sys.path`에 넣도록 수정한다.
4. Consumer map과 ADR에 `PRESERVE_CURRENT_LAYOUT`과 결과를 기록한다.
5. Commit: `fix(hybrid): stabilize current src bootstrap` 또는 실패가 없으면 docs-only `NO-OP` 기록.

**Verification**

```bash
PYTHONPATH=algorithm/hybrid python3 -c 'import pathlib, src; assert pathlib.Path(src.__file__).resolve() == pathlib.Path("algorithm/hybrid/src/__init__.py").resolve()'
python3 -m pytest -q algorithm/hybrid/tests/external_pipeline/test_bootstrap_import_contract.py
git diff --name-status -- algorithm/common algorithm/event algorithm/frame algorithm/hybrid/src
```

Expected: expected Hybrid owner resolves; 마지막 command에 rename/delete가 없고 RC-100A가 승인한 bootstrap 외 source 이동은 0건.

## Task RC-100B: Introduce a Quantization-only named package

**Dependencies**: RC-030, RC-100A

**Status**: GATED EPIC. `docs/aegis/plans/rc-100b-quantization-package-slices.md`에서 exact move/export/shim slice를 검토한 뒤 실행한다.

**Files**

- Create: `quantization/hgpipe_quantization/`
- Modify: `quantization/pyproject.toml`
- Keep temporarily: `quantization/src/` source-checkout compatibility shim
- Create: `tests/packaging/test_named_package_discovery.py`
- Modify: Quantization paths listed in `docs/analysis/HBTXR-PACKAGE-CONSUMER-MAP.md`

**Why**

Installable project `hgpipe-quantization`이 public package `src`를 배포하는 ownership 불일치를 Quantization 경계 안에서 해결한다.

**Impact/Compatibility**

- `hgpipe-quant` executable/subcommand/schema 유지
- Built distribution은 `hgpipe_quantization*`만 포함하고 `src*` 제외
- Hybrid/algorithm 경로와 import는 변경하지 않음

**Steps**

1. Consumer map의 Quantization import/export allowlist를 확정한다.
2. `quantization/src` implementation을 `hgpipe_quantization` owner로 이동하고 public export parity를 검사한다.
3. `hgpipe-quant = "hgpipe_quantization.cli:main"`으로 entry point를 바꾼다.
4. Old `quantization/src`는 source-checkout-only re-export shim으로 축소한다.
5. Package discovery와 Quantization component regression을 실행한다.
6. Commit: `refactor(quantization): name the installable package owner`.

**Verification**

```bash
PYTHONPATH=algorithm/hybrid:quantization python3 -c 'import src, hgpipe_quantization; print(src.__file__, hgpipe_quantization.__file__)'
python3 -m unittest discover -s tests/packaging -p 'test_named_package_discovery.py'
(cd quantization && python3 -m unittest discover -s tests -p 'test_*.py')
```

Expected: `src` resolves to Hybrid under the combined diagnostic path, `hgpipe_quantization` resolves independently, distribution discovery contains no `src` package, CLI contract remains identical.

## Task RC-100C: Conditionally migrate the Hybrid namespace

**Dependencies**: RC-100A, RC-100B plus trigger evidence and explicit approval

**Status**: CONDITIONAL/BLOCKED. Default disposition is `NO-OP`.

**Trigger requirements**

1. A documented supported workflow requires Hybrid and another `src` owner in one interpreter or requires Hybrid as an installable public distribution.
2. The failure is reproduced in a clean subprocess through official entrypoints, not an arbitrary `PYTHONPATH` ordering experiment.
3. Evidence records command, `sys.path`, expected/actual `src.__file__`, error, consumer list, compatibility window, and rollback.
4. `docs/track/ADR-package-namespaces.md` and the user explicitly approve migration.

If all triggers pass, create and review `docs/aegis/plans/rc-100c-hybrid-package-migration-slices.md`. This parent plan does not authorize source moves. If any trigger is absent, record `disposition=no-op`; do not create `hbtxr_hybrid`, shims, or a Hybrid pyproject.

## Task RC-110: Replace the quantization CLI wrapper chain

**Dependencies**: RC-010, RC-100B

**Files**

- Modify: `quantization/hgpipe_quantization/cli.py`
- Create: `quantization/hgpipe_quantization/commands/{verify,artifact,calibration,audit}.py`
- Modify: `quantization/tests/test_cli.py`

**Why**

네 번 재정의되는 `build_parser/main`의 순서 결합을 제거한다.

**Architecture**

각 command module은 `register(subparsers) -> None`과 `run(args) -> int`만 노출한다. `cli.py`는 parser 생성, dispatch, exit code만 소유한다.

**Steps**

1. Current `--help`, subcommand list, defaults, exit codes를 snapshot한다.
2. Command별 parser registration을 네 모듈로 이동한다.
3. 단일 `build_parser()`와 `main(argv=None)`을 작성한다.
4. Old wrapper alias와 `_original_*` chain을 제거한다.
5. CLI/package tests와 snapshot diff를 실행한다.
6. Commit: `refactor(quantization): register cli commands explicitly`.

**Retirement Track**

- Remove: chained `build_parser/main` definitions
- Keep: executable name와 public `main`
- Delete trigger: parser snapshot identical except approved ordering

**Verification**

```bash
(cd quantization && python3 -m unittest tests.test_cli tests.test_lut_calibration tests.test_quantization_scheme tests.test_completion_audit)
rg -n '^def (build_parser|main)' quantization/hgpipe_quantization/cli.py
```

Expected: 각 정의 1개, CLI regression pass.

## Task RC-120: Retire active algorithm FACET names while preserving reference identity

**Dependencies**: RC-000, RC-030

**Status**: GATED EPIC. 먼저 active/reference/history match를 line-level allowlist로 고정하고 아래 세 slice를 순서대로 검토·실행한다. 이 task는 `EvEye` package나 modality directory를 rename하지 않는다.

**Files**

- Create during execution: `docs/analysis/HBTXR-ALGORITHM-LEGACY-NAME-CONSUMER-MAP.md`
- Create during execution: `docs/provenance/algorithm-legacy-name-allowlist.tsv`
- Create during execution: `tools/validation/validate_algorithm_legacy_names.py`
- Create during execution: `tests/validation/test_algorithm_legacy_names.py`
- Create during execution: `algorithm/common/tests/test_runtime_control_contracts.py`
- Move: `algorithm/common/scripts/facet/{train,validate,validate10times,inference}.py` to `algorithm/common/scripts/`
- Move: `algorithm/common/scripts/facet/inference_visualize.ipynb` to `algorithm/common/scripts/`
- Move: all 12 tracked files under `algorithm/common/tests/facet/` to `algorithm/common/tests/`
- Move: `algorithm/common/scripts/facet_main.py` to `algorithm/common/scripts/sagemaker_launcher.py`
- Modify: the moved train/validate scripts and the exact active-name files listed below
- Modify: current operational docs that point to moved paths or active runtime names
- Preserve: `algorithm/common/src/EvEye`, all `EvEye.*` imports, reference/history zones and source-faithful reference literals

**Frozen reference/history boundary**

- `references/**`
- `algorithm/analysis/**`
- `algorithm/archive/**`
- `algorithm/artifacts/**`
- `algorithm/hybrid/hardware_reference/**`
- `algorithm/**/docs/analysis/**`
- `algorithm/**/tests/handover/**`
- `algorithm/common/PROVENANCE.md`의 legacy source names와 `algorithm/docs/track/**`
- 논문·원본 프로젝트 고유명, `references/**/FACET` 입력 경로와 FACET-named reference artifact filenames

Frozen zone은 승인된 baseline commit `ebe862b11506e819c5bd5abc925299fc5fbb6f1a` 대비 content/path/blob diff와 untracked addition이 0이어야 한다. Maintained Python/YAML 안의 reference literal은 `algorithm-legacy-name-allowlist.tsv`에 `path`, `match_text`, `expected_count`, `category`, `reason`을 기록한 경우에만 허용한다. Validator는 tracked text의 path/token count와 frozen-zone baseline path/blob hashes를 확인하고 어떤 drift도 fail closed한다. Active temporary path, environment variable, class/API name, resize policy, output contract는 allowlist 대상이 아니다.

**Canonical replacements**

| Before | After |
|---|---|
| `common/scripts/facet/*` | `common/scripts/*` |
| `common/tests/facet/*` | `common/tests/*` |
| `common/scripts/facet_main.py` | `common/scripts/sagemaker_launcher.py` |
| common train/validate `FACET_DISABLE_CUDNN` | environment override 제거; existing `runtime.disable_cudnn` 사용 |
| common train/validate `FACET_DEVICES` | environment override 제거; existing `trainer.devices` 사용 |
| common train `FACET_CKPT_PATH` | environment override 제거; existing `train.ckpt_path` 사용 |
| config-driven evaluator `FACET_DISABLE_CUDNN` | environment override 제거; loaded config의 existing `runtime.disable_cudnn` 사용 |
| standalone dataset builder `FACET_DISABLE_CUDNN` | environment override 제거; semantic `--disable-cudnn` CLI option 사용 |
| `facet_square_direct` | `roi_square_direct` |
| `EventPatchEmbeddingFACET` | `EventPatchEmbedding` |
| `/tmp/facet_unet_dataset_smoke` | `/tmp/unet_dataset_smoke` |

Old-name compatibility alias와 새 product-prefixed replacement environment variable은 최종 상태에 남기지 않는다. 이 결정은 직전 계획의 `FACET_* → HBTXR_*` replacement만 취소한다. 기존 HBTXR project/model/class/config identity와 기존 test/reference controls를 전역 rename하는 작업은 RC-120 범위가 아니다. RC-030/consumer map에서 repository 밖 required consumer가 확인되면 해당 slice를 중단하고 별도 compatibility 결정을 요청한다.

**Slice A — common path flattening**

1. Source/target basename collision이 0임을 다시 확인한다.
2. Baseline matches를 manifest에 고정하고 validator의 allowed/excluded/new-match/count-drift cases를 unit test한다.
3. 5개 scripts와 12개 tests를 `git mv`로 평탄화한다.
4. SageMaker launcher를 역할 기반 이름으로 이동한다.
5. `algorithm/docs/Validation.md`, `algorithm/docs/Execution.md`, `algorithm/docs/Modality-Reorganization.md`의 current path를 갱신한다.
6. `algorithm/common/PROVENANCE.md`는 historical snapshot으로 그대로 보존한다. 새 path 결정은 ADR-009와 current operational docs에만 기록한다.
7. Commit: `refactor(algorithm): flatten common scripts and tests`.

**Slice B — common active identifiers**

1. Moved `train.py`, `validate.py`, `validate10times.py`에서 FACET environment lookup을 제거하고 existing `runtime.disable_cudnn`, `trainer.devices`, `train.ckpt_path`만 읽는다.
2. `algorithm/common/src/EvEye/utils/scripts/build_full_dean_dataset_with_unet.py`는 `env_flag()`와 environment lookup을 제거하고 parser에 `--disable-cudnn`을 추가한다. `configure_torch_backend(disable_cudnn: bool)`로 명시적으로 전달하고 manifest key를 `disable_cudnn`으로 변경한다.
3. `algorithm/common/src/EvEye/utils/scripts/evaluate_epnet_checkpoint.py`는 existing `--config`를 backend 설정 전에 load하고 `runtime.disable_cudnn`을 사용한다. 새 environment variable이나 CLI option을 만들지 않는다.
4. `algorithm/common/tests/test_runtime_control_contracts.py`에서 config precedence, devices parsing, checkpoint forwarding, evaluator backend config, builder `--disable-cudnn`, `disable_cudnn` metadata를 characterization한다.
5. 나머지 reference-coupled utility의 source-faithful FACET literals는 line-level allowlist로만 보존한다.
6. Current operational docs에서 old environment invocation을 제거하고 canonical config keys 및 CLI option을 기록한다.
7. Commit: `refactor(algorithm): replace legacy environment overrides with explicit controls`.

**Slice C — Hybrid active runtime names**

1. `facet_square_direct`를 다음 exact files에서 `roi_square_direct`로 변경한다.
   - `algorithm/hybrid/src/config/runtime_config.py`
   - `algorithm/hybrid/src/data/dataset.py`
   - `algorithm/hybrid/src/data/transform.py`
   - `algorithm/hybrid/src/preprocess/build_manifests.py`
   - `algorithm/hybrid/scripts/external_pipeline/prepare_ev_eye.py`
   - `algorithm/hybrid/scripts/external_pipeline/build_groundedsam_dataset.py`
   - `algorithm/hybrid/configs/external/base.yaml`
   - `algorithm/hybrid/tests/external_pipeline/{test_dataset,test_interpolation,test_output_contract_snapshot,test_preprocess,test_runtime_e2e,test_script_workflows,test_target_fps_dataset,test_train_pipeline}.py`
   - `algorithm/event/docs/exp/README.md`
2. `algorithm/hybrid/src/models/patch_embeddings.py`의 public class/export를 `EventPatchEmbedding`으로 변경한다.
3. `algorithm/frame/configs/DavisEyeEllipse_RGBUNet_local_train_smoke.yaml`의 active `/tmp/facet_*` path를 `/tmp/unet_dataset_smoke`로 변경한다.
4. Reference-root YAML values와 analysis/history 문구는 수정하지 않는다.
5. Spatial transform matrix, output shape, interpolation, model state-dict key set이 이름 변경 전후 동일한지 characterization한다.
6. Commit: `refactor(hybrid): retire active branded contracts`.

**Verification**

```bash
python3 -m compileall -q algorithm/common/scripts algorithm/common/tests algorithm/common/src/EvEye algorithm/hybrid/src algorithm/hybrid/scripts
python3 -m unittest tests.validation.test_algorithm_legacy_names
python3 tools/validation/validate_algorithm_legacy_names.py \
  --root . \
  --baseline ebe862b11506e819c5bd5abc925299fc5fbb6f1a \
  --allowlist docs/provenance/algorithm-legacy-name-allowlist.tsv
PYTHONPATH=algorithm/common/src python3 -c 'import EvEye'
PYTHONPATH=algorithm/common/src python3 -m pytest -q algorithm/common/tests/test_runtime_control_contracts.py
PYTHONPATH=algorithm/hybrid python3 -m pytest -q \
  algorithm/hybrid/tests/external_pipeline/test_dataset.py \
  algorithm/hybrid/tests/external_pipeline/test_interpolation.py \
  algorithm/hybrid/tests/external_pipeline/test_output_contract_snapshot.py \
  algorithm/hybrid/tests/external_pipeline/test_preprocess.py \
  algorithm/hybrid/tests/external_pipeline/test_runtime_e2e.py \
  algorithm/hybrid/tests/external_pipeline/test_script_workflows.py \
  algorithm/hybrid/tests/external_pipeline/test_target_fps_dataset.py \
  algorithm/hybrid/tests/external_pipeline/test_train_pipeline.py
find algorithm/common/scripts algorithm/common/tests -iname '*facet*' -print
rg -n '(FACET|HBTXR)_(DISABLE_CUDNN|DEVICES|CKPT_PATH)|facet_square_direct|EventPatchEmbeddingFACET|/tmp/(facet|hbtxr)_unet_dataset_smoke' algorithm \
  --glob '!algorithm/analysis/**' --glob '!algorithm/archive/**' --glob '!algorithm/hybrid/hardware_reference/**'
git diff --exit-code ebe862b11506e819c5bd5abc925299fc5fbb6f1a -- \
  references algorithm/analysis algorithm/archive algorithm/artifacts \
  algorithm/hybrid/hardware_reference algorithm/event/docs/analysis \
  algorithm/frame/docs/analysis algorithm/hybrid/docs/analysis \
  algorithm/hybrid/tests/handover algorithm/common/PROVENANCE.md algorithm/docs/track
git status --short --untracked-files=all -- \
  references algorithm/analysis algorithm/archive algorithm/artifacts \
  algorithm/hybrid/hardware_reference algorithm/event/docs/analysis \
  algorithm/frame/docs/analysis algorithm/hybrid/docs/analysis \
  algorithm/hybrid/tests/handover algorithm/common/PROVENANCE.md algorithm/docs/track
```

Expected: moved paths exist only at neutral targets; active branded control search is empty; no replacement `HBTXR_*` environment variable was introduced; existing config controls and the standalone builder `--disable-cudnn` behavior are characterized; focused Hybrid tests pass; `EvEye` imports; frozen zones have no baseline-to-current path/blob drift or untracked addition; every remaining case-insensitive `facet` match resolves to the checked allowlist.

**Retirement Track**

- Remove: affected FACET environment controls and any prohibited HBTXR-prefixed replacement environment controls, active legacy paths, resize-policy value and patch-embedding class name
- Keep: source-faithful reference identity and historical/provenance text
- Delete trigger: consumer map complete, focused characterization pass, remaining match allowlist exact
- Rollback: slice별 `git revert`; reference/history를 edit해 rollback하지 않는다

## Task RC-130: Remove the Hybrid `src.pools` facade

**Dependencies**: RC-030, RC-100A

**Status**: GATED EPIC. `src.pools`는 문서상 public surface였으므로 의도적 breaking retirement로 기록한다. Permanent compatibility shim은 만들지 않는다.

**Current evidence**

- `algorithm/hybrid/src/pools`: 6 tracked modules, 331 LOC
- Maintained runtime consumer: `algorithm/hybrid/src/training/trainer.py`의 LR scheduler import/call 한 곳
- `src.pools` test consumer: 0
- External consumer: unknown
- `src.optim.pool.py`: 별도의 optimizer candidate/report feature이며 retirement 대상이 아님

**Canonical owner map**

| Removed pool module | Final owner/action |
|---|---|
| `pools/heads.py` | 재생성하지 않음; existing `models/tracker/head_factory.py`와 `models/tracker/registry.py` 유지 |
| `pools/losses.py` | 재생성하지 않음; existing `loss/stage.py`, `stage1.py`, `stage2.py` 유지 |
| `pools/optimizers.py` | 재생성하지 않음; existing `optim/registry.py`와 `optim/pool.py` 유지 |
| `pools/lr_schedulers.py` | `optim/lr_schedulers.py`로 이동 |
| `pools/runtime_schedulers.py` | 재생성하지 않음; existing `models/controller.py` owner 유지 |
| `pools/__init__.py` | 제거; `src/__init__.py` export에서도 제거 |

`models/heads/` package나 `models/*/registry.py`를 새로 만들지 않는다. Existing `models/heads.py`와 filesystem name collision을 만들고 사용되지 않는 registry owner를 재생성하기 때문이다. Domain `__init__.py`는 필요한 stable API만 re-export하며 registry 구현을 담지 않는다.

**Files**

- Move: `algorithm/hybrid/src/pools/lr_schedulers.py` to `algorithm/hybrid/src/optim/lr_schedulers.py`
- Create: `algorithm/hybrid/tests/external_pipeline/test_lr_schedulers.py`
- Modify: `algorithm/hybrid/src/optim/__init__.py`
- Modify: `algorithm/hybrid/src/training/trainer.py`
- Modify: `algorithm/hybrid/src/__init__.py`
- Delete after consumer-zero gate: `algorithm/hybrid/src/pools/{__init__,heads,losses,optimizers,runtime_schedulers}.py`
- Preserve: `algorithm/hybrid/src/optim/pool.py`, optimizer-pool configs, report names and consumers
- Preserve as history: `algorithm/hybrid/docs/analysis/hgtxr_pool_integration.md`; ADR-009 records that its public-surface description is superseded

**Steps**

1. RC-030 consumer map에 `src.pools`, `from src import pools`, dynamic import and docs exposure를 추가한다.
2. LR scheduler의 `none`, `cosine`, `step`, `plateau`, metric `min/max`, external-scheduler suppression, unknown-name behavior를 characterization한다.
3. LR scheduler module을 `src.optim.lr_schedulers`로 이동하고 `trainer.py` import를 바꾼다.
4. `src.optim.__init__`에서 필요한 scheduler API만 re-export한다.
5. Unused head/loss/optimizer/runtime facades는 다른 package로 복제하지 않고 삭제한다.
6. `src.__all__`에서 `pools`를 제거하고 `src.pools` import가 실패함을 검증한다.
7. Full Hybrid regression 후 directory를 제거한다.
8. Commit A: `refactor(hybrid): move lr schedulers to optim`.
9. Commit B: `refactor(hybrid): remove component pool facades`.

**Verification**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=algorithm/hybrid python3 -m pytest -p no:cacheprovider -q \
  algorithm/hybrid/tests/external_pipeline/test_lr_schedulers.py \
  algorithm/hybrid/tests/external_pipeline/test_train_pipeline.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=algorithm/hybrid python3 -m pytest -p no:cacheprovider -q algorithm/hybrid/tests
test ! -d algorithm/hybrid/src/pools
test -f algorithm/hybrid/src/optim/pool.py
rg -n 'src\.pools|from[[:space:]]+src[.]pools|from[[:space:]]+src[[:space:]]+import[[:space:]]+pools' algorithm/hybrid \
  --glob '!algorithm/hybrid/hardware_reference/**' --glob '!algorithm/hybrid/docs/analysis/hgtxr_pool_integration.md'
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=algorithm/hybrid python3 -c \
  'from src.optim.lr_schedulers import list_lr_scheduler_names; assert list_lr_scheduler_names() == ["cosine", "none", "plateau", "step"]'
```

Expected: `src/pools` directory와 maintained consumer가 0; `src.pools`는 intentionally unavailable; LR scheduler behavior와 Hybrid tests 통과; `optim/pool.py`와 optimizer experiment/report contract 유지.

**Retirement Track**

- Remove: all six `src/pools` modules and `src.__all__` export
- Keep: canonical domain owners and `src/optim/pool.py`
- Delete trigger: consumer map has no required external consumer, trainer uses `src.optim.lr_schedulers`, focused/full tests pass
- Stop rule: required external consumer가 확인되면 removal을 중단하고 compatibility 기간을 사용자에게 재승인받는다

## Task RC-200: Split hardware validation and signoff owners

**Dependencies**: RC-010, RC-020

**Status**: GATED EPIC. Golden diagnostics와 `docs/analysis/HBTXR-HARDWARE-RULE-CLUSTERS.md`를 만든 뒤 `docs/aegis/plans/rc-200-hardware-validation-slices.md`에서 exact symbol/line/test/commit 단위로 나누기 전에는 구현하지 않는다.

**Files**

- Create: `hardware/tools/validation/{models,loaders,rules,renderers}.py`
- Create: `hardware/tools/signoff/{stages,runner}.py`
- Modify: `hardware/tools/write_final_evidence_manifest.py`
- Modify: `hardware/tools/run_third_goal_final_signoff.py`
- Modify: `hardware/tools/validate_final_operator_handoff.py`
- Modify: `hardware/tools/validate_final_signoff_bundle.py`
- Modify: `hardware/tests/test_write_final_evidence_manifest.py`
- Modify: `hardware/tests/test_run_third_goal_final_signoff.py`
- Modify: `hardware/tests/test_validate_final_operator_handoff.py`
- Modify: `hardware/tests/test_validate_final_signoff_bundle.py`

**Why**

Validation policy, filesystem collection, orchestration, rendering이 한 함수에 결합되어 작은 변경도 전체 회귀를 요구한다.

**Architecture**

- `models.py`: immutable result/diagnostic dataclasses
- `loaders.py`: JSON/path parsing only
- `rules.py`: pure `check_*` predicates returning diagnostics
- `renderers.py`: JSON/Markdown projection only
- `stages.py`: signoff stage definitions and dependencies
- `runner.py`: stage orchestration and command execution boundary

**Steps**

1. Existing fixture output의 JSON key/order/error code를 golden으로 고정한다.
2. `write_final_evidence_manifest.build_consistency_checks`의 rule cluster를 순수 함수로 한 묶음씩 이동한다.
3. `validate_handoff`와 `validate_bundle`이 동일 rule API를 사용하게 한다.
4. `run_pipeline`의 stage metadata와 executor를 분리한다.
5. Old functions를 facade로 유지하고 golden test를 실행한다.
6. Facade 내부 branch proxy가 25 이하가 될 때까지 orchestration만 남긴다.
7. Commit을 rule cluster별로 나눈다.

**Repair Track**

- Root cause: append-only validation growth와 duplicated report shaping
- Stable repair: typed diagnostics+pure rules
- Compatibility: output key/message/exit status 유지

**Retirement Track**

- Old owner: monolithic rule blocks
- Keep reason: public script entry points
- Delete trigger: facade가 delegation만 수행하고 모든 tests pass

**Verification**

```bash
python3 -m unittest \
  hardware.tests.test_write_final_evidence_manifest \
  hardware.tests.test_run_third_goal_final_signoff \
  hardware.tests.test_validate_final_operator_handoff \
  hardware.tests.test_validate_final_signoff_bundle
python3 .agents/recon/2026-07-15-semantic-census/semantic_scan.py --root . --out /tmp/hbtxr-after-hardware
```

Expected: golden output 동일, 네 facade 각각 200LOC 미만, 새 pure rule 함수가 focused tests를 가짐.

## Task RC-210: Split hybrid data and training responsibilities

**Dependencies**: RC-100A, RC-130

**Status**: GATED EPIC. RC-100A가 current layout을 고정한 뒤 `docs/aegis/plans/rc-210-hybrid-owner-slices.md`에 exact symbol/fixture/commit slice를 작성·검토한다. RC-100C를 요구하지 않는다.

**Files**

- Reuse/modify: `algorithm/hybrid/src/data/contracts.py`, `components.py`, `event_builder.py`
- Conditional create after residual-responsibility proof: `algorithm/hybrid/src/data/event_normalization.py`, `target_assembly.py`
- Reuse/modify: `algorithm/hybrid/src/training/trainer.py`, `checkpoints.py`, `step_runner.py`
- Conditional create after residual-responsibility proof: `algorithm/hybrid/src/training/state.py`
- Modify: `algorithm/hybrid/tests/external_pipeline/test_dataset.py`, `test_train_pipeline.py`, `test_runtime_e2e.py`

**Why**

EventInputBuilder와 TrainingSession이 validation, state, transformation, orchestration을 동시에 소유한다.

**Architecture**

- event contract: existing `data/contracts.py` input shape/dtype/domain validation
- normalization/target assembly: existing `data/components.py`를 우선 재사용하고 residual 책임이 입증될 때만 새 owner 생성
- training state: epoch/step/metric data
- epoch runner: existing `training/step_runner.py`
- checkpointing: existing `training/checkpoints.py`
- TrainingSession: dependency orchestration only

**Steps**

1. Event builder input/output shape, dtype, error message를 parameterized fixture로 고정한다.
2. Existing contracts/components에서 담당하지 않는 pure normalization/assembly responsibility만 추출한다.
3. Existing checkpoints/step_runner를 재사용하고 TrainingSession의 residual state/logging만 분리한다.
4. Existing public constructor와 `trainer.train()` facade를 유지한다.
5. Dataset/train pipeline/runtime tests를 실행한다.
6. Commit A: `refactor(hybrid): separate event input contracts`.
7. Commit B: `refactor(hybrid): separate training state and epoch runner`.

**Retirement Track**

- Old owner: builder/session 내부의 mixed responsibilities
- Delete trigger: old private helpers caller 0, facade branch proxy 감소

**Verification**

```bash
python3 -m pytest -q \
  algorithm/hybrid/tests/external_pipeline/test_dataset.py \
  algorithm/hybrid/tests/external_pipeline/test_train_pipeline.py \
  algorithm/hybrid/tests/external_pipeline/test_runtime_e2e.py
```

Expected: record/shape/checkpoint behavior 동일, facade는 orchestration만 담당.

## Task RC-220: Break deferred cycles and narrow error/path boundaries

**Dependencies**: RC-100A, RC-100B

**Files**

- Evidence: `.agents/recon/2026-07-15-semantic-census/import-scc-evidence.json`
- Conditional create: `quantization/hgpipe_quantization/contracts.py`
- Conditional modify: `quantization/hgpipe_quantization/{api,artifact_imagenet,artifact_patch}.py`
- Conditional create: `algorithm/hybrid/src/preprocess/contracts.py`
- Conditional modify: `algorithm/hybrid/src/preprocess/{event_generation,target_fps_build,annotation_backends,groundedsam_build}.py`
- Modify: `algorithm/hybrid/src/preprocess/io_utils.py`
- Modify: `hardware/tools/gen_cleanup_dry_run_2026_06_09.py`
- Modify: `quantization/tests/test_artifact_patch.py`, `quantization/tests/test_package_api.py`, `quantization/tests/test_cli.py`
- Modify: `algorithm/hybrid/tests/external_pipeline/test_preprocess.py`, `test_target_fps_build.py`, `test_v2e_experiment.py`, `test_interpolation.py`, `test_raw_ellipse_blink.py`

**Why**

Lazy reverse imports, broad exceptions, fixed host paths는 실패 원인을 숨기고 owner를 흐린다.

**Steps**

1. SCC 후보별로 재현 가능한 import 실패, 중복 contract owner, 또는 측정 가능한 유지보수 이득이 있는지 먼저 기록한다.
2. 필요성이 입증된 SCC만 shared data/protocol을 lower-level module로 이동하고 import graph를 재생성한다. 입증되지 않은 SCC는 accepted lazy boundary로 문서화하고 변경하지 않는다.
3. `io_utils`에서 `JSONDecodeError`, `KeyError`, `TypeError`, `ValueError`만 expected-data 오류로 분류한다.
4. 기타 예외는 context를 추가해 다시 raise한다.
5. Cleanup root를 required CLI argument 또는 repo-relative default로 바꾼다.
6. 각 component focused tests를 실행한다.
7. Commit을 SCC/error/path별로 나눈다.

**Verification**

```bash
python3 -m pytest -q algorithm/hybrid/tests
(cd quantization && python3 -m unittest discover -s tests -p 'test_*.py')
python3 -m pytest -q \
  algorithm/hybrid/tests/external_pipeline/test_preprocess.py \
  algorithm/hybrid/tests/external_pipeline/test_target_fps_build.py \
  algorithm/hybrid/tests/external_pipeline/test_v2e_experiment.py \
  algorithm/hybrid/tests/external_pipeline/test_interpolation.py \
  algorithm/hybrid/tests/external_pipeline/test_raw_ellipse_blink.py
rg -n '/home/user/project/PRJXR' hardware/tools
```

Expected: 필요성이 입증된 SCC만 제거되고 나머지는 documented accepted lazy boundary이며, fixed host root 0.

## Task RC-300: Consolidate active duplicates and API owners

**Dependencies**: RC-100A, RC-200

**Status**: GATED EPIC. `docs/analysis/HBTXR-ACTIVE-DUPLICATE-ALLOWLIST.md`와 `docs/aegis/plans/rc-300-duplicate-owner-slices.md`가 exact caller/symbol/test/commit을 승인하기 전에는 변경하지 않는다.

**Files**

- Create: `hardware/tools/_shared/{json_io,hashing,paths}.py`
- Modify: `docs/analysis/HBTXR-ACTIVE-DUPLICATE-ALLOWLIST.md`에 승인된 exact caller path만
- Modify: allowlist가 지정한 exact algorithm loss export/registry/shim path만

**Why**

Active/active 반복 구현과 여러 공개 loss owner가 변경을 여러 곳에 복제하게 만든다.

**Steps**

1. Duplicate CSV에서 active/active, 12LOC 이상, identical error behavior 그룹만 승인 목록으로 만든다.
2. Hardware domain-local utility를 만들고 3개 caller씩 작은 commit으로 전환한다.
3. CLI 단독 실행에서도 import 가능한 package-relative fallback을 검증한다.
4. Current `algorithm/hybrid/src/loss`를 공식 Hybrid loss owner로 문서화한다.
5. Root/training loss shim의 consumer를 측정하고 deprecation metadata를 추가한다.
6. Commit을 utility family와 loss owner로 분리한다.

**Retirement Track**

- Reference copy는 코드 공통화 대상이 아니라 provenance 대상
- Active duplicate는 caller 0 후 제거
- Quantization compatibility shim만 usage 0과 deprecation window 종료 후 제거; Hybrid `src`는 RC-100C가 선택되지 않으면 유지

**Verification**

```bash
python3 -m unittest discover -s hardware/tests -p 'test_*.py'
python3 -m pytest -q algorithm/hybrid/tests
rg -n '^def (load_json|sha256_file|normalize_roots)' hardware/tools
```

Expected: approved active duplicate만 감소하고 script entry points 유지.

## Task CL-000: Create a repository asset/provenance manifest

**Dependencies**: RC-000

**Files**

- Create: `tools/provenance/build_repository_asset_manifest.py`
- Create: `tests/provenance/test_repository_asset_manifest.py`
- Create: `docs/provenance/repository-assets.json`

**Why**

References 보존과 non-reference binary cleanup은 source, rights, hash, consumer를 기록하지 않으면 비교 무결성이나 rollback을 검증할 수 없다.

**Change Necessity**

기존 handover registry의 fail-closed 원칙을 재사용하되 repository asset type을 확장한다.

**Required schema**

- `path`, `kind`, `authority`, `bytes`, `sha256`
- `source_repo`, `source_revision`, `license_status`
- `consumers`, `comparison_entrypoints`, `mirror`, `retention_class`
- `decision`: `keep`, `permanent-comparison-reference`, `restricted`, `externalize-binary`, `review`
- `verified_at_commit`
- 모든 `references/**`: `retention_class=permanent_comparison_reference`, `decision=keep`; replacement/delete/rename/deduplicate 금지

**Steps**

1. Manifest schema validation tests를 작성한다.
2. Git blob과 working-tree SHA 일치를 검사한다.
3. Missing rights/source/consumer는 reference activation/distribution과 non-reference externalize/delete 결정을 거부한다.
4. 모든 references와 current binary/duplicate 후보를 서로 다른 retention class로 기록한다.
5. `references/**` destructive action이 schema-invalid인지 negative fixture로 고정한다.
6. Commit: `feat(provenance): add repository asset manifest`.

**Verification**

```bash
python3 -m unittest tests.provenance.test_repository_asset_manifest
python3 tools/provenance/build_repository_asset_manifest.py --check docs/provenance/repository-assets.json
```

Expected: 누락 provenance는 non-zero, 완전한 keep/externalize-binary entry는 zero, 모든 destructive reference fixture는 non-zero.

## Task CL-100: Catalog permanent comparison references

**Dependencies**: CL-000

**Files**

- Modify: `docs/provenance/repository-assets.json`
- Modify: `references/README.md`
- Create: `references/MANIFEST.md`
- Create: `docs/provenance/reference-assets.json`
- Create: `docs/analysis/HBTXR-REFERENCE-COMPARISON-CATALOG.md`

**Why**

5,821파일/약 1.18GB의 legacy tree와 다른 reference roots는 비교·porting·재현 입력이다. 크기는 삭제 근거가 아니며, 실험에서 어떤 snapshot을 왜 사용하는지 catalog가 필요하다.

**Permanent retention invariants**

1. 모든 tracked `references/**` path와 payload SHA-256 유지
2. 삭제, rename, move, pointer replacement, deduplication 금지
3. Upstream repo/revision/license status와 comparison purpose 기록
4. Active runtime import 금지; adapter는 `analysis/comparison/` 또는 별도 approved experiment surface에 둠
5. 결과·cache·checkpoint는 references 밖에 저장
6. Unclear license/privacy는 activation/distribution을 막지만 local retention을 제거하지 않음

**Steps**

1. Baseline path, Git blob, SHA-256, bytes를 `reference-assets.json`에 고정한다.
2. Source revision, license status, modality/task, entrypoint/config, required environment, expected input/output을 comparison catalog에 기록한다.
3. Active runtime import와 generated output 유입을 별도로 검사한다.
4. Optional external mirror가 있으면 backup metadata로만 기록하고 authoritative path는 repository로 유지한다.
5. Commit: `docs(references): catalog permanent comparison assets`.

**Rollback**

이 task는 reference payload를 수정·이동·삭제하지 않는다. Governance metadata만 추가한다.

**Verification**

```bash
git grep -n 'references/legacy-codebase' -- ':!references/legacy-codebase/**'
python3 tools/provenance/build_repository_asset_manifest.py --check docs/provenance/repository-assets.json
git diff --name-status -- references | awk '$1 ~ /^[DR]/ {bad=1} END {exit bad}'
```

Expected: comparison consumers와 provenance gaps가 열거되고, reference D/R 0건, baseline payload hash drift 0건.

## Tasks CL-101 and CL-102: Cancelled reference transfer-as-replacement and deletion

**Status**: CANCELLED by the permanent comparison-reference decision.

An optional external mirror may be created only as a non-authoritative backup under a separate backup plan. It never replaces the in-repository copy and cannot unlock deletion. CL-101/102 cannot appear in a selected batch.

## Task CL-110: Introduce a hardware artifact locator

**Dependencies**: CL-000

**Files**

- Create: `hardware/artifacts/manifest.json`
- Create: `hardware/pynq/hgtxr/artifacts.py`
- Create: `hardware/tests/pynq/test_artifact_locator.py`
- Create: `hardware/tests/integration/test_artifact_locator_compat.py`
- Modify: direct consumers enumerated by `rg -n 'artifact_paths|bitfile|hwhfile' hardware/pynq hardware/tools hardware/tests`
- Keep: tracked `.bit/.hwh/.bin` local fallback

**Why**

약 190MB binary를 source path에 직접 결합하지 않고도 board/runtime이 찾을 수 있어야 한다.

**Contract**

`resolve_artifact(name, *, cache_dir=None, verify_sha256=True) -> Path`

Manifest는 board, design, bitstream, hwh, source commit, SHA-256, bytes, retrieval URL을 포함한다.

**Steps**

1. 현재 direct path consumer를 목록화한다.
2. Locator와 checksum tests를 추가한다.
3. Consumers를 locator로 전환하되 local checked-in path fallback을 유지한다.
4. 이 task에서는 binary 전송·삭제를 수행하지 않는다.

**Verification**

```bash
python3 -m unittest hardware.tests.pynq.test_artifact_locator
python3 -m unittest hardware.tests.integration.test_artifact_locator_compat
rg -n 'hardware/pynq/hgtxr/.*\.(bit|hwh)|artifact_paths\(' hardware/pynq hardware/tools --glob '*.py'
```

Expected: checksum mismatch fail-closed, offline checked-in fallback 동작, direct hard-coded consumer path 0, binary 변경·삭제 0.

## Task CL-111: Transfer hardware binaries to an approved immutable target

**Dependencies**: CL-110; **Status**: BLOCKED until target/owner/retention/access and transfer are approved

Approved allowlist의 `.bit/.hwh/.bin`만 복사하고 source/destination SHA-256와 read-back/restore drill을 기록한다. 저장소 fallback은 유지한다.

## Task CL-112: Remove approved checked-in hardware binary fallbacks

**Dependencies**: CL-111; **Status**: BLOCKED until separate deletion approval

Locator 전환율 100%, offline/retrieval contract, restore drill, exact deletion allowlist가 승인된 뒤 별도 deletion commit으로만 수행한다.

## Task CL-120: Record exact-copy relationships while retaining reference snapshots

**Dependencies**: CL-000

**Files**

- Modify: `docs/provenance/repository-assets.json`
- Create: `docs/analysis/HBTXR-EXACT-COPY-RELATIONSHIPS.md`
- Review: `quantization/references/HG-PIPE-Quantization`, `algorithm/archive/imports`, and exact paths from `inventory/duplicate-blobs.csv`

**Why**

Active/reference/archive exact copies는 owner와 lineage를 혼동시킬 수 있다. Reference copy는 비교 snapshot으로 유지하고 active owner와의 관계만 명시한다.

**Steps**

1. Git blob hash가 동일한 cross-authority group만 exact 후보로 선택한다.
2. Active owner와 source revision을 지정한다.
3. Reference path에는 `comparison_snapshot_of` 관계를 기록하되 bytes/path를 유지한다.
4. Non-reference archive duplicate는 별도 future workstream 후보로만 분류한다.
5. Non-identical copy는 semantic lineage가 확인될 때까지 유지한다.
6. Pointer replacement와 파일 제거를 수행하지 않는다.

**Verification**

```bash
python3 tools/provenance/build_repository_asset_manifest.py --check docs/provenance/repository-assets.json
git diff --check
```

Expected: 모든 관계에 blob hash·active owner·comparison purpose가 있고 reference path/payload 변경·삭제 0.

## Task CL-121: Cancelled reference-copy replacement/removal

**Status**: CANCELLED. Any future non-reference archive deduplication requires a new task ID and a validator that rejects every `references/` prefix.

## Task CL-130: Audit documentation, naming, and evidence drift

**Dependencies**: CL-000

**Files**

- Create: `docs/analysis/HBTXR-DOC-PATH-EVIDENCE-AUDIT.md`
- Create: `docs/track/PATH-MIGRATION-subject-independent.md`
- Create: `docs/provenance/evidence-supersedes.json`
- Review: `README.md`, `hardware/README.md`, `hardware/docs/architecture/DIRECTORY_LAYOUT.md`, `quantization/README.md`, `third/SAM3-I/predictions.json`

**Why**

문서와 경로가 실제 owner를 잘못 설명하면 cleanup이 잘못된 source를 삭제할 수 있다.

**Steps**

1. README/layout의 current/target/deferred drift를 audit 문서에 기록한다.
2. `subjet_independent` 27경로의 reference graph와 old→new rename map을 만든다.
   `references/**`에서는 rename하지 않고 alias metadata만 기록한다.
3. Evidence tree를 hash/producer/timestamp/supersedes로 분류한다.
4. `predictions.json`이 upstream source인지 local output인지 판정한다.
5. 이 task에서는 README 수정, rename, artifact move를 수행하지 않는다.

**Verification**

```bash
rg -n 'future implementation area|\.\./ICCAD24-HG-PIPE|subjet_independent' README.md hardware quantization algorithm docs
python3 -m unittest discover -s tests/provenance -p 'test_*.py'
```

Expected: 모든 drift/rename/evidence 후보가 exact path, owner, proposed action을 가지며 파일 이동은 0건.

## Task CL-131: Apply approved documentation and path migrations

**Dependencies**: RC-100A, RC-100B, CL-110, CL-130; add RC-100C only when selected; **Status**: BLOCKED until active/docs migration approval

CL-130의 exact active/docs allowlist만 수정한다. `references/**`는 audit/alias metadata만 허용하고 rename/move 대상에서 제외한다. README/layout correction, active typo-path rename, evidence supersedes 적용을 서로 다른 commits로 나누고 old→new link check와 provenance tests를 각 commit에서 실행한다.

## Task RC-900: Validate, record delta, and enforce retirement gates

**Dependencies**: every task ID in the validator-approved `dependency_closure`; no unselected task is implied

**Files**

- Create: `.agents/recon/2026-07-15-semantic-refactor-delta/codebase-recon.json`
- Create: `.agents/recon/2026-07-15-semantic-refactor-delta/codebase-recon.md`
- Update: `docs/track/PROGRESS.md`, `CHANGELOG.md`, `ADR.md`, `log.md`

**Steps**

1. Selection manifest를 `--phase completion`으로 검증해 closure의 모든 task가 completed이고 evidence가 존재하는지 확인한다.
2. 선택된 task가 요구하는 component regression만 manifest evidence에 기록하고 실행한다.
3. Semantic scanner를 재실행하고 baseline 대비 delta를 만든다.
4. 선택된 public import, CLI, schema, error code, binary locator compatibility를 비교한다.
5. Old Quantization shim/facade의 consumer count와 references baseline path/hash invariants를 기록한다.
6. RC-120이 선택되었으면 active-name search가 allowlist 밖 0건이고 frozen reference/history diff가 0인지 검증한다.
7. RC-130이 선택되었으면 `src/pools`와 maintained `src.pools` consumer가 0이고 `src/optim/pool.py`가 유지되는지 검증한다.
8. Retirement trigger를 충족하지 못한 old path는 제거하지 않는다.
9. `git diff --check`와 recon validator를 실행한다.
10. Commit: `docs(refactor): record semantic cleanup delta`.

**Verification**

```bash
python3 -m unittest discover -s tests/validation -p 'test_refactor_cleanup_batch.py'
python3 tools/validation/validate_refactor_cleanup_batch.py \
  --phase completion \
  --catalog docs/track/refactor-cleanup-task-catalog.json \
  --schema docs/schemas/refactor-cleanup-selected-batch.schema.json \
  --batch docs/track/refactor-cleanup-selected-batch.json \
  --expected-baseline ebe862b11506e819c5bd5abc925299fc5fbb6f1a
python3 tools/validation/validate_algorithm_legacy_names.py \
  --root . \
  --baseline ebe862b11506e819c5bd5abc925299fc5fbb6f1a \
  --allowlist docs/provenance/algorithm-legacy-name-allowlist.tsv
bash /mnt/c/Users/User/.codex/plugins/cache/awesome-codex-plugins/agentops/3.2.0/skills-codex/codebase-recon/scripts/validate-output.sh \
  .agents/recon/2026-07-15-semantic-refactor-delta/codebase-recon.json
python3 .agents/recon/2026-07-15-semantic-census/validate_recon.py \
  .agents/recon/2026-07-15-semantic-refactor-delta/codebase-recon.json
git diff --check
git status --short --branch
```

Expected: unknown/missing-dependency/unapproved/incomplete batch, trigger 없는 RC-100C execution, selected CANCELLED task, destructive `references/**` path는 non-zero; selected closure is complete with resolvable evidence; reference D/R/hash drift 0; stored Python equivalent validation passes; official shell validator pass/block status is recorded separately; baseline_verified true and unapproved non-reference deletion is 0.

## Risk Register

| Risk | 영향 | 완화 |
|---|---|---|
| 외부 `src` consumer | package break | current Hybrid owner 유지; Quantization consumer map+shim+deprecation |
| 외부 FACET-name/`src.pools` consumer | explicit breaking migration으로 호출 실패 | RC-030 consumer map; required external consumer 발견 시 RC-120/130 중단 후 별도 compatibility 승인 |
| 제거된 environment override/metadata 외부 consumer | launcher override 또는 parser break | RC-030 consumer scan, config/CLI characterization, `facet_disable_cudnn` external consumer 발견 시 slice 중단 |
| Reference identity의 오염 | 비교 자산 경로·인용 불일치 | line-level allowlist와 frozen-zone zero-diff gate |
| `optim/pool.py` 오삭제 | optimizer sweep/report 회귀 | RC-130 preserve assertion과 consumer regression |
| 근거 없는 Hybrid migration | 대규모 churn | RC-100C trigger+approval, 기본 NO-OP |
| CLI/parser order drift | automation break | help/parser snapshot |
| JSON message/key drift | validation consumers break | golden fixture |
| Hardware binary removal | PYNQ runtime break | locator+checksum+offline cache |
| Reference license/source 누락 | comparison activation/distribution risk | local bytes 유지, fail-closed activation/distribution manifest |
| False dead-code inference | runtime break | 이름 heuristic은 삭제 금지 |
| Giant refactor batch | review/rollback 어려움 | rule cluster별 commit |
| Archive/reference non-identical | provenance loss | reference bytes/path permanent keep; lineage metadata만 추가 |

## Retirement Summary

- 즉시 retirement 후보: shadowed 앞 정의, import-time top-level write block, invalid type comments
- Consumer/characterization 후 retirement: active `FACET/facet` paths/contracts와 Hybrid `src.pools` facade
- 호환 window 후 retirement: Quantization `src` shim, CLI wrapper chain, monolithic private rule blocks
- Manifest/승인 후 retirement: non-reference hardware binaries와 active/docs typo paths
- 영구 유지: 모든 `references/**` path/payload와 source-faithful FACET identity, `algorithm/common/src/EvEye`, current Hybrid `src` root unless RC-100C is triggered, `src/optim/pool.py`, non-identical evidence, board-specific contract paths, dynamic-call dead-code 후보

## Plan Progress

- Semantic baseline: complete
- Evidence matrix: complete
- Review corrections: committed-blob isolation, duplicate-blob evidence, SCC inference gates, current algorithm layout preservation, conditional Hybrid migration, and permanent references incorporated
- Repair execution: not started
- Refactor execution: not started
- Active algorithm name and `src.pools` retirement: plan revised on 2026-07-21; implementation not started
- Reference manifest/comparison catalog: not started; permanent-retention policy accepted
- Reference transfer-as-replacement/deletion/dedup: cancelled
- Non-reference binary transfer/deletion and active/docs migration: blocked pending explicit approval
