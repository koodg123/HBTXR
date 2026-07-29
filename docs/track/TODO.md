# TODO

## Decision Gates

- Decide whether Spec-Kit scaffolding may be initialized after a clean checkpoint.
- Decide Git LFS versus external artifact storage if large artifacts are required.
- Approve any future push separately.
- Completed: approve and execute the T-010/T-020/T-100/T-110/T-210 code/file-only slice.
- Approve dependency installation before T-000 creates the test environment.

## Analysis Backlog

- [x] Classify the 10 changed/missing HANDOVER/HBTXR files.
- [x] Analyze HGTXR recovery, ensemble, support-adaptive, teacher, and metric flows.
- [x] Identify HGTXR regression/golden-vector adaptation priorities.
- [x] Analyze XR_Accel cyclic/ZCU104 automation and config schema.
- [x] Implement data and evaluation contract gates as T-100 and T-110.
- [ ] Implement checkpoint compatibility and blending as T-120 after environment approval.
- [ ] Adapt HGTXR Stage-1/Stage-2 regression fixtures behind current interfaces.
- [x] Implement XR cyclic configuration validation as T-210.
- [ ] Implement the T-220 XR automation dry-run planner after a new selection.
- [ ] Map licenses for ERVT, TENNs-Eye, TDTracker, BRAT, FECET, and Retina sources.
- [ ] Audit CRLF/tooling governance and record decisions as T-600.
- [x] Write the concrete follow-up implementation plan with Task Cards and gates.
- [x] Classify standalone HBTXR annotation and `references/impl` content.
- [x] Write the standalone HBTXR selective integration plan and file matrix.
- [ ] SI-000: generalize provenance validation for an explicit external repo and committed source view.
- [ ] SI-010: verify standalone authorship/license and freeze no-op decisions.
- [ ] SI-020: resolve 346x260, 346x240, and 640x480 coordinate frames in ADR-005.
- [ ] SI-100: add annotation record contracts after SI-010/SI-020 approval.
- [ ] SI-110: add dependency-free annotation metrics after SI-100.
- [ ] SI-120: add dependency-free uncertainty metrics after SI-110.
- [ ] SI-130: curate annotation evidence without importing numeric results.
- [ ] SI-200: consider SWIFT anti-blink/checkpoint behavior only through a separate license/runtime-approved plan.
- Permanent no-op: do not re-import FECET archives or HG-PIPE code/tests/configs.
- Permanent exclusions: standalone samples, datasets, results, tables, figures,
  weights, locks, environments, downloads, reproduction, and experiments.
- Scope Fence: HBTXR listed paths; HANDOVER read-only; no external mutation.
- Baseline Lock: verify branch, HEAD, tools, licenses, and inputs before each wave.
- Approved Behavior: the selected five-task slice is complete; every deferred task needs a new selection and its applicable gate.
- Compatibility: current defaults, output ABI, checkpoint load, HLS top, and XR references stay stable.
- Retirement: remove conditional adapters or never add them when characterization proves them unnecessary.
- Execution: sequential waves with implementer, specification review, and quality review.
- Drift: stop on owner, source, license, tool, split, or board drift and revise the plan.
- Evidence: preserve commands, staged manifests, reviews, commit SHAs, and uncovered risks.
- Advisory boundary: branch-local completion is not merge, release, or push authority.

## Semantic Refactor and Cleanup Backlog

- [x] Generate baseline file/function/class/line semantic census at `ebe862b`.
- [x] Validate architecture, Python, and non-Python findings through three read-only agents.
- [x] Write the RC/CL execution plan and cleanup decision matrix.
- [ ] RC-010: characterize and remove shadowed artifact-patch definitions.
- [ ] RC-020: make dated hardware update scripts import-safe and fix invalid type comments.
- [ ] RC-030: map all `src`, CLI, PYTHONPATH, and external package consumers.
- [ ] RC-100A: preserve current Hybrid ownership and minimally harden the bootstrap/import contract.
- [ ] RC-100B: migrate only Quantization from generic `src` to `hgpipe_quantization` with a measured compatibility shim.
- [ ] RC-100C: migrate Hybrid namespace only after supported-workflow collision evidence and explicit approval; otherwise record NO-OP.
- [ ] RC-110: replace the quantization CLI wrapper chain with explicit command registration.
- [ ] RC-120: flatten common script/test paths, retire active repository-owned FACET names, and absorb old environment controls into existing config or neutral CLI controls without `HBTXR_*` replacements.
- [ ] RC-130: move LR schedulers to `src.optim`, remove `src.pools`, and preserve `src.optim.pool` after consumer characterization.
- [ ] RC-200: split hardware validator/signoff rules, I/O, orchestration, and rendering.
- [ ] RC-210: after RC-130, reuse existing contracts/components/checkpoints/step-runner owners and split only residual hybrid data/training responsibilities.
- [ ] RC-220: evaluate deferred-cycle necessity; refactor only evidenced SCCs, and narrow broad exception/fixed-path boundaries.
- [ ] RC-300: consolidate approved active duplicates and canonical API owners.
- [ ] CL-000: add a fail-closed repository asset/provenance manifest.
- [ ] CL-100: catalog `references/**` as permanent in-repository comparison assets with provenance and hashes.
- [x] CL-101: CANCELLED — no transfer-as-replacement for `references/**`.
- [x] CL-102: CANCELLED — no deletion of `references/**`.
- [ ] CL-110: add a hardware artifact locator while retaining checked-in fallbacks.
- [ ] CL-111: transfer approved hardware binaries only after immutable target and read-back approval.
- [ ] CL-112: remove checked-in binary fallbacks only after a separate exact-allowlist approval.
- [ ] CL-120: record exact-copy owner/provenance relationships without changing reference snapshots.
- [x] CL-121: CANCELLED for `references/**`; non-reference duplication requires a separate plan.
- [ ] CL-130: audit README/layout/naming/evidence drift without moving files.
- [ ] CL-131: apply only separately approved documentation/path/evidence migrations.
- [ ] RC-900: run component regression and produce a validated delta recon.
- Gate: `references/**` deletion, rename, deduplication, external-pointer replacement, and hash drift are prohibited.
- Gate: no destructive non-reference cleanup, external archive mutation, commit, push, or merge without separate authorization.
- Gate: RC-120/RC-130 planning is complete but implementation requires a new selected batch; required external old-name consumer stops retirement for a separate compatibility decision.

## Algorithm Modular Ownership — SUPERSEDED (2026-07-22)

> **정정 2026-07-29.** 이 섹션의 모든 항목이 `[ ]`(미착수)로 표기되어 있었으나 **사실이
> 아니었습니다.** AM-000부터 AM-900까지 2026-07-21에 실행되었고, 그 다음 날
> flat-functional 재작성이 목표 트리 자체를 대체했습니다. 세 문서(이 파일,
> `PROGRESS.md`, `HANDOVER.md`)가 서로 다른 상태를 말하고 있었고 전부 틀렸습니다.
>
> **근거 (전부 현재 브랜치 계보 위):**
> - `c7d6bfe` AM-000 baseline freeze · `5d94dc5` AM-010 package origins
> - `41f5456` EvEye 호환 계층 은퇴 · `5070c87` src.pools 해체 · `2a451be` frame/event 실행 가능
> - `6b52483` AM-090 documented no-op · `7944b47` AM-900 integration/preservation gate 통과
> - **`6c1cc54` (07-22) eveye 네임스페이스 제거 — AM-010이 세운 `eveye.*` 패키징을 되돌림**
>
> **현재 실제 상태**: `algorithm/`에 `eveye`도 `pyproject.toml`도 없습니다. bare package
> (`common` `dataset` `engine` `models` `quantization` `utils` …) 구조이고, 이것이
> flat-functional 재작성의 결과입니다. 브랜치 이름 `rewrite/flat-functional`이 그것입니다.

- [x] AM planning: define target roots, meanings, source mapping and preservation.
- [x] AM-000 ~ AM-900: 실행 완료 (`c7d6bfe` … `7944b47`).
- [~] `eveye.*` 패키징: **대체됨** — `6c1cc54`가 네임스페이스를 제거하고 bare package로
      전환. 재도입하려면 새 결정이 필요합니다.
- [ ] **RC-/CL-/SI-/T- 섹션도 같은 감사가 필요합니다.** 이 섹션이 6일간 틀린 상태로
      있었으므로 다른 섹션도 실제 상태와 대조되지 않았을 가능성이 높습니다. 특히
      RC-100B(`hgpipe_quantization` 이관)는 Q7이 `references/hardware/`로 옮긴 것과
      충돌할 수 있습니다.
