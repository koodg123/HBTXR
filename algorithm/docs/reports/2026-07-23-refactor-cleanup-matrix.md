# HBTXR Refactoring and Cleanup Decision Matrix

| ID | 우선 | 분류 | 근거 | 결정 | 실행 전 게이트 | 보존/Retirement 기준 |
|---|---|---|---|---|---|---|
| RF-001 | P0 | correctness | `quantization/src/artifact_patch.py:240,331,446,663` | 뒤 정의를 canonical로 검증하고 가려진 앞 정의 제거 | 두 버전 output/schema 비교, artifact patch tests | 앞 정의 직접 소비자 없음 확인 |
| RF-002 | P0 | side effect | `hardware/tools/update_e2e_full_docs_2026_06_09.py:13-141` | import-safe `main(argv)`로 이동 | import 전후 filesystem snapshot 동일 | top-level write block 제거 |
| RF-003 | P0 | side effect | `hardware/tools/update_e2e_full_vit_2026_06_09.py:33-63` | import-safe 또는 migration archive | import 전후 config/TCL 동일 | one-shot script owner 확정 |
| RF-004 | P0 | packaging | `algorithm/hybrid/.../_bootstrap.py:7-12`, Hybrid·Quantization의 일반명 `src` | modality 구조와 `algorithm/hybrid/src` 보존; Hybrid bootstrap 최소 보강; Quantization만 named package 정리; Hybrid migration은 조건부 NO-OP | 지원 workflow/consumer, clean-process 충돌 재현, 별도 승인 | Quantization shim 사용 0건; Hybrid trigger 미충족 시 현 구조 유지 |
| RF-005 | P1 | CLI | `quantization/src/cli.py:60-949` | `register_*_commands()`와 단일 `main()` | parser snapshot 및 CLI tests | wrapper chain 제거 |
| RF-006 | P1 | complexity | `write_final_evidence_manifest.py:601-2554` | pure checks, collectors, renderer 분리 | golden JSON key/order, hardware tests | 원 함수 facade만 남기거나 제거 |
| RF-007 | P1 | complexity | `run_third_goal_final_signoff.py:597-2315` | stage runner와 policy 분리 | dry-run command/output snapshot | monolithic pipeline 제거 |
| RF-008 | P1 | complexity | `validate_final_operator_handoff.py:394-1569` | schema/predicate/report 분리 | validation error code·message 호환 | monolithic validator 제거 |
| RF-009 | P1 | complexity | `validate_final_signoff_bundle.py:518-1425` | bundle loader/rules/render 분리 | bundle tests와 golden diagnostics | monolithic validator 제거 |
| RF-010 | P1 | domain class | `event_builder.py:30-543` | input contract, normalization, assembly 분리 | dataset/event builder characterization | facade 책임 1개 이하 |
| RF-011 | P1 | orchestration | `trainer.py:886-1294` | state, epoch runner, checkpoint/logging 분리 | train pipeline/runtime tests | `TrainingSession`은 orchestration만 소유 |
| RF-012 | P1 | dependency | medium-confidence import SCC 후보 3개 | 구체적 실패·중복 owner가 있을 때만 protocol/data contract/factory owner 추출 | edge evidence와 재현 실패 또는 owner 중복 입증 | 필요성 미입증 시 lazy boundary 문서화·유지 |
| RF-013 | P1 | error handling | `preprocess/io_utils.py:379-417` | expected exception만 포착하고 오류 유형 보고 | malformed input cases | broad `Exception` 제거 |
| RF-014 | P1 | path | `gen_cleanup_dry_run_2026_06_09.py:10` | CLI/config root 주입 | no default write outside root | fixed host root 제거 |
| RF-015 | P2 | type hygiene | 두 `Predict.py:64` | type comment를 일반 주석/유효 annotation으로 변경 | AST type-comment parse | invalid type comment 0건 |
| RF-016 | P2 | duplication | hardware `load_json`, `sha256_file` 반복 | domain-local shared utility | exact output/error tests | 중복 active body 제거 |
| RF-017 | P2 | API owner | loss shim/registry alias 중복 | canonical API와 compatibility alias 선언 | import consumer inventory | deprecated shim 사용 0건 |
| RF-018 | P2 | observability | maintained `print` 후보 | CLI writer와 library logger 구분 | stdout contract tests | library direct print 최소화 |
| CL-001 | P0 | governance | subject-specific reference data paths | restricted comparison reference로 저장소 안에 보존; activation/distribution만 차단 | rights, anonymization, checksum, access policy | 삭제·rename·hash drift 0건 |
| CL-002 | P0 | repository size | `references/legacy-codebase` 5,821파일/1.18GB | 영구 in-repository comparison catalog와 provenance manifest | source URL/revision/license/SHA/internal links | tree 유지; 삭제·pointer 대체·rename·deduplicate 금지 |
| CL-003 | P1 | binary artifact | hardware `.bit/.hwh/.bin` 약 190MB | locator는 CL-110; 전송/삭제는 BLOCKED CL-111/112 | PYNQ/package consumers, immutable target, read-back/restore, 별도 승인 | locator 전환율 100%와 exact 삭제 allowlist 승인 |
| CL-004 | P1 | duplication | 0-byte 제외 cross-authority 동일 blob 519그룹; maintained↔archive/reference 228그룹 | CL-120에서 관계/provenance만 기록; reference replacement/removal CL-121은 취소 | `duplicate-blobs.csv`, consumer graph, authority 구분 | reference 삭제·rename·hash drift 0건; non-reference는 별도 승인 |
| CL-005 | P1 | source duplication | `quantization/src`와 nested HG-PIPE copy | active owner와 영구 comparison snapshot을 함께 유지하고 revision 관계 기록 | license/revision/semantic diff | reference copy는 유지하고 active runtime 의존성만 0 |
| CL-006 | P1 | docs drift | root/hardware/quantization README | 현재 구현과 경로로 수정 | link/path checker | 존재하지 않는 active path 0 |
| CL-007 | P2 | naming | `subjet_independent` 27경로 | active/docs만 migration; `references/**`는 alias metadata만 기록 | internal/external references | reference path drift 0; active alias는 승인 후 종료 |
| CL-008 | P2 | evidence | handover/final evidence trees | 파일별 supersedes manifest | hash/producer/timestamp 비교 | canonical/latest/archive 판정 후 |
| CL-009 | P2 | vendor output | `third/SAM3-I/predictions.json` | upstream/local-generated 판정 후 artifact 이동 | upstream revision comparison | generated 판정 시 vendor tree 제거 |
| CL-010 | P3 | local cache | ignored `__pycache__` | 작업 트리에서만 정리 | `git ls-files` 비추적 확인 | 즉시 삭제 가능하나 커밋 대상 아님 |

## 판정 규칙

- `P0`: 후속 구조 변경 전에 해결해야 하는 correctness, side-effect, package, rights 문제
- `P1`: 동작 호환성 테스트 아래에서 분해할 핵심 유지보수 부채
- `P2`: owner가 확정된 뒤 수행할 중복·문서·경로 정리
- `P3`: 로컬 또는 낮은 위험의 위생 작업

`dead-code-candidates.csv`의 695개 항목은 이 매트릭스의 삭제 대상이 아니다. 동적 import·외부 소비자·registry를 확인해 높은 신뢰도로 승격된 항목만 별도 행으로 추가한다.

저장소 크기나 exact duplicate 자체는 `references/**` 정리 우선순위 또는 삭제 권한을 만들지 않는다. 외부 mirror가 있더라도 비권위 백업일 뿐 저장소 원본을 대체하지 않는다.
