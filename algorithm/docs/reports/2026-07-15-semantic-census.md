# HBTXR Function/Class/Line Semantic Census

## 1. 조사 목적과 기준

이 문서는 메인 저장소의 리팩터링과 정리 우선순위를 정하기 위한 정적 의미 전수조사 결과다.

- 대상: `/mnt/d/dataset/EV_Eye/paper_works/HBTXR-Pool/HBTXR`
- 브랜치: `refactor/hbtxr-structure`
- 기준 커밋: `ebe862b11506e819c5bd5abc925299fc5fbb6f1a`
- 조사 방식: `git ls-tree`/`git cat-file` 커밋 blob 전수 인벤토리, Python stdlib AST, 비-Python 선언 패턴, import/중복/라인 규칙 분석
- 소스 격리: dirty working tree 내용은 분석 입력으로 읽지 않음
- 금지 범위: 프로젝트 모듈 import, 테스트·빌드·학습·합성·실험 실행, 파일 삭제, 코드 리팩터링

`function/class/line 전수조사`는 다음 두 계층으로 구현했다.

1. 모든 추적 파일을 파일·언어·권위 구역·크기·라인 단위로 수집한다.
2. 모든 Python 함수·클래스의 시작/끝 라인과 구조를 수집하고, maintained/analysis/test 구역은 본문 복잡도·호출·중복·라인 위험까지 심층 분석한다.

Reference/vendor 코드는 전체 심볼과 라인 범위를 수집하지만 메인 코드의 리팩터링 부채로 합산하지 않는다.

## 2. 전수 범위

### 전체 저장소

| 항목 | 수량 |
|---|---:|
| Git 추적 파일 | 12,856 |
| 추적 파일 logical bytes | 2,183,270,747 |
| 디코딩된 텍스트 라인 | 10,705,773 |
| 코드 라인 proxy | 8,967,553 |
| Python 파일 | 3,721 |
| Python 함수 | 26,617 |
| Python 클래스 | 3,732 |
| Python async 함수 | 2 |
| 비-Python 구조 심볼 | 20,423 |

### 권위 구역

| 구역 | 파일 | bytes | 텍스트 라인 | 의미 |
|---|---:|---:|---:|---|
| maintained | 740 | 199,616,877 | 173,086 | 현재 유지보수 대상 |
| test | 179 | 1,960,248 | 33,273 | 검증 코드 |
| analysis | 883 | 36,136,591 | 151,170 | 분석 도구·중간 분석 코드 |
| artifact | 371 | 94,544,767 | 42,579 | 결과·보고서·생성물 |
| archive | 684 | 20,970,514 | 139,177 | 비활성 보존 코드 |
| reference | 6,295 | 1,314,245,464 | 9,170,421 | 논문·legacy·증거 입력 |
| vendor | 3,247 | 506,134,716 | 824,853 | 외부 third-party 코드 |
| docs | 457 | 9,661,570 | 171,214 | 문서 |

`references/legacy-codebase`만 5,821파일, 1,180,862,505 logical bytes, 9,012,195 텍스트 라인이다. 저장소의 의미적 복잡도보다 보존 데이터의 물리적 규모가 훨씬 크다.

### Python 권위별 심볼

| 구역 | 추적 Python 파일 | 심볼 보유 파일 | 함수·클래스 |
|---|---:|---:|---:|
| maintained | 409 | 351 | 3,832 |
| test | 121 | 121 | 1,205 |
| analysis | 149 | 124 | 589 |
| artifact | 2 | 2 | 21 |
| archive | 163 | 154 | 1,411 |
| reference | 1,466 | 1,006 | 9,658 |
| vendor | 1,411 | 1,107 | 13,635 |

핵심 실행 경계인 `algorithm/hybrid/src`, `quantization/src`, `hardware/tools`, `hardware/pynq/hgtxr`, `tools`에는 Python 255파일과 심볼 2,768개가 있다.

## 3. 파싱과 라인 품질

모든 Python blob 3,721개에 파싱을 시도했다. 아래 3,711+4개에서 함수·클래스 line span을 수집했고 실패 6개는 finding으로 남겼다.

- Python 3,711개는 type-comment 모드로 정상 파싱됐다.
- 4개는 Python 실행 구문은 유효하지만 잘못된 `# type:` 주석을 포함한다.
- 6개는 모두 `references/legacy-codebase/.../PyAedatTools`의 Python 2/legacy 구문 오류다.
- maintained의 두 type-comment 오류:
  - `algorithm/common/src/EvEye/model/DavisEyeEllipse/EPNet/Predict.py:64`
  - `algorithm/common/src/EvEye/model/DavisEyeEllipse/HBTXR/Predict.py:64`

두 줄은 `# type: np.array -> torch.tensor`이며 유효한 타입 표현식이 아니다. 일반 실행에는 영향이 없지만 AST/type checker와 정적 도구의 전수 분석을 방해한다.

### 심층 구역 라인 finding

| 규칙 | 수량 | 해석 |
|---|---:|---|
| hard-coded absolute path | 507 | target/provenance/변환 문자열을 포함하므로 개별 판정 필요 |
| very long line | 324 | 자동 생성·테이블 선언과 일반 코드 분리 필요 |
| large function | 316 | 함수 60/120라인 임계치 |
| library print | 290 | maintained 비-script 모듈의 직접 출력 후보 |
| high branch density | 278 | AST 분기 proxy, McCabe 점수는 아님 |
| work marker | 214 | TODO/FIXME/HACK/XXX 문자열 |
| broad exception | 75 | Exception/BaseException 포착 |
| too many parameters | 69 | 10개 초과 |
| large class | 53 | 클래스 150/300라인 임계치 |
| dynamic execution | 52 | eval/exec 이름 호출 후보, 개별 확인 필요 |
| wildcard import | 34 | export/namespace 불명확성 |
| bare except | 6 | 모두 analysis 구역 |
| mutable default | 5 | maintained이나 strict active에는 없음 |
| invalid type comment | 4 | 실행 구문은 유효 |
| true Python syntax error | 6 | reference 구역만 해당 |

각 행의 정확한 파일·라인·심볼·근거는 `.agents/recon/2026-07-15-semantic-census/inventory/line-findings.csv`에 있다. 이 숫자만으로 자동 수정하거나 삭제하면 안 된다.

## 4. Function/Class hotspot

### 최상위 복잡도

| 우선 | 심볼 | 범위 | LOC | branch proxy | 판단 |
|---|---|---|---:|---:|---|
| P0 | `build_consistency_checks` | `hardware/tools/write_final_evidence_manifest.py:601-2554` | 1,954 | 375 | 규칙·I/O·보고서가 한 함수에 결합 |
| P0 | `run_pipeline` | `hardware/tools/run_third_goal_final_signoff.py:597-2315` | 1,719 | 163 | orchestration과 실행 정책 결합 |
| P0 | `validate_handoff` | `hardware/tools/validate_final_operator_handoff.py:394-1569` | 1,176 | 230 | 검증 규칙과 렌더링 결합 |
| P0 | `validate_bundle` | `hardware/tools/validate_final_signoff_bundle.py:518-1425` | 908 | 113 | bundle schema와 정책 결합 |
| P1 | `main` | `quantization/src/cli.py:244-792` | 549 | 89 | dispatch와 command 구현 결합 |
| P1 | `EventInputBuilder` | `algorithm/hybrid/src/data/event_builder.py:30-543` | 514 | 62 | 입력 계약·변환·검증 결합 |
| P1 | `HBTXRTracker` | `algorithm/hybrid/src/models/hybrid_tracker.py:27-470` | 444 | 24 | 모델 조립과 실행 경계 결합 |
| P1 | `TrainingSession` | `algorithm/hybrid/src/training/trainer.py:886-1294` | 409 | 87 | 상태·logging·epoch orchestration 결합 |
| P1 | `SOAP` | `algorithm/hybrid/src/optim/soap.py:14-301` | 288 | 68 | optimizer policy와 state update 결합 |
| P1 | `TrackerTokenEncoder` | `algorithm/hybrid/src/models/tracker/encoder.py:11-291` | 281 | 60 | shape/attention/variant logic 결합 |

LOC와 branch는 분해 필요성을 보여주는 신호다. 동작 중복이나 버그를 직접 증명하지는 않는다.

### Shadowed definition

`quantization/src/artifact_patch.py`에는 alias 없이 같은 top-level 이름이 반복된다.

- `validate_artifact_patch_matrix_manifest`: `:240`, `:446`
- `write_artifact_patch_matrix_manifest_validation_markdown`: `:331`, `:663`

Python에서는 뒤 정의가 앞 정의를 덮어쓴다. 현재 전수조사에서 가장 높은 신뢰도로 확인된 dead implementation이다.

`quantization/src/cli.py`도 `build_parser`와 `main`을 각각 네 번 정의하지만 `_original_*`로 앞 구현을 연결한다. 이는 dead code가 아니라 의도된 wrapper chain이며, 정의 순서 의존성이 큰 리팩터링 대상이다.

### 중복 구현

심층 구역의 normalized body-equivalent 결과는 245그룹/638행이다. 주요 유형은 다음과 같다.

- `hardware/tools`의 반복 `load_json`, `sha256_file`, root normalization, report writer
- `quantization/src`와 `quantization/references/HG-PIPE-Quantization/src`의 동일 구현
- `algorithm/common`의 EPNet/HBTXR loss·metric 공통 본문
- analysis handover/result 복사본의 motion labeler·evaluation script

작은 getter와 protocol-shaped method도 포함되므로 그룹 전체를 자동 통합하지 않는다. active/reference 동일 구현은 canonical active owner와 영구 reference snapshot의 관계를 provenance로 기록하고, active/active 반복만 공통화 대상으로 승격한다. 이 기록은 reference 삭제·rename·pointer 대체 권한이 아니다.

Git blob 기준으로 권위 구역을 가로지르는 non-empty exact duplicate는 519그룹/1,280행이다. 0-byte placeholder 177개는 의미 있는 정리 중복으로 보지 않아 이 CSV에서 제외했다. Non-empty 그룹 중 maintained와 archive/reference가 직접 교차하는 그룹은 228개다. 전체 근거는 `inventory/duplicate-blobs.csv`이며, 이 수치 역시 소비자·license·provenance 검토 없이 삭제 권한을 부여하지 않는다.

## 5. Class/Module semantic findings

### 패키지 경계 충돌

- Hybrid의 실질 실행 코드는 `algorithm/hybrid/src`지만 `algorithm/pyproject.toml` 설치 대상은 `algorithm/common/src`다.
- Hybrid external pipeline은 `_bootstrap.py:7-12`에서 `sys.path`를 조작한다.
- Quantization도 최상위 패키지명을 `src`로 사용한다.

동일 interpreter에서 import shadowing 가능성은 있지만 지원 workflow의 재현 실패나 동시 배포 요구는 확인되지 않았다. 기본 결정은 modality 구조와 `algorithm/hybrid/src` 보존이다. Quantization만 named package 대상으로 계획하고, Hybrid namespace migration은 clean supported-workflow 재현과 별도 승인이 모두 있을 때만 수행하며 그 전에는 NO-OP이다.

### Import-time side effect

다음 모듈은 import만으로 파일을 읽고 쓸 수 있다.

- `hardware/tools/update_e2e_full_docs_2026_06_09.py:13-17,19-66,136-141`
- `hardware/tools/update_e2e_full_vit_2026_06_09.py:33-42,54-63`

이 코드는 `main(argv)`와 `if __name__ == "__main__"` 경계로 이동하거나 one-shot migration archive로 명시해야 한다.

### Dependency cycle 후보

정적 edge 검토에서 다음 세 묶음이 SCC 후보로 관찰됐다.

- `quantization/src/{api.py, artifact_imagenet.py, artifact_patch.py}`
- `algorithm/hybrid/src/preprocess/{event_generation.py, target_fps_build.py}`
- `algorithm/hybrid/src/preprocess/{annotation_backends.py, groundedsam_build.py}`

역방향 edge는 함수 내부 lazy import이므로 현재 import-time 실패 증거는 아니다. 이는 공통 protocol, data contract, factory owner가 불명확할 수 있다는 medium-confidence 추론이다. 각 edge·line·scope와 refactor 필요성 게이트는 `.agents/recon/2026-07-15-semantic-census/import-scc-evidence.json`에 기록했다. 구체적 실패나 중복 owner가 입증되지 않으면 현 lazy boundary를 문서화하고 유지한다.

### Exception/path handling

- `algorithm/hybrid/src/preprocess/io_utils.py:379-383,398-417`은 모든 예외를 skip count로 축약해 프로그래밍 오류까지 숨길 수 있다.
- `hardware/tools/gen_cleanup_dry_run_2026_06_09.py:10`은 `/home/user/project/PRJXR`를 실제 ROOT로 고정한다.
- `algorithm/hybrid/src/optim/registry.py:62`의 Windows 경로는 runtime path라기보다 provenance 메타데이터다.
- `/home/xilinx/...` 일부는 board target 계약이므로 무조건 제거하면 안 된다.

경로 finding 507개는 runtime/config/provenance/test/transformer 문자열로 분류한 후 처리해야 한다.

## 6. Test reachability와 dead-code 한계

이름 기반 heuristic으로 695개 public maintained 심볼이 낮은 참조 수를 보였다. 그러나 다음을 해석하지 못한다.

- dynamic import와 registry
- CLI entry point
- reflection과 callback
- 외부 저장소/사용자 import
- attribute alias와 protocol 구현

따라서 695개는 삭제 목록이 아니라 후속 call graph 확인 목록이다. Alias 없이 가려진 `artifact_patch.py` 앞 정의 두 개만 높은 신뢰도의 retirement 후보다.

엄격한 활성 경계의 public top-level 심볼 가운데 동일 이름이 테스트 AST에서 보이지 않는 항목도 많지만 이는 coverage가 아니다. 실제 삭제나 API 변경 전 focused characterization test가 필요하다.

## 7. Repository Cleanup census

### Legacy/reference

- `references/legacy-codebase`: 5,821파일, 약 1.18GB logical bytes
- 해당 영역 Verilog: 30파일, 약 636만 줄
- `BlockSequence_bb.v`: 68,767,039 bytes, 2,115,721줄
- `BlockSequence_bb_replaced.v`: 68,737,966 bytes, 2,115,721줄
- 코드가 아닌 archive, model, event data, bitstream, subject data가 함께 존재

`references/README.md`는 이 영역을 비실행 traceability로 정의한다. 사용자 결정에 따라 비교 실험용 영구 in-repository reference로 보존한다. Source revision·license·SHA-256·active owner 관계는 catalog에 기록하되 삭제, rename, deduplicate, 외부 pointer 대체는 수행하지 않는다. 외부 archive가 필요하면 비권위 백업 mirror로만 다룬다.

### Hardware binaries

| 확장자 | 파일 | bytes |
|---|---:|---:|
| `.bit` | 9 | 173,801,106 |
| `.hwh` | 9 | 3,897,659 |
| `.bin` | 1 | 12,192,768 |
| `.npy` | 4 | 786,992 |
| `.npz` | 2 | 49,190 |

Bitstream은 PYNQ와 packaging 소비자가 있을 수 있으므로 artifact locator와 checksum manifest를 먼저 도입해야 한다.

### 문서/레이아웃 drift

- 루트 `README.md:9`는 hardware를 future 영역으로 설명하지만 실제 HLS/tools/tests가 존재한다.
- `hardware/docs/architecture/DIRECTORY_LAYOUT.md`의 target 경로 일부가 아직 생성되지 않았다.
- `hardware/src/*`는 placeholder이고 실제 구현은 `hardware/hls`다.
- `quantization/README.md`는 존재하지 않는 `../ICCAD24-HG-PIPE`를 참조한다.
- `subjet_independent` 오탈자가 포함된 추적 경로가 27개 있다.

Active/docs 경로 이름 변경은 내부 링크와 실험 증거 인용을 함께 바꾸는 migration으로 수행한다. `references/**` 오탈자 경로는 변경하지 않고 alias metadata로만 설명한다.

## 8. 판단 요약

### 즉시 계획에 넣을 항목

1. Shadowed quantization 정의의 동작 characterization과 단일 owner화
2. Import-time write 제거
3. `src` namespace와 packaging 경계 ADR
4. Quantization CLI command registry
5. 거대 hardware validator의 pure rule/I/O/render 분리
6. Training/data class 분해
7. Broad exception과 실행 경로 주입
8. Active/reference 중복 owner 지정
9. Manifest-first legacy/binary 외부화
10. README/layout 정합성 복구

### 자동 정리하지 않을 항목

- 695개 dead-code heuristic 후보
- reference/vendor의 syntax/style 문제
- board-specific 경로
- 서로 내용이 다른 evidence tree
- source/license/hash가 없는 legacy 파일
- 현재 PYNQ 소비자를 확인하지 않은 bitstream

## 9. 산출물 인덱스

- Recon manifest: `.agents/recon/2026-07-15-semantic-census/codebase-recon.json`
- Recon report: `.agents/recon/2026-07-15-semantic-census/codebase-recon.md`
- Scanner: `.agents/recon/2026-07-15-semantic-census/semantic_scan.py`
- Summary: `.agents/recon/2026-07-15-semantic-census/inventory/semantic-summary.json`
- File census: `.agents/recon/2026-07-15-semantic-census/inventory/file-inventory.csv`
- Python symbols: `.agents/recon/2026-07-15-semantic-census/inventory/python-symbols.csv`
- Non-Python symbols: `.agents/recon/2026-07-15-semantic-census/inventory/nonpython-symbols.csv`
- Line findings: `.agents/recon/2026-07-15-semantic-census/inventory/line-findings.csv`
- Duplicate functions: `.agents/recon/2026-07-15-semantic-census/inventory/duplicate-functions.csv`
- Dead-code candidates: `.agents/recon/2026-07-15-semantic-census/inventory/dead-code-candidates.csv`
- Import edges: `.agents/recon/2026-07-15-semantic-census/inventory/import-edges.csv`
- Exact duplicate blobs: `.agents/recon/2026-07-15-semantic-census/inventory/duplicate-blobs.csv`
- Import SCC evidence: `.agents/recon/2026-07-15-semantic-census/import-scc-evidence.json`
- Stored fallback validator: `.agents/recon/2026-07-15-semantic-census/validate_recon.py`

설치된 공식 recon validator는 설치 파일의 CRLF 줄 끝과 `jq` 부재로 실행되지 않았다. 저장한 Python 동등 검사는 통과했으나 이를 공식 validator 통과로 표현하지 않는다.

## 10. 미조사 범위

- 바이너리·모델·데이터·이미지의 내부 의미
- PDF와 notebook cell 실행 의미
- runtime plugin/registry 해석
- 외부 소비자와 배포 환경
- 테스트·빌드·합성·실험 결과

이 조사는 리팩터링 계획의 근거이며 동작 동일성이나 삭제 안전성을 확정하는 실행 검증은 아니다.
