# HBTXR baseline codebase recon

## 기준

- 모드: `baseline`
- 커밋: `ebe862b11506e819c5bd5abc925299fc5fbb6f1a`
- 권위 순서: Git 추적 트리 → 컴포넌트 README/manifest/schema → 활성 구현 → 테스트 → archive/reference/vendor
- 소스 뷰: `git ls-tree`와 `git cat-file`로 읽은 커밋 blob; dirty working tree 내용은 읽지 않음
- 방법: 프로젝트 모듈을 import하지 않는 AST/텍스트 정적 분석

## Mental model

저장소는 `algorithm`, `quantization`, `hardware`, `provenance`라는 활성 실행 표면과 `analysis`, `archive`, `references`, `third`라는 증거·보존 표면이 한 Git 이력 안에 공존한다. 활성 코드를 평가할 때 reference/vendor의 크기와 코드 냄새를 합산하면 실제 유지보수 우선순위를 왜곡하므로 권위 구역을 분리했다.

대표 흐름은 다음과 같다.

1. Hybrid 학습: `train_hbtxr.py` → `trainer.train()` → `TrainingSession` → data/model/optimizer → external-pipeline tests.
2. Evaluation: `src.evaluation` 계약 → `convert_record()` → HANDOVER bridge tests.
3. Quantization: CLI → package API/pipeline → case/LUT/integer verification → JSON/Markdown report → CLI/API tests.
4. Hardware config: experiment JSON → `normalize_experiment()` → strict schema/manifest → Vivado/HLS command surface → schema tests.
5. Provenance: registry JSON → validator → rights/path/SHA/committed-blob checks → fail-closed tests.

## Bounded audit

기계 인벤토리는 다음 파일에 있다.

- `inventory/file-inventory.csv`: 모든 추적 파일 12,856개
- `inventory/python-symbols.csv`: Python 함수 26,617개, 클래스 3,732개, async 함수 2개
- `inventory/nonpython-symbols.csv`: 보수적 패턴으로 수집한 비-Python 구조 심볼 20,423개
- `inventory/line-findings.csv`: maintained/analysis/test 중심 라인 발견사항 2,233개
- `inventory/duplicate-functions.csv`: 심층 구역의 body-equivalent 심볼 638행/245그룹
- `inventory/duplicate-blobs.csv`: 권위 구역을 가로지르는 0-byte 제외 동일 Git blob 1,280행/519그룹
- `inventory/dead-code-candidates.csv`: 낮은 신뢰도의 이름 기반 후보 695개
- `inventory/import-edges.csv`: 심층 구역 import edge 4,913개

Python blob 3,721개 모두에 정적 파싱을 시도했다. 3,711개는 type-comment 모드로 정상 파싱됐고, 4개는 실행 구문은 유효하지만 잘못된 `# type:` 주석 때문에 fallback 파싱했으며, 6개는 reference 영역의 실제 legacy 구문 오류로 실패했다.

## Pattern evidence

### 유지보수 경계

- Hybrid와 quantization이 모두 일반명 `src`를 사용한다. Hybrid는 `_bootstrap.py`의 `sys.path` 주입에도 의존한다.
- Hardware의 실질 behavioral owner는 `hardware/hls`이며 `hardware/src`와 `hardware/rtl`은 현재 placeholder/migration 경계다.
- Quantization의 `quantization/references/HG-PIPE-Quantization`은 증거 복사본이고 실행 owner는 `quantization/src`다.

### 우선 수정 후보

- `quantization/src/artifact_patch.py`는 두 공개 함수를 두 번 정의하며 뒤 구현이 앞 구현을 가린다.
- `hardware/tools/update_e2e_full_docs_2026_06_09.py`와 `update_e2e_full_vit_2026_06_09.py`는 import 시 파일을 수정한다.
- Hardware validation/signoff 함수 4개는 각각 908~1,954줄이다. 라인·분기·본문 검토는 validation policy, I/O, orchestration, rendering 책임이 결합됐을 가능성을 보여주지만, 이는 분해 전 characterization이 필요한 medium-confidence 추론이다.
- `quantization/src/cli.py`는 parser/main wrapper chain으로 기존 동작을 보존하지만 정의 순서에 강하게 결합된다.
- `algorithm/hybrid/src/data/event_builder.py`와 `training/trainer.py`는 데이터 계약, orchestration, logging/state를 함께 보유한다.

### Cleanup 경계

- `references/legacy-codebase`는 비실행 traceability 영역이며 5,821파일, 약 1.18GB logical bytes다.
- legacy Verilog 30개가 약 636만 줄을 차지하며 큰 두 generated black-box 파일만 각각 약 211만 줄이다.
- 추적 hardware `.bit/.hwh/.bin`은 외부 artifact locator와 hash manifest 없이 삭제하거나 이동하면 안 된다.
- 활성/보존 경계를 가로지르는 동일 blob과 active/reference 소스 복제가 존재한다. canonical owner와 provenance manifest를 먼저 정해야 한다.
- root/hardware/quantization 문서에는 실제 구현·경로와 어긋난 설명이 있다.

## Synthesis

첫 리팩터링 묶음은 correctness/safety에 한정한다. Shadowed definition과 import-time write 뒤 Quantization named-package 정리, CLI, 거대 validator를 별도 호환성 작업으로 진행한다. Hybrid namespace migration은 지원 workflow 충돌이 재현되고 별도 승인될 때만 수행하며 기본은 NO-OP이다. Cleanup은 `references/**` 영구 catalog와 active/reference 관계 기록을 우선하고, 외부화·삭제 검토는 별도 승인된 non-reference binary에만 적용한다.

현재 정적 조사만으로 실제 dead code라고 확정할 수 있는 대상은 alias 없이 가려진 두 앞 정의 정도다. 이름 기반 미사용 후보 695개와 test-name 미도달 후보는 dynamic import와 외부 호출을 반영하지 못하므로 삭제 권한을 부여하지 않는다.

## 조사하지 않은 범위

- 바이너리·모델·데이터·이미지 내용
- 논문/PDF 본문과 notebook cell 실행 의미
- 동적 plugin/registry의 런타임 해석
- 외부 저장소/사용자의 import 및 CLI 소비
- 테스트, 빌드, 학습, 합성, 실험 실행

따라서 이 recon은 코드 변경 완료 판정이 아니라 리팩터링 계획의 근거다.

## 검증기 상태

저장소에 함께 저장한 `validate_recon.py`의 동등 검사는 통과했다. 설치된 공식 `validate-output.sh`는 현재 설치본의 CRLF 줄 끝과 로컬 `jq` 부재 때문에 실행되지 않았으므로 공식 검증기 통과로 기록하지 않는다.
