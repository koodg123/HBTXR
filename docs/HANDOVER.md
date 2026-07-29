# HBTXR Algorithm Modular Refactor HANDOVER

> **작성** 2026-07-21 (Asia/Seoul) · **갱신** 2026-07-29
> **상태** **CONSUMED / SUPERSEDED — 실행 지침으로 따르지 마십시오**
> **소유** repo

> ## ⚠️ 2026-07-29 만료 고지
>
> **이 핸드오프는 소비되었습니다.** 아래 본문은 2026-07-21 시점의 상태를 기록한 것이며,
> 그 이후 두 번 무효화되었습니다:
>
> 1. **AM 작업이 계속 진행됨.** 본문의 "AM-030 완료, AM-040 착수 전"은 그날 안에
>    낡았습니다. AM-900까지 실행되었습니다 (`6b52483` AM-090 no-op, `7944b47` AM-900 gate).
> 2. **목표 트리가 대체됨.** 2026-07-22 `6c1cc54`가 `eveye` 네임스페이스를 제거하고 bare
>    package로 전환했습니다 — 이 문서가 전제하는 `eveye.*` 패키징이 더 이상 목표가
>    아닙니다. 현재 브랜치 `rewrite/flat-functional`이 그 결과입니다.
>
> 또한 본문이 전제하는 **"의도적인 미커밋 상태"는 존재하지 않습니다.** 당시의 dirty
> 상태는 이후 전부 커밋·푸시되었습니다.
>
> **현재 상태를 알고 싶다면**: [track/PROGRESS.md](track/PROGRESS.md) ·
> [track/TODO.md](track/TODO.md) · [../algorithm/docs/track/log.md](../algorithm/docs/track/log.md)
>
> 아래 본문은 **역사 기록으로만** 보존합니다. 편집하지 않습니다.
> 핸드오프는 소비되면 만료되어야 한다는 규칙은
> [governance/DOC-CONVENTIONS.md](governance/DOC-CONVENTIONS.md)에 있습니다.

---

> 작성일: 2026-07-21 (Asia/Seoul)
> 상태: AM-030 완료 및 승인, AM-040 착수 전 *(당시 기록 — 위 만료 고지 참조)*
> 목적: 다른 Codex 세션이 현재의 의도적인 미커밋 상태를 손상하지 않고 순차 리팩토링을 계속하기 위한 단일 권위 문서

## 1. 즉시 읽을 것과 첫 행동

다음 순서로 읽는다.

1. 이 `HANDOVER.md`
2. 주 작업 트리의 `docs/aegis/plans/2026-07-21-algorithm-modular-ownership-refactor.md`
3. 실행 작업 트리의 `docs/aegis/work/2026-07-21-algorithm-modular-ownership-execution/10-intent.md`
4. 같은 디렉터리의 `20-checkpoint.md`와 `90-evidence.md`
5. `docs/track/algorithm-modular-baseline.md`

첫 행동은 **편집이 아니라 실행 작업 트리 존재 여부와 Git 상태 확인**이다.

```bash
test -d /tmp/hbtxr-algorithm-modular-ownership-exec/.git || \
  test -f /tmp/hbtxr-algorithm-modular-ownership-exec/.git
test "$(git -C /tmp/hbtxr-algorithm-modular-ownership-exec branch --show-current)" = \
  "refactor/algorithm-modular-ownership-exec"
test "$(git -C /tmp/hbtxr-algorithm-modular-ownership-exec rev-parse HEAD)" = \
  "5d94dc5aa745d09b25c8414f42e5dbe3e3ccd5b4"
git -C /tmp/hbtxr-algorithm-modular-ownership-exec \
  merge-base --is-ancestor 5d94dc5aa745d09b25c8414f42e5dbe3e3ccd5b4 HEAD
git -C /tmp/hbtxr-algorithm-modular-ownership-exec status --short --branch
git -C /tmp/hbtxr-algorithm-modular-ownership-exec diff HEAD --stat
git -C /tmp/hbtxr-algorithm-modular-ownership-exec diff HEAD --check
git -C /tmp/hbtxr-algorithm-modular-ownership-exec diff --cached --check
```

`/tmp/hbtxr-algorithm-modular-ownership-exec`가 없으면 **중단한다**. 현재의
case-only `EvEye`/`eveye` 이중 상태는 일반적인 checkout이나 patch 재적용으로
안전하게 복원할 수 없다. 새 worktree를 만들거나 변경을 추측하여 재구성하지 말고
사용자에게 상태 유실을 보고한다.

## 2. 목표와 사용자 결정

목표는 `algorithm`을 다음 상위 구조 중심으로 정리하는 것이다.

- 유지되는 활성 영역: `common`, `dataset`, `utils`, `engine`, `configs`,
  `tests`, `frame`, `event`, `hybrid`
- 그대로 보존: `analysis`, `archive`, `artifacts`, `requirements`,
  `README.md`, `pyproject.toml`
- 활성 코드에서 제거할 이름: `facet`, `EvEye`, `HBTXR_*`
- `hybrid/src/pools`는 제거하되 `losses.py`, `optimizer.py`,
  `lr_schedulers.py`, `runtime_schedulers.py`의 도메인 API는 유지한다.
- frame/event 공통 코드는 common으로 두고 Hybrid 고유의 결합, 검색/추적,
  runtime/training 동작은 Hybrid에 유지한다.
- `references/**`와 연구/이력 자료는 이름을 바꾸거나 이동하지 않는다.

사용자는 2026-07-21에 **리팩토링 전 테스트 RED를 중단 조건으로 삼지 말고,
리팩토링 결과를 대상으로 테스트하라**고 결정했다. 기존 실패는 숨기거나 면제하지
않고 AM-900에서 결과 차이를 비교한다.

## 3. 권위 계획과 기준선

- 계획 파일: `/mnt/d/dataset/EV_Eye/paper_works/HBTXR-Pool/HBTXR/docs/aegis/plans/2026-07-21-algorithm-modular-ownership-refactor.md`
- 계획 SHA-256: `c73cd5ee0e4f9dd0798b2dbd38bf3c9a50c25e81b996d634c4bc776b022b97b4`
- 승인 기준 commit: `ebe862b11506e819c5bd5abc925299fc5fbb6f1a`
- AM-000 기준선 commit: `c7d6bfe2bd1e3875bb76a2e556b9d4ece3944d2b`
- AM-010 패키징 commit: `5d94dc5aa745d09b25c8414f42e5dbe3e3ccd5b4`

계획 파일의 checksum을 재확인한다.

```bash
sha256sum /mnt/d/dataset/EV_Eye/paper_works/HBTXR-Pool/HBTXR/docs/aegis/plans/2026-07-21-algorithm-modular-ownership-refactor.md
```

## 4. 이 작업과 관련된 세 worktree의 정확한 역할

### 4.1 주 작업 트리 — 절대 정리하지 말 것

- 경로: `/mnt/d/dataset/EV_Eye/paper_works/HBTXR-Pool/HBTXR`
- branch: `refactor/hbtxr-structure`
- HEAD: `ebe862b11506e819c5bd5abc925299fc5fbb6f1a`
- 상태: planning/recon 문서가 의도적으로 modified/untracked인 dirty 상태

이 작업 트리에서는 `reset`, `checkout`, `restore`, `clean`, `stash`, stage,
commit을 하지 않는다. 기존 planning/recon 변경은 사용자의 작업이다. 이
`HANDOVER.md`와 `HANDOVER-PROMPT.md`도 요청에 따라 여기에 생성되었지만 stage나
commit하지 않았다.

### 4.2 AM-000 보존 worktree

- 경로: `/mnt/d/dataset/EV_Eye/paper_works/HBTXR-Pool/.worktrees/HBTXR-algorithm-modular-ownership`
- branch: `refactor/algorithm-modular-ownership`
- HEAD: `c7d6bfe2bd1e3875bb76a2e556b9d4ece3944d2b`
- 역할: AM-000 기준선 및 보존 manifest의 커밋 경계

등록된 네 번째 worktree인
`/mnt/d/dataset/EV_Eye/paper_works/HBTXR-Pool/.worktrees/HBTXR-integration-sparse`
(branch `refactor/hbtxr-code-file-integration`, HEAD
`c24030f969a29f47b25f1771bbf6a15550afe926`)는 이 작업과 무관하므로 건드리지 않는다.

### 4.3 활성 실행 worktree — 여기서만 계속 작업

- 경로: `/tmp/hbtxr-algorithm-modular-ownership-exec`
- branch: `refactor/algorithm-modular-ownership-exec`
- HEAD: `5d94dc5aa745d09b25c8414f42e5dbe3e3ccd5b4`
- HEAD commit: `build(algorithm): define modular package origins`
- 상태: AM-020/AM-030 변경이 staged, unstaged, untracked로 함께 존재하는
  의도적 dirty 상태

2026-07-21 인계 시점의 census:

- porcelain status summary row: 127
- staged tracked entry: 64
- unstaged tracked entry: 78
- porcelain `??` summary row: 18, 실제 untracked file: 54
- porcelain 열 marker는 index 82/worktree 96이지만 `??`가 양쪽 열에 잡히므로
  실제 staged/unstaged entry 수로 해석하면 안 된다.
- ignored `__pycache__`/`.pyc`: 127
- `git diff HEAD --stat`: tracked 116 files, `+276/-318`; untracked 54개는 제외됨

다음으로 재확인할 수 있다.

```bash
cd /tmp/hbtxr-algorithm-modular-ownership-exec
printf 'status_rows='; git status --porcelain=v1 | wc -l
printf 'staged_tracked='; git diff --cached --name-status | wc -l
printf 'unstaged_tracked='; git diff --name-status | wc -l
printf 'untracked_files='; git ls-files --others --exclude-standard | wc -l
printf 'ignored_python_cache='; \
  git ls-files --others -i --exclude-standard | \
  rg '(^|/)(__pycache__/|.*\.pyc$)' | wc -l
git diff HEAD --numstat | awk \
  'BEGIN{f=0;a=0;d=0} {f++; if($1~/^[0-9]+$/)a+=$1; if($2~/^[0-9]+$/)d+=$2} \
   END{print "files=" f, "added=" a, "deleted=" d}'
```

숫자가 달라졌다면 바로 reset하지 말고, 먼저 변경 주체와 차이를 조사한다.

## 5. 중요한 파일시스템 위험

현재 source tree에는 대소문자만 다른 `EvEye`와 `eveye`가 동시에 존재한다.
Windows/DrvFS나 `/mnt/c`의 case-folding 임시 경로를 쓰면 두 package payload가
충돌한다.

모든 packaging/editable-install 테스트 전에 Linux 임시 경로를 강제한다.

```bash
export TMPDIR=/tmp
export TEMP=/tmp
export TMP=/tmp
```

AM-060에서 `algorithm/common/src/EvEye`가 완전히 제거되기 전에는 AM-020 이후의
dual-case migration state를 commit하지 않는다. 현재 index/worktree를
정리하거나 재배치하지 않는다.

실행 worktree에는 ignored `__pycache__`/`.pyc` 파일 127개가 있다. 이는 검증 중
생성된 cache이며 AM 작업 산출물이 아니다. stage하지 말고, 삭제도 명시적인 안전
gate와 사용자 권한 없이 수행하지 않는다.

## 6. 완료된 작업과 검증 결과

### AM-000 — 기준선 및 보존 경계 고정: 완료/커밋됨

- commit: `c7d6bfe`
- 기준선 tracked file 12,856개
- 보존 대상 tracked blob 8,044개를 mode/object ID/path와 함께 기록
- mapping 20개와 compatibility wrapper 1개 기록
- 산출물:
  - `docs/track/algorithm-modular-baseline.md`
  - `docs/provenance/algorithm-modular-paths.tsv`
- preserved tree와 TSV의 equality, preserved-zone diff/status 모두 통과

### AM-010 — package/import 계약: 완료/커밋됨

- commit: `5d94dc5`
- 단순 multi-root discovery 대신 explicit package-to-source mapping과
  exact/descendant allowlist를 적용
- compatibility test 3개 통과
- synthetic 6-owner wheel build/clean install/arbitrary-CWD import 통과
- spec review 및 quality review 승인

### AM-020 — dataset owner 이동: 완료/승인됨/미커밋

- canonical owner: `algorithm/dataset/src/eveye/dataset`
- legacy `EvEye.dataset`는 explicit alias wrapper만 유지
- DavisEyeCenter 실행 demo를
  `eveye.engine.tools.inspect_davis_eye_center`로 분리
- focused regression: **20 passed**
- registry name/origin, wrapper identity, NPY shape/dtype/center,
  ellipse coordinate 계약 통과
- spec review 및 quality review 승인

### AM-030 — reusable utils 이동: 완료/승인됨/미커밋

- canonical owner: `algorithm/utils/src/eveye/utils`
- maintained consumer와 canonical dataset import를 `eveye.utils`로 변경
- focused gate: **24 passed, 1 skipped**
- skip은 matplotlib/seaborn 미설치로 인한 optional plotting runtime identity;
  structural wrapper 계약은 통과
- compileall, import census, notebook JSON/AST 25개, preserved-zone 검사 통과
- spec review 및 quality review 승인
- AM-040으로 남긴 세 파일:
  - `algorithm/common/src/EvEye/utils/cache/NpyCacheFrameStack.py`
  - `algorithm/common/src/EvEye/utils/cache/MemmapCacheFrameStack.ipynb`
  - `algorithm/common/src/EvEye/utils/tonic/tonicLearning.ipynb`
  이들은 dataset/orchestration 의존성이 있어 leaf utils로 이동하면 안 된다.

주의: `20-checkpoint.md` 본문의 오래된 한 줄에는 다음 작업이 AM-030이라고 적혀
있지만, 같은 문서의 active slice와 `90-evidence.md`가 보여 주듯 AM-030은 이미
완료되었다. **재개 지점은 AM-040이다.** 현재
`common/src/eveye/common`, `event/src/eveye/event`,
`algorithm/tests/{common,engine,event}`는 아직 생성되지 않았다. AM-020에서 만든
engine inspect tool만 존재하므로 AM-040 구현은 부분 착수된 상태가 아니다.

## 7. 리팩토링 전 테스트 기준선

AM-000에서 Hybrid 전체 테스트 수집은 4개 collection error로 종료되었다.
해당 4개를 제외한 부분 실행은 다음과 같다.

- 16 failed
- 128 passed
- 28 subtests passed

주요 기존 결함은 잘못된 `tests.test_*` import 경로, parent path 계산, snapshot
drift 3건, runtime run-root 불일치 2건, `hybrid/tests` 아래로 잘못 잡힌 help
경로다. 이것을 AM-040의 별도 선행 복구 조건으로 사용하지 않는다. AM-900에서
리팩토링 후 collection error/fail/pass/subtest 수와 실패 identity delta를 이
기준선에 맞춰 보고한다.

## 8. 현재 작업: AM-040 Task Card

```yaml
task_card:
  task_id: AM-040
  sub_agent: codex-native
  role: implementer
  objective: common, engine, event의 canonical ownership을 분리하고 registry 및 소비자 계약을 보존한다.
  file_ownership:
    - algorithm/common/src/EvEye/callback/**
    - algorithm/common/src/EvEye/logger/**
    - algorithm/common/src/EvEye/model/**
    - algorithm/common/src/EvEye/utils/scripts/**
    - algorithm/common/src/EvEye/utils/cache/NpyCacheFrameStack.py
    - algorithm/common/src/EvEye/utils/cache/MemmapCacheFrameStack.ipynb
    - algorithm/common/src/EvEye/utils/tonic/tonicLearning.ipynb
    - algorithm/common/src/eveye/common/**
    - algorithm/engine/src/eveye/engine/**
    - algorithm/event/src/eveye/event/**
    - algorithm/tests/common/**
    - algorithm/tests/engine/**
    - algorithm/tests/event/**
  assigned_skill:
    - aegis:executing-plans
    - aegis:verification-before-completion
  inputs:
    - approved AM-040 plan
    - live AM-020 and AM-030 dirty state
    - current model registry names and constructor behavior
  outputs:
    - common shared model owner
    - engine callback/logger/factory/tools owner
    - event-only EPNet, ElNet, TennSt owner
    - focused owner and compatibility tests
  validation:
    - registry names and constructor forwarding snapshot
    - focused pytest for common, engine, event
    - dependency-direction rg gates
    - compileall and import-origin checks
    - preserved-zone drift zero
    - independent spec review, then quality review
  dependencies:
    - AM-020 approved
    - AM-030 approved
```

### AM-040 정확한 owner 이동

- `common/src/EvEye/callback/**` → `engine/src/eveye/engine/callback/**`
- `common/src/EvEye/logger/**` → `engine/src/eveye/engine/logger/**`
- `common/src/EvEye/model/model_factory.py` →
  `engine/src/eveye/engine/model_factory.py`
- `common/src/EvEye/utils/scripts/**` →
  `engine/src/eveye/engine/tools/**`
- HBTXR `Predict.py::main`을 먼저
  `engine/src/eveye/engine/tools/predict_shared_ellipse.py`로 분리
- EPNet, ElNet, TennSt → `event/src/eveye/event/models/**`
- 나머지 shared/uncertain model → `common/src/eveye/common/models/**`

세 AM-030 제외 파일은 내용을 다시 분류한다. orchestration tool 성격이면
`eveye.engine.tools`로 이동하거나 그곳에 실행 부분을 추출한다. leaf utils로
억지 이동하지 않는다.

### EllipseMobileNet 미결정 사항

`algorithm/common/src/EvEye/model/DavisEyeEllipse/EllipseMobileNet.py`에는 비활성
legacy import가 하나 남아 있다.

```python
from EvEye.dataset.DavisEyeEllipse.losses import cal_loss
```

AM-020에서 주석만 추가했으며 maintained consumer는 0이다. AM-040에서
`cal_loss`의 의미를 확인한 뒤 다음 중 증거에 맞는 처리를 선택한다.

- 실제 shared model 계약이면 canonical shared/common 위치로 명시적으로 이동
  또는 import 변경
- event-only 계약이면 event owner의 공개 API로 이동
- dead demo/import라면 사용 증거와 test를 남긴 뒤 제거

근거 없이 Hybrid loss에 연결하거나 새 registry/fallback owner를 만들지 않는다.

### AM-040 구현 순서

1. 현재 model registry 이름, module origin, constructor forwarding을 테스트로
   고정한다.
2. HBTXR prediction `main`과 dependency-bearing 실행 코드를 engine tool로 먼저
   분리한다.
3. callback/logger/factory, shared model, event model을 각각 한 번만 이동한다.
4. maintained import를 canonical `eveye.*` 경로로 고친다.
5. consumer-proven legacy wrapper만 explicit alias/`__all__`로 유지한다.
6. 세 제외 파일과 EllipseMobileNet disposition을 명시적으로 기록한다.
7. Linux `/tmp`를 강제한 focused tests와 dependency gate를 실행한다.
8. spec-compliance review 통과 후 code-quality review를 별도로 받는다.
9. AM-060 전이므로 commit하지 않고 AM-050으로 넘긴다.

Focused verification의 계획상 핵심 명령은 다음과 같다.

```bash
export TMPDIR=/tmp TEMP=/tmp TMP=/tmp
cd /tmp/hbtxr-algorithm-modular-ownership-exec
am_pkg_tmp="$(mktemp -d)"
python3 -m venv --system-site-packages "$am_pkg_tmp/edit-venv"
"$am_pkg_tmp/edit-venv/bin/python" -c \
  'import pytest, torch, lightning, pytorch_lightning'
"$am_pkg_tmp/edit-venv/bin/python" -m pip install --no-deps -e algorithm
PYTHONDONTWRITEBYTECODE=1 "$am_pkg_tmp/edit-venv/bin/python" \
  -m pytest -p no:cacheprovider -q \
  algorithm/tests/common algorithm/tests/engine algorithm/tests/event
! rg -n '^[[:space:]]*(from|import)[[:space:]]+(eveye\.(dataset|common|engine|event|hybrid)|src)([.[:space:]]|$)' \
  algorithm/utils/src/eveye/utils
! rg -n '^[[:space:]]*(from|import)[[:space:]]+(eveye\.(event|hybrid)|src)([.[:space:]]|$)' \
  algorithm/dataset/src/eveye/dataset algorithm/common/src/eveye/common
! rg -n '^[[:space:]]*(from|import)[[:space:]]+(eveye\.hybrid|src)([.[:space:]]|$)' \
  algorithm/engine/src/eveye/engine
```

## 9. 이후 순서

AM-040 이후 계획 순서를 바꾸지 않는다.

1. AM-050: common configs/tests 이동, `facet` launcher flattening,
   `facet_main.py` → `sagemaker_launcher.py`
2. AM-060: maintained `EvEye` import 0, wheel payload 검사 후 legacy
   `common/src/EvEye/**` 제거 및 package discovery를 `eveye*`로 축소
3. AM-070: `hybrid/src/pools` 제거, loss/optimizer/LR/runtime scheduler의
   domain API 유지, `optim/pool.py` 보존
4. AM-080: frame/event config 계약과 runnable smoke surface 검증;
   빈 `frame/src`는 만들지 않음
5. AM-090: 두 번째 실제 modality consumer와 parity fixture가 있을 때만 Hybrid
   pure leaf를 shared로 추출; 없으면 문서화된 NO-OP
6. AM-900: compileall, 전체 tests, wheel/install/import, CLI help,
   forbidden-name 및 preserved-zone 최종 gate와 AM-000 대비 테스트 delta 보고

## 10. 절대 보존 경계

다음 경로는 수정, 이동, 이름 변경 또는 정리하지 않는다.

- `references`
- `algorithm/analysis`
- `algorithm/archive`
- `algorithm/artifacts`
- `algorithm/requirements`
- `algorithm/hybrid/hardware_reference`
- `algorithm/event/docs/analysis`
- `algorithm/frame/docs/analysis`
- `algorithm/hybrid/docs/analysis`
- `algorithm/hybrid/tests/handover`
- `algorithm/common/PROVENANCE.md`
- `algorithm/docs/track`

각 slice 후 다음 검사 결과가 비어 있어야 한다.

```bash
git status --porcelain --untracked-files=all -- \
  references algorithm/analysis algorithm/archive algorithm/artifacts \
  algorithm/requirements algorithm/hybrid/hardware_reference \
  algorithm/event/docs/analysis algorithm/frame/docs/analysis \
  algorithm/hybrid/docs/analysis algorithm/hybrid/tests/handover \
  algorithm/common/PROVENANCE.md algorithm/docs/track
```

## 11. 안전 및 권한 경계

- push, merge, tag, PR 생성, release를 하지 않는다.
- primary worktree나 보존 worktree를 cleanup하지 않는다.
- `git reset --hard`, `git clean`, checkout/restore를 사용하지 않는다.
- dual-case 상태에서는 commit하지 않는다.
- 새 owner, wildcard wrapper, `sys.modules` alias, `__path__` trick 또는
  묵시적 fallback을 만들지 않는다.
- test가 실패하면 결과와 failure identity를 기록한다. 기존 RED라는 이유로
  새로운 회귀를 숨기지 않는다.
- 계획과 live tree가 충돌하면 변경하지 말고 사용자에게 drift를 보고한다.

## 12. 재개 완료 조건

다음 세션의 첫 checkpoint에는 최소한 다음이 포함되어야 한다.

- 세 worktree의 branch/HEAD와 활성 worktree의 정확한 status census
- 계획 SHA-256 일치 여부
- preserved-zone status가 비어 있는지
- AM-020/030 변경이 그대로 존재하는지
- AM-040 registry characterization test 결과
- 세 제외 파일과 EllipseMobileNet `cal_loss` 처리 결정 및 근거
- focused tests, dependency gates, spec review, quality review 결과
