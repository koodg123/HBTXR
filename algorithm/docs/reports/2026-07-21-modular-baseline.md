# Algorithm Modular Ownership Baseline (AM-000)

## 기준선 잠금

| 항목 | 값 |
|---|---|
| 작업 트리 | `/mnt/d/dataset/EV_Eye/paper_works/HBTXR-Pool/.worktrees/HBTXR-algorithm-modular-ownership` (capture metadata) |
| Git 경계 | HBTXR 저장소의 이 격리 worktree만 해당 |
| 권위 계획 | `/mnt/d/dataset/EV_Eye/paper_works/HBTXR-Pool/HBTXR/docs/aegis/plans/2026-07-21-algorithm-modular-ownership-refactor.md` |
| 권위 계획 SHA-256 | `c73cd5ee0e4f9dd0798b2dbd38bf3c9a50c25e81b996d634c4bc776b022b97b4` |
| 브랜치 | `refactor/algorithm-modular-ownership` |
| 승인 기준선 | `ebe862b11506e819c5bd5abc925299fc5fbb6f1a` |
| 측정 HEAD | `ebe862b11506e819c5bd5abc925299fc5fbb6f1a` |
| 추적 파일 | 12,856개 |
| 기준선 판정 | exact match; ancestry 예외를 사용하지 않음 |

worktree는 실행 기록을 만들기 전에 clean이었다. AM-000 측정 시 HEAD는 계속
승인 기준선과 정확히 같고, 현재 비-clean 상태는 실행 기록과 이 AM-000
산출물로만 구성된다. production, test, config, README, `pyproject.toml`은
수정하지 않았다. 위 절대 worktree path는 2026-07-21 host의 capture metadata이며, 다른 host/session의 portable path contract가 아니다.

## 보존 구역

기준선 tree에는 아래 구역을 합쳐 8,044개의 추적 blob이 있다. 각 blob의
mode, object ID, 경로와 `preserve` disposition은
`docs/provenance/algorithm-modular-paths.tsv`의 `preserved_blob` 행에 있다.
Git은 빈 디렉터리를 추적하지 않으므로 `algorithm/artifacts`의 0은 기준선에서
추적 blob이 없다는 뜻이다.

| 경로 | 추적 blob 수 | 처리 |
|---|---:|---|
| `references` | 5,846 | preserve |
| `algorithm/analysis` | 1,219 | preserve |
| `algorithm/archive` | 680 | preserve |
| `algorithm/artifacts` | 0 | preserve empty-path contract |
| `algorithm/requirements` | 3 | preserve |
| `algorithm/hybrid/hardware_reference` | 280 | preserve |
| `algorithm/event/docs/analysis` | 1 | preserve |
| `algorithm/frame/docs/analysis` | 1 | preserve |
| `algorithm/hybrid/docs/analysis` | 2 | preserve |
| `algorithm/hybrid/tests/handover` | 2 | preserve |
| `algorithm/common/PROVENANCE.md` | 1 | preserve |
| `algorithm/docs/track` | 9 | preserve |

다음 두 검사는 모두 exit 0이었다.

```bash
git diff --quiet ebe862b11506e819c5bd5abc925299fc5fbb6f1a -- \
  references algorithm/analysis algorithm/archive algorithm/artifacts \
  algorithm/requirements algorithm/hybrid/hardware_reference \
  algorithm/event/docs/analysis algorithm/frame/docs/analysis \
  algorithm/hybrid/docs/analysis algorithm/hybrid/tests/handover \
  algorithm/common/PROVENANCE.md algorithm/docs/track

test -z "$(git status --porcelain --untracked-files=all -- \
  references algorithm/analysis algorithm/archive algorithm/artifacts \
  algorithm/requirements algorithm/hybrid/hardware_reference \
  algorithm/event/docs/analysis algorithm/frame/docs/analysis \
  algorithm/hybrid/docs/analysis algorithm/hybrid/tests/handover \
  algorithm/common/PROVENANCE.md algorithm/docs/track)"
```


## Tree-to-TSV equality proof

아래 command는 승인 commit의 12개 보존 경로 tree를 TSV schema로 변환한 뒤
`preserved_blob` 행과 직접 비교한다.

```bash
diff -u \
  <(git ls-tree -r ebe862b11506e819c5bd5abc925299fc5fbb6f1a -- \
    references algorithm/analysis algorithm/archive algorithm/artifacts \
    algorithm/requirements algorithm/hybrid/hardware_reference \
    algorithm/event/docs/analysis algorithm/frame/docs/analysis \
    algorithm/hybrid/docs/analysis algorithm/hybrid/tests/handover \
    algorithm/common/PROVENANCE.md algorithm/docs/track | \
    awk -F '\t' 'BEGIN {OFS="\t"} {split($1, meta, " "); print "preserved_blob", $2, $2, meta[1], meta[3], "preserve", "baseline ebe862b11506e819c5bd5abc925299fc5fbb6f1a"}') \
  <(awk -F '\t' '$1 == "preserved_blob"' \
    docs/provenance/algorithm-modular-paths.tsv)
```

Fresh result: exit 0, empty diff. Row counts were `preserved_rows=8044`,
`mapping_rows=20`, `compatibility_rows=1`.

## 소비자 기준선

모든 수치는 승인 기준선 commit의 tracked content를 대상으로 계산했다.
`EvEye`/`src` import는 실제 Python import 줄만 세기 위해 줄 시작의 공백 뒤
`from`/`import`를 요구한다. 연구 증거인 `algorithm/analysis`의 `EvEye` import
15줄/7파일은 보존하되 migration consumer 수에는 포함하지 않았다.

| 표면 | 측정값 | 이관 판정 |
|---|---:|---|
| `algorithm/common/src/EvEye` | 추적 154파일, Python 130파일 | 구현 owner를 이동; 소비자 증명된 경로만 import-only wrapper 허용 |
| active `EvEye.*` imports | 176줄 / 57파일 | AM-020~AM-060 consumer gate |
| active Hybrid `src.*` imports | 302줄 / 96파일 | Hybrid namespace 유지; `src.pools`만 별도 gate |
| common CLI 표면 | 6파일 | 5개 CLI/notebook을 flatten하고 launcher를 rename |
| common/frame/event configs | 2 / 12 / 24파일 | common 2개만 `algorithm/configs`로 이동 |
| notebooks | 32파일, 모두 `algorithm/common` 아래 | import/path consumer로 재검사 |
| `algorithm/hybrid/src` | Python 122파일 | Hybrid-local owner 유지; pools slice만 예외 |
| broad dynamic-import risk surface | 44줄 / 9파일 | 아래 exact query로 추적; frozen handover 포함 |
| dynamic load/exec subset | 31줄 / 9파일 | 좁힌 보조 수치; broad 44가 fail-closed census |

common CLI 기준선은 다음 6개다.

- `algorithm/common/scripts/facet/inference.py`
- `algorithm/common/scripts/facet/inference_visualize.ipynb`
- `algorithm/common/scripts/facet/train.py`
- `algorithm/common/scripts/facet/validate.py`
- `algorithm/common/scripts/facet/validate10times.py`
- `algorithm/common/scripts/facet_main.py`

외부 저장소의 `EvEye.*`/`src.*` 소비자는 이 worktree만으로 알 수 없다. 따라서
AM-060과 Hybrid namespace retirement는 외부 consumer가 0임이 별도로 증명될
때까지 fail closed다.

## Census 재현 명령과 기대 결과

아래 Bash block은 승인 commit의 tracked content만 읽고 모든 AM-000 census를
계산한 뒤 기대값을 assertion한다.

```bash
baseline_sha=ebe862b11506e819c5bd5abc925299fc5fbb6f1a
ev_pattern='^[[:space:]]*(from[[:space:]]+EvEye([.]|[[:space:]]|$)|import[[:space:]]+EvEye([.]|[[:space:]]|$))'
src_pattern='^[[:space:]]*(from[[:space:]]+src([.]|[[:space:]]|$)|import[[:space:]]+src([.]|[[:space:]]|$))'
dynamic_pattern='(import_module|find_spec|__import__|spec_from_file_location|module_from_spec|sys[.]modules)'
load_exec_pattern='(import_module|find_spec|__import__|spec_from_file_location|module_from_spec|exec_module)'
active_globs=('algorithm/common/**/*.py' 'algorithm/frame/**/*.py' \
  'algorithm/event/**/*.py' 'algorithm/hybrid/**/*.py')

ev_lines=$(git grep -n -E "$ev_pattern" "$baseline_sha" -- \
  "${active_globs[@]}" | awk 'END {print NR}')
ev_files=$(git grep -l -E "$ev_pattern" "$baseline_sha" -- \
  "${active_globs[@]}" | awk 'END {print NR}')
src_lines=$(git grep -n -E "$src_pattern" "$baseline_sha" -- \
  "${active_globs[@]}" | awk 'END {print NR}')
src_files=$(git grep -l -E "$src_pattern" "$baseline_sha" -- \
  "${active_globs[@]}" | awk 'END {print NR}')
dynamic_lines=$(git grep -n -E "$dynamic_pattern" "$baseline_sha" -- \
  "${active_globs[@]}" \
  ':(exclude)algorithm/hybrid/hardware_reference/**' | awk 'END {print NR}')
dynamic_files=$(git grep -l -E "$dynamic_pattern" "$baseline_sha" -- \
  "${active_globs[@]}" \
  ':(exclude)algorithm/hybrid/hardware_reference/**' | awk 'END {print NR}')
load_exec_lines=$(git grep -n -E "$load_exec_pattern" "$baseline_sha" -- \
  "${active_globs[@]}" \
  ':(exclude)algorithm/hybrid/hardware_reference/**' | awk 'END {print NR}')
load_exec_files=$(git grep -l -E "$load_exec_pattern" "$baseline_sha" -- \
  "${active_globs[@]}" \
  ':(exclude)algorithm/hybrid/hardware_reference/**' | awk 'END {print NR}')

cli_files=$(git ls-tree -r --name-only "$baseline_sha" -- \
  algorithm/common/scripts | awk 'END {print NR}')
common_configs=$(git ls-tree -r --name-only "$baseline_sha" -- \
  algorithm/common/configs | awk '/[.]yaml$/ {n++} END {print n+0}')
frame_configs=$(git ls-tree -r --name-only "$baseline_sha" -- \
  algorithm/frame/configs | awk '/[.]yaml$/ {n++} END {print n+0}')
event_configs=$(git ls-tree -r --name-only "$baseline_sha" -- \
  algorithm/event/configs | awk '/[.]yaml$/ {n++} END {print n+0}')
notebooks=$(git ls-tree -r --name-only "$baseline_sha" -- algorithm | \
  awk '/[.]ipynb$/ {n++} END {print n+0}')
eveye_py=$(git ls-tree -r --name-only "$baseline_sha" -- \
  algorithm/common/src/EvEye | awk '/[.]py$/ {n++} END {print n+0}')
hybrid_src_py=$(git ls-tree -r --name-only "$baseline_sha" -- \
  algorithm/hybrid/src | awk '/[.]py$/ {n++} END {print n+0}')

printf 'active_EvEye=%s/%s\n' "$ev_lines" "$ev_files"
printf 'hybrid_src=%s/%s\n' "$src_lines" "$src_files"
printf 'broad_dynamic=%s/%s\n' "$dynamic_lines" "$dynamic_files"
printf 'load_exec=%s/%s\n' "$load_exec_lines" "$load_exec_files"
printf 'CLI=%s configs=%s/%s/%s notebooks=%s EvEye_py=%s Hybrid_src_py=%s\n' \
  "$cli_files" "$common_configs" "$frame_configs" "$event_configs" \
  "$notebooks" "$eveye_py" "$hybrid_src_py"

test "$ev_lines/$ev_files" = '176/57'
test "$src_lines/$src_files" = '302/96'
test "$dynamic_lines/$dynamic_files" = '44/9'
test "$load_exec_lines/$load_exec_files" = '31/9'
test "$cli_files/$common_configs/$frame_configs/$event_configs" = '6/2/12/24'
test "$notebooks/$eveye_py/$hybrid_src_py" = '32/130/122'
```

기대 출력과 fresh 결과는 다음과 같고 모든 assertion은 exit 0이다.

```text
active_EvEye=176/57
hybrid_src=302/96
broad_dynamic=44/9
load_exec=31/9
CLI=6 configs=2/12/24 notebooks=32 EvEye_py=130 Hybrid_src_py=122
```

## Mapping disposition

승인 계획의 current-to-target 20개 행은 TSV의 `mapping` 행으로 1:1 기록했다.
그중 18개는 `move`, `pools/heads.py`와 `pools/__init__.py` 행은 `remove`,
`optim/pool.py`는 `preserve`다. `EvEye.*` 호환 경로는 별도의 `compatibility`
행에서 `wrapper`로 표시했으며, wrapper는 business logic을 소유할 수 없고
maintained/required external consumer가 0일 때 제거한다.

## Baseline test environment and raw evidence

`/tmp/hbtxr-am-venv`는 저장소 밖 disposable environment다.

| Component | Captured version |
|---|---|
| Python | `3.11.14` |
| pytest | `9.1.1` |
| torch | `2.2.2+cu121` |
| torchvision | `0.17.2+cu121` |
| lightning | `2.6.5` |
| pytorch-lightning | `2.6.5` |
| numpy | `1.26.4` |

FULL collection command:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=algorithm/hybrid:algorithm/hybrid/scripts/external_pipeline:algorithm/hybrid/tests/external_pipeline \
  /tmp/hbtxr-am-venv/bin/python -m pytest -s -p no:cacheprovider -q \
  --junitxml=/tmp/hbtxr-am000-collection.xml algorithm/hybrid/tests
```

Result: exit 2, 4 collection errors. Raw JUnit:
`/tmp/hbtxr-am000-collection.xml`, size 3,976 bytes, SHA-256
`fed04906cda3a24595741dbf5ed85858af07b7986a6ca9ebabcd3222df192466`.

PARTIAL command:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=algorithm/hybrid:algorithm/hybrid/scripts/external_pipeline:algorithm/hybrid/tests/external_pipeline \
  /tmp/hbtxr-am-venv/bin/python -m pytest -s -p no:cacheprovider -q \
  --ignore=algorithm/hybrid/tests/external_pipeline/test_target_fps_canonical.py \
  --ignore=algorithm/hybrid/tests/external_pipeline/test_target_fps_dataset.py \
  --ignore=algorithm/hybrid/tests/external_pipeline/test_target_fps_manifest.py \
  --ignore=algorithm/hybrid/tests/external_pipeline/test_timelens_xl_finetune.py \
  --junitxml=/tmp/hbtxr-am000-partial.xml algorithm/hybrid/tests
```

Result: exit 1, 16 failed, 128 passed, 28 subtests passed. Raw JUnit:
`/tmp/hbtxr-am000-partial.xml`, size 59,950 bytes, SHA-256
`13de54df9132a1e39ce5dbaf8ddbf2c7512433315b1e2cd899e0ec6e0825644b`.

PARTIAL의 네 ignore path는 초기 전달본의 잘못된 test-root 경로에서 실제
`algorithm/hybrid/tests/external_pipeline/` 파일 위치로 정정했다. 이 경로가
raw JUnit의 `errors=0` 및 현재 tracked tree와 일치한다.

이 절대 worktree path와 `/tmp` path는 2026-07-21 host의 capture metadata다.
다른 host/session에서 동일 절대 경로가 존재한다는 portable contract가 아니다.
RED baseline과 AM-010 전 사용자 결정 gate는 계속 유효하다.

## 정지 조건과 환경 한계

- HEAD가 승인 기준선에서 설명 없이 바뀌면 즉시 정지한다.
- 보존 구역의 path/blob/status가 하나라도 달라지면 즉시 정지한다.
- 새 implementation owner, dynamic alias 또는 계획되지 않은 adapter가 생기면
  해당 slice를 정지한다.
- 시스템 Python에는 `pytest`, `torch`, `lightning`, `pytorch_lightning`,
  `numpy`가 없다. controller는 저장소 밖의 disposable
  `/tmp/hbtxr-am-venv`로 추가 baseline을 측정했으며 저장소에는 dependency를
  설치하지 않았다.
- compileall은 exit 0, 51개 YAML parse는 exit 0이었다. system Python의 direct
  `train.py --help`는 missing `torch`로 exit 1이었다.
- full Hybrid suite는 capture `FileNotFoundError`로 exit 1/0 tests, capture-off
  보강 run은 남은 4개 collection error로 exit 2였다. 그 4개를 ignore한
  partial suite는 exit 1, 16 failed, 128 passed, 28 subtests passed였다.
- 실패는 exact baseline에 이미 존재하지만 baseline policy상 AM-010 시작 전에
  사용자 진행 결정을 기다린다. 남은 collection error는 잘못된
  `tests.test_*` import, test failure는 parent 경로 계산, snapshot 3건, runtime
  run-root 2건과 help-path 계산 문제를 포함한다.
