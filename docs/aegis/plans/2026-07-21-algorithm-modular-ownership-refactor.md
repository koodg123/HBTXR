# Goal

`algorithm/`의 active code ownership을 `common`, `dataset`, `utils`, `engine`,
`configs`, `tests`, `frame`, `event`, `hybrid`로 명확히 나누되, 연구 증거와
패키징 표면은 제자리에서 보존한다. 이 계획은 기존 동작을 먼저 고정한 뒤
작은 수직 slice로 이동하며, 한 번의 대규모 tree rewrite를 허용하지 않는다.

성공 조건은 다음과 같다.

- `algorithm/analysis`, `algorithm/archive`, `algorithm/artifacts`,
  `algorithm/requirements`의 baseline 대비 path/blob drift가 0이다.
- `algorithm/README.md`와 `algorithm/pyproject.toml`은 같은 경로에 남는다.
  README 설명과 package discovery 내용은 새 구조에 맞게 갱신할 수 있다.
- active top-level owner는 `common`, `dataset`, `utils`, `engine`, `docs`,
  `configs`, `tests`, `frame`, `event`, `hybrid`로 표현된다.
- 공통 Python 구현은 bare `dataset`, `utils`, `engine` import를 만들지 않고
  `eveye.*` named namespace를 사용한다.
- `frame`은 frame-only surface, `event`는 event-only surface, `hybrid`는
  frame-event fusion/search-track surface를 소유한다.
- 실제 두 번째 modality consumer가 없는 Hybrid 코드는 `common`, `dataset`,
  `utils`, `engine`으로 이동하지 않는다.
- `algorithm/hybrid/src/pools`와 `src.pools`는 제거되지만 다음 API 구현은
  domain owner로 이동해 유지된다.
  - `hybrid/src/loss/losses.py`
  - `hybrid/src/optim/optimizer.py`
  - `hybrid/src/optim/lr_schedulers.py`
  - `hybrid/src/runtime/runtime_schedulers.py`
- `hybrid/src/optim/pool.py`의 optimizer candidate/report 동작은 유지된다.
- 최종 active source에는 `FACET/facet` 경로·식별자와 새 `HBTXR_*` 환경
  제어가 없으며 reference/history identity는 변경하지 않는다.

# Architecture

Modular-monolith ownership을 사용한다. 낮은 계층은 높은 계층이나 modality를
import하지 않는다.

```text
common contracts
  ├─> dataset
  ├─> utils
  └─> engine
        ├─> frame
        ├─> event
        └─> hybrid
```

`common`은 공유 contract와 다중 modality model을, `dataset`은 sample/loading,
`utils`는 leaf helper, `engine`은 실행 lifecycle을 소유한다. `frame`, `event`,
`hybrid`는 각 modality의 특수 동작만 소유한다. `hybrid`는 shared layer를
사용할 수 있지만 shared layer는 `hybrid`를 import할 수 없다.

# Tech Stack

- Python 3.10+
- setuptools with PEP 420 namespace package discovery
- PyTorch, Lightning, pytest
- YAML configuration
- Git path/blob verification and semantic census

# Baseline/Authority Refs

- Commit: `ebe862b11506e819c5bd5abc925299fc5fbb6f1a`
- `algorithm/pyproject.toml`
- `algorithm/README.md`
- `algorithm/docs/Modality-Reorganization.md`
- `docs/aegis/plans/2026-07-15-hbtxr-semantic-refactor-cleanup.md`
- `.agents/recon/2026-07-15-semantic-census/**`
- 2026-07-21 user decisions:
  - preserve `analysis`, `archive`, `artifacts`, `requirements`, `README.md`,
    `pyproject.toml`
  - use the active ownership roots named in this plan
  - keep Hybrid-only behavior in Hybrid
  - remove `pools` while retaining loss/optimizer/LR/runtime scheduler APIs

BaselineUsageDraft:

- Required baseline refs: package manifest, current import consumers, current tree,
  semantic census, preserved-zone baseline
- Delivered context refs: live tree at `ebe862b`, prior active-name/pools plan
- Acknowledged before plan refs: all required refs above
- Cited in plan refs: all required refs above
- Missing refs: repository-external `EvEye.*` and `src.*` consumers
- Decision: continue with consumer gates; required external consumer discovery pauses
  the affected retirement slice

# Compatibility Boundary

Preserve exactly:

- dataset sample keys, shapes, coordinates, augmentation behavior
- model factory names and constructor parameters
- train/validate/inference CLI arguments and exit semantics
- checkpoint keys, metrics, optimizer report schema
- Hybrid stage loss aliases, optimizer metadata, LR scheduler behavior, runtime FSM
- reference/history paths and blobs

Intentionally retire after consumer gates:

- active `common/scripts/facet`, `common/tests/facet`, `facet_main.py`
- active FACET runtime identifiers covered by RC-120
- `EvEye.*` internal imports after all maintained consumers move to `eveye.*`
- `src.pools` and `algorithm/hybrid/src/pools`

Compatibility policy:

- `EvEye` may contain explicit import-only wrappers during migration. Wrappers may
  not contain business logic and are removed when maintained and required external
  consumers are zero.
- Dynamic `sys.modules` aliases and shared mutable `__path__` tricks are forbidden.
- Hybrid `src` namespace is unchanged by the first ownership migration. A Hybrid
  named-package migration requires its own consumer proof and atomic cutover.

# TDD Route

- Mode: off
- Decision: skipped
- Strict authority: not applicable
- Test posture: characterization baseline and post-change regression
- Reason: strict TDD was not requested; this is a behavior-preserving migration
- Verification: import-origin, config/model/dataset parity, CLI smoke, focused and
  full regression

# Verification

Every slice runs focused verification from an editable or isolated wheel
environment. AM-900 contains the complete final gate; raw repository-root
imports without installation are not accepted as package evidence.

# Scope and readiness

Requirement Ready Check:

- Requirement source: latest user decision plus prior semantic-refactor plan
- Goals/scope: active algorithm ownership only; preserved evidence remains in place
- Acceptance: exact target tree, import mapping, behavior regression and retirement
- Open blocker: external legacy consumers are unknown
- Decision: ready for planning; every source-move batch remains gated by its
  consumer manifest and focused baseline

Change Necessity:

- User-visible need: make ownership and modality boundaries visible in the tree
- No-change option: current config-only frame/event surfaces keep code mixed in
  `EvEye` and do not satisfy the requested ownership model
- Minimum change: split `EvEye` by responsibility, then retire wrappers only after
  import parity; keep Hybrid-local code in place except `pools`
- Decision: code-change, staged by owner

Existence Check:

- New owners: `dataset`, `utils`, `engine`, `configs`, `tests`, `eveye.*`
- Existing reuse: all implementation moves from current `EvEye`; no duplicate
  implementation owner is created
- Creation proof: current `EvEye` combines dataset, utility, model, callback,
  logger and execution responsibilities and frame/event have no Python owner
- Entropy control: wrappers are import-only and have a zero-consumer retirement
  gate
- Decision: add-with-proof

Plan Pressure Test:

- Owner/contract/retirement: exact owner and wrapper retirement are specified
- Architecture integrity: no shared layer imports a modality package
- Verification: wheel/import/CLI/data/model/Hybrid gates are explicit
- Executability: tasks are serialized at package-owner boundaries
- Pressure result: proceed as a gated epic; do not combine tasks into one move

# Target code tree and meanings

```text
algorithm/
├── README.md                         # retained root guide; path unchanged
├── pyproject.toml                    # retained build owner; path unchanged
├── requirements/                     # frozen dependency input files
│   ├── analysis.txt                  # report/analysis dependencies
│   ├── eval.txt                      # evaluation dependencies
│   └── train.txt                     # training/runtime dependencies
├── analysis/                         # frozen results, reports, study scripts
│   ├── RESULTS/
│   ├── report/
│   ├── scripts/
│   └── references-analysis/
├── archive/                          # frozen provenance/history imports
│   └── imports/
├── artifacts/                        # frozen output sink contract
├── docs/                             # active plans, specs, ADR and validation
│
├── common/                           # shared contracts, shared models, launchers
│   ├── PROVENANCE.md
│   ├── scripts/
│   │   ├── train.py
│   │   ├── validate.py
│   │   ├── validate10times.py
│   │   ├── inference.py
│   │   ├── inference_visualize.ipynb
│   │   └── sagemaker_launcher.py
│   └── src/eveye/common/
│       ├── models/
│       │   ├── CitiBike/
│       │   ├── DavisEyeEllipse/HBTXR/
│       │   ├── DavisEyeEllipse/UNet/
│       │   └── DavisWithMask/
│       └── contracts/
│
├── dataset/                          # shared sample, loading and data factory owner
│   └── src/eveye/dataset/
│       ├── base_dataset.py
│       ├── dataset_factory.py
│       ├── CitiBike/
│       ├── DavisEyeCenter/
│       ├── DavisEyeEllipse/
│       ├── DavisWithMask/
│       └── Test/
│
├── utils/                            # pure processing, cache, I/O and visualization
│   └── src/eveye/utils/
│       ├── PupilTracker.py
│       ├── cache/
│       ├── dvs_common_utils/
│       ├── processor/
│       ├── tonic/
│       └── visualization/
│
├── engine/                           # modality-neutral execution lifecycle
│   └── src/eveye/engine/
│       ├── callback/
│       ├── logger/
│       ├── model_factory.py
│       └── tools/                    # dataset/model/report orchestration CLIs
│
├── configs/                          # cross-modality/common configs only
│   ├── OutputGroundTruth.yaml
│   └── TestDataset.yaml
│
├── tests/                            # shared ownership and compatibility tests
│   ├── common/
│   ├── dataset/
│   ├── utils/
│   ├── engine/
│   └── compatibility/
│
├── frame/                            # frame-only configs/docs and proven code only
│   ├── configs/
│   ├── docs/
│   ├── scripts/
│   └── tests/
│
├── event/                            # event-only model/config/runtime surface
│   ├── configs/
│   ├── docs/
│   ├── scripts/
│   ├── tests/
│   └── src/eveye/event/models/
│       ├── EPNet/
│       ├── ElNet/
│       └── TennSt.py
│
└── hybrid/                           # frame-event fusion/search-track owner
    ├── configs/
    ├── docs/
    ├── hardware_reference/           # frozen in place
    ├── scripts/
    ├── tests/
    └── src/
        ├── config/
        ├── data/
        ├── evaluation/
        ├── loss/
        │   ├── losses.py             # stable loss registry facade
        │   ├── stage.py
        │   ├── stage1.py
        │   └── stage2.py
        ├── models/
        ├── optim/
        │   ├── optimizer.py          # stable optimizer facade
        │   ├── lr_schedulers.py      # stable LR scheduler owner
        │   ├── pool.py               # optimizer experiments/reports; preserved
        │   └── registry.py
        ├── preprocess/
        ├── runtime/
        │   ├── runtime_schedulers.py # stable runtime FSM facade
        │   ├── similarity.py
        │   └── tracker.py
        ├── training/
        └── utils/
```

`frame`에는 현재 독립 Python implementation이 확인되지 않았으므로 빈
`src` package를 만들지 않는다. Event-only config가 직접 선택하는 EPNet,
ElNet, TennSt만 event owner로 이동한다. HBTXR와 UNet은 frame/event 양쪽에서
소비되므로 shared model owner에 남는다.

## Owner meanings and included code

| Owner | Meaning | Included code/data | Must not contain |
|---|---|---|---|
| `common` | modality-neutral contracts and shared models | shared model implementations, provenance, CLI launchers | Hybrid fusion/runtime logic |
| `dataset` | sample construction and loading | dataset classes, base dataset, factory, common transforms | training orchestration |
| `utils` | low-level reusable helpers | cache, event representation, file processors, visualization | model imports, dataset orchestration or mode policy |
| `engine` | model/execution lifecycle | callback, logger, model factory, evaluation/report/dataset-build tools | concrete Hybrid stage logic |
| `configs` | cross-modality/common configuration | the two current common YAML files | modality-private Hybrid configs |
| `tests` | shared owner and compatibility verification | common tests, import-origin, dataset/utils/engine tests | immutable handover/reference tests |
| `frame` | APS/RGB/cached-frame specialization | frame configs/docs and future proven frame-only implementation | event or fusion behavior |
| `event` | event-only specialization | EPNet, ElNet, TennSt, event configs and tests | frame-event fusion |
| `hybrid` | frame-event combination and search/track runtime | Hybrid data/model/loss/optim/runtime/training/preprocess | generic shared code without a Hybrid consumer |
| `docs` | current authority and execution evidence | plans, specs, validation, ADR, progress | generated model artifacts |
| `analysis` | frozen research evidence | result packages, reports, analysis/reference study scripts | active runtime owner |
| `archive` | frozen provenance/history | imported legacy implementations and conversations | active imports |
| `artifacts` | generated output sink | checkpoints/reports/large local outputs by policy | source implementation |
| `requirements` | frozen dependency profiles | train/eval/analysis requirement files | runtime code |
| `README.md` | root navigation | install/run/layout overview | implementation logic |
| `pyproject.toml` | root build metadata | distribution metadata and package discovery | model/runtime logic |

# Current-to-target mapping

| Current | Target | Rule |
|---|---|---|
| `common/src/EvEye/dataset/**` | `dataset/src/eveye/dataset/**` | implementation move |
| `common/src/EvEye/utils/scripts/**` | `engine/src/eveye/engine/tools/**` | operational tools may depend on dataset/model |
| remaining `common/src/EvEye/utils/**` | `utils/src/eveye/utils/**` | low-level implementation move |
| `common/src/EvEye/callback/**` | `engine/src/eveye/engine/callback/**` | lifecycle owner |
| `common/src/EvEye/logger/**` | `engine/src/eveye/engine/logger/**` | lifecycle owner |
| `common/src/EvEye/model/model_factory.py` | `engine/src/eveye/engine/model_factory.py` | factory owner |
| `common/src/EvEye/model/DavisEyeEllipse/EPNet/**` | `event/src/eveye/event/models/EPNet/**` | event-only |
| `common/src/EvEye/model/DavisEyeEllipse/ElNet/**` | `event/src/eveye/event/models/ElNet/**` | event-only |
| `common/src/EvEye/model/DavisEyeCenter/TennSt.py` | `event/src/eveye/event/models/TennSt.py` | event-only |
| remaining `common/src/EvEye/model/**` | `common/src/eveye/common/models/**` | shared/uncertain owner |
| `common/configs/**` | `configs/**` | common config owner |
| `common/tests/facet/**` | `tests/common/**` | active name retired |
| `common/scripts/facet/**` | `common/scripts/**` | launcher flattening |
| `common/scripts/facet_main.py` | `common/scripts/sagemaker_launcher.py` | semantic launcher name |
| `pools/losses.py` | `loss/losses.py` | API preserved |
| `pools/optimizers.py` | `optim/optimizer.py` | singular requested name |
| `pools/lr_schedulers.py` | `optim/lr_schedulers.py` | API preserved |
| `pools/runtime_schedulers.py` | `runtime/runtime_schedulers.py` | API preserved |
| `pools/heads.py`, `pools/__init__.py` | removed | existing model owner/no facade root |
| `optim/pool.py` | unchanged | optimizer experiment feature |

# Authoritative dependency DAG

```text
AM-000 baseline and preservation lock
  ├─ AM-010 package/import consumer contract
  │    ├─ AM-020 dataset migration
  │    │    └─ AM-030 utils migration
  │    │         └─ AM-040 common/engine/event ownership migration
  │    │              └─ AM-050 config/test/script migration
  │    │                   └─ AM-060 EvEye retirement
  │    └─ AM-070 Hybrid pools domain migration
  ├─ AM-080 frame/event runnable-surface validation
  └─ AM-090 conditional Hybrid shared-leaf extraction

all selected tasks -> AM-900 final gate
```

# Task AM-000: Freeze baseline and preserved zones

**Files**

- Create during execution: `docs/track/algorithm-modular-baseline.md`
- Create during execution: `docs/provenance/algorithm-modular-paths.tsv`
- Verify only: `references`, `algorithm/{analysis,archive,artifacts,requirements}`,
  `algorithm/hybrid/hardware_reference`, `algorithm/**/docs/analysis`,
  `algorithm/hybrid/tests/handover`, `algorithm/common/PROVENANCE.md`,
  `algorithm/docs/track`

**Steps**

1. Record branch, HEAD, status, tracked paths and blob IDs for preserved zones.
2. Record current `EvEye`, `src`, CLI, config, notebook and dynamic import consumers.
3. Mark every mapping row `move`, `wrapper`, `preserve`, or `remove`.
4. Stop if the live HEAD differs from the approved baseline without an explained
   ancestry update.

**Verification**

```bash
git status --short --branch
git rev-parse HEAD
git ls-tree -r ebe862b11506e819c5bd5abc925299fc5fbb6f1a -- \
  references algorithm/analysis algorithm/archive algorithm/artifacts \
  algorithm/requirements algorithm/hybrid/hardware_reference \
  algorithm/event/docs/analysis algorithm/frame/docs/analysis \
  algorithm/hybrid/docs/analysis algorithm/hybrid/tests/handover \
  algorithm/common/PROVENANCE.md algorithm/docs/track
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

# Task AM-010: Fix the package and import contract

**Files**

- Modify during execution: `algorithm/pyproject.toml`
- Create during execution: `algorithm/tests/compatibility/test_import_origins.py`
- Create during execution: `docs/analysis/ALGORITHM-IMPORT-CONSUMERS.md`

**Change**

Use PEP 420 portions under the physical owner roots. During EvEye migration the
package discovery block is:

```toml
[tool.setuptools.packages.find]
where = ["common/src", "dataset/src", "utils/src", "engine/src", "event/src"]
include = ["eveye*", "EvEye*"]
namespaces = true
```

After AM-060 it becomes `include = ["eveye*"]`. `configs`, `tests`, `docs`,
`analysis`, `archive`, `artifacts`, `requirements`, and `hybrid/src` are not
discovered as top-level packages in this slice.

**Verification**

```bash
rg -n '^[[:space:]]*(from|import)[[:space:]]+(EvEye|dataset|utils|engine)([.[:space:]]|$)' \
  algorithm --glob '!algorithm/analysis/**' --glob '!algorithm/archive/**' \
  --glob '!algorithm/hybrid/hardware_reference/**'
python3 -m pip wheel --no-deps --no-build-isolation -w /tmp/eveye-wheel algorithm
```

# Task AM-020: Move the dataset owner

**Files**

- Move: `algorithm/common/src/EvEye/dataset/**`
- To: `algorithm/dataset/src/eveye/dataset/**`
- Extract first: `algorithm/common/src/EvEye/dataset/DavisEyeCenter/DavisEyeCenterDataset.py::main`
  to `algorithm/engine/src/eveye/engine/tools/inspect_davis_eye_center.py`
- Modify: dataset imports, factory imports, common launchers and model utilities
- Test: `algorithm/tests/dataset/**`

**Steps**

1. Characterize every registered dataset name and representative sample contract.
2. Extract the dependency-bearing DavisEyeCenter demo `main` into the engine tool
   before moving any dataset implementation; the dataset module keeps no event or
   engine import.
3. Move implementation once; do not copy implementation into both packages.
4. Rewrite maintained imports to `eveye.dataset.*`.
5. Add only consumer-proven `EvEye.dataset.*` import wrappers.
6. Verify factory names, shapes, dtypes and coordinates.

**Verification**

```bash
am_pkg_tmp="$(mktemp -d)"
python3 -m venv --system-site-packages "$am_pkg_tmp/edit-venv"
"$am_pkg_tmp/edit-venv/bin/python" -c \
  'import pytest, torch, lightning, pytorch_lightning'
"$am_pkg_tmp/edit-venv/bin/python" -m pip install --no-deps -e algorithm
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=algorithm/hybrid \
  "$am_pkg_tmp/edit-venv/bin/python" \
  -m pytest -p no:cacheprovider -q \
  algorithm/tests/dataset algorithm/hybrid/tests/external_pipeline/test_dataset.py
```

# Task AM-030: Move reusable utilities

**Files**

- Move: `algorithm/common/src/EvEye/utils/{PupilTracker.py,cache,dvs_common_utils,processor,tonic,visualization}`
- To: `algorithm/utils/src/eveye/utils/**`
- Exclude: `algorithm/common/src/EvEye/utils/scripts/**`; AM-040 owns it
- Test: `algorithm/tests/utils/**`

**Steps**

1. Characterize cache, event representation, processor and visualization imports
   without running dataset generation.
2. Move implementations and rewrite imports to `eveye.utils.*`.
3. Preserve source-faithful reference paths and artifact names in allowlisted tools.
4. Reject any moved module that imports `eveye.dataset`, `eveye.common.models`,
   `eveye.event`, `src.*` or `eveye.hybrid`; route it to AM-040 tools instead.
5. Do not move Hybrid utilities in this task.

**Verification**

```bash
python3 -m compileall -q algorithm/utils/src/eveye/utils
am_pkg_tmp="$(mktemp -d)"
python3 -m venv --system-site-packages "$am_pkg_tmp/edit-venv"
"$am_pkg_tmp/edit-venv/bin/python" -c \
  'import pytest, torch, lightning, pytorch_lightning'
"$am_pkg_tmp/edit-venv/bin/python" -m pip install --no-deps -e algorithm
PYTHONDONTWRITEBYTECODE=1 "$am_pkg_tmp/edit-venv/bin/python" \
  -m pytest -p no:cacheprovider -q algorithm/tests/utils
```

# Task AM-040: Split common, engine and event ownership

**Files**

- Move callbacks/logger/factory according to the mapping table
- Move `algorithm/common/src/EvEye/utils/scripts/**` to
  `algorithm/engine/src/eveye/engine/tools/**`
- Extract first: `algorithm/common/src/EvEye/model/DavisEyeEllipse/HBTXR/Predict.py::main`
  to `algorithm/engine/src/eveye/engine/tools/predict_shared_ellipse.py`
- Move EPNet, ElNet and TennSt to `event/src/eveye/event/models`
- Move remaining shared/uncertain model implementation to
  `common/src/eveye/common/models`
- Modify: model factory registrations and maintained model imports
- Test: `algorithm/tests/{common,engine,event}/**`

**Steps**

1. Snapshot exact model registry names and constructor forwarding.
2. Extract the dependency-bearing HBTXR prediction demo into the engine tool
   before moving any shared/event model implementation; the shared model module
   keeps no event or engine import.
3. Move event-only implementations and shared implementations to their owners.
4. Make `eveye.engine.model_factory` depend on public common/event model modules.
5. Rewrite operational tools to `eveye.dataset`, `eveye.utils`,
   `eveye.common.models`, `eveye.event.models`, and `eveye.engine` imports.
6. Enforce the dependency direction: utils imports no higher owner; dataset and
   common import neither event/hybrid nor legacy `src.*`; engine imports no
   Hybrid implementation.

**Verification**

```bash
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

# Task AM-050: Move common configs/tests and flatten launchers

**Files**

- Move `algorithm/common/configs/**` to `algorithm/configs/**`
- Move `algorithm/common/tests/facet/**` to `algorithm/tests/common/**`
- Move `algorithm/common/scripts/facet/**` to `algorithm/common/scripts/**`
- Rename `facet_main.py` to `sagemaker_launcher.py`
- Modify config path resolution and `algorithm/README.md`

**Steps**

1. Reject any target basename collision before moving.
2. Accept explicit config paths independent of current working directory.
3. Apply the neutral control mapping already defined by RC-120.
4. Update root README without moving or deleting it.

**Verification**

```bash
find algorithm/common/scripts algorithm/tests -iname '*facet*' -print
am_pkg_tmp="$(mktemp -d)"
python3 -m venv --system-site-packages "$am_pkg_tmp/edit-venv"
"$am_pkg_tmp/edit-venv/bin/python" -c \
  'import pytest, torch, lightning, pytorch_lightning'
"$am_pkg_tmp/edit-venv/bin/python" -m pip install --no-deps -e algorithm
PYTHONDONTWRITEBYTECODE=1 "$am_pkg_tmp/edit-venv/bin/python" \
  -m pytest -p no:cacheprovider -q \
  algorithm/tests/common algorithm/tests/compatibility
```

# Task AM-060: Retire the EvEye compatibility owner

**Files**

- Remove after gate: `algorithm/common/src/EvEye/**`
- Modify: `algorithm/pyproject.toml` to include only `eveye*`
- Modify: docs and maintained consumers

**Retirement gate**

1. Maintained `EvEye` import count is zero.
2. Required external consumer count is zero or separately migrated.
3. Wheel contains `eveye/*` and no `EvEye/*` payload.
4. Arbitrary-CWD import, editable install and direct-script help all pass.

**Verification**

```bash
test ! -d algorithm/common/src/EvEye
rg -n '^[[:space:]]*(from|import)[[:space:]]+EvEye([.[:space:]]|$)' algorithm \
  --glob '!algorithm/analysis/**' --glob '!algorithm/archive/**' \
  --glob '!algorithm/hybrid/hardware_reference/**'

am_pkg_tmp="$(mktemp -d)"
am_repo_root="$(pwd -P)"
python3 -m pip wheel --no-deps --no-build-isolation \
  -w "$am_pkg_tmp/dist" algorithm
python3 -m venv "$am_pkg_tmp/wheel-venv"
"$am_pkg_tmp/wheel-venv/bin/python" -m pip install --no-deps \
  "$am_pkg_tmp"/dist/hbtxr_algorithm-*.whl
(
  cd "$am_pkg_tmp"
  "$am_pkg_tmp/wheel-venv/bin/python" -I - <<'PY'
from importlib import util
assert util.find_spec("eveye") is not None
assert util.find_spec("eveye.dataset") is not None
assert util.find_spec("eveye.engine") is not None
assert util.find_spec("EvEye") is None
assert util.find_spec("dataset") is None
assert util.find_spec("utils") is None
assert util.find_spec("engine") is None
PY
)
python3 - "$am_pkg_tmp"/dist/hbtxr_algorithm-*.whl <<'PY'
import sys
from zipfile import ZipFile
with ZipFile(sys.argv[1]) as wheel:
    names = set(wheel.namelist())
assert any(name.startswith("eveye/dataset/") for name in names)
assert any(name.startswith("eveye/engine/") for name in names)
assert not any(name.startswith(("EvEye/", "dataset/", "utils/", "engine/")) for name in names)
PY
python3 -m venv --system-site-packages "$am_pkg_tmp/help-venv"
"$am_pkg_tmp/help-venv/bin/python" -c \
  'import pytest, torch, lightning, pytorch_lightning'
"$am_pkg_tmp/help-venv/bin/python" -m pip install --no-deps -e algorithm
(
  cd "$am_pkg_tmp"
  "$am_pkg_tmp/help-venv/bin/python" \
    "$am_repo_root/algorithm/common/scripts/train.py" --help
  "$am_pkg_tmp/help-venv/bin/python" \
    "$am_repo_root/algorithm/common/scripts/validate.py" --help
)
```

# Task AM-070: Remove pools and retain domain APIs

**Files**

- Move: `hybrid/src/pools/losses.py` to `hybrid/src/loss/losses.py`
- Move/rename: `hybrid/src/pools/optimizers.py` to
  `hybrid/src/optim/optimizer.py`
- Move: `hybrid/src/pools/lr_schedulers.py` to
  `hybrid/src/optim/lr_schedulers.py`
- Move: `hybrid/src/pools/runtime_schedulers.py` to
  `hybrid/src/runtime/runtime_schedulers.py`
- Delete after move: `hybrid/src/pools/heads.py`, `pools/__init__.py`, directory
- Modify: domain `__init__.py`, `src/__init__.py`, `training/trainer.py`
- Preserve: `hybrid/src/optim/pool.py`
- Create tests:
  - `hybrid/tests/external_pipeline/test_loss_registry.py`
  - `hybrid/tests/external_pipeline/test_optimizer_api.py`
  - `hybrid/tests/external_pipeline/test_lr_schedulers.py`
  - `hybrid/tests/external_pipeline/test_runtime_schedulers.py`

**Behavior obligations**

- Loss: six aliases, build/compute delegation, unknown-name `KeyError`
- Optimizer: name/metadata/build and pool-report delegation
- LR scheduler: none/cosine/step/plateau, metric mode and unknown type
- Runtime scheduler: alias list, build and representative FSM transition

**Verification**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=algorithm/hybrid python3 -m pytest \
  -p no:cacheprovider -q algorithm/hybrid/tests
test ! -d algorithm/hybrid/src/pools
test -f algorithm/hybrid/src/optim/pool.py
```

# Task AM-080: Make frame/event surfaces runnable and testable

**Files**

- Modify: `algorithm/frame/docs/**`, `algorithm/event/docs/**`
- Create during execution:
  - `algorithm/frame/tests/test_config_contracts.py`
  - `algorithm/event/tests/test_config_contracts.py`
- Modify: common launchers to accept explicit config paths

**Steps**

1. Document one train/validate smoke command per modality.
2. Load every YAML and verify referenced dataset/model registry names.
3. Do not create an empty `frame/src` package.
4. Event tests verify the event model owner and cached-event dataset path.

**Verification**

```bash
am_pkg_tmp="$(mktemp -d)"
python3 -m venv --system-site-packages "$am_pkg_tmp/edit-venv"
"$am_pkg_tmp/edit-venv/bin/python" -c \
  'import pytest, torch, lightning, pytorch_lightning'
"$am_pkg_tmp/edit-venv/bin/python" -m pip install --no-deps -e algorithm
PYTHONDONTWRITEBYTECODE=1 "$am_pkg_tmp/edit-venv/bin/python" \
  -m pytest -p no:cacheprovider -q \
  algorithm/frame/tests algorithm/event/tests
```

# Task AM-090: Conditionally extract Hybrid shared leaves

**Status**: CONDITIONAL. No move is permitted until a second active modality
consumer exists and parity fixtures pass.

Candidate order:

1. pure path normalization and JSON/JSONL I/O
2. cache/component registry
3. state6 geometry
4. transform/evaluation contracts

Never move in this task:

- Hybrid config/mode contracts
- Hybrid dataset assembly/event builder
- Hybrid models/loss/training/runtime tracker
- Hybrid preprocess orchestration

If no second consumer exists, AM-090 completes as a documented NO-OP.

# Task AM-900: Final integration and preservation gate

**Dependencies**: every selected AM task.

**Verification**

```bash
am_pkg_tmp="$(mktemp -d)"
am_repo_root="$(pwd -P)"
python3 -m venv --system-site-packages "$am_pkg_tmp/edit-venv"
"$am_pkg_tmp/edit-venv/bin/python" -c \
  'import pytest, torch, lightning, pytorch_lightning'
"$am_pkg_tmp/edit-venv/bin/python" -m pip install --no-deps -e algorithm
"$am_pkg_tmp/edit-venv/bin/python" -m compileall -q \
  algorithm/common algorithm/dataset algorithm/utils algorithm/engine \
  algorithm/event algorithm/hybrid/src
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=algorithm/hybrid \
  "$am_pkg_tmp/edit-venv/bin/python" \
  -m pytest -p no:cacheprovider -q \
  algorithm/tests algorithm/frame/tests algorithm/event/tests algorithm/hybrid/tests
python3 -m pip wheel --no-deps --no-build-isolation \
  -w "$am_pkg_tmp/dist" algorithm
python3 -m venv "$am_pkg_tmp/wheel-venv"
"$am_pkg_tmp/wheel-venv/bin/python" -m pip install --no-deps \
  "$am_pkg_tmp"/dist/hbtxr_algorithm-*.whl
(
  cd "$am_pkg_tmp"
  "$am_pkg_tmp/wheel-venv/bin/python" -I - <<'PY'
from importlib import util
assert util.find_spec("eveye") is not None
assert util.find_spec("eveye.common") is not None
assert util.find_spec("eveye.dataset") is not None
assert util.find_spec("eveye.engine") is not None
assert util.find_spec("eveye.event") is not None
assert util.find_spec("eveye.utils") is not None
assert util.find_spec("EvEye") is None
assert util.find_spec("dataset") is None
assert util.find_spec("utils") is None
assert util.find_spec("engine") is None
PY
)
(
  cd "$am_pkg_tmp"
  "$am_pkg_tmp/edit-venv/bin/python" \
    "$am_repo_root/algorithm/common/scripts/train.py" --help
  "$am_pkg_tmp/edit-venv/bin/python" \
    "$am_repo_root/algorithm/common/scripts/validate.py" --help
)
test -f algorithm/README.md
test -f algorithm/pyproject.toml
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
git diff --check
```

Fail closed if any preserved-zone path/blob differs from baseline, any active
legacy import remains outside its allowlist, `src.pools` resolves, any retained
Hybrid facade API differs, or `optim/pool.py` is missing.

# Risks and rollback

| Risk | Effect | Mitigation |
|---|---|---|
| import namespace split | class/module identity mismatch | explicit wrappers only; no alias tricks |
| package discovery omits a portion | wheel works differently from checkout | inspect wheel and isolated import |
| model ownership misclassification | config/model factory regression | registry snapshot before each move |
| shared layer imports Hybrid | circular dependency | import graph gate |
| pools API moved but behavior changed | training/runtime regression | four focused API tests + full Hybrid test |
| preserved evidence drift | provenance loss | baseline path/blob comparison, fail closed |

Each implementation task is committed separately. Rollback uses `git revert` on
the affected task commit; reset, destructive checkout and preserved-zone cleanup
are not part of this plan.

# Retirement summary

- Retire: active FACET paths/contracts, `EvEye` implementation owner after wrapper
  gate, `src.pools`, `pools/heads.py`, `pools/__init__.py`
- Preserve: `analysis`, `archive`, `artifacts`, `requirements`, README/pyproject
  paths, reference/history identity, Hybrid special behavior, `optim/pool.py`
- Conditional/NO-OP without evidence: Hybrid-to-shared extraction and Hybrid
  namespace migration
