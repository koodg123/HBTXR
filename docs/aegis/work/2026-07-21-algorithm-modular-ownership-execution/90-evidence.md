# Evidence Bundle Draft

## AM-000

- Worktree branch: `refactor/algorithm-modular-ownership`.
- Baseline HEAD: `ebe862b11506e819c5bd5abc925299fc5fbb6f1a`.
- Tracked file count: 12,856.
- Initial worktree status: clean.
- Preserved tree: 8,044 tracked blobs across the exact AM-000 allowlist;
  `docs/provenance/algorithm-modular-paths.tsv` records mode/object ID/path.
- Preserved-zone baseline diff: exit 0.
- Preserved-zone porcelain status: exit 0.
- Fresh tree-to-TSV equality: transformed `git ls-tree -r
  ebe862b11506e819c5bd5abc925299fc5fbb6f1a -- <12 preserved paths>` versus
  `awk -F '\t' '$1 == "preserved_blob"'
  docs/provenance/algorithm-modular-paths.tsv` under `diff -u`: exit 0, empty
  diff. Counts: `preserved_rows=8044`, `mapping_rows=20`,
  `compatibility_rows=1`. The full reproducible command is in
  `docs/track/algorithm-modular-baseline.md`.
- Consumer census: active `EvEye` imports 176 lines/57 files; active `src`
  imports 302 lines/96 files; broad dynamic-import risk surface 44 lines/9
  files; load/exec subset 31 lines/9 files.
- Surface census: common CLI 6; common/frame/event configs 2/12/24; common
  notebooks 32; `EvEye` Python files 130; Hybrid `src` Python files 122.
- Mapping inventory: 20/20 approved mapping rows recorded as 18 `move`, one
  `remove`, one `preserve`; conditional `EvEye` compatibility recorded as a
  separate `wrapper` row.
- Authority plan: `/mnt/d/dataset/EV_Eye/paper_works/HBTXR-Pool/HBTXR/docs/aegis/plans/2026-07-21-algorithm-modular-ownership-refactor.md`; SHA-256 `c73cd5ee0e4f9dd0798b2dbd38bf3c9a50c25e81b996d634c4bc776b022b97b4`.
- Absolute worktree and `/tmp` paths are capture metadata from the 2026-07-21
  host, not portable path contracts.
- Exact census reproduction: the executable Bash block in
  `docs/track/algorithm-modular-baseline.md` produced `active_EvEye=176/57`,
  `hybrid_src=302/96`, `broad_dynamic=44/9`, `load_exec=31/9`, `CLI=6`,
  `configs=2/12/24`, `notebooks=32`, `EvEye_py=130`, `Hybrid_src_py=122`;
  all value assertions exited 0.
- Test environment: Python 3.11.14, pytest 9.1.1, torch 2.2.2+cu121,
  torchvision 0.17.2+cu121, lightning 2.6.5, pytorch-lightning 2.6.5,
  numpy 1.26.4.
- FULL command: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=algorithm/hybrid:algorithm/hybrid/scripts/external_pipeline:algorithm/hybrid/tests/external_pipeline /tmp/hbtxr-am-venv/bin/python -m pytest -s -p no:cacheprovider -q --junitxml=/tmp/hbtxr-am000-collection.xml algorithm/hybrid/tests`; exit 2, four collection errors.
- PARTIAL command: the same environment/interpreter/pytest prefix plus
  `--ignore=algorithm/hybrid/tests/external_pipeline/test_target_fps_canonical.py
  --ignore=algorithm/hybrid/tests/external_pipeline/test_target_fps_dataset.py
  --ignore=algorithm/hybrid/tests/external_pipeline/test_target_fps_manifest.py
  --ignore=algorithm/hybrid/tests/external_pipeline/test_timelens_xl_finetune.py
  --junitxml=/tmp/hbtxr-am000-partial.xml algorithm/hybrid/tests`; exit 1,
  16 failed, 128 passed, 28 subtests passed. The four ignore paths correct the
  earlier non-existent test-root paths to the tracked `external_pipeline` files.
- Raw JUnit: `/tmp/hbtxr-am000-collection.xml`, 3,976 bytes, SHA-256
  `fed04906cda3a24595741dbf5ed85858af07b7986a6ca9ebabcd3222df192466`;
  `/tmp/hbtxr-am000-partial.xml`, 59,950 bytes, SHA-256
  `13de54df9132a1e39ce5dbaf8ddbf2c7512433315b1e2cd899e0ec6e0825644b`.
- Remaining exact-baseline defects: four wrong-path `tests.test_*` imports; parent path resolution, three snapshot drifts, two runtime run-root mismatches and help paths rooted under `hybrid/tests`.
- `/tmp/hbtxr-am-venv` is disposable and outside the repository; no repository dependency, production, test or config file changed.
- File-mode result: host `stat` remains `0777` because `/mnt/d` is a 9p/DrvFS mount without metadata/file-mode support and `core.filemode=false`; no mount change was made. All five artifacts are staged, and `git diff --cached --summary` confirms `create mode 100644` for each.
- User baseline-policy decision (2026-07-21): continue the sequential refactor
  and test the refactored result. AM-000 RED results are comparison evidence,
  not a structural-slice blocker.
- Reporting obligation: AM-900 must report all post-refactor collection errors,
  failures, passes and subtests, plus failure-identity deltas against the AM-000
  baseline of 4 collection errors and 16 failed / 128 passed / 28 subtests
  passed. Existing failures are neither waived nor hidden.
- Final AM-000 verification: exact HEAD, preserved diff/status, tree-to-TSV equality, manifest row/count checks, census assertions and owned-file allowlist passed. Five `git diff --no-index --check /dev/null <artifact>` checks returned exit 1 because each artifact is a new file, with whitespace diagnostics of 0 bytes. `git diff --cached --check` returned exit 0; cached summary lists five `create mode 100644` entries.


## AM-010 package and import contract

- Changed only `algorithm/pyproject.toml`,
  `algorithm/tests/compatibility/test_import_origins.py`, and
  `docs/analysis/ALGORITHM-IMPORT-CONSUMERS.md`.
- Original plan correction: five discovery roots without explicit package
  mappings failed once multiple `eveye` portions existed. The verified contract
  maps `EvEye` and every `eveye.<owner>` to its physical root and uses paired
  exact/descendant package-name patterns.
- Focused compatibility tests: 3 passed under Python 3.11.14. The test parser
  uses setuptools >=68 and does not depend on Python 3.11-only `tomllib`.
- Synthetic nested six-owner wheel: build, clean install, arbitrary-CWD imports,
  approved payload, lookalike exclusion and bare-package exclusion all passed.
- Current wheel: `EvEye` payload retained; no premature `eveye` or bare owner
  payload.
- Host safety: default pip temp under `/mnt/c` case-folded `EvEye`/`eveye`; the
  documented gate forces `TMPDIR`, `TEMP`, and `TMP` under Linux `/tmp`. No
  intermediate source-migration commit may contain both case-only payloads.
- Consumer census remained `EvEye` 176 lines/57 files and Hybrid `src` 302
  lines/96 files. Repository-external consumers remain unknown.
- Review: amended-spec APPROVED; code-quality APPROVED.
- Drift: no implementation owner move, Hybrid edit, preserved-zone change,
  repository build artifact, dependency change or premature retirement.
