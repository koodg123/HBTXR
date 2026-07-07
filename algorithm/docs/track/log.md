# Work Log

## 2026-07-07

- Created branch `refactor/hbtxr-structure` from `main` because the working
  tree was dirty and commit policy avoids committing directly to `main`.
- Committed the dirty import state as
  `51b18b6 ref(archive): checkpoint imported HBTXR materials`.
- Moved imported/legacy algorithm material out of active surfaces into
  `algorithm/archive/imports`.
- Committed archive separation as
  `0770c47 refactor(algorithm): archive imported legacy materials`.
- Verified active hybrid source syntax with `compileall`.
- Verified active `algorithm` tree no longer exposes archive-excluded legacy,
  import, conversation, or old branch-name patterns.
- Moved the refactor summary from root `.agents` to
  `algorithm/docs/track/refactor` for root layout consistency.

## 2026-07-03

- Documented current HBTXR/FACET model-comparison state.
- Added project tracking notes for conversation state and progress.
- Corrected BRAT full-test inference package to use the same Subject 37-48 test
  set as EIDet.
- Verified BRAT full-test raw prediction row count: `366,171`.
- Verified BRAT non-Blink joined row count: `360,495`.
- Prepared all current changes for git commit on `etri-server`.
- Reworked the push-ready commit after GitHub rejected large row-level result
  CSVs, keeping those files local and ignored.
