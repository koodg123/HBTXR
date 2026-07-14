# Progress

Last updated: 2026-07-07

## Completed

- [x] Analyze FACET codebase and paper.
- [x] Document FACET reproduction plan.
- [x] Prepare subject-independent DeanDataset split.
- [x] Train/evaluate HBTXR subject-independent variants.
- [x] Package HBTXR full-UNet `img128_patch4` results.
- [x] Package EPNet subject-independent `img64` results.
- [x] Package TennSt and TENNs-Eye results.
- [x] Package TDTracker subject-independent `img64` results.
- [x] Adapt EIDet/ElNet to avoid unavailable native DCNv2.
- [x] Package EIDet subject-independent test results.
- [x] Fix BRAT test export to include both left and right eye sessions.
- [x] Re-run BRAT on the same Subject 37-48 full test set as EIDet.
- [x] Package corrected BRAT subject-independent test results.
- [x] Update model inference package summary.
- [x] Exclude large row-level CSV/H5/weight/log artifacts from git tracking so
      the branch can be pushed to GitHub.
- [x] Preserve the current dirty HBTXR import state in commit `51b18b6`.
- [x] Move imported conversations, HGTXR project docs, and legacy hybrid
      docs/scripts/configs/tests into `algorithm/archive/imports`.
- [x] Keep active `algorithm/hybrid` scripts/configs/tests/docs focused on
      maintained package surfaces.
- [x] Move the refactor summary under `algorithm/docs/track/refactor` so root
      package structure remains clean.
- [x] Import selected `HANDOVER/HBTXR` EV-Eye and EX-Gaze hybrid analysis
      scripts, configs, reports, metadata, and compact result summaries.
- [x] Archive `IMPL_REPOS/SW/FECET-HBTXR` and `SWIFT-HBTXR` as optional
      future porting sources under `algorithm/archive/imports`.
- [x] Document the selective import and large-artifact exclusions in
      `algorithm/docs/track/IMPL_REPOS_HANDOVER_IMPORT_2026_07_08.md`.

## In Progress Or Next

- [ ] Retrain BRAT with corrected left/right train and validation export for a
      fully fair comparison.
- [ ] Decide whether ignored large generated artifacts should be uploaded to
      Git LFS or external artifact storage.
- [ ] Continue standardizing remaining target models when executable training
      code and dataset adapters are available.
- [ ] Continue Phase 2 active-code refactor for `frame`, `event`, `hybrid`,
      `hardware`, and `quantization`.
- [ ] Promote archived legacy scripts/tests only after API adaptation and
      validation.
- [ ] Adapt the imported EV-Eye/EX-Gaze scripts to the current
      `algorithm/hybrid/src` package before treating them as active workflows.

## Current Key Metrics

| Model | Valid N | Weighted Mean Error | Weighted Median Error | Weighted Mean IoU |
|---|---:|---:|---:|---:|
| EIDet | 360,255 | 3.849677 | 2.386649 | 0.357419 |
| BRAT | 360,495 | 0.879130 | 0.477422 | 0.770778 |
