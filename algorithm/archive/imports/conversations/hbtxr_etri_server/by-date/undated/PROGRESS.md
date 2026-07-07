# Progress

Last updated: 2026-07-03

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

## In Progress Or Next

- [ ] Retrain BRAT with corrected left/right train and validation export for a
      fully fair comparison.
- [ ] Decide whether ignored large generated artifacts should be uploaded to
      Git LFS or external artifact storage.
- [ ] Continue standardizing remaining target models when executable training
      code and dataset adapters are available.

## Current Key Metrics

| Model | Valid N | Weighted Mean Error | Weighted Median Error | Weighted Mean IoU |
|---|---:|---:|---:|---:|
| EIDet | 360,255 | 3.849677 | 2.386649 | 0.357419 |
| BRAT | 360,495 | 0.879130 | 0.477422 | 0.770778 |
