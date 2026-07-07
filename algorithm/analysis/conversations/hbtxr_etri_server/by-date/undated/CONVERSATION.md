# Conversation Summary

Last updated: 2026-07-03

## Current Working Branch

- `etri-server`

## Major User Decisions

- Use `/home/kjm26/project/PRJXR/HBTXR` as the main working directory.
- Store FACET-related documents under `references/report/FACET`.
- Use the subject-independent DeanDataset split:
  - train subjects `1-32`
  - validation subjects `33-36`
  - test subjects `37-48`
- Use HBTXR subject-independent `img64_patch4` as the reference condition for
  comparable model setup.
- Exclude Blink rows from error-distribution tables.
- Remove `FECET` from the target list because it was a typo.
- Treat EIDet as the local FACET `ElNet.py` model.
- Evaluate BRAT on the same Subject 37-48 full test set as EIDet.

## Work Completed

- FACET codebase and paper were analyzed and documented.
- FACET data-generation flow, split behavior, event-only representation, and
  U-Net label expansion path were documented.
- HBTXR, EPNet, TennSt, TENNs-Eye, TDTracker, EIDet, and BRAT result packages
  were prepared under `analysis/RESULTS` where available.
- Subject-wise and motion-wise error-distribution workbooks were generated for
  packaged models.
- EIDet/ElNet was adapted away from unavailable native DCNv2 by adding a
  torchvision deform-convolution path.
- BRAT full-test export was corrected to preserve left/right eye sessions and
  prevent overwrite.
- BRAT was re-run on GPU1 using the corrected full test set.

## Current Output Locations

- FACET reports and operation logs:
  `references/report/FACET`
- Model-comparison reports:
  `references/report`
- Result packages:
  `analysis/RESULTS`
- BRAT corrected full-test export:
  `references/codebase/software/ais2025/Event-based-Eye-Tracking-Challenge-Solution/event_data_hbtxr_img64_fulltest`

## Important Caveats

- BRAT full-test evaluation is corrected, but a fully fair BRAT comparison still
  requires retraining BRAT with corrected left/right train/validation export.
- Center-only models should be compared primarily by center pixel error. IoU for
  those models is a proxy unless otherwise stated.
- Some result artifacts are large generated files.

