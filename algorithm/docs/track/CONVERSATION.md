# Conversation Summary

Last updated: 2026-07-07

## Current Working Branch

- `refactor/hbtxr-structure`

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
- Preserve the dirty tree before proceeding with the HBTXR structure refactor.
- Use generalized active package names; keep source-specific historical names
  only where needed for archive provenance.
- Move imported and legacy materials out of active package surfaces instead of
  deleting them.

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
- HGTXR hybrid integration and imported HBTXR/HGTXR materials were checkpointed.
- Imported conversations, HGTXR project docs, and legacy hybrid material were
  archived under `algorithm/archive/imports`.
- Active `algorithm/hybrid` scripts/configs/tests/docs were reduced to
  maintained package surfaces plus README boundary notes.

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
- Archive-internal documents intentionally preserve original source names and
  historical paths for provenance.
