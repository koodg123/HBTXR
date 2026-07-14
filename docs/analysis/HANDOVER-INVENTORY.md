# HANDOVER repository inventory

## Purpose and interpretation

This document records the inspected HANDOVER repository snapshots and the
path/blob overlap evidence available for assessing their relationship to the
current HBTXR repository. Counts below are inventory evidence only. A matching
path or byte-identical blob does not establish API, data-contract, license,
build, board, or behavioral compatibility. Conversely, a byte difference does
not by itself prove that the HANDOVER variant should replace the current file.

## Repository snapshots

| Source | Branch / revision | Tracked paths | Observed top-level scope |
| --- | --- | ---: | --- |
| `HANDOVER/HBTXR` | `etri-server` / `7d1b0cace624` | 6,336 | `references`: 5,934; `analysis`: 320; `third`: 76 |
| `HANDOVER/HGTXR` | `kjm26/feat/capture-hgtxr-current-state` / `70c9718117ca` | 1,881 | software: 859; hardware: 701; docs: 188; quantization: 128 |
| `HANDOVER/HGPIPE/HG-PIPE-Quantization` | `master` / `3127ac6e2223` | 128 | quantization source and supporting artifacts |
| `HANDOVER/HGPIPE/ICCAD24-HG-PIPE` | `main` / `19d74e851722` | 1,210 | ICCAD24 HG-PIPE implementation and supporting artifacts |
| `HANDOVER/ViT_Accel` | `main` / `90a82ddecd32` | 2,923 | ViT accelerator source, workspaces, automation, configuration, and documents |
| `HANDOVER/XR_Accel` | `koodg123/feat/tune-zcu104-cyclic-maxperf` / `5041401aec53` | 1,312 | XR accelerator source, workspaces, automation, configuration, and documents |

The revision identifiers make this a reproducible point-in-time inventory.
Later commits in any source require a new comparison rather than silently
extending these counts.

## Source-to-target mapping

| HANDOVER surface | Current HBTXR comparison surface | Recorded overlap | Inventory judgment |
| --- | --- | --- | --- |
| `HANDOVER/HBTXR/analysis` | `algorithm/analysis` | 320 common paths; 320 byte-identical | Already represented at the inspected snapshots; no inventory-driven copy is indicated. |
| `HANDOVER/HBTXR/third` | corresponding current third-party archive | 71 common paths; 71 byte-identical | Common material is already represented. Unmatched paths still require separate provenance and need analysis. |
| `HANDOVER/HBTXR/references` | `references/legacy-codebase` | 5,048 common mapped paths; 5,037 identical; 11 changed | Predominantly overlapping legacy evidence. The 11 changed paths require file-level review; the count alone does not select a preferred version. |
| `HANDOVER/HGTXR/software/src/hbtxr` | `algorithm/hybrid/src` | 113 common paths; 2 identical; 111 changed | Strong path continuity but extensive content divergence. Treat as a semantic-audit surface, not as a bulk-copy candidate. |
| `HANDOVER/HGTXR/hardware` | current HBTXR hardware tree | 394 common paths; 45 identical; 349 changed | Strong path continuity with substantial evolution. Module-level review and regression evidence are required before reuse. |
| `HANDOVER/HGPIPE/HG-PIPE-Quantization` | current quantization archive / implementation | 125 unique blobs all present; implementation has 33 common paths, 29 identical and 4 changed | Archived content is materially represented. Review the four changed implementation paths only if behavior or provenance requires reconciliation. |
| `HANDOVER/HGPIPE/ICCAD24-HG-PIPE` | current legacy/reference material | 964 unique blobs; 928 present | Most content is represented. Of the 36 absent blobs, the observed concentration is 33 SPINAL paths and 3 Python cache artifacts. |
| `HANDOVER/ViT_Accel` | current accelerator/reference material | 1,532 unique blobs; 827 present | 705 unique blobs are not represented by the compared blob set. Their absence is an audit queue, not an integration recommendation. |
| `HANDOVER/XR_Accel` | current accelerator/reference material | 1,044 unique blobs; 824 present | 220 unique blobs are not represented by the compared blob set. Their absence is an audit queue, not an integration recommendation. |

## Overlap evidence and triage consequences

### Material already represented

- The 320 common HBTXR analysis paths and 71 common third-party paths are
  byte-identical. Re-copying those common paths would add no content and risks
  obscuring provenance.
- All 125 unique blobs from the standalone HG-PIPE quantization repository are
  present in the compared current material. The narrower implementation-path
  comparison leaves four changed paths for targeted review.
- The ICCAD24 comparison found 928 of 964 unique blobs represented. The three
  Python cache artifacts among the absent set are generated material and do not
  constitute source-code gaps. The 33 SPINAL paths remain a reference-scope
  question; this inventory does not establish an active HBTXR dependency on
  them.

### Material requiring semantic review

- HGTXR software has 111 changed files among 113 common source paths. Shared
  names show ancestry or structural correspondence, but the divergence is too
  high for path-based promotion. Interfaces, tensor/data contracts, evaluation
  behavior, tests, and licenses must be checked before any selective port.
- HGTXR hardware has 349 changed files among 394 common paths. Board targets,
  tool versions, generated outputs, register/interface contracts, timing
  assumptions, and golden vectors must be evaluated at module granularity.
- The 11 changed HANDOVER/HBTXR legacy-reference paths are suitable for
  provenance comparison. They are not evidence that active HBTXR source is
  stale.

### Material absent from the compared target blob set

| Source | Unique blobs | Present | Not represented | Present ratio |
| --- | ---: | ---: | ---: | ---: |
| ICCAD24 HG-PIPE | 964 | 928 | 36 | 96.3% |
| ViT accelerator | 1,532 | 827 | 705 | 54.0% |
| XR accelerator | 1,044 | 824 | 220 | 78.9% |

“Not represented” means that an identical blob was not found in the compared
target material. It does not mean the target lacks equivalent functionality:
renames, refactoring, generated workspaces, configuration variants, or a
different implementation can all produce this result.

## Audit boundaries and limitations

This inventory intentionally does not claim any active integration outcome.
Its evidence is limited as follows:

1. Counts describe the named revisions only and may become stale when a source
   advances.
2. Path overlap measures file naming and placement. Blob overlap measures byte
   identity. Neither validates runtime behavior or numerical equivalence.
3. Unique-blob counts do not preserve one-to-one path cardinality because the
   same content can occur at multiple paths.
4. The snapshot counts do not validate licenses, attribution requirements,
   dataset redistribution rights, or third-party provenance.
5. Generated workspaces, caches, model weights, datasets, logs, and tool output
   must be classified separately from maintainable source and hand-written
   configuration.
6. Hardware reuse requires target-board and toolchain validation; software reuse
   requires explicit data, metric, checkpoint, and CLI/API contract tests.
7. No file in this document is approved for bulk replacement. Changed and
   absent sets are inputs to the software, hardware, license/artifact, and
   integration audits.

## Inventory conclusion

The comparison shows three distinct states: large byte-identical regions that
should remain no-op, highly divergent but structurally corresponding HGTXR
software/hardware regions that need semantic review, and partially represented
accelerator repositories that need artifact classification before selective
reuse. The evidence supports targeted audits and test-backed adaptation; it
does not support copying an entire HANDOVER repository into active HBTXR.
