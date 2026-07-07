# Integrated Reference Analysis Package

Date: 2026-06-26

## Purpose

This package integrates the existing `analysis/vit-accel` codebase/paper reports and extends the same analysis style to the local reference corpora under:

- `/home/kjm26/project/PRJXR/References/ViT`
- `/home/kjm26/project/PRJXR/References/Hardware`
- `/home/kjm26/project/PRJXR/References/HW_Framework`

`analysis/curated-2026-06-26` is intentionally preserved as a separate curated category package. This directory is the integrated source-by-source analysis tree requested after that package.

## Directory Map

| Directory | Contents | Intended use |
|---|---|---|
| `vit-accel/codebases/` | 29 per-codebase directories plus integrated roll-up | Reuse previous detailed source analyses without flattening provenance |
| `vit-accel/papers/` | 18 per-paper directories plus integrated roll-up | Paper-to-experiment mapping and hardware relevance |
| `vit/codebases/` | 10 per-codebase directories for `/References/ViT` | Algorithm, quantization, deployment, and accuracy-oriented candidates |
| `hardware/codebases/` | 37 per-codebase directories for `/References/Hardware` | HLS/RTL/FPGA architecture and resource-policy candidates |
| `hw-framework/frameworks/` | 11 per-framework directories for `/References/HW_Framework` | HLS/DSL/compiler/toolflow candidates |
| `manifests/` | Coverage, provenance, and evidence boundary | Audit trail and continuation handoff |

## Per-Source Analysis Layout

Each source has its own directory and `analysis.md`:

```text
analysis/integrated-2026-06-26/
  vit-accel/codebases/<Codebase>/analysis.md
  vit-accel/papers/<Paper>/analysis.md
  vit/codebases/<Codebase>/analysis.md
  hardware/codebases/<Codebase>/analysis.md
  hw-framework/frameworks/<Framework>/analysis.md
```

The machine-readable source list is stored in `manifests/per_source_manifest.json`; the tabular form is `manifests/per_source_index.tsv`.

## Best-Output Criteria

| Criterion | Check |
|---|---|
| Completeness | Covers existing `vit-accel`, `References/ViT`, `References/Hardware`, and `References/HW_Framework` top-level corpora |
| Evidence | Uses local inventories, existing analysis reports, and read-only sub-agent inventory results |
| Executability | Produces Markdown documents and manifest files under the writable hardware tree |
| Consistency | Keeps C3b/HLS evidence separate from new reference-derived experiment ideas |
| Safety | Does not copy credentials or modify external reference repositories |
| Maintainability | Uses a stable directory layout that can accept per-source deep-dive files later |

## Evidence Boundary

- This is static analysis and integration documentation.
- No HLS synthesis, Vivado implementation, board run, model training, or benchmark reproduction was executed.
- New reference-derived ideas must pass the existing HGTXR gates before becoming performance claims:
  - csim/csynth evidence for HLS changes
  - routed timing/power for implementation claims
  - PYNQ/board-smoke JSON for board-level claims
