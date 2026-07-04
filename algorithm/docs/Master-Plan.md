# HBTXR Reorganization Master Plan

Date: 2026-07-05

## Goal

Reconstruct the three HBTXR branch workspaces into a single `HBTXR` directory
with a clean root layout and a consolidated algorithm package.

## Source Branches

- `HBTXR-etri-server/HBTXR`: baseline algorithm, evaluation, motion labels, and
  subject-independent result packages.
- `HBTXR-etri-desktop/HBTXR`: Retina and ERVT packaging additions.
- `HBTXR-home/HBTXR`: annotation analysis and frame/crop HBTXR experiment
  additions.

## Target Boundaries

- Root: `README.md`, `.gitignore`, `hardware/`, `quantization/`, `third/`,
  `references/`, and `algorithm/`.
- Algorithm code: `algorithm/src/EvEye`.
- Algorithm analysis and reports: `algorithm/analysis`.
- Legacy papers and codebases: `references/papers` and
  `references/legacy-codebase`.

## Strategy

1. Preserve original source branches.
2. Copy server branch as the baseline.
3. Merge desktop Retina/ERVT artifacts into `algorithm/analysis`.
4. Merge home annotation and frame/crop experiment additions.
5. Exclude large runtime artifacts from the repository tree where feasible.
6. Verify the final tree and import-level package metadata.
