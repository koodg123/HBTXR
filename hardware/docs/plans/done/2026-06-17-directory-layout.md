> **작성** 2026-06-17 · **갱신** 2026-07-29
> **상태** superseded-by:../../../README.md
> **소유** hardware
>
> ## ⚠️ 대체됨
>
> 이 문서가 정의한 목표 레이아웃(`src/ configs/ scripts/ tests/ experiments/ artifacts/
> generated/ external/ archive/`)은 **채워지지 않았습니다.** 2026-07-29 census가 그중
> **24개 디렉토리가 `.gitkeep` 하나만 든 채로 남아 있었음**을 확인했습니다
> ([census §2](../../reports/2026-07-29-hardware-census.md)).
>
> 현재 레이아웃은 `config/ module/ build/ deploy/ tools/ docs/ workspace/` 이며
> [hardware/README.md](../../../README.md)가 정의합니다.
>
> 본문은 편집하지 않습니다 — 당시 결정의 기록입니다.

---

# HGTXR Hardware Directory Layout

Date: 2026-06-17

## Purpose

This document defines the target directory layout for the HGTXR hardware tree
and records which legacy paths are intentionally kept for compatibility.

## Target Layout

```text
hardware/
├── README.md
├── AGENTS.md
├── src/
├── configs/
├── scripts/
├── tests/
├── docs/
├── experiments/
├── artifacts/
├── generated/
├── external/
└── archive/
```

## Directory Policy

| Path | Role | Git policy |
|---|---|---|
| `hls/` | Current HLS source and testbench compatibility path | tracked |
| `rtl/` | Current RTL source compatibility path | tracked |
| `pynq/` | Current PYNQ host and overlay compatibility path | tracked |
| `refs/` | Golden vectors, weights, and reference manifests | tracked when required by tests or reproducibility |
| `src/` | New normalized source tree placeholder | tracked |
| `configs/` | Board, synthesis, validation, and experiment configs | tracked |
| `scripts/` | Build, run, validation, report, and maintenance entrypoints | tracked |
| `tests/` | Unit, integration, HLS, PYNQ, and fixture tests | tracked |
| `docs/` | Architecture, experiment, report, reference, decision, and tracking docs | tracked |
| `experiments/` | Active, completed, blocked, and template experiment definitions | tracked |
| `artifacts/` | Preserved bitstreams, HWH, smoke bundles, manifests, and reports | tracked selectively |
| `generated/` | Rebuildable HLS/Vivado/PYNQ/signoff outputs and logs | mostly ignored; signoff evidence may be tracked |
| `external/` | External papers, codebase summaries, and legacy references | tracked selectively |
| `archive/` | Deprecated or migration-only material | tracked selectively |

## Compatibility Paths Kept

The following paths are heavily referenced by tools, tests, and documents and
must not be moved without a planned path migration:

- `hls/`
- `pynq/`
- `refs/`
- `analysis/vit-accel/`
- `generated/pynq/`
- `generated/signoff/`
- `vivado/scripts/`

The current cleanup therefore keeps those paths in place and only moves files
that are clearly root-level generated logs or empty accidental files.

## Completed Cleanup Actions

- Created normalized target directories:
  - `src/`, `experiments/`, `artifacts/`, `external/`, `archive/`
  - structured subdirectories under `docs/`, `scripts/`, and `tests/`
- Moved root HLS log:
  - `vitis_hls.log` -> `generated/logs/hls/vitis_hls.log`
- Moved root Vivado logs and journals:
  - `vivado.log` -> `generated/logs/vivado/vivado.log`
  - `vivado.jou` -> `generated/logs/vivado/vivado.jou`
  - `vivado_42.backup.log` -> `generated/logs/vivado/vivado_42.backup.log`
  - `vivado_42.backup.jou` -> `generated/logs/vivado/vivado_42.backup.jou`
- Moved empty accidental root files:
  - `=318` -> `archive/migration-logs/=318.empty`
  - `=332` -> `archive/migration-logs/=332.empty`

## Deferred Migration

The following migrations are useful but intentionally deferred because current
tools and tests still refer to the legacy paths directly:

1. Move HLS source from `hls/` to `src/hls/`.
2. Move PYNQ host code from `pynq/` to `src/host/pynq/`.
3. Move selected overlay `.bit` and `.hwh` files to `artifacts/bitstreams/`
   and `artifacts/hwh/`, then update packaging tools.
4. Move `analysis/vit-accel/` to `docs/references/vit-accel/` or
   `external/codebases/vit-accel/`, then update source-audit tools.
5. Split `refs/` into `tests/fixtures/` and `artifacts/manifests/` only after
   static validation and bundle packaging tools are updated.

## Validation Gates For Future Path Migration

Before moving compatibility paths, update all direct references and run:

```bash
python3 -m unittest discover -s tests
git diff --check -- .
python3 tools/check_third_goal_preflight.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR
```
