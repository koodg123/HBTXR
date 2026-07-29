# Provenance and Verification

Date: 2026-06-26

## Inputs

| Input | Method | Notes |
|---|---|---|
| `analysis/vit-accel` | Local file inventory and representative report reads | Existing detailed analyses were not modified |
| `/home/kjm26/project/PRJXR/References/reports` | Local file inventory and index read | Used as target analysis-style reference |
| `/home/kjm26/project/PRJXR/References/ViT` | Local inventory plus read-only sub-agent exploration | 10 top-level repos covered |
| `/home/kjm26/project/PRJXR/References/Hardware` | Local inventory plus read-only sub-agent exploration | 37 top-level repos covered |
| `/home/kjm26/project/PRJXR/References/HW_Framework` | Local inventory plus read-only sub-agent exploration | 11 top-level repos covered |

## Sub-Agent Task Cards

```yaml
task_card:
  task_id: T-101
  sub_agent: "gpt5.3-codex-spark"
  role: "research"
  objective: "Read-only inventory and HGTXR relevance classification for References/ViT"
  file_ownership: []
  assigned_skill: ["code-analyzer"]
  inputs: ["/home/kjm26/project/PRJXR/References/ViT"]
  outputs: ["ViT repo inventory, category map, priority recommendations"]
  validation: ["read-only rg/find style inventory"]
  dependencies: []
```

```yaml
task_card:
  task_id: T-102
  sub_agent: "gpt5.3-codex-spark"
  role: "research"
  objective: "Read-only inventory and HGTXR hardware relevance classification for References/Hardware"
  file_ownership: []
  assigned_skill: ["algo2fpga", "code-analyzer"]
  inputs: ["/home/kjm26/project/PRJXR/References/Hardware"]
  outputs: ["Hardware repo inventory, key HDL/HLS files, priority recommendations"]
  validation: ["read-only rg/find style inventory"]
  dependencies: []
```

```yaml
task_card:
  task_id: T-103
  sub_agent: "gpt5.3-codex-spark"
  role: "research"
  objective: "Read-only inventory and HGTXR framework relevance classification for References/HW_Framework"
  file_ownership: []
  assigned_skill: ["code-analyzer", "artifact-provenance-manager"]
  inputs: ["/home/kjm26/project/PRJXR/References/HW_Framework"]
  outputs: ["Framework repo inventory, key toolflow files, priority recommendations"]
  validation: ["read-only rg/find style inventory"]
  dependencies: []
```

```yaml
task_card:
  task_id: T-201
  sub_agent: "gpt5.3-codex-spark"
  role: "evaluator"
  objective: "Read-only cross-check of vit-accel codebase/paper counts and per-source metadata fields"
  file_ownership: []
  assigned_skill: ["code-analyzer"]
  inputs: ["/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel"]
  outputs: ["codebase count, paper count, required per-source metadata fields, naming caveats"]
  validation: ["read-only local file inventory"]
  dependencies: []
```

## Commands Used

| Purpose | Command shape |
|---|---|
| Existing analysis inventory | `find analysis/vit-accel -maxdepth 3 -type f` |
| Reference report style check | `find /home/kjm26/project/PRJXR/References/reports -maxdepth 2 -type f` and `sed -n` on index files |
| Reference top-level inventories | `find /home/kjm26/project/PRJXR/References/{ViT,Hardware,HW_Framework} -mindepth 1 -maxdepth 1 -type d` |
| Per-source split generation | Generated one `<source>/analysis.md` per codebase, paper, or framework under `analysis/integrated-2026-06-26` |
| Per-source validation | `find analysis/integrated-2026-06-26 -path '*/analysis.md' -type f | wc -l` returned `105` |

## Evidence Boundary

- This package is static documentation and integration analysis.
- No code in external reference repositories was modified.
- No synthesis, implementation, board run, training, or benchmark reproduction was executed.
- HGTXR performance/resource claims remain tied to existing generated HLS/Vivado/board artifacts, not to this reference analysis.
