# HBTXR Reorganization Sub-Plan

## Task Cards

```yaml
task_card:
  task_id: T-001
  sub_agent: "codex-native"
  role: "implementer"
  objective: "Create the requested HBTXR root layout."
  file_ownership: ["HBTXR/"]
  assigned_skill: ["code-refactoring-refactor-clean"]
  inputs: ["HBTXR-etri-server/HBTXR", "HBTXR-etri-desktop/HBTXR", "HBTXR-home/HBTXR"]
  outputs: ["HBTXR/README.md", "HBTXR/.gitignore", "HBTXR/algorithm", "HBTXR/references", "HBTXR/third"]
  validation: ["tree output", "key file existence checks"]
  dependencies: []
```

```yaml
task_card:
  task_id: T-002
  sub_agent: "codex-native"
  role: "integrator"
  objective: "Merge branch-specific algorithm additions without overwriting the server baseline blindly."
  file_ownership: ["HBTXR/algorithm/src", "HBTXR/algorithm/configs", "HBTXR/algorithm/analysis"]
  assigned_skill: ["code-refactoring-refactor-clean"]
  inputs: ["server baseline", "desktop Retina/ERVT additions", "home frame/annotation additions"]
  outputs: ["merged configs", "merged analysis scripts", "frame dataset registration"]
  validation: ["diff review", "tree output", "Python syntax/import checks when dependencies allow"]
  dependencies: ["T-001"]
```
