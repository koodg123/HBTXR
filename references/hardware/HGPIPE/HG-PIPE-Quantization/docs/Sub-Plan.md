# Sub Plan

```yaml
task_card:
  task_id: T-001
  sub_agent: codex-native
  role: analyst
  objective: Identify quantization evidence in ICCAD24-HG-PIPE.
  file_ownership: []
  assigned_skill: [code-analyzer, dnn-accelerator-expert]
  inputs: [../ICCAD24-HG-PIPE]
  outputs: [artifact inventory, implementation hooks]
  validation: [exact file paths inspected]
  dependencies: []
```

```yaml
task_card:
  task_id: T-002
  sub_agent: codex-native
  role: analyst
  objective: Identify current HG-PIPE-Quantization implementation gaps.
  file_ownership: []
  assigned_skill: [code-analyzer]
  inputs: [.]
  outputs: [gap report]
  validation: [directory/file evidence]
  dependencies: []
```

```yaml
task_card:
  task_id: T-003
  sub_agent: main
  role: implementer
  objective: Implement quantization reconstruction package and verification CLI.
  file_ownership: [hgpipe_quantization/, tests/, docs/, README.md, pyproject.toml]
  assigned_skill: [code-analyzer, dnn-accelerator-expert, fpga-asic-design-expert]
  inputs: [../ICCAD24-HG-PIPE/src, ../ICCAD24-HG-PIPE/case/refs, ../ICCAD24-HG-PIPE/statistics]
  outputs: [Python package, reports, docs]
  validation: [unit tests, full refs verification]
  dependencies: [T-001, T-002]
```
