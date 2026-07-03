# HBTXR Retina/ERVT Result Packaging Sub-Plan

Date: 2026-07-03

## Task Cards

```yaml
task_card:
  task_id: T-001
  sub_agent: "codex-native"
  role: "implementer"
  objective: "Generate and rebuild Retina subject37-48 pixel-error package using HBTXR joined motion labels"
  file_ownership:
    - "analysis/results/Retina/**"
    - "analysis/scripts/*retina*"
  assigned_skill: []
  inputs:
    - "analysis/results/HBTXR/HBTXR_subject_independent_img64_patch4_test_sample_metadata.csv"
    - "analysis/results/HBTXR/HBTXR_subject37_48_test_joined_motion_error.csv"
    - "analysis/results/Retina/checkpoints/epoch=66-val_loss=2.8817.ckpt"
  outputs:
    - "analysis/results/Retina/Retina_subject37_48_*.csv"
    - "analysis/results/Retina/JETCAS_REPLY_TABLES (Error-Distributions)_Retina_subject37_48.xlsx"
  validation:
    - "No subject37-48 Saccade row is blank"
    - "Joined motion rows match the HBTXR label-map sample space"
  dependencies: []
```

```yaml
task_card:
  task_id: T-002
  sub_agent: "codex-native"
  role: "implementer"
  objective: "Generate and rebuild ERVT subject37-48 pixel-error package using HBTXR joined motion labels"
  file_ownership:
    - "analysis/results/ERVT/**"
    - "analysis/scripts/*ervt*"
  assigned_skill: []
  inputs:
    - "analysis/results/HBTXR/HBTXR_subject_independent_img64_patch4_test_sample_metadata.csv"
    - "analysis/results/HBTXR/HBTXR_subject37_48_test_joined_motion_error.csv"
    - "analysis/results/ERVT/checkpoints/best_epoch012_val_distance_8.5869.pth"
  outputs:
    - "analysis/results/ERVT/ERVT_subject37_48_*.csv"
    - "analysis/results/ERVT/JETCAS_REPLY_TABLES (Error-Distributions)_ERVT_subject37_48.xlsx"
  validation:
    - "No subject37-48 Saccade row is blank"
    - "Sequence-segment unmatched rows are documented"
  dependencies:
    - "T-001 label-map decision"
```

## Execution Notes

No separate native sub-agent was launched. The work was integrated in the main agent because both model packages share the same scripts and the same HBTXR label-map join logic.
