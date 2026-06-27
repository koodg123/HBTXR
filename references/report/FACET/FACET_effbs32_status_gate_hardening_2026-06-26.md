# FACET HBTXR Effective-Batch-32 Status Gate Hardening

Date: 2026-06-26

## Summary

The HBTXR effective-batch-32 follow-up run was already configured, launched as a waiter, and assigned an evaluation watcher. The reproduction status checker now also treats that run as part of the final completion gate.

## Status Checker Changes

Updated:

```text
references/codebase/software/FACET/EvEye/utils/scripts/check_reproduction_status.py
```

Added checkpoint/completion gates:

```text
Phase 4B HBTXR effective-batch-32 checkpoint
Phase 4B HBTXR effective-batch-32 completion
```

Added final artifact gates:

```text
FACET_hbtxr_effbs32_reproduction_results_*.json
FACET_hbtxr_effbs32_reproduction_results_*.md
FACET_epnet_vs_hbtxr_effbs32_comparison_*.json
FACET_epnet_vs_hbtxr_effbs32_comparison_*.md
```

## Rationale

The active HBTXR baseline uses effective batch size 4 while EPNet uses 32. The effective-batch-32 run is the stricter comparison path, so the final reproduction status should not silently ignore it once the experiment has been introduced and automated.

## Runtime Note

No active training process was changed. This is a status/checker hardening only.
