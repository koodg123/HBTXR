# FACET Status And Summary Synchronization

Date: 2026-06-26

## Summary

The final evaluation runners now synchronize the top-level reproduction summary
and reproduction status artifacts after evaluation. This reduces stale output
where `FACET_reproduction_results_2026-06-26.md` points to an older
`FACET_reproduction_status_2026-06-26.json`, or the status report has not yet
seen newly generated evaluation artifacts.

No training process was stopped or restarted by this update.

## Added Script

```text
references/codebase/software/FACET/EvEye/utils/scripts/sync_reproduction_status_summary.py
```

The script performs:

```text
1. build FACET_reproduction_summary_<date>.json and FACET_reproduction_results_<date>.md
2. refresh FACET_reproduction_status_<date>.json and FACET_reproduction_status_<date>.md
3. rebuild the summary so it embeds the refreshed status snapshot
4. refresh the status again so it validates the latest summary artifact
```

Default date:

```text
2026-06-26
```

## Runner Integration

Updated runners:

```text
references/report/FACET/run_full_checkpoint_evaluation_2026-06-26.sh
references/report/FACET/run_epnet_fpn_dw_checkpoint_evaluation_2026-06-26.sh
references/report/FACET/run_hbtxr_effbs32_checkpoint_evaluation_2026-06-26.sh
```

Each runner now calls:

```text
sync_reproduction_status_summary.py --date 2026-06-26
```

after its own evaluation and comparison work.

## Validation

Static validation passed:

```text
python -m py_compile sync_reproduction_status_summary.py build_reproduction_summary.py check_reproduction_status.py
bash -n run_full_checkpoint_evaluation_2026-06-26.sh run_epnet_fpn_dw_checkpoint_evaluation_2026-06-26.sh run_hbtxr_effbs32_checkpoint_evaluation_2026-06-26.sh
```

Search validation confirmed all three runners call:

```text
sync_reproduction_status_summary.py --date 2026-06-26
```

The sync script was not executed during this update because it would create or
rewrite partial final summary artifacts while the long-running training gates
are still incomplete.

## Remaining State

The reproduction goal is still incomplete. This update only improves the final
artifact synchronization path that will run after the planned evaluation runners
are triggered by the watcher sessions.
