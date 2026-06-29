# FACET Evaluation Runner Completion Gate

Date: 2026-06-26

## Summary

Final evaluation runners now refuse to evaluate checkpoints unless the matching
training log contains the strict `max_epochs=70` completion marker. This guards
against manually running an evaluation script on intermediate checkpoints and
accidentally producing artifacts that look like final reproduction results.

No training process was stopped or restarted by this update.

## Updated Scripts

```text
references/report/FACET/run_full_checkpoint_evaluation_2026-06-26.sh
references/report/FACET/run_epnet_fpn_dw_checkpoint_evaluation_2026-06-26.sh
references/report/FACET/run_hbtxr_effbs32_checkpoint_evaluation_2026-06-26.sh
```

Default gates:

```text
FACET_FULL_EVAL_REQUIRE_COMPLETED=1
FACET_FPN_DW_EVAL_REQUIRE_COMPLETED=1
FACET_EFFBS32_EVAL_REQUIRE_COMPLETED=1
```

Each runner uses the same strict completion marker pattern as the watchdogs:

```text
`max_epochs=70` reached
max_epochs=70 reached
Trainer.fit stopped: ... max_epochs=70 ... reached
```

The environment override should only be used for explicit debugging:

```text
FACET_FULL_EVAL_REQUIRE_COMPLETED=0
FACET_FPN_DW_EVAL_REQUIRE_COMPLETED=0
FACET_EFFBS32_EVAL_REQUIRE_COMPLETED=0
```

## Runtime Gate Check

The current incomplete runs were checked. The scripts refused evaluation before
starting GPU validation:

```text
EPNet full training is not complete; refusing final evaluation without completion marker
EPNet fpn_dw training is not complete; refusing final evaluation without completion marker
HBTXR effbs32 training is not complete; refusing final evaluation without completion marker
```

## Regression Smoke

Added:

```text
references/report/FACET/test_evaluation_runner_completion_gate_2026-06-26.sh
```

The smoke checks that each evaluation runner:

- defaults its completion gate to enabled
- defines `training_complete()`
- refuses final evaluation without completion marker
- points at the expected training log
- synchronizes status after final evaluation
- regenerates `FACET_reproduction_completion_audit_2026-06-26.{json,md}`

Update on 2026-06-26 15:35 KST:

```text
The final evaluation runners now regenerate the completion audit immediately
after `sync_reproduction_status_summary.py`. This prevents a successful final
evaluation from leaving `can_mark_goal_complete` stale until the next hourly
guard refresh.
```

The smoke was added to:

```text
references/report/FACET/run_validation_smoke_suite_2026-06-26.sh
```

## Validation

Observed suite output:

```text
[1/4] Python syntax checks
[2/4] Shell syntax checks
[3/4] Artifact validation smoke
artifact validation smoke passed
[4/4] Pairwise, completion marker, and routing smokes
pairwise input validation smoke passed
training completion marker smoke passed
evaluation runner completion gate smoke passed
hourly refresh guard routing smoke passed
FACET validation smoke suite passed
```

## Reproduction Impact

This strengthens the Phase 4 and Phase 4B final evaluation gates. Completion
still requires full training completion markers, full-validation evaluation
artifacts, follow-up fpn_dw/effective-batch-32 artifacts, and all status checker
items to pass.
