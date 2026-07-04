#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/kjm26/project/PRJXR/HBTXR"
FACET_ROOT="$ROOT/references/codebase/software/FACET"
PY="$ROOT/.facet-train-venv/bin/python"
CONFIG_ONLY_LOG="/tmp/facet_training_completion_config_only.log"
NOT_REACHED_LOG="/tmp/facet_training_completion_not_reached.log"
COMPLETED_LOG="/tmp/facet_training_completion_completed.log"
STOPPED_BAD_LOG="/tmp/facet_training_completion_stopped_bad.log"
STOPPED_GOOD_LOG="/tmp/facet_training_completion_stopped_good.log"
SHELL_COMPLETION_PATTERN='`max_epochs=70` reached|max_epochs=70 reached|Trainer\.fit stopped:.*max_epochs=70.*reached'

export PYTHONPATH="$FACET_ROOT"
export PYTHONPYCACHEPREFIX=/tmp/facet_training_completion_marker_pycache

printf '%s\n' 'trainer config: max_epochs=70' >"$CONFIG_ONLY_LOG"
printf '%s\n' 'trainer config: max_epochs=70 not reached' >"$NOT_REACHED_LOG"
printf '%s\n' '`max_epochs=70` reached.' >"$COMPLETED_LOG"
printf '%s\n' 'Trainer.fit stopped unexpectedly after user interrupt' >"$STOPPED_BAD_LOG"
printf '%s\n' 'Trainer.fit stopped: `max_epochs=70` reached.' >"$STOPPED_GOOD_LOG"

"$PY" - "$CONFIG_ONLY_LOG" "$NOT_REACHED_LOG" "$COMPLETED_LOG" "$STOPPED_BAD_LOG" "$STOPPED_GOOD_LOG" <<'PY'
import sys
from pathlib import Path

from EvEye.utils.scripts.check_reproduction_status import training_complete

config_only = Path(sys.argv[1])
not_reached = Path(sys.argv[2])
completed = Path(sys.argv[3])
stopped_bad = Path(sys.argv[4])
stopped_good = Path(sys.argv[5])

if training_complete(config_only):
    raise SystemExit("config-only max_epochs=70 log was incorrectly accepted")
if training_complete(not_reached):
    raise SystemExit("max_epochs=70 not reached log was incorrectly accepted")
if training_complete(stopped_bad):
    raise SystemExit("Trainer.fit stopped without max_epochs reached was incorrectly accepted")
if not training_complete(completed):
    raise SystemExit("Lightning completion marker was not accepted")
if not training_complete(stopped_good):
    raise SystemExit("Trainer.fit stopped max_epochs reached marker was not accepted")

PY

if rg -q "$SHELL_COMPLETION_PATTERN" "$CONFIG_ONLY_LOG"; then
  echo "shell marker accepted config-only max_epochs=70 log" >&2
  exit 1
fi
if rg -q "$SHELL_COMPLETION_PATTERN" "$NOT_REACHED_LOG"; then
  echo "shell marker accepted max_epochs=70 not reached log" >&2
  exit 1
fi
if rg -q "$SHELL_COMPLETION_PATTERN" "$STOPPED_BAD_LOG"; then
  echo "shell marker accepted Trainer.fit stopped without max_epochs reached" >&2
  exit 1
fi
if ! rg -q "$SHELL_COMPLETION_PATTERN" "$COMPLETED_LOG"; then
  echo "shell marker rejected real completion log" >&2
  exit 1
fi
if ! rg -q "$SHELL_COMPLETION_PATTERN" "$STOPPED_GOOD_LOG"; then
  echo "shell marker rejected Trainer.fit stopped max_epochs reached log" >&2
  exit 1
fi

echo "training completion marker smoke passed"
