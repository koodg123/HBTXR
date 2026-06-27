#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/kjm26/project/PRJXR/HBTXR"
REPORT_ROOT="$ROOT/references/report/FACET"

python3 - "$REPORT_ROOT" <<'PY'
import re
import sys
from pathlib import Path

report_root = Path(sys.argv[1])
pattern = re.compile(
    r'(?:MIN_INTERVAL_SECONDS|INTERVAL_SECONDS|WAIT_INTERVAL_SECONDS)='
    r'"\$\{([A-Z0-9_]+):-([0-9]+)\}"'
)
required_envs = {
    "FACET_STATUS_REFRESH_MIN_INTERVAL_SECONDS",
    "FACET_TRAINING_WATCHDOG_INTERVAL_SECONDS",
    "FACET_WATCH_INTERVAL_SECONDS",
    "FACET_FOLLOWUP_WATCHDOG_INTERVAL_SECONDS",
    "FACET_FPN_DW_WATCH_INTERVAL_SECONDS",
    "FACET_EFFBS32_WATCH_INTERVAL_SECONDS",
    "FACET_FPN_DW_WAIT_INTERVAL_SECONDS",
    "FACET_EFFBS32_WAIT_INTERVAL_SECONDS",
    "FACET_HBTXR_PROBE_INTERVAL_SECONDS",
}

found = {}
bad_defaults = []
for path in sorted(report_root.glob("*.sh")):
    text = path.read_text(encoding="utf-8")
    for env_name, default in pattern.findall(text):
        found[env_name] = str(path)
        if default != "3600":
            bad_defaults.append((path, env_name, default))

missing = sorted(required_envs - found.keys())
if missing:
    raise SystemExit("missing monitored interval defaults: " + ", ".join(missing))

if bad_defaults:
    lines = [
        f"{path}: {env_name} default is {default}, expected 3600"
        for path, env_name, default in bad_defaults
    ]
    raise SystemExit("\n".join(lines))
PY

echo "monitoring interval defaults smoke passed"
