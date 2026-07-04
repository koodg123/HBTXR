from __future__ import annotations

import subprocess
from pathlib import Path


SOFTWARE_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = SOFTWARE_ROOT / "scripts" / "v3" / "run_raw_event_count_train.sh"
CANONICAL_ROOT = "/home/kjm26/project/dataset/EV_Eye/canonical"


def test_raw_event_count_wrapper_passes_canonical_root_to_both_stages():
    result = subprocess.run(
        ["sh", str(WRAPPER), "--dry-run", "--skip-prepare"],
        cwd=SOFTWARE_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    override = f"--override data.canonical_root={CANONICAL_ROOT}"
    stage_lines = [line for line in result.stdout.splitlines() if "scripts/external/run_train.sh" in line]

    assert len(stage_lines) == 2
    assert all(override in line for line in stage_lines)
