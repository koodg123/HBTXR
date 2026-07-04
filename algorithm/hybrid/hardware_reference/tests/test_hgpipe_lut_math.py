from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_hgpipe_lut_math_validator(tmp_path: Path):
    out = tmp_path / "hgpipe_lut_math_validation.json"
    subprocess.run(
        [sys.executable, "hardware/tools/validate_hgpipe_lut_math.py", "--out", str(out)],
        cwd=ROOT,
        check=True,
    )
    payload = json.loads(out.read_text())
    assert payload["status"] == "pass"
    assert payload["tables"]["gelu_entries"] == 16
    assert payload["tables"]["exp_entries"] == 16
    assert payload["tables"]["rsqrt_entries"] == 32
    assert "HG-PIPE" in payload["source_pattern"]
