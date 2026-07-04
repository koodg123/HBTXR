from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SOFTWARE_ROOT = Path(__file__).resolve().parents[1]
CHECKER = SOFTWARE_ROOT / "scripts" / "v3" / "check_raw_event_count_contract.py"


def _write_contract_fixture(root: Path, *, interpolated: bool = False) -> tuple[Path, Path]:
    manifest_root = root / "manifest1"
    manifest_root.mkdir()
    summary = {
        "data_mode": "mode1",
        "canonical_name": "canonical1",
        "manifest_name": "manifest1",
        "frame_source": "original",
        "resize_policy": "facet_square_direct",
        "event_policy": "fixed_count",
        "event_count_target": 5000,
        "accumulation": "fast_causal_linear",
    }
    (manifest_root / "manifest_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    row = {
        "data_mode": "mode1",
        "canonical_name": "canonical1",
        "manifest_name": "manifest1",
        "frame_source": "original",
        "resize_policy": "facet_square_direct",
        "event_window": {
            "policy": "fixed_count",
            "event_count_target": 5000,
            "accumulation": "fast_causal_linear",
        },
        "source_kind": "interpolated" if interpolated else None,
    }
    for split in ("train", "val", "test"):
        (manifest_root / f"{split}_manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    config = root / "stage2.yaml"
    config.write_text(
        "\n".join(
            [
                "training:",
                "  stage: stage2",
                "data:",
                "  mode: mode1",
                "  canonical_name: canonical1",
                "  manifest_name: manifest1",
                "  mode1:",
                "    event_builder:",
                "      policy: fixed_count",
                "      event_count_target: 5000",
                "      accumulation: fast_causal_linear",
            ]
        ),
        encoding="utf-8",
    )
    return manifest_root, config


def test_raw_event_count_contract_accepts_raw_fixed_count(tmp_path: Path):
    manifest_root, config = _write_contract_fixture(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "--manifest-root",
            str(manifest_root),
            "--stage2-config",
            str(config),
        ],
        cwd=SOFTWARE_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )

    assert "[PASS] raw event-count contract" in result.stdout


def test_raw_event_count_contract_rejects_interpolation_marker(tmp_path: Path):
    manifest_root, config = _write_contract_fixture(tmp_path, interpolated=True)

    result = subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "--manifest-root",
            str(manifest_root),
            "--stage2-config",
            str(config),
        ],
        cwd=SOFTWARE_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
    )

    assert result.returncode == 1
    assert "interpolation marker present" in result.stdout
