#!/usr/bin/env python3
"""Write a compact final-signoff blocker audit from preflight JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BLOCKER_ACTIONS = {
    "C3b AXIS/DMA physical smoke result": {
        "kind": "physical-board-smoke",
        "action": "Run the C3b bundle on ZCU104 PYNQ, validate it on-board, then import the JSON on host.",
        "commands": [
            "tar -xzf e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz",
            "cd e2e_axis_dma_c3b_mem16_smoke_bundle",
            "./run_e2e_axis_dma_c3b_mem16_file_smoke.sh",
            "./validate_e2e_axis_dma_c3b_mem16_file_smoke.sh",
            "python3 tools/import_pynq_smoke_result.py /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --preset axis-c3b-mem16",
        ],
    },
    "requested PAPER_PRJXR DeiT image": {
        "kind": "reference-input",
        "action": "Restore the requested PAPER_PRJXR image path or record an approved replacement source.",
        "commands": [],
    },
    "requested XR-VITs sibling": {
        "kind": "reference-input",
        "action": "Restore the XR-VITs sibling checkout or record an approved replacement source.",
        "commands": [],
    },
}


def load_preflight(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("preflight JSON must be an object")
    checks = payload.get("checks")
    if not isinstance(checks, list):
        raise ValueError("preflight JSON missing checks list")
    return payload


def build_audit(preflight: dict[str, Any]) -> dict[str, Any]:
    checks = [check for check in preflight["checks"] if isinstance(check, dict)]
    failures = [check for check in checks if check.get("status") == "fail"]
    warnings = [check for check in checks if check.get("status") == "warn"]
    blockers = []
    next_commands: list[str] = []

    for check in failures:
        name = str(check.get("name", "unknown"))
        action = BLOCKER_ACTIONS.get(name, {"kind": "generic", "action": "Resolve this failed preflight check.", "commands": []})
        commands = list(action["commands"])
        next_commands.extend(commands)
        blockers.append(
            {
                "name": name,
                "kind": action["kind"],
                "detail": str(check.get("detail", "")),
                "path": check.get("path", ""),
                "action": action["action"],
                "commands": commands,
            }
        )

    return {
        "status": "pass" if not blockers else "blocked",
        "source_mode": preflight.get("mode", ""),
        "source_summary": preflight.get("summary", {}),
        "blocker_count": len(blockers),
        "warning_count": len(warnings),
        "blockers": blockers,
        "warnings": [
            {
                "name": str(check.get("name", "")),
                "detail": str(check.get("detail", "")),
                "path": check.get("path", ""),
            }
            for check in warnings
        ],
        "next_commands": next_commands,
    }


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# HGTXR Final Signoff Audit",
        "",
        f"- status: `{audit['status']}`",
        f"- source_mode: `{audit['source_mode']}`",
        f"- blockers: `{audit['blocker_count']}`",
        f"- warnings: `{audit['warning_count']}`",
        "",
        "## Blockers",
        "",
    ]

    blockers = audit["blockers"]
    if not blockers:
        lines.append("- None.")
    else:
        for blocker in blockers:
            path = blocker["path"] or "n/a"
            lines.append(f"- `{blocker['name']}` ({blocker['kind']}): {blocker['detail']}")
            lines.append(f"  - path: `{path}`")
            lines.append(f"  - action: {blocker['action']}")

    lines.extend(["", "## Next Commands", ""])
    commands = audit["next_commands"]
    if commands:
        lines.append("```sh")
        lines.extend(commands)
        lines.append("```")
    else:
        lines.append("- None.")

    lines.extend(["", "## Warnings", ""])
    warnings = audit["warnings"]
    if warnings:
        for warning in warnings:
            path = warning["path"] or "n/a"
            lines.append(f"- `{warning['name']}`: {warning['detail']} (`{path}`)")
    else:
        lines.append("- None.")

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-json", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--markdown-out", required=True)
    args = parser.parse_args(argv)

    audit = build_audit(load_preflight(Path(args.preflight_json)))
    json_out = Path(args.json_out)
    markdown_out = Path(args.markdown_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(audit, indent=2) + "\n")
    markdown_out.write_text(render_markdown(audit))
    print(f"[final-signoff-audit] status={audit['status']} blockers={audit['blocker_count']} warnings={audit['warning_count']}")
    return 1 if audit["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
