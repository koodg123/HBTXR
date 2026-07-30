#!/usr/bin/env python3
"""Audit that the user-selected A2->A1 and C paths are evidenced."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import write_e2e_resource_matrix


DATE_TAG = "2026_06_10"
HGTXR_ROOT = Path(__file__).resolve().parents[2]


def add_check(checks: list[dict[str, Any]], name: str, ok: bool, detail: str) -> None:
    checks.append({"name": name, "status": "pass" if ok else "fail", "detail": detail})


def load_matrix(root: Path, matrix_path: Path | None) -> dict[str, Any]:
    if matrix_path is not None:
        payload = json.loads(matrix_path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"{matrix_path} must contain a JSON object")
        return payload
    return write_e2e_resource_matrix.build_matrix(root)


def row_by_id(matrix: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = matrix.get("rows", [])
    if not isinstance(rows, list):
        return {}
    return {str(row.get("id")): row for row in rows if isinstance(row, dict)}


def resource(row: dict[str, Any], key: str) -> int | None:
    hls = row.get("hls", {})
    resources = hls.get("resources", {}) if isinstance(hls, dict) else {}
    value = resources.get(key) if isinstance(resources, dict) else None
    return value if isinstance(value, int) else None


def latency(row: dict[str, Any]) -> int | None:
    hls = row.get("hls", {})
    value = hls.get("latency_cycles") if isinstance(hls, dict) else None
    return value if isinstance(value, int) else None


def artifacts_ready(row: dict[str, Any]) -> bool:
    artifacts = row.get("artifacts", {})
    if not isinstance(artifacts, dict):
        return False
    bit = artifacts.get("bit", {})
    hwh = artifacts.get("hwh", {})
    return (
        isinstance(bit, dict)
        and isinstance(hwh, dict)
        and bit.get("exists") is True
        and hwh.get("exists") is True
    )


def build_audit(root: Path, matrix_path: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    matrix = load_matrix(root, matrix_path)
    rows = row_by_id(matrix)
    checks: list[dict[str, Any]] = []

    a2 = rows.get("A2", {})
    a1 = rows.get("A1", {})
    c1 = rows.get("C1", {})
    c3b = rows.get("C3b", {})

    add_check(checks, "resource_matrix_pass", matrix.get("status") == "pass", str(matrix.get("status")))
    add_check(checks, "a2_present", "A2" in rows, ",".join(sorted(rows)))
    add_check(checks, "a1_present", "A1" in rows, ",".join(sorted(rows)))
    add_check(checks, "c1_present", "C1" in rows, ",".join(sorted(rows)))
    add_check(checks, "c3b_present", "C3b" in rows, ",".join(sorted(rows)))
    add_check(checks, "e_pending_policy", matrix.get("policy", {}).get("e_pending") is True, str(matrix.get("policy", {}).get("e_pending")))

    add_check(checks, "path1_a2_selection_first", a2.get("selection_path") == "Path 1, first", str(a2.get("selection_path")))
    add_check(checks, "path1_a1_after_a2", a1.get("selection_path") == "Path 1, after A2", str(a1.get("selection_path")))
    add_check(checks, "path1_a2_artifacts_ready", artifacts_ready(a2), str(a2.get("artifacts", {})))
    add_check(checks, "path1_a1_artifacts_ready", artifacts_ready(a1), str(a1.get("artifacts", {})))
    add_check(checks, "path1_a2_parallelism_8", a2.get("parallelism") == 8, str(a2.get("parallelism")))
    add_check(checks, "path1_a1_parallelism_8", a1.get("parallelism") == 8, str(a1.get("parallelism")))
    add_check(checks, "path1_a2_uram_positive", (resource(a2, "uram") or 0) > 0, str(resource(a2, "uram")))
    add_check(checks, "path1_a1_dsp_positive", (resource(a1, "dsp") or 0) > 0, str(resource(a1, "dsp")))

    add_check(checks, "path2_c1_par16", c1.get("parallelism") == 16, str(c1.get("parallelism")))
    add_check(checks, "path2_c3b_par16", c3b.get("parallelism") == 16, str(c3b.get("parallelism")))
    add_check(checks, "path2_c3b_mem16", c3b.get("memory_banks") == 16, str(c3b.get("memory_banks")))
    add_check(checks, "path2_c3b_artifacts_ready", artifacts_ready(c3b), str(c3b.get("artifacts", {})))
    add_check(checks, "path2_c_dsp_gain_vs_a1", (resource(c3b, "dsp") or 0) > (resource(a1, "dsp") or 0), f"{resource(c3b, 'dsp')} > {resource(a1, 'dsp')}")
    add_check(checks, "path2_c_latency_gain_vs_a1", (latency(c3b) or 0) < (latency(a1) or 0), f"{latency(c3b)} < {latency(a1)}")
    add_check(checks, "path2_c3b_lut_below_c1", (resource(c3b, "lut") or 0) < (resource(c1, "lut") or 0), f"{resource(c3b, 'lut')} < {resource(c1, 'lut')}")
    add_check(checks, "path2_c3b_recommended", matrix.get("summary", {}).get("recommended_board_smoke_variant") == "C3b", str(matrix.get("summary", {}).get("recommended_board_smoke_variant")))

    fail_count = sum(1 for check in checks if check["status"] != "pass")
    observed = {
        "a2": {
            "parallelism": a2.get("parallelism"),
            "memory_banks": a2.get("memory_banks"),
            "latency_cycles": latency(a2),
            "dsp": resource(a2, "dsp"),
            "lut": resource(a2, "lut"),
            "uram": resource(a2, "uram"),
            "board_status": a2.get("board_status"),
        },
        "a1": {
            "parallelism": a1.get("parallelism"),
            "memory_banks": a1.get("memory_banks"),
            "latency_cycles": latency(a1),
            "dsp": resource(a1, "dsp"),
            "lut": resource(a1, "lut"),
            "uram": resource(a1, "uram"),
            "board_status": a1.get("board_status"),
        },
        "c3b": {
            "parallelism": c3b.get("parallelism"),
            "memory_banks": c3b.get("memory_banks"),
            "latency_cycles": latency(c3b),
            "dsp": resource(c3b, "dsp"),
            "lut": resource(c3b, "lut"),
            "uram": resource(c3b, "uram"),
            "board_status": c3b.get("board_status"),
        },
    }
    return {
        "status": "pass" if fail_count == 0 else "fail",
        "root": str(root),
        "date_tag": DATE_TAG,
        "selection": {
            "path_1": "A2 then A1",
            "path_2": "C, with C3b as board-smoke candidate",
            "e": "pending",
        },
        "source": {
            "resource_matrix": str(matrix_path.resolve()) if matrix_path is not None else "generated-from-reports",
            "choice_doc": str(root / "docs" / "track" / "CHOICE.md"),
        },
        "observed": observed,
        "pass_count": len(checks) - fail_count,
        "fail_count": fail_count,
        "check_count": len(checks),
        "checks": checks,
        "safety": {
            "executes_hls_or_vivado": False,
            "executes_board_smoke": False,
            "creates_board_result": False,
            "creates_xr_vits_policy": False,
            "writes_canonical_inputs": False,
        },
    }


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# HGTXR Selected Path Execution Audit",
        "",
        f"- status: `{audit['status']}`",
        f"- root: `{audit['root']}`",
        f"- Path 1: `{audit['selection']['path_1']}`",
        f"- Path 2: `{audit['selection']['path_2']}`",
        f"- E: `{audit['selection']['e']}`",
        f"- checks: `{audit['pass_count']}/{audit['check_count']}`",
        "",
        "## Observed",
        "",
        "| Variant | PAR | Mem Banks | Latency | DSP | LUT | URAM | Board |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for key in ("a2", "a1", "c3b"):
        row = audit["observed"][key]
        lines.append(
            f"| {key.upper()} | {row['parallelism']} | {row['memory_banks']} | "
            f"{row['latency_cycles']} | {row['dsp']} | {row['lut']} | {row['uram']} | "
            f"`{row['board_status']}` |"
        )
    lines.extend(["", "## Checks", "", "| Check | Status | Detail |", "|---|---|---|"])
    for check in audit["checks"]:
        detail = str(check["detail"]).replace("|", "\\|")
        lines.append(f"| {check['name']} | `{check['status']}` | {detail} |")
    lines.extend(
        [
            "",
            "## Safety",
            "",
            "- Does not run HLS or Vivado.",
            "- Does not run board smoke.",
            "- Does not create board result JSON.",
            "- Does not create XR-VITs replacement policy.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit selected HGTXR A/C execution paths.")
    parser.add_argument("--root", type=Path, default=HGTXR_ROOT)
    parser.add_argument("--resource-matrix", type=Path)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    audit = build_audit(args.root, args.resource_matrix)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    args.markdown_out.write_text(render_markdown(audit))
    print(
        f"[selected-path-execution-audit] status={audit['status']} "
        f"checks={audit['pass_count']}/{audit['check_count']}"
    )
    return 0 if audit["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
