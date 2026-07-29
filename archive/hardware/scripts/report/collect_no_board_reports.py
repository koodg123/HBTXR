#!/usr/bin/env python3
"""Collect no-board HLS/Vivado report evidence into normalized tables.

This script intentionally avoids board/PYNQ artifacts. It parses existing
Vitis HLS csynth XML files and emits CSV/JSON/Markdown summaries that can be
used as the baseline for no-board DSE.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET


RESOURCES = ("BRAM_18K", "DSP", "FF", "LUT", "URAM")
TOP_MODULES = {
    "hgtxr_e2e_axis_top",
    "hgtxr_search_profile_top",
    "hgtxr_track_profile_top",
}


@dataclass
class HlsModuleReport:
    run_id: str
    experiment_id: str
    variant: str
    variant_id: str
    solution: str
    module: str
    status: str
    source_type: str
    recorded_utc: str
    top_model: str
    xml_path: str
    csynth_exists: bool
    csim_log: str | None
    csim_exists: bool
    target_clock_ns: float | None
    estimated_clock_ns: float | None
    latency_best_cycles: int | None
    latency_avg_cycles: int | None
    latency_worst_cycles: int | None
    interval_min_cycles: int | None
    interval_max_cycles: int | None
    bram_18k: int | None
    dsp: int | None
    ff: int | None
    lut: int | None
    uram: int | None
    bram_18k_available: int | None
    dsp_available: int | None
    ff_available: int | None
    lut_available: int | None
    uram_available: int | None

    def pct(self, used: int | None, avail: int | None) -> float | None:
        if used is None or avail in (None, 0):
            return None
        return used * 100.0 / avail

    def to_row(self) -> dict[str, object]:
        row = asdict(self)
        for key in ("bram_18k", "dsp", "ff", "lut", "uram"):
            row[f"{key}_pct"] = self.pct(row[key], row[f"{key}_available"])
        if self.latency_worst_cycles is not None and self.target_clock_ns:
            row["latency_worst_ms_at_target"] = (
                self.latency_worst_cycles * self.target_clock_ns / 1_000_000.0
            )
        else:
            row["latency_worst_ms_at_target"] = None
        if self.module == "hgtxr_search_profile_top" or self.top_model == "hgtxr_search_profile_top":
            row["mode_target_ms"] = 4.0
        elif self.module == "hgtxr_track_profile_top" or self.top_model == "hgtxr_track_profile_top":
            row["mode_target_ms"] = 1.0
        else:
            row["mode_target_ms"] = None
        row["mode_target_met"] = (
            row["latency_worst_ms_at_target"] <= row["mode_target_ms"]
            if row["latency_worst_ms_at_target"] is not None
            and row["mode_target_ms"] is not None
            else None
        )
        row["deterministic_latency"] = (
            self.latency_best_cycles == self.latency_avg_cycles == self.latency_worst_cycles
            if self.latency_best_cycles is not None
            else None
        )
        return row


def text(root: ET.Element, path: str) -> str | None:
    node = root.find(path)
    if node is None or node.text is None:
        return None
    value = node.text.strip()
    return value if value else None


def to_int(value: str | None) -> int | None:
    if value is None or value in {"-", "N/A"}:
        return None
    try:
        return int(float(value.replace(",", "")))
    except ValueError:
        return None


def to_float(value: str | None) -> float | None:
    if value is None or value in {"-", "N/A"}:
        return None
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return None


def variant_from_path(xml_path: Path, generated_root: Path) -> tuple[str, str]:
    try:
        rel = xml_path.relative_to(generated_root)
        parts = rel.parts
    except ValueError:
        parts = xml_path.parts
    variant = parts[0] if len(parts) > 0 else "unknown"
    solution = parts[1] if len(parts) > 1 and parts[1].startswith("solution") else "unknown"
    return variant, solution


def module_from_path(xml_path: Path, top_model: str | None) -> str:
    name = xml_path.name
    if name.endswith("_csynth.xml"):
        return name[: -len("_csynth.xml")]
    if name == "csynth.xml":
        return top_model or "top"
    return xml_path.stem


def csim_log_for(xml_path: Path, generated_root: Path) -> Path | None:
    variant, solution = variant_from_path(xml_path, generated_root)
    module = module_from_path(xml_path, None)
    candidates = [
        generated_root / variant / solution / "csim" / "report" / f"{module}_csim.log",
        generated_root / variant / solution / "csim" / "report" / "hgtxr_e2e_axis_top_csim.log",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def parse_hls_xml(
    xml_path: Path,
    generated_root: Path,
    run_id: str,
    experiment_id: str,
    recorded_utc: str,
) -> HlsModuleReport | None:
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return None

    variant, solution = variant_from_path(xml_path, generated_root)
    top_model = text(root, "UserAssignments/TopModelName") or module_from_path(xml_path, None)
    module = module_from_path(xml_path, top_model)
    csim_log = csim_log_for(xml_path, generated_root)

    resources = {}
    available = {}
    for res in RESOURCES:
        resources[res] = to_int(text(root, f"AreaEstimates/Resources/{res}"))
        available[res] = to_int(text(root, f"AreaEstimates/AvailableResources/{res}"))

    return HlsModuleReport(
        run_id=run_id,
        experiment_id=experiment_id,
        variant=variant,
        variant_id=variant,
        solution=solution,
        module=module,
        status="parsed",
        source_type="hls_csynth_xml",
        recorded_utc=recorded_utc,
        top_model=top_model,
        xml_path=str(xml_path),
        csynth_exists=xml_path.exists(),
        csim_log=str(csim_log) if csim_log else None,
        csim_exists=bool(csim_log),
        target_clock_ns=to_float(text(root, "UserAssignments/TargetClockPeriod")),
        estimated_clock_ns=to_float(
            text(root, "PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod")
        ),
        latency_best_cycles=to_int(
            text(root, "PerformanceEstimates/SummaryOfOverallLatency/Best-caseLatency")
        ),
        latency_avg_cycles=to_int(
            text(root, "PerformanceEstimates/SummaryOfOverallLatency/Average-caseLatency")
        ),
        latency_worst_cycles=to_int(
            text(root, "PerformanceEstimates/SummaryOfOverallLatency/Worst-caseLatency")
        ),
        interval_min_cycles=to_int(
            text(root, "PerformanceEstimates/SummaryOfOverallLatency/Interval-min")
        ),
        interval_max_cycles=to_int(
            text(root, "PerformanceEstimates/SummaryOfOverallLatency/Interval-max")
        ),
        bram_18k=resources["BRAM_18K"],
        dsp=resources["DSP"],
        ff=resources["FF"],
        lut=resources["LUT"],
        uram=resources["URAM"],
        bram_18k_available=available["BRAM_18K"],
        dsp_available=available["DSP"],
        ff_available=available["FF"],
        lut_available=available["LUT"],
        uram_available=available["URAM"],
    )


def collect_reports(
    generated_root: Path,
    run_id: str,
    experiment_id: str,
) -> list[dict[str, object]]:
    reports: list[dict[str, object]] = []
    recorded_utc = datetime.now(timezone.utc).isoformat()
    for xml_path in sorted(generated_root.glob("*/solution*/syn/report/*_csynth.xml")):
        parsed = parse_hls_xml(xml_path, generated_root, run_id, experiment_id, recorded_utc)
        if parsed:
            reports.append(parsed.to_row())
    return reports


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def is_top_row(row: dict[str, object]) -> bool:
    module = str(row["module"])
    top_model = str(row["top_model"])
    xml_path = str(row["xml_path"])
    return (
        module in TOP_MODULES
        or top_model in TOP_MODULES
        or any(xml_path.endswith(f"{top}_csynth.xml") for top in TOP_MODULES)
    )


def p0_violations(rows: list[dict[str, object]]) -> list[str]:
    """Return hard no-board gate failures for P0 Search/Track evidence."""

    top_rows = [r for r in rows if is_top_row(r)]
    violations: list[str] = []
    required_tops = {
        "hgtxr_search_profile_top": "Search",
        "hgtxr_track_profile_top": "Track",
    }

    required_rows: list[dict[str, object]] = []
    for top, label in required_tops.items():
        matches = [
            r
            for r in top_rows
            if r["module"] == top or r["top_model"] == top
        ]
        if not matches:
            violations.append(f"{label}: missing top csynth row for {top}")
            continue
        if not any(r["mode_target_met"] is True for r in matches):
            violations.append(f"{label}: mode target is not met")
        required_rows.extend(matches)

    resource_pairs = (
        ("bram_18k", "bram_18k_available"),
        ("dsp", "dsp_available"),
        ("ff", "ff_available"),
        ("lut", "lut_available"),
        ("uram", "uram_available"),
    )
    for r in required_rows:
        for used_key, avail_key in resource_pairs:
            used = r[used_key]
            avail = r[avail_key]
            if isinstance(used, int) and isinstance(avail, int) and used > avail:
                violations.append(
                    f"{r['variant']}/{r['module']}: {used_key} {used}>{avail}"
                )

    return violations


def write_markdown(out_dir: Path, rows: list[dict[str, object]]) -> None:
    top_rows = [r for r in rows if is_top_row(r)]
    c3b_rows = [
        r
        for r in rows
        if r["variant"] == "hgtxr_e2e_axis_par16_c3b_mem16"
        and r["module"]
        in {
            "hgtxr_e2e_axis_top",
            "hgtxr_e2e_controller_run",
            "hgtxr_e2e_attn_unit_0_s",
            "hgtxr_e2e_attn_unit_1_s",
            "hgtxr_e2e_project_qkv",
            "hgtxr_e2e_attention_core",
            "hgtxr_e2e_mlp_unit_0_s",
            "hgtxr_e2e_mlp_unit_1_s",
            "hgtxr_e2e_mlp_head",
        }
    ]

    lines = [
        "# No-Board HLS Report Collection",
        "",
        "Evidence level: existing Vitis HLS `csynth.xml` reports only. Board inference is excluded.",
        "",
        "## Top Variants",
        "",
        "| Variant | Module | Latency cycles | Latency ms @ target | Mode target ms | Target met | LUT | LUT % | DSP | DSP % | BRAM_18K | BRAM % | URAM | URAM % | XML |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in sorted(top_rows, key=lambda x: str(x["variant"])):
        lines.append(
            "| {variant} | {module} | {lat} | {lat_ms} | {target_ms} | {target_met} | {lut} | {lut_pct} | {dsp} | {dsp_pct} | {bram} | {bram_pct} | {uram} | {uram_pct} | `{xml}` |".format(
                variant=r["variant"],
                module=r["module"],
                lat=fmt(r["latency_worst_cycles"]),
                lat_ms=fmt(r["latency_worst_ms_at_target"]),
                target_ms=fmt(r["mode_target_ms"]),
                target_met=fmt(r["mode_target_met"]),
                lut=fmt(r["lut"]),
                lut_pct=fmt(r["lut_pct"]),
                dsp=fmt(r["dsp"]),
                dsp_pct=fmt(r["dsp_pct"]),
                bram=fmt(r["bram_18k"]),
                bram_pct=fmt(r["bram_18k_pct"]),
                uram=fmt(r["uram"]),
                uram_pct=fmt(r["uram_pct"]),
                xml=r["xml_path"],
            )
        )

    lines.extend(
        [
            "",
            "## C3b Block Rows",
            "",
            "| Module | Latency cycles | LUT | DSP | BRAM_18K | URAM | XML |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for r in sorted(c3b_rows, key=lambda x: str(x["module"])):
        lines.append(
            "| {module} | {lat} | {lut} | {dsp} | {bram} | {uram} | `{xml}` |".format(
                module=r["module"],
                lat=fmt(r["latency_worst_cycles"]),
                lut=fmt(r["lut"]),
                dsp=fmt(r["dsp"]),
                bram=fmt(r["bram_18k"]),
                uram=fmt(r["uram"]),
                xml=r["xml_path"],
            )
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `latency_worst_ms_at_target` is derived from HLS target clock period and worst-case cycles.",
            "- p95/p99 and DMA bandwidth are not emitted here because they require runtime or board instrumentation.",
            "- Use this table as the baseline for `NB-P0-01` parallelism and `NB-P0-03` memory-binding DSE.",
            "",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-root", default="generated")
    parser.add_argument("--run-id", default="p0_no_board_report_collection_2026_06_26")
    parser.add_argument("--experiment-id", default="NB-P0-00")
    parser.add_argument(
        "--output-dir",
        default="analysis/no-board-results/p0_report_collection_2026_06_26",
    )
    parser.add_argument(
        "--enforce-p0",
        action="store_true",
        help="Fail if Search/Track mode-profile targets or ZCU104 HLS capacity gates fail.",
    )
    args = parser.parse_args()

    generated_root = Path(args.generated_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_reports(generated_root, args.run_id, args.experiment_id)
    (out_dir / "hls_modules.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_csv(out_dir / "hls_modules.csv", rows)
    write_markdown(out_dir, rows)

    top_rows = [r for r in rows if is_top_row(r)]
    write_csv(out_dir / "hls_top_variants.csv", top_rows)

    print(f"reports={len(rows)}")
    print(f"top_variants={len(top_rows)}")
    print(f"output_dir={out_dir}")
    if args.enforce_p0:
        violations = p0_violations(rows)
        if violations:
            for violation in violations:
                print(f"P0_VIOLATION: {violation}")
            return 1
        print("P0 gates: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
