from __future__ import annotations

import csv
import os
import subprocess
from datetime import date
from pathlib import Path


ROOT = Path("/home/user/project/PRJXR")
HGTXR = ROOT / "HGTXR"
OUT = HGTXR / "docs" / "cleanup"
STAMP = "2026_06_09"


def du_bytes(path: Path) -> int:
    try:
        out = subprocess.check_output(
            ["du", "-sb", str(path)], text=True, stderr=subprocess.DEVNULL
        )
        return int(out.split()[0])
    except Exception:
        return path.stat().st_size if path.is_file() else 0


def human(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{num_bytes} B"


def mtime(path: Path) -> str:
    try:
        return date.fromtimestamp(path.stat().st_mtime).isoformat()
    except Exception:
        return ""


def classify(path: Path) -> tuple[str, str, str, str]:
    name = path.name
    rel = "./" + str(path.relative_to(ROOT))
    cls = "review"
    action = "manual review before cleanup"
    risk = "medium"
    reason = "Unclassified artifact; inspect before moving or deleting."

    keep_rel = {
        "./HGTXR",
        "./HGTXR/hardware",
        "./HGTXR/software",
        "./HGTXR/configs",
        "./HGTXR/tools",
        "./HGTXR/tests",
        "./HGTXR/docs",
        "./HGTXR/pynq",
        "./HGTXR/README.md",
        "./HGTXR/requirements.txt",
        "./HGTXR/pyproject.toml",
    }
    if rel in keep_rel or rel.startswith("./HGTXR/docs/cleanup"):
        return (
            "keep",
            "preserve in place",
            "low",
            "Source, config, docs, tests, packaged overlay, or cleanup provenance.",
        )

    if rel in {"./PAPER_PRJXR", "./impl_repos", "./REF"}:
        return (
            "keep-review",
            "preserve; optionally archive snapshots later",
            "high",
            "Research papers, resources, references, or implementation evidence.",
        )

    if rel == "./old":
        return (
            "archive-candidate",
            "compress or move to archive after confirming no unique source is needed",
            "high",
            "Large legacy tree with checkpoints, old HG-PIPE outputs, and experiments.",
        )

    if rel == "./workspace":
        return (
            "archive-candidate",
            "archive synthesis workspace after preserving reports and bitstreams",
            "medium",
            "Generated synthesis workspace; may contain reproducibility evidence.",
        )

    if rel in {"./HGTXR/hardware/generated/hgtxr_hls", "./HGTXR/hardware/generated/hgtxr_e2e_hls", "./HGTXR/hardware/generated/build"}:
        return (
            "archive-candidate",
            "archive after copying required reports to docs/resources",
            "medium",
            "Generated HLS/Vivado output; expensive to regenerate but not source.",
        )

    if rel == "./HGTXR/hardware/generated/hgtxr_e2e_hls/solution_e2e_q4w8a/syn/report":
        return (
            "keep",
            "preserve report evidence in place or mirror to docs/resources",
            "low",
            "Latest PAR5 E2E csynth evidence.",
        )

    if name in {".venv", ".pytest_cache", ".Xil", "__pycache__"} or name.endswith(".pyc"):
        return (
            "regenerate-candidate",
            "delete after rebuild command is documented and approved",
            "low",
            "Tool/cache/dependency output that can be regenerated.",
        )

    if name.endswith((".log", ".jou", ".str")):
        return (
            "archive-or-delete-candidate",
            "compress or delete duplicate tool logs after review",
            "medium",
            "Tool log; may contain provenance but is often duplicated by reports.",
        )

    empty_marker_names = {
        "4",
        "8",
        "16",
        "32",
        "HLS",
        "PYNQ",
        "Vitis",
        "Vivado",
        "true",
        "ap_ctrl_hs",
        "ap_memory",
        "ap_vld",
    }
    if name in empty_marker_names and du_bytes(path) == 0:
        return (
            "candidate-delete",
            "delete only after user approval",
            "low",
            "Empty accidental-looking marker path.",
        )

    if name.endswith((".npz", ".json")) and path.parent == HGTXR:
        return (
            "archive-or-delete-candidate",
            "archive or delete after docs/resources copy is confirmed",
            "low",
            "Small root-level smoke artifact outside normal artifact folders.",
        )

    return cls, action, risk, reason


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    entries: list[Path] = []
    entries.extend(sorted(ROOT.iterdir(), key=lambda item: item.name.lower()))
    entries.extend(sorted(HGTXR.iterdir(), key=lambda item: item.name.lower()))
    latest_report = HGTXR / "hgtxr_e2e_hls" / "solution_e2e_q4w8a" / "syn" / "report"
    if latest_report.exists():
        entries.append(latest_report)

    rows = []
    seen: set[Path] = set()
    for path in entries:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        cls, action, risk, reason = classify(path)
        size = du_bytes(path)
        rows.append(
            {
                "path": str(path),
                "relative_path": "./" + str(path.relative_to(ROOT)),
                "scope": "HGTXR" if str(path).startswith(str(HGTXR)) else "PRJXR",
                "kind": "dir" if path.is_dir() else "file",
                "size_bytes": size,
                "size_human": human(size),
                "mtime": mtime(path),
                "class": cls,
                "recommended_action": action,
                "risk": risk,
                "reason": reason,
            }
        )

    manifest = OUT / f"cleanup_manifest_{STAMP}.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    largest: list[tuple[int, Path]] = []
    for dirpath, _, filenames in os.walk(ROOT):
        for filename in filenames:
            path = Path(dirpath) / filename
            try:
                largest.append((path.stat().st_size, path))
            except OSError:
                continue
    largest.sort(reverse=True, key=lambda item: item[0])

    largest_csv = OUT / f"largest_files_{STAMP}.csv"
    with largest_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["size_bytes", "size_human", "path", "class_hint", "reason"])
        for size, path in largest[:150]:
            cls, _, _, reason = classify(path)
            writer.writerow([size, human(size), str(path), cls, reason])

    class_sizes: dict[str, int] = {}
    for row in rows:
        class_sizes[row["class"]] = class_sizes.get(row["class"], 0) + int(row["size_bytes"])

    plan = [
        "# Cleanup Dry-Run Plan - 2026-06-09",
        "",
        "Scope:",
        "- /home/user/project/PRJXR",
        "- /home/user/project/PRJXR/HGTXR",
        "",
        "Safety boundary:",
        "- No files were deleted, moved, compressed, or renamed in this pass.",
        "- This is a dry-run classification manifest only.",
        "- candidate-delete means deletion is plausible after user approval, not that it should be deleted now.",
        "",
        "Generated artifacts:",
        f"- docs/cleanup/cleanup_manifest_{STAMP}.csv",
        f"- docs/cleanup/largest_files_{STAMP}.csv",
        f"- docs/cleanup/cleanup_plan_{STAMP}.md",
        f"- docs/cleanup/cleanup_summary_{STAMP}.md",
        "",
        "Class size summary:",
    ]
    for cls, total in sorted(class_sizes.items(), key=lambda item: item[1], reverse=True):
        plan.append(f"- {cls}: {human(total)}")
    plan.extend(
        [
            "",
            "Recommended cleanup phases:",
            "",
            "1. Preserve evidence first",
            "   - Keep HGTXR source, hardware/, software/, docs/, and root environment files in place.",
            "   - Keep docs/resources and latest hgtxr_e2e_hls PAR5 csynth report evidence.",
            "   - Preserve PAPER_PRJXR and impl_repos as research provenance unless separately archived.",
            "",
            "2. Low-risk regenerate cleanup after approval",
            "   - Remove __pycache__, .pyc, .pytest_cache, and .Xil caches.",
            "   - Remove empty accidental-looking marker paths only after a final path list review.",
            "",
            "3. Archive generated hardware workspaces after approval",
            "   - Archive HGTXR/hardware/generated/hgtxr_hls, HGTXR/hardware/generated/hgtxr_e2e_hls, HGTXR/hardware/generated/build, and workspace/synthesis after reports/bitstreams are mirrored.",
            "   - Do not delete HLS/Vivado projects until report-backed CSV rows and key bit/hwh artifacts are verified.",
            "",
            "4. Legacy/reference storage decision",
            "   - old is 39G and should be archive-first, not delete-first.",
            "   - REF and impl_repos are research inputs; preserve or archive by topic.",
            "   - Large venv directories can be deleted only if dependency rebuild commands are documented.",
            "",
            "5. Final verification after any real cleanup",
            "   - Re-run HGTXR static validation.",
            "   - Check that docs/resources/e2e_axis_baseline_2026_06_09.md and the PAR5 csynth CSV row remain present.",
            "   - Check that hardware/pynq/hgtxr bit/hwh artifacts remain present if board validation is still needed.",
        ]
    )
    (OUT / f"cleanup_plan_{STAMP}.md").write_text("\n".join(plan) + "\n")

    summary = ["# Cleanup Dry-Run Summary - 2026-06-09", "", "No files were deleted, moved, compressed, or renamed.", ""]
    summary.append("Largest classified entries:")
    summary.append("")
    for row in sorted(rows, key=lambda item: int(item["size_bytes"]), reverse=True)[:20]:
        summary.append(
            f"- {row['size_human']} {row['relative_path']} [{row['class']}] {row['recommended_action']}"
        )
    summary.append("")
    summary.append("Immediate low-risk candidates for a later approved cleanup:")
    summary.append("")
    for row in rows:
        if row["class"] in {"candidate-delete", "regenerate-candidate"}:
            summary.append(f"- {row['size_human']} {row['relative_path']} [{row['class']}] {row['reason']}")
    summary.append("")
    summary.append("High-risk preserve/archive items:")
    summary.append("")
    for row in rows:
        if row["risk"] == "high":
            summary.append(f"- {row['size_human']} {row['relative_path']} [{row['class']}] {row['reason']}")
    (OUT / f"cleanup_summary_{STAMP}.md").write_text("\n".join(summary) + "\n")

    print(manifest)
    print(largest_csv)
    print(OUT / f"cleanup_plan_{STAMP}.md")
    print(OUT / f"cleanup_summary_{STAMP}.md")


if __name__ == "__main__":
    main()
