import argparse
import subprocess
import sys
from pathlib import Path


FACET_ROOT = Path(__file__).resolve().parents[3]
ROOT = FACET_ROOT.parents[3]
REPORT_ROOT = ROOT / "references/report/FACET"
SCRIPTS_ROOT = FACET_ROOT / "EvEye/utils/scripts"


def run(command: list[str]):
    completed = subprocess.run(command, text=True, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def build_summary(date: str):
    run(
        [
            sys.executable,
            str(SCRIPTS_ROOT / "build_reproduction_summary.py"),
            "--status-json",
            str(REPORT_ROOT / f"FACET_reproduction_status_{date}.json"),
            "--result",
            f"EPNet_full_unet:{REPORT_ROOT / f'FACET_reproduction_results_{date}.json'}",
            "--result",
            f"HBTXR_full_unet:{REPORT_ROOT / f'FACET_hbtxr_reproduction_results_{date}.json'}",
            "--result",
            f"EPNet_fpn_dw_full_unet:{REPORT_ROOT / f'FACET_epnet_fpn_dw_reproduction_results_{date}.json'}",
            "--result",
            f"HBTXR_full_unet_effbs32:{REPORT_ROOT / f'FACET_hbtxr_effbs32_reproduction_results_{date}.json'}",
            "--comparison",
            f"EPNet_vs_HBTXR:{REPORT_ROOT / f'FACET_epnet_vs_hbtxr_comparison_{date}.json'}",
            "--comparison",
            f"EPNet_vs_HBTXR_effbs32:{REPORT_ROOT / f'FACET_epnet_vs_hbtxr_effbs32_comparison_{date}.json'}",
            "--output-json",
            str(REPORT_ROOT / f"FACET_reproduction_summary_{date}.json"),
            "--output-md",
            str(REPORT_ROOT / f"FACET_reproduction_results_{date}.md"),
        ]
    )


def refresh_status(date: str):
    run(
        [
            sys.executable,
            str(SCRIPTS_ROOT / "check_reproduction_status.py"),
            "--output-json",
            str(REPORT_ROOT / f"FACET_reproduction_status_{date}.json"),
            "--output-md",
            str(REPORT_ROOT / f"FACET_reproduction_status_{date}.md"),
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Synchronize FACET reproduction summary and status artifacts."
    )
    parser.add_argument("--date", default="2026-06-26")
    args = parser.parse_args()

    build_summary(args.date)
    refresh_status(args.date)
    build_summary(args.date)
    refresh_status(args.date)


if __name__ == "__main__":
    main()
