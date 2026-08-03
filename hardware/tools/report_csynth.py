#!/usr/bin/env python3
"""S9 — turn the csynth reports into one table (SPEC §9).

    python3 hardware/tools/report_csynth.py
    python3 hardware/tools/report_csynth.py --out hardware/workspace/hls mha_core

Two questions, and only the second needed the tool:

  - resources and timing per unit;
  - did every pipelined loop actually reach its target II. `#pragma HLS pipeline II=1` is a
    REQUEST. g++ discards unknown pragmas, so V1 cannot tell a met request from an ignored
    one, and a loop that quietly schedules at II=4 is a fourfold throughput loss that no
    element-wise comparison fails on.

Stdlib only, like `export_hls_golden.py` — this runs wherever the tool ran.
"""
import argparse
import io
import os
import re
import sys

# `|ap_clk  |  3.33 ns|  2.404 ns|     0.90 ns|`
_CLK = re.compile(r"\|\s*ap_clk\s*\|\s*([\d.]+) ns\s*\|\s*([\d.]+) ns\s*\|\s*([\d.]+) ns")
# The summary total: `|Total            |        0|    32|    2517|    7577|    0|`
_TOTAL = re.compile(r"^\|Total\s*\|\s*(\S+)\|\s*(\S+)\|\s*(\S+)\|\s*(\S+)\|\s*(\S+)\|", re.M)

# A loop that could not be pipelined AT ALL reports achieved "-" and reads exactly like an
# outer tile that was never asked to. The scheduler says so in the log and nowhere else, so
# the log is the other half of the II check.
_SCHED = re.compile(r"^WARNING: \[(SCHED|HLS) 20\d-\d+\].*"
                    r"(unable to (enforce|schedule)|II Violation|cannot be flushed|"
                    r"Unable to satisfy|has been increased)", re.M | re.I)

FIELDS = ("BRAM18", "DSP", "FF", "LUT", "URAM")


def unit_report(out, unit):
    return os.path.join(out, "syn_" + unit, "sol", "syn", "report",
                        "syn_%s_csynth.rpt" % unit)


def read(path):
    with io.open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def summarize(path):
    """(estimated ns, {BRAM18, DSP, FF, LUT, URAM}) from a top-level report."""
    text = read(path)
    clk = _CLK.search(text)
    ns = clk.group(2) if clk else "-"
    # AFTER the utilization header: the same `|Total` shape appears in the Instance detail
    # table further down, and the first one is the summary.
    tail = text.split("== Utilization Estimates", 1)
    tot = _TOTAL.search(tail[1]) if len(tail) > 1 else None
    vals = dict(zip(FIELDS, tot.groups())) if tot else {f: "-" for f in FIELDS}
    return ns, vals


def loop_rows(text):
    """Yield (column-name -> index, cells) for every data row of every `* Loop:` table.

    Header-driven, because the column COUNT varies: a loop table is 8 columns but the
    Performance Estimates summary above it is 7, and a fixed-index parse reads that summary
    as a loop named "3" with a target II of 536870914. It did, and the invented misses were
    convincing enough to nearly ship.
    """
    for chunk in text.split("* Loop:")[1:]:
        head = None
        for line in chunk.splitlines():
            s = line.strip()
            if s.startswith("+"):
                continue                       # the rule above or below a header
            if not s.startswith("|"):
                if head is not None:
                    break                      # the table ended
                continue
            cells = [c.strip() for c in s.strip("|").split("|")]
            if head is None:
                # The first header row spans merged columns and has no "Loop Name"; the
                # second one names every column and is the one to key on.
                if "Loop Name" in cells:
                    head = dict((c, i) for i, c in enumerate(cells))
                continue
            yield head, cells


def loop_misses(out, unit):
    """Loops whose achieved II differs from the target, across every module of the unit.

    Sub-modules, not just the top: every unit is `INLINE off` on purpose, so its loops are
    reported in its own file rather than the top's.
    """
    d = os.path.dirname(unit_report(out, unit))
    misses = []
    if not os.path.isdir(d):
        return misses
    for name in sorted(os.listdir(d)):
        if not name.endswith("_csynth.rpt"):
            continue
        for head, cells in loop_rows(read(os.path.join(d, name))):
            try:
                loop = cells[head["Loop Name"]].lstrip("-+ ")
                achieved = cells[head["achieved"]]
                target = cells[head["target"]]
            except (KeyError, IndexError):
                continue
            # Both are "-" on a loop that was never asked to pipeline, which is the outer
            # tiles and is the intent. A loop that ASKED and failed outright is also "-"
            # here and is caught from the log instead -- see `sched_warnings`.
            if achieved.isdigit() and target.isdigit() and achieved != target:
                misses.append((loop, achieved, target, name[:-len("_csynth.rpt")]))
    return misses


def sched_warnings(out, unit):
    """Scheduler complaints from the run log — the half of the II answer the report drops."""
    p = os.path.join(out, unit + ".log")
    if not os.path.isfile(p):
        return []
    return [m.group(0).strip() for m in _SCHED.finditer(read(p))]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("units", nargs="*", help="default: every unit with a report")
    ap.add_argument("--out", default=os.path.join("hardware", "workspace", "hls"))
    args = ap.parse_args(argv)

    units = args.units
    if not units:
        if not os.path.isdir(args.out):
            sys.stderr.write("no reports under %s -- run run_csynth.sh first\n" % args.out)
            return 2
        units = sorted(n[len("syn_"):] for n in os.listdir(args.out)
                       if n.startswith("syn_"))
    if not units:
        sys.stderr.write("no reports under %s\n" % args.out)
        return 2

    rows, missing = [], []
    for u in units:
        p = unit_report(args.out, u)
        if not os.path.isfile(p):
            missing.append(u)
            rows.append((u, "-", {f: "-" for f in FIELDS}))
            continue
        ns, vals = summarize(p)
        rows.append((u, ns, vals))

    w = max(len(r[0]) for r in rows)
    head = "| %-*s | %6s | %6s | %5s | %7s | %7s | %4s |" % (
        w, "unit", "est ns", "BRAM18", "DSP", "FF", "LUT", "URAM")
    print(head)
    print("|%s|%s|%s|%s|%s|%s|%s|" % ("-" * (w + 2), "-" * 8, "-" * 8, "-" * 7, "-" * 9,
                                      "-" * 9, "-" * 6))
    for u, ns, v in rows:
        print("| %-*s | %6s | %6s | %5s | %7s | %7s | %4s |" % (
            w, u, ns, v["BRAM18"], v["DSP"], v["FF"], v["LUT"], v["URAM"]))

    print("")
    print("II misses (achieved != target):")
    total = 0
    for u in units:
        for loop, ach, tgt, mod in loop_misses(args.out, u):
            print("  %-14s %-28s achieved %s, target %s   [%s]" % (u, loop, ach, tgt, mod))
            total += 1
    if total == 0:
        print("  none -- every pipelined loop met its target")

    print("")
    print("scheduler warnings:")
    warned = 0
    for u in units:
        for w in sched_warnings(args.out, u):
            print("  %-14s %s" % (u, w[:160]))
            warned += 1
    if warned == 0:
        print("  none")

    if missing:
        print("")
        print("no report: %s" % " ".join(missing))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
