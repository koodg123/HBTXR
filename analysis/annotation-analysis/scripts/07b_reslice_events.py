"""
07b_reslice_events.py — re-slice events for ALREADY-collected windows.

Fixes the earlier n_events=0 bug (events.txt is 'ts x y pol', 4 space-separated
fields; the leading integer is the TIMESTAMP, not an index). Frames/labels stay
as-is; only samples/event/*.npz and manifest_windows.csv (n_events) are rewritten.

Usage:
  python 07b_reslice_events.py --root /mnt/e/DATASET/eveye --out ../samples
"""
import os, csv, argparse, sys
import evlib as ev

try:
    import numpy as np
except Exception:
    np = None


def slice_events(events_txt, windows, pad_us=0):
    """One pass over events.txt (4-field 'ts x y pol'); route to windows by [ts_lo,ts_hi]."""
    ranges = [(int(w["ts_lo"]) - pad_us, int(w["ts_hi"]) + pad_us, w["key"]) for w in windows]
    buf = {w["key"]: {"t": [], "x": [], "y": [], "p": []} for w in windows}
    if not os.path.isfile(events_txt) or not ranges:
        return buf
    tmin = min(r[0] for r in ranges)
    tmax = max(r[1] for r in ranges)
    with open(events_txt, "r", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                ts = int(parts[0])
            except ValueError:
                continue
            if ts < tmin:
                continue
            if ts > tmax:
                break  # events.txt is time-sorted ascending
            x, y, pol = int(parts[1]), int(parts[2]), int(parts[3])
            for lo, hi, key in ranges:
                if lo <= ts <= hi:
                    b = buf[key]
                    b["t"].append(ts); b["x"].append(x); b["y"].append(y); b["p"].append(pol)
    return buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="EV-Eye root, e.g. /mnt/e/DATASET/eveye")
    ap.add_argument("--out", default="../samples")
    ap.add_argument("--event-pad-us", type=int, default=0)
    args = ap.parse_args()
    if np is None:
        print("[!] numpy required", file=sys.stderr); sys.exit(2)

    man = ev.p(args.out, "manifest_windows.csv")
    if not os.path.isfile(man):
        print(f"[!] {man} not found", file=sys.stderr); sys.exit(1)
    with open(man, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"[i] {len(rows)} windows in manifest")

    ev.ensure_dir(ev.p(args.out, "event"))
    bysession = {}
    for r in rows:
        bysession.setdefault((r["user"], r["eye"], r["session"]), []).append(r)

    total_ev = 0
    n_by_key = {}
    for (u, eye, session), wins in bysession.items():
        events_txt = ev.p(ev.davis_root(args.root), u, eye, session, "events", "events.txt")
        buf = slice_events(events_txt, wins, args.event_pad_us)
        for w in wins:
            b = buf.get(w["key"], {"t": [], "x": [], "y": [], "p": []})
            n = len(b["t"])
            n_by_key[w["key"]] = n
            total_ev += n
            np.savez_compressed(
                ev.p(args.out, "event", w["key"] + ".npz"),
                t=np.asarray(b["t"], dtype=np.int64),
                x=np.asarray(b["x"], dtype=np.int16),
                y=np.asarray(b["y"], dtype=np.int16),
                p=np.asarray(b["p"], dtype=np.int8),
                ts_lo=int(w["ts_lo"]), ts_hi=int(w["ts_hi"]))
        med = sorted(n_by_key[w["key"]] for w in wins)[len(wins) // 2]
        print(f"  [{u}/{eye}/{session}] {len(wins)} windows, median n_events={med}")

    # rewrite manifest_windows.csv with updated n_events
    for r in rows:
        r["n_events"] = n_by_key.get(r["key"], 0)
    with open(man, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    nz = sum(1 for r in rows if int(r["n_events"]) > 0)
    print(f"\n[done] total events={total_ev:,}  windows_with_events={nz}/{len(rows)}")
    print(f"       updated {man} and samples/event/*.npz")
    if nz == 0:
        print("[!] still zero — check events.txt path/format for one session.")


if __name__ == "__main__":
    main()
