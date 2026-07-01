"""
03_analyze_tobii_gaze.py — Tobii Pro Glasses 3 gaze GT + clock-sync diagnostics (Q4).

For each Tobii session it parses the JSON-lines `gazedata` (handles plain text or
.gz), computes sampling rate, validity (NaN/missing), gaze2d ranges and pupil
diameters; reads tobiisend.txt / tobiittl.txt for the cross-device TTL offset.

Outputs:
  <out>/03_tobii_gaze.csv       per-session gaze stats
  <out>/03_tobii_sync.csv       per-user TTL send/recv offsets
  <out>/03_tobii_summary.json

Usage:
  python 03_analyze_tobii_gaze.py --root E:/DATASET/eveye --out ../results
"""
import os, csv, json, gzip, argparse, sys, glob, re
import evlib as ev


def open_maybe_gz(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", errors="ignore")
    return open(path, "r", errors="ignore")


def find_gaze_file(session_dir):
    """gazedata may be a file (no ext), gazedata.gz, or inside a folder."""
    for cand in ("gazedata", "gazedata.gz", "gazedata.txt"):
        pth = ev.p(session_dir, cand)
        if os.path.isfile(pth):
            return pth
    # folder named gazedata
    fold = ev.p(session_dir, "gazedata")
    if os.path.isdir(fold):
        hits = glob.glob(ev.p(fold, "*"))
        if hits:
            return hits[0]
    return None


def parse_gaze(path, cap=0):
    ts, g2x, g2y, pdl, pdr = [], [], [], [], []
    n_valid = n_total = 0
    with open_maybe_gz(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            try:
                o = json.loads(line)
            except Exception:
                # maybe the 6-col txt extract
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        ts.append(float(parts[1])); g2x.append(float(parts[2]))
                        g2y.append(float(parts[3])); n_valid += 1
                    except ValueError:
                        pass
                continue
            if o.get("type") != "gaze":
                continue
            d = o.get("data", {})
            t = o.get("timestamp")
            gd = d.get("gaze2d")
            if t is not None:
                ts.append(float(t))
            if isinstance(gd, list) and len(gd) == 2 and gd[0] == gd[0]:  # not NaN
                g2x.append(float(gd[0])); g2y.append(float(gd[1])); n_valid += 1
            for side, arr in (("eyeleft", pdl), ("eyeright", pdr)):
                pe = d.get(side, {})
                pdv = pe.get("pupildiameter")
                if isinstance(pdv, (int, float)) and pdv == pdv:
                    arr.append(float(pdv))
            if cap and n_total >= cap:
                break

    def stats(a):
        if not a:
            return None
        a2 = sorted(a)
        return {"n": len(a), "min": round(a2[0], 4), "max": round(a2[-1], 4),
                "mean": round(sum(a)/len(a), 4), "median": round(a2[len(a2)//2], 4)}

    rate = None
    if len(ts) >= 2:
        span = ts[-1] - ts[0]
        rate = round((len(ts)-1)/span, 2) if span > 0 else None
    return {
        "n_records": n_total, "n_valid_gaze": n_valid,
        "valid_pct": round(100*n_valid/n_total, 2) if n_total else 0,
        "sample_rate_hz": rate,
        "t_start_s": round(ts[0], 4) if ts else None,
        "t_end_s": round(ts[-1], 4) if ts else None,
        "gaze2d_x": stats(g2x), "gaze2d_y": stats(g2y),
        "pupil_diam_left_mm": stats(pdl), "pupil_diam_right_mm": stats(pdr),
    }


def read_float_list(path):
    if not os.path.isfile(path):
        return []
    out = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
            if m:
                try:
                    out.append(float(m[-1]))
                except ValueError:
                    pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="../results")
    ap.add_argument("--cap", type=int, default=0, help="max gaze rows/session (0=all)")
    ap.add_argument("--max-users", type=int, default=0)
    args = ap.parse_args()
    ev.ensure_dir(args.out)

    troot = ev.tobii_root(args.root)
    if not os.path.isdir(troot):
        print(f"[!] Data_tobii not found: {troot}", file=sys.stderr)
        sys.exit(1)

    users = sorted([d for d in os.listdir(troot) if re.fullmatch(r"user\d+", d)],
                   key=ev.user_id)
    if args.max_users:
        users = users[: args.max_users]

    gaze_rows, sync_rows = [], []
    for u in users:
        ubase = ev.p(troot, u)
        send = read_float_list(ev.p(ubase, "tobiisend.txt"))
        ttl = read_float_list(ev.p(ubase, "tobiittl.txt"))
        offset = None
        if send and ttl:
            offset = round(ttl[0] - send[0], 6)
        sync_rows.append({"user": u, "n_send": len(send), "n_ttl": len(ttl),
                          "send0": send[0] if send else None,
                          "ttl0": ttl[0] if ttl else None,
                          "ttl_minus_send0": offset})
        for s in ev.SESSIONS:
            sd = ev.p(ubase, s)
            if not os.path.isdir(sd):
                continue
            gpath = find_gaze_file(sd)
            if not gpath:
                continue
            try:
                st = parse_gaze(gpath, cap=args.cap)
            except Exception as e:
                st = {"error": str(e)}
            gaze_rows.append({"user": u, "session": s,
                              "session_type": ev.SESSION_MEANING.get(s, "?"),
                              "gaze_file": os.path.basename(gpath),
                              **{k: (json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v)
                                 for k, v in st.items()}})
        print(f"  [{u}] done")

    if gaze_rows:
        with open(ev.p(args.out, "03_tobii_gaze.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(gaze_rows[0].keys()))
            w.writeheader(); w.writerows(gaze_rows)
    if sync_rows:
        with open(ev.p(args.out, "03_tobii_sync.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(sync_rows[0].keys()))
            w.writeheader(); w.writerows(sync_rows)
    ev.dump_json({"users": len(users), "gaze_sessions": len(gaze_rows),
                  "note": "gaze2d is normalized [0,1] scene coords; timestamp is "
                          "recording-relative seconds. DAVIS uses absolute us UNIX; "
                          "align via tobiisend/tobiittl TTL offset."},
                 ev.p(args.out, "03_tobii_summary.json"))
    print(f"\n[i] gaze sessions={len(gaze_rows)}  users={len(users)}")
    print(f"[i] wrote results to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
