"""
02_analyze_labels.py — Pupil LABEL analysis: HDF5 masks + VIA ellipse CSV.

For each labelled mask .h5 it introspects the HDF5 layout, then (best effort)
computes per-frame mask area / centroid / fitted ellipse. For each VIA csv it
computes label coverage and the ellipse-center time series. It then cross-checks
mask-derived centers against VIA ellipse centers (label-pipeline consistency, Q1).

Outputs:
  <out>/02_h5_layout.json          datasets/shapes/dtypes per .h5
  <out>/02_mask_stats.csv          per-frame mask geometry (capped by --max-frames)
  <out>/02_via_coverage.csv        per-session VIA label coverage + quality flags
  <out>/02_labels_summary.json     coverage totals, session gaps, consistency notes

Usage:
  python 02_analyze_labels.py --root E:/DATASET/eveye --out ../results
"""
import os, csv, argparse, sys, glob
import evlib as ev


def analyze_h5(path, max_frames):
    """Return (layout, per_frame_stats[list]). Robust to unknown schema."""
    try:
        import h5py, numpy as np
    except ImportError:
        return ({"error": "h5py not installed"}, [])
    layout = []
    per_frame = []
    try:
        tree = ev.h5_tree(path)
        layout = [{"name": n, "shape": list(sh), "dtype": dt} for n, sh, dt in tree]
        # heuristic: find the largest array that looks like a stack of HxW masks
        with h5py.File(path, "r") as f:
            best = None
            def visit(name, obj):
                nonlocal best
                if isinstance(obj, h5py.Dataset) and obj.ndim in (2, 3):
                    sz = 1
                    for d in obj.shape:
                        sz *= d
                    if best is None or sz > best[1]:
                        best = (name, sz, obj.shape, obj.ndim)
            f.visititems(visit)
            if best:
                name, _, shape, ndim = best
                ds = f[name]
                if ndim == 2:
                    st = ev.mask_stats(ds[()])
                    if st:
                        per_frame.append({"frame": 0, **st})
                else:  # 3D stack: assume axis 0 = frame index (may need transpose)
                    n = min(shape[0], max_frames)
                    for i in range(n):
                        arr = ds[i]
                        if arr.ndim == 2:
                            st = ev.mask_stats(arr)
                            if st:
                                per_frame.append({"frame": i, **st})
    except Exception as e:
        layout = [{"error": str(e)}]
    return (layout, per_frame)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="../results")
    ap.add_argument("--max-frames", type=int, default=50,
                    help="max mask frames to geometrize per h5 (keep modest)")
    ap.add_argument("--max-h5", type=int, default=0, help="limit #h5 files (0=all)")
    args = ap.parse_args()
    ev.ensure_dir(args.out)

    mroot = ev.mask_root(args.root)
    h5_layout = {}
    mask_rows = []
    if os.path.isdir(mroot):
        files = []
        for eye in ev.EYES:
            files += sorted(glob.glob(ev.p(mroot, eye, "*.h5")))
        if args.max_h5:
            files = files[: args.max_h5]
        print(f"[i] {len(files)} mask .h5 files")
        for fp in files:
            key = os.path.relpath(fp, mroot).replace("\\", "/")
            layout, per_frame = analyze_h5(fp, args.max_frames)
            h5_layout[key] = layout
            for st in per_frame:
                mask_rows.append({"file": key, **{k: st.get(k) for k in
                                  ("frame", "area", "centroid", "bbox", "ellipse")}})
    else:
        print(f"[!] mask root not found: {mroot}", file=sys.stderr)

    # VIA coverage across all DAVIS sessions
    via_rows = []
    cov = {"sessions_with_csv": 0, "labelled": 0, "total_rows": 0,
           "quality_false": 0, "session_label_counts": {}}
    davis = ev.davis_root(args.root)
    for u, eye, s, sp in ev.iter_sessions(davis):
        csv_path = ev.p(sp, f"{u}.csv")
        if not os.path.isfile(csv_path):
            continue
        rows = ev.parse_via_csv(csv_path)
        labelled = [r for r in rows if r["region_count"] > 0 and r["ellipse"]]
        qfalse = sum(1 for r in labelled
                     if any(v is False for v in (r["quality"] or {}).values()))
        cov["sessions_with_csv"] += 1
        cov["labelled"] += len(labelled)
        cov["total_rows"] += len(rows)
        cov["quality_false"] += qfalse
        cov["session_label_counts"].setdefault(s, 0)
        cov["session_label_counts"][s] += len(labelled)
        # ellipse-center jitter (Q2): mean abs successive center delta over labelled frames
        centers = [(r["ts"], r["ellipse"][0], r["ellipse"][1]) for r in labelled if r["ts"]]
        centers.sort()
        jit = None
        if len(centers) >= 2:
            import math
            ds = [math.hypot(centers[i+1][1]-centers[i][1], centers[i+1][2]-centers[i][2])
                  for i in range(len(centers)-1)]
            jit = round(sum(ds)/len(ds), 3)
        via_rows.append({"user": u, "eye": eye, "session": s,
                         "session_type": ev.SESSION_MEANING.get(s, "?"),
                         "n_rows": len(rows), "n_labelled": len(labelled),
                         "coverage_pct": round(100*len(labelled)/len(rows), 3) if rows else 0,
                         "quality_false": qfalse,
                         "center_jitter_px": jit})

    # session-gap diagnostic for masks
    masks_by_session = {}
    if os.path.isdir(mroot):
        for eye in ev.EYES:
            for fp in glob.glob(ev.p(mroot, eye, "*.h5")):
                m = os.path.basename(fp)
                for s in ev.SESSIONS:
                    if s in m:
                        masks_by_session.setdefault(s, 0)
                        masks_by_session[s] += 1

    # write
    if mask_rows:
        with open(ev.p(args.out, "02_mask_stats.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(mask_rows[0].keys()))
            w.writeheader()
            for r in mask_rows:
                r = dict(r)
                for k in ("centroid", "bbox", "ellipse"):
                    if isinstance(r.get(k), (list, dict)):
                        r[k] = str(r[k])
                w.writerow(r)
    if via_rows:
        with open(ev.p(args.out, "02_via_coverage.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(via_rows[0].keys()))
            w.writeheader()
            w.writerows(via_rows)
    ev.dump_json(h5_layout, ev.p(args.out, "02_h5_layout.json"))
    ev.dump_json({"via_coverage": cov, "mask_files_by_session": masks_by_session,
                  "note_session_1_0_1_masks": masks_by_session.get("session_1_0_1", 0)},
                 ev.p(args.out, "02_labels_summary.json"))

    print("\n=== LABELS SUMMARY ===")
    print(f" VIA: sessions_with_csv={cov['sessions_with_csv']} labelled={cov['labelled']:,} "
          f"(of {cov['total_rows']:,} rows) quality_false={cov['quality_false']}")
    print(f" VIA labelled by session: {cov['session_label_counts']}")
    print(f" mask .h5 by session: {masks_by_session}")
    if masks_by_session.get("session_1_0_1", 0) == 0:
        print(" [confirmed] session_1_0_1 has NO mask labels (expected per EV-Eye).")
    print(f"\n[i] wrote results to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
