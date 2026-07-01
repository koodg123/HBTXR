"""
04_analyze_tracking_results.py — introspect Frame_event_pupil_track_result/*.mat
and (optionally) Data_davis_predict/*_mask.gif predicted masks.

.mat files (update_20_point_userN_session_X.mat) hold the frame+event hybrid
pupil-tracking trajectories. We load them version-agnostically and dump each
variable's shape/dtype + basic numeric ranges so you can see the schema, then
extract any 2D (T x 2) trajectory as a pupil-center time series.

Outputs:
  <out>/04_mat_layout.json       variables/shapes per .mat
  <out>/04_track_summary.json    coverage + trajectory length stats
  <out>/04_predict_inventory.csv (optional) predicted-mask GIF counts per session

Usage:
  python 04_analyze_tracking_results.py --root E:/DATASET/eveye --out ../results
  python 04_analyze_tracking_results.py --root E:/DATASET/eveye --out ../results --predict
"""
import os, csv, argparse, sys, glob
import evlib as ev


def summarize_var(v):
    try:
        import numpy as np
        a = np.asarray(v)
        info = {"shape": list(a.shape), "dtype": str(a.dtype)}
        if a.dtype.kind in "fiu" and a.size:
            info["min"] = float(np.nanmin(a)); info["max"] = float(np.nanmax(a))
            info["mean"] = round(float(np.nanmean(a)), 4)
        return info
    except Exception as e:
        return {"repr": str(type(v)), "note": str(e)[:120]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="../results")
    ap.add_argument("--max-mat", type=int, default=0, help="limit #mat files (0=all)")
    ap.add_argument("--predict", action="store_true",
                    help="also inventory Data_davis_predict mask GIFs (can be slow)")
    args = ap.parse_args()
    ev.ensure_dir(args.out)

    troot = ev.track_root(args.root)
    layout = {}
    traj_lens = []
    cov = {"files": 0, "by_session": {}, "by_eye": {}, "engines": {}}
    if os.path.isdir(troot):
        files = []
        for eye in ev.EYES:
            files += sorted(glob.glob(ev.p(troot, eye, "*.mat")))
        if args.max_mat:
            files = files[: args.max_mat]
        print(f"[i] {len(files)} tracking .mat files")
        for fp in files:
            key = os.path.relpath(fp, troot).replace("\\", "/")
            d, engine = ev.load_mat_any(fp)
            cov["engines"][engine] = cov["engines"].get(engine, 0) + 1
            layout[key] = {k: summarize_var(v) for k, v in d.items()}
            cov["files"] += 1
            for eye in ev.EYES:
                if key.startswith(eye):
                    cov["by_eye"][eye] = cov["by_eye"].get(eye, 0) + 1
            for s in ev.SESSIONS:
                if s in key:
                    cov["by_session"][s] = cov["by_session"].get(s, 0) + 1
            # longest 2-col array = candidate (T x 2) pupil-center trajectory
            try:
                import numpy as np
                best = None
                for k, v in d.items():
                    a = np.asarray(v)
                    if a.ndim == 2 and 2 in a.shape and a.dtype.kind in "fiu":
                        T = a.shape[0] if a.shape[1] == 2 else a.shape[1]
                        if best is None or T > best:
                            best = T
                if best:
                    traj_lens.append(best)
            except Exception:
                pass
    else:
        print(f"[!] tracking root not found: {troot}", file=sys.stderr)

    predict_rows = []
    if args.predict:
        proot = ev.predict_root(args.root)
        if os.path.isdir(proot):
            # path: Data_davis_predict/userN/userN/{left,right}/session_X/predict/*.gif
            for u in os.listdir(proot):
                inner = ev.p(proot, u, u)  # doubled userN
                base = inner if os.path.isdir(inner) else ev.p(proot, u)
                for eye in ev.EYES:
                    for s in ev.SESSIONS:
                        pd = ev.p(base, eye, s, "predict")
                        if os.path.isdir(pd):
                            n = len(glob.glob(ev.p(pd, "*.gif"))) + len(glob.glob(ev.p(pd, "*.png")))
                            predict_rows.append({"user": u, "eye": eye, "session": s,
                                                 "n_predicted_masks": n})
            print(f"[i] predict sessions found: {len(predict_rows)}")
        else:
            print(f"[!] predict root not found: {proot}", file=sys.stderr)

    ev.dump_json(layout, ev.p(args.out, "04_mat_layout.json"))
    summ = {"tracking_files": cov["files"], "coverage": cov}
    if traj_lens:
        traj_lens.sort()
        summ["trajectory_length"] = {
            "n": len(traj_lens), "min": traj_lens[0], "max": traj_lens[-1],
            "median": traj_lens[len(traj_lens)//2]}
    ev.dump_json(summ, ev.p(args.out, "04_track_summary.json"))
    if predict_rows:
        with open(ev.p(args.out, "04_predict_inventory.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(predict_rows[0].keys()))
            w.writeheader(); w.writerows(predict_rows)

    print("\n=== TRACKING SUMMARY ===")
    print(f" files={cov['files']} engines={cov['engines']} by_session={cov['by_session']}")
    print(f"[i] wrote results to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
