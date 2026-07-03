"""Inventory scan of EV-Eye Data_davis for per-motion-type analysis."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from common import (USERS, EYES, SESSIONS, CACHE_DIR, session_dir, list_frames,
                    load_gt_ellipses, load_data_segments, parse_frame_filename)

def scan():
    segs = load_data_segments()
    rows = []
    for u in USERS:
        for eye in EYES:
            for code, (folder, motion, pattern) in SESSIONS.items():
                d = session_dir(u, eye, code)
                if not os.path.isdir(d):
                    rows.append(dict(user=u, eye=eye, code=code, motion=motion, pattern=pattern,
                                     exists=0, n_frames=0, dur_s=0, fps=0, n_ann=0,
                                     seg_blink="", seg_window="")); continue
                frames = list_frames(u, eye, code); nf = len(frames)
                if nf >= 2:
                    _, t0 = parse_frame_filename(frames[0]); _, t1 = parse_frame_filename(frames[-1])
                    dur = (t1 - t0) / 1e6
                    ts = np.array([parse_frame_filename(f)[1] for f in frames])
                    dt = np.median(np.diff(ts)) / 1e6; fps = (1.0/dt) if dt > 0 else 0
                else:
                    dur = 0; fps = 0
                gt = load_gt_ellipses(u, eye, code); n_ann = len(gt)
                seg = segs[u][code]
                win = (f"{seg.get('sacc_start')}-{seg.get('sacc_end')}" if motion=="saccade"
                       else f"{seg.get('smooth_start')}-{seg.get('smooth_end')}")
                rows.append(dict(user=u, eye=eye, code=code, motion=motion, pattern=pattern,
                                 exists=1, n_frames=nf, dur_s=round(dur,1), fps=round(fps,1),
                                 n_ann=n_ann, seg_blink=str(seg.get("blink")), seg_window=win))
    return pd.DataFrame(rows)

def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    df = scan()
    out = os.path.join(CACHE_DIR, "inventory.csv"); df.to_csv(out, index=False)
    print(f"wrote {out}  ({len(df)} rows)")
    print("\n==== totals ====")
    print(f"users={df.user.nunique()}  sessions_present={int(df.exists.sum())}/{len(df)}")
    print(f"total frames={int(df.n_frames.sum()):,}  total annotated GT frames={int(df.n_ann.sum()):,}")
    print("\n==== by motion ====")
    g = df.groupby("motion").agg(sessions=("exists","sum"), frames=("n_frames","sum"), ann=("n_ann","sum")).reset_index()
    print(g.to_string(index=False))
    print("\n==== by session code ====")
    g2 = df.groupby(["code","motion","pattern"]).agg(present=("exists","sum"), frames=("n_frames","sum"),
        ann=("n_ann","sum"), ann_per_sess=("n_ann","mean")).reset_index()
    g2["ann_per_sess"] = g2["ann_per_sess"].round(1)
    print(g2.to_string(index=False))
    print("\n==== GT-evaluable error samples per motion (annotated only) ====")
    print(df[df.n_ann>0].groupby("motion").n_ann.sum().to_string())
    print("\nNOTE: session 101 (saccade) has no manual GT -> saccade GT comes only from 201.")

if __name__ == "__main__":
    main()
