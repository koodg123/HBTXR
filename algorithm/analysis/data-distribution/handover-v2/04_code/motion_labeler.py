"""Motion labeling for EV-Eye per-motion analysis.

Methods:
  (1) data_segments.xlsx windows -> label GT frames inside the labeled saccade/
      smooth window; carry the blink flag.
  (2) velocity at each GT frame via cv2 central-difference over local neighbors
      (efficient: reads only +-W frames around each GT frame) -> I-VT class.
  (3) velocity binning (low/med/high by data-driven percentiles).

Output: cache/velocity_at_gt.csv  (one row per GT frame, resumable per session).
"""
import os, sys, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, cv2
from common import (USERS, EYES, SESSIONS, CACHE_DIR, session_dir,
                    load_gt_ellipses, load_data_segments, parse_frame_filename)
from pupil_detect import detect_center

W = 2  # neighbor half-window in frames (+-2 ~ +-80ms at 25Hz)

def _frame_map(user, eye, code):
    d = os.path.join(session_dir(user, eye, code), "frames")
    return {parse_frame_filename(f)[0]: f for f in glob.glob(os.path.join(d, "*.png"))}

def _center_at(fmap, idx, cache):
    if idx in cache:
        return cache[idx]
    f = fmap.get(idx)
    if f is None:
        cache[idx] = (np.nan, np.nan, 0.0, np.nan); return cache[idx]
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    _, ts = parse_frame_filename(f)
    if img is None:
        cache[idx] = (np.nan, np.nan, 0.0, np.nan); return cache[idx]
    cx, cy, conf = detect_center(img)
    cache[idx] = (cx, cy, conf, ts); return cache[idx]

def velocity_for_session(user, eye, code, segs):
    motion = SESSIONS[code][1]
    gt = load_gt_ellipses(user, eye, code)
    if len(gt) == 0:
        return pd.DataFrame()
    fmap = _frame_map(user, eye, code)
    seg = segs[user][code]
    if motion == "saccade":
        w0, w1, blink = seg.get("sacc_start"), seg.get("sacc_end"), seg.get("blink")
    else:
        w0, w1, blink = seg.get("smooth_start"), seg.get("smooth_end"), seg.get("blink")
    try:
        w0 = int(w0); w1 = int(w1)
    except Exception:
        w0, w1 = -1, -1
    cache = {}
    rows = []
    for _, r in gt.iterrows():
        i = int(r.frame_idx)
        # central difference using nearest available neighbors within +-W
        prev = next_ = None
        for d in range(1, W + 1):
            if prev is None:
                c = _center_at(fmap, i - d, cache)
                if not np.isnan(c[0]):
                    prev = c
            if next_ is None:
                c = _center_at(fmap, i + d, cache)
                if not np.isnan(c[0]):
                    next_ = c
        cur = _center_at(fmap, i, cache)
        speed = np.nan
        if prev is not None and next_ is not None and not np.isnan(prev[3]) and not np.isnan(next_[3]):
            dt = (next_[3] - prev[3]) / 1e6  # s
            if dt > 0:
                disp = np.hypot(next_[0] - prev[0], next_[1] - prev[1])
                speed = disp / dt  # px/s
        in_win = (w0 <= i <= w1) if w0 >= 0 else False
        rows.append(dict(user=user, eye=eye, code=code, motion=motion,
                         frame_idx=i, ts=int(r.ts), gt_cx=r.cx, gt_cy=r.cy,
                         det_x=cur[0], det_y=cur[1], det_conf=cur[2],
                         speed_pxps=speed, in_seg_window=in_win, seg_blink=str(blink)))
    return pd.DataFrame(rows)

def build(users=None, time_budget=38.0):
    import time
    t0 = time.time()
    segs = load_data_segments()
    out_path = os.path.join(CACHE_DIR, "velocity_at_gt.csv")
    done = set(); all_rows = []
    if os.path.exists(out_path):
        prev = pd.read_csv(out_path)
        done = set(zip(prev.user, prev.eye, prev.code.astype(str)))
        all_rows = [prev]
    users = users or USERS
    todo = [(u, eye, code) for u in users for eye in EYES for code in ["102","201","202"]
            if (u, eye, str(code)) not in done]
    processed = 0
    for (u, eye, code) in todo:
        if time.time() - t0 > time_budget:
            break
        df = velocity_for_session(u, eye, code, segs)
        if len(df):
            all_rows.append(df)
        processed += 1
    res = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    res.to_csv(out_path, index=False)
    remaining = len(todo) - processed
    return res, out_path, processed, remaining

if __name__ == "__main__":
    res, p, processed, remaining = build()
    print(f"processed {processed} sessions this run; remaining={remaining}; "
          f"total rows={len(res)}; users_done={res.user.nunique()}")
