"""Lightweight cv2 dark-pupil detector to produce a DENSE pupil-center trajectory
(no torch). Used ONLY for motion labeling / velocity (NOT for accuracy claims)."""
import os, glob
import numpy as np, cv2
from common import list_frames, parse_frame_filename

def detect_center(img_gray):
    """Return (cx, cy, conf) of darkest blob (pupil) or (nan,nan,0)."""
    g = cv2.GaussianBlur(img_gray, (5, 5), 0)
    # adaptive dark threshold: pupil is among darkest pixels
    thr = np.percentile(g, 3)              # darkest ~3%
    _, m = cv2.threshold(g, thr, 255, cv2.THRESH_BINARY_INV)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.nan, np.nan, 0.0
    # pick the most circular sizable contour
    best = None; best_score = -1
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 30 or a > 0.25 * img_gray.size:
            continue
        peri = cv2.arcLength(c, True)
        if peri == 0:
            continue
        circ = 4 * np.pi * a / (peri * peri)   # 1=circle
        score = circ * np.sqrt(a)
        if score > best_score:
            best_score = score; best = c
    if best is None:
        return np.nan, np.nan, 0.0
    M = cv2.moments(best)
    if M["m00"] == 0:
        return np.nan, np.nan, 0.0
    cx = M["m10"] / M["m00"]; cy = M["m01"] / M["m00"]
    return float(cx), float(cy), float(best_score)

def trajectory(user, eye, code, max_frames=None, stride=1):
    frames = list_frames(user, eye, code)
    if stride > 1:
        frames = frames[::stride]
    if max_frames:
        frames = frames[:max_frames]
    rec = []
    for f in frames:
        idx, ts = parse_frame_filename(f)
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is None:
            rec.append((idx, ts, np.nan, np.nan, 0.0)); continue
        cx, cy, conf = detect_center(img)
        rec.append((idx, ts, cx, cy, conf))
    import pandas as pd
    return pd.DataFrame(rec, columns=["frame_idx", "ts", "x", "y", "conf"])

if __name__ == "__main__":
    # sanity-check detector against GT on user1/left/201 (saccade, 49 GT pts)
    import pandas as pd
    from common import load_gt_ellipses
    gt = load_gt_ellipses(1, "left", "201")
    traj = trajectory(1, "left", "201")
    m = traj.set_index("frame_idx")
    errs = []
    for _, r in gt.iterrows():
        if r.frame_idx in m.index:
            row = m.loc[r.frame_idx]
            if not np.isnan(row.x):
                errs.append(np.hypot(row.x - r.cx, row.y - r.cy))
    errs = np.array(errs)
    print(f"detector vs GT on user1/left/201: n={len(errs)}, "
          f"median={np.median(errs):.1f}px, mean={errs.mean():.1f}px, p90={np.percentile(errs,90):.1f}px")
    print(f"trajectory valid frames: {traj.x.notna().sum()}/{len(traj)}")
