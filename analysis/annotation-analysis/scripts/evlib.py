"""
evlib.py — Shared helpers for EV-Eye annotation/label analysis.

This module is intentionally dependency-light at import time. Heavy deps
(h5py, scipy, cv2, pandas, matplotlib) are imported lazily inside functions
so that scripts which don't need them still run.

Dataset layout assumed (EV-Eye, Ningreka/EV-Eye):
  <root>/raw_data/Data_davis/userN/{left,right}/session_X/{events,frames}/...
  <root>/raw_data/Data_davis/userN/{left,right}/session_X/user_N.csv      (VIA ellipse)
  <root>/raw_data/Data_davis_labelled_with_mask/{left,right}/userN_session_X.h5
  <root>/raw_data/Data_tobii/userN/{tobiisend.txt,tobiittl.txt,session_X/...}
  <root>/processed_data/Data_davis_predict/userN/userN/{left,right}/session_X/predict/*_mask.gif
  <root>/processed_data/Frame_event_pupil_track_result/{left,right}/update_20_point_userN_session_X.mat
"""
from __future__ import annotations
import os
import re
import json
import glob

SESSIONS = ["session_1_0_1", "session_1_0_2", "session_2_0_1", "session_2_0_2"]
EYES = ["left", "right"]
SESSION_MEANING = {
    "session_1_0_1": "saccade+fixation",
    "session_1_0_2": "saccade+fixation",
    "session_2_0_1": "smooth_pursuit",
    "session_2_0_2": "smooth_pursuit",
}

# ----------------------------------------------------------------------------- paths
def p(*parts):
    return os.path.join(*parts)

def davis_root(root):
    return p(root, "raw_data", "Data_davis")

def mask_root(root):
    return p(root, "raw_data", "Data_davis_labelled_with_mask")

def tobii_root(root):
    return p(root, "raw_data", "Data_tobii")

def predict_root(root):
    return p(root, "processed_data", "Data_davis_predict")

def track_root(root):
    return p(root, "processed_data", "Frame_event_pupil_track_result")

def list_users(davis_dir):
    """Return sorted user dir names like ['user1','user2',...] (natural sort)."""
    if not os.path.isdir(davis_dir):
        return []
    users = [d for d in os.listdir(davis_dir) if re.fullmatch(r"user\d+", d)]
    return sorted(users, key=lambda s: int(s[4:]))

def user_id(uname):
    m = re.search(r"\d+", uname)
    return int(m.group()) if m else -1

def iter_sessions(davis_dir):
    """Yield (user, eye, session, session_abs_path) for every existing DAVIS session."""
    for u in list_users(davis_dir):
        for eye in EYES:
            base = p(davis_dir, u, eye)
            if not os.path.isdir(base):
                continue
            for s in SESSIONS:
                sp = p(base, s)
                if os.path.isdir(sp):
                    yield u, eye, s, sp

# ----------------------------------------------------------------------------- events.txt
def parse_events_line(line):
    """events.txt row: 'timestamp x y polarity' (space-separated; ts is us UNIX).
    Returns (ts_us, x, y, pol) or None.
    NOTE: the leading integer is the TIMESTAMP, not an index column."""
    parts = line.split()
    if len(parts) < 4:
        return None
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))
    except ValueError:
        return None

def events_summary(events_txt, sample_cap=None):
    """Stream events.txt once. Returns dict with count, t_start_us, t_end_us,
    duration_s, rate_hz, pol_on, pol_off, x_max, y_max. sample_cap limits the
    rows scanned (for a fast estimate); None = full pass."""
    n = on = off = 0
    t0 = t1 = None
    xmax = ymax = 0
    if not os.path.isfile(events_txt):
        return None
    with open(events_txt, "r", errors="ignore") as f:
        for line in f:
            rec = parse_events_line(line)
            if rec is None:
                continue
            ts, x, y, pol = rec
            if t0 is None:
                t0 = ts
            t1 = ts
            n += 1
            if pol:
                on += 1
            else:
                off += 1
            if x > xmax:
                xmax = x
            if y > ymax:
                ymax = y
            if sample_cap and n >= sample_cap:
                break
    if n == 0:
        return None
    dur = (t1 - t0) / 1e6 if (t0 is not None and t1 is not None) else 0.0
    return {
        "count": n,
        "t_start_us": t0,
        "t_end_us": t1,
        "duration_s": round(dur, 6),
        "rate_hz": round(n / dur, 1) if dur > 0 else None,
        "pol_on": on,
        "pol_off": off,
        "on_off_ratio": round(on / off, 4) if off else None,
        "x_max": xmax,
        "y_max": ymax,
        "sampled": bool(sample_cap),
    }

def estimate_event_count_by_size(events_txt, probe_lines=2000):
    """Fast estimate of event count = filesize / avg_line_bytes (no full pass)."""
    if not os.path.isfile(events_txt):
        return None
    size = os.path.getsize(events_txt)
    nb = nl = 0
    with open(events_txt, "rb") as f:
        for raw in f:
            nb += len(raw)
            nl += 1
            if nl >= probe_lines:
                break
    if nl == 0:
        return 0
    avg = nb / nl
    return int(size / avg)

# ----------------------------------------------------------------------------- frames
def frame_files(frames_dir):
    if not os.path.isdir(frames_dir):
        return []
    return sorted(glob.glob(p(frames_dir, "*.png")))

def parse_frame_timestamp(fname):
    """'000123_1657711084437019.png' -> (123, 1657711084437019)"""
    base = os.path.basename(fname)
    m = re.match(r"(\d+)_(\d+)\.png", base)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))

def read_timestamps_txt(path):
    """frames/timestamps.txt -> list of (index, ts_us)."""
    out = []
    if not os.path.isfile(path):
        return out
    with open(path, "r", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    out.append((int(parts[0]), int(parts[1])))
                except ValueError:
                    pass
    return out

# ----------------------------------------------------------------------------- VIA csv
def parse_via_csv(csv_path):
    """Parse a VIA export user_N.csv. Returns list of dict rows:
    {filename, ts, region_count, ellipse:(cx,cy,rx,ry,theta)|None, quality:dict}."""
    import csv as _csv
    rows = []
    if not os.path.isfile(csv_path):
        return rows
    with open(csv_path, "r", errors="ignore", newline="") as f:
        reader = _csv.DictReader(f)
        for r in reader:
            fn = r.get("filename", "")
            rc = r.get("region_count", "0")
            try:
                rc = int(rc)
            except ValueError:
                rc = 0
            ell = None
            shape = r.get("region_shape_attributes", "") or "{}"
            try:
                sd = json.loads(shape)
                if sd.get("name") == "ellipse":
                    ell = (
                        float(sd.get("cx", "nan")),
                        float(sd.get("cy", "nan")),
                        float(sd.get("rx", "nan")),
                        float(sd.get("ry", "nan")),
                        float(sd.get("theta", 0.0)),
                    )
            except Exception:
                pass
            quality = {}
            try:
                qa = json.loads(r.get("region_attributes", "") or "{}")
                quality = qa.get("image_quality", {}) if isinstance(qa, dict) else {}
            except Exception:
                pass
            tsm = re.search(r"_(\d+)\.png", fn)
            ts = int(tsm.group(1)) if tsm else None
            rows.append({"filename": fn, "ts": ts, "region_count": rc,
                         "ellipse": ell, "quality": quality})
    return rows

# ----------------------------------------------------------------------------- h5 / mat
def h5_tree(path, max_items=200):
    """Return a list of (name, shape, dtype) for every dataset in an HDF5 file."""
    import h5py
    out = []
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            out.append((name, tuple(obj.shape), str(obj.dtype)))
    with h5py.File(path, "r") as f:
        f.visititems(visit)
    return out[:max_items]

def load_mat_any(path):
    """Load a .mat regardless of version. Returns (dict_of_vars, engine).
    engine in {'scipy','h5py'}."""
    try:
        from scipy.io import loadmat
        d = loadmat(path, squeeze_me=True, struct_as_record=False)
        return ({k: v for k, v in d.items() if not k.startswith("__")}, "scipy")
    except NotImplementedError:
        # v7.3 -> HDF5
        import h5py, numpy as np
        out = {}
        with h5py.File(path, "r") as f:
            for k in f.keys():
                try:
                    out[k] = np.array(f[k])
                except Exception:
                    out[k] = "<unreadable>"
        return (out, "h5py")
    except Exception as e:
        return ({"__error__": str(e)}, "error")

# ----------------------------------------------------------------------------- mask geometry
def mask_stats(mask2d):
    """Given a 2D binary/intensity mask, return area, centroid, bbox, fitted ellipse."""
    import numpy as np
    m = np.asarray(mask2d)
    if m.ndim != 2:
        return None
    b = m > (m.max() / 2.0 if m.max() > 1 else 0.5)
    area = int(b.sum())
    if area == 0:
        return {"area": 0, "centroid": None, "bbox": None}
    ys, xs = np.where(b)
    cx, cy = float(xs.mean()), float(ys.mean())
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    res = {"area": area, "centroid": [round(cx, 3), round(cy, 3)], "bbox": bbox}
    # optional cv2 ellipse fit
    try:
        import cv2
        contours, _ = cv2.findContours(b.astype("uint8"), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            if len(c) >= 5:
                (ex, ey), (MA, ma), ang = cv2.fitEllipse(c)
                res["ellipse"] = {"cx": round(ex, 2), "cy": round(ey, 2),
                                  "rx": round(MA / 2, 2), "ry": round(ma / 2, 2),
                                  "theta_deg": round(ang, 2)}
    except Exception:
        pass
    return res

def ellipse_centroid_distance(e1, e2):
    """Euclidean distance between two ellipse centers (cx,cy,...)."""
    import math
    if not e1 or not e2:
        return None
    return math.hypot(e1[0] - e2[0], e1[1] - e2[1])

# ----------------------------------------------------------------------------- io
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)
    return d

def dump_json(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path
