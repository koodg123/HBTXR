"""Common constants & helpers for the EV-Eye per-motion-type analysis.

Mapping (confirmed from EX-Gaze/data/data_segments.xlsx + ev_eye_dataset_utils.py):
    folder          code  pattern  motion
    session_1_0_1   101   p1       saccade+fixation  (NO manual GT)
    session_1_0_2   102   p2       smooth pursuit
    session_2_0_1   201   p1       saccade+fixation
    session_2_0_2   202   p2       smooth pursuit
=> pattern 1 = saccade, pattern 2 = smooth.

GT annotations (user_*.csv, VGG ellipse cx,cy,rx,ry,theta) are SPARSE:
~27-49 frames/session, only in 102/201/202. Frame stream ~25 Hz (40 ms).
"""
import os, glob, json, re
import numpy as np
import pandas as pd

EVEYE_ROOT = os.environ.get("EVEYE_ROOT", "/sessions/busy-happy-newton/mnt/eveye")
DATA_DAVIS = os.path.join(EVEYE_ROOT, "raw_data", "Data_davis")
DATA_MASK  = os.path.join(EVEYE_ROOT, "raw_data", "Data_davis_labelled_with_mask")
CODES      = os.environ.get("MT_CODES", os.path.join(EVEYE_ROOT, "codes"))
MOTION_DIR = os.path.join(CODES, "motion_type")
CACHE_DIR  = os.path.join(MOTION_DIR, "cache")
REPORTS_DIR= os.path.join(MOTION_DIR, "reports")
def _find_segments():
    import glob as _g
    cands=[os.environ.get("MT_SEGMENTS",""),
           os.path.join(CODES,"EX-Gaze","data","data_segments.xlsx"),
           os.path.join(CODES,"codebase","EX-Gaze-main","data","data_segments.xlsx")]
    for c in cands:
        if c and os.path.exists(c): return c
    return cands[1]
DATA_SEGMENTS_XLSX = _find_segments()
PROCESSED_DIR = os.path.join(EVEYE_ROOT, "processed_data")
PE_DIR = os.path.join(PROCESSED_DIR, "Pixel_error_evaluation")   # frame|event / left|right
EVEYE_PRETRAINED = os.path.join(PROCESSED_DIR, "Pre-trained_models")  # left|right/user{N}.pth

IMG_SHAPE = (260, 346)  # H, W

SESSIONS = {
    "101": ("session_1_0_1", "saccade", 1),
    "102": ("session_1_0_2", "smooth",  2),
    "201": ("session_2_0_1", "saccade", 1),
    "202": ("session_2_0_2", "smooth",  2),
}
EYES = ["left", "right"]
USERS = list(range(1, 49))

SEG_COLS = {
    "101": {"sacc_start": 1,  "sacc_end": 2,  "blink": 3},
    "102": {"smooth_start": 4,  "smooth_end": 5,  "blink": 6,  "dir": 7},
    "201": {"sacc_start": 8,  "sacc_end": 9,  "blink": 10},
    "202": {"smooth_start": 11, "smooth_end": 12, "blink": 13, "dir": 14, "count": 15},
}


def parse_frame_filename(fn):
    base = os.path.basename(fn)
    base = base[:-4] if base.endswith(".png") else base
    parts = base.split("_")
    return int(parts[0]), int(parts[-1])


def session_dir(user, eye, code):
    return os.path.join(DATA_DAVIS, f"user{user}", eye, SESSIONS[code][0])


def list_frames(user, eye, code):
    d = os.path.join(session_dir(user, eye, code), "frames")
    return sorted(glob.glob(os.path.join(d, "*.png")))


def load_gt_ellipses(user, eye, code):
    """DataFrame of annotated frames: frame_idx, ts, cx, cy, rx, ry, theta, filename."""
    d = session_dir(user, eye, code)
    csvs = glob.glob(os.path.join(d, "user_*.csv"))
    cols = ["frame_idx", "ts", "cx", "cy", "rx", "ry", "theta", "filename"]
    if not csvs:
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(csvs[0])
    rows = []
    for _, r in df.iterrows():
        if int(r["region_count"]) <= 0:
            continue
        base = str(r["filename"])
        base = base[:-4] if base.endswith(".png") else base
        if not re.match(r"^\d+_\d+$", base):
            continue  # skip non-standard filenames (e.g. placeholder rows)
        try:
            shape = json.loads(r["region_shape_attributes"])
        except Exception:
            continue
        if shape.get("name") != "ellipse":
            continue
        idx, ts = parse_frame_filename(r["filename"])
        rows.append(dict(frame_idx=idx, ts=ts, cx=float(shape["cx"]), cy=float(shape["cy"]),
                         rx=float(shape.get("rx", np.nan)), ry=float(shape.get("ry", np.nan)),
                         theta=float(shape.get("theta", np.nan)), filename=r["filename"]))
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values("frame_idx").reset_index(drop=True)


def load_data_segments():
    """dict[user]->dict[code]->dict(fields). Frame-index windows (single repr. events)."""
    df = pd.read_excel(DATA_SEGMENTS_XLSX, header=None)
    out = {}
    for u in USERS:
        row = df.iloc[u + 2]  # user1 -> iloc[3]
        urec = {}
        for code, cols in SEG_COLS.items():
            rec = {}
            for field, ci in cols.items():
                rec[field] = row.iloc[ci]
            urec[code] = rec
        out[u] = urec
    return out


def euclidean(ax, ay, bx, by):
    return float(np.sqrt((ax - bx) ** 2 + (ay - by) ** 2))
