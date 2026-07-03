"""Shared loaders + stats for EV-Eye per-motion error analysis (sections 1-5).
Authoritative center-distance error = processed_data/Pixel_error_evaluation/*.mat
(matlab Euclidean px error vs manual GT). Tracks: frame (UNet on real frame), event.
"""
import os, glob, numpy as np, pandas as pd, scipy.io as sio
from common import USERS, EYES, SESSIONS, PE_DIR, CACHE_DIR

MOTION_OF = {"102": "smooth", "201": "saccade", "202": "smooth"}
GT_QUANT_FLOOR_MEAN = 0.383   # perfect predictor vs integer GT (MC, see section 5)

def load_pe_allframes(force=False):
    """DataFrame: user,eye,code,motion,track,err  (one row per evaluated GT frame)."""
    cache = os.path.join(CACHE_DIR, "pe_allframes.csv")
    if os.path.exists(cache) and not force:
        return pd.read_csv(cache)
    rows = []
    for track in ["frame", "event"]:
        for u in USERS:
            for eye in EYES:
                for code in ["102", "201", "202"]:
                    f = os.path.join(PE_DIR, track, eye, f"user{u}_{SESSIONS[code][0]}.mat")
                    if not os.path.exists(f):
                        continue
                    pe = sio.loadmat(f)["matcell"].ravel().astype(float)
                    pe = pe[np.isfinite(pe)]
                    for v in pe:
                        rows.append((u, eye, code, MOTION_OF[code], track, float(v)))
    df = pd.DataFrame(rows, columns=["user", "eye", "code", "motion", "track", "err"])
    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_csv(cache, index=False)
    return df

def describe(s):
    s = np.asarray(s, float); s = s[np.isfinite(s)]
    if len(s) == 0:
        return dict(n=0, mean=np.nan, median=np.nan, p95=np.nan, p99=np.nan, max=np.nan)
    return dict(n=int(len(s)), mean=round(float(s.mean()), 3), median=round(float(np.median(s)), 3),
                p95=round(float(np.percentile(s, 95)), 3), p99=round(float(np.percentile(s, 99)), 3),
                max=round(float(s.max()), 3))

def boot_ci(s, stat=np.mean, n_boot=2000, ci=95, seed=0):
    s = np.asarray(s, float); s = s[np.isfinite(s)]
    if len(s) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    bs = [stat(rng.choice(s, len(s), replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(bs, [(100-ci)/2, 100-(100-ci)/2])
    return (round(float(lo), 3), round(float(hi), 3))

def subject_level_ci(per_subject_vals, ci=95):
    """CI of the across-subject mean (t-based), more conservative than frame bootstrap."""
    v = np.asarray(per_subject_vals, float); v = v[np.isfinite(v)]
    if len(v) < 2:
        return (np.nan, np.nan)
    from scipy import stats
    m, se = v.mean(), v.std(ddof=1)/np.sqrt(len(v))
    t = stats.t.ppf(1-(1-ci/100)/2, len(v)-1)
    return (round(float(m-t*se), 3), round(float(m+t*se), 3))
