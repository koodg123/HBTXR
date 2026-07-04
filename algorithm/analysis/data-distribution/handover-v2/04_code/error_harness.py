"""Per-motion ERROR aggregation harness.

Takes a predictions source with columns [user,eye,code,frame_idx,pred_x,pred_y]
plus the motion labels (speed) cached in velocity_at_gt.csv (which also carries
GT cx/cy and the cv2-detector center). Produces per-motion error tables:
  (A) session-level  : saccade-session vs smooth-session
  (B) I-VT           : fixation / saccade / smooth
  (C) velocity bins  : low / med / high
Each with mean / median / p95 / n, raw and EV-Eye-style filtered (<34 px).

Default predictor = the cv2 classical detector already in the cache (a CLASSICAL
FRAME-BASED BASELINE -- not EX-Gaze/EV-Eye trained models).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from common import CACHE_DIR, REPORTS_DIR

FILTER_PX = 34.0  # EV-Eye analysis.py convention

def load_labeled(pred_df=None, method_name="cv2_classical"):
    df = pd.read_csv(os.path.join(CACHE_DIR, "velocity_at_gt.csv"))
    df = df[df.speed_pxps.notna()].copy()
    if pred_df is None:
        df["pred_x"], df["pred_y"] = df.det_x, df.det_y
    else:
        df = df.merge(pred_df, on=["user","eye","code","frame_idx"], how="inner")
    df["method"] = method_name
    df = df[df.pred_x.notna()].copy()
    df["err"] = np.hypot(df.pred_x - df.gt_cx, df.pred_y - df.gt_cy)
    # motion labels
    sacc = df[df.motion=="saccade"].speed_pxps
    v_fix = np.percentile(sacc, 60); v_sacc = np.percentile(sacc, 90)
    df["ivt"] = np.where(df.speed_pxps<v_fix,"fixation",
                  np.where(df.speed_pxps>v_sacc,"saccade","smooth"))
    edges = np.percentile(df.speed_pxps,[33.3,66.6])
    df["vbin"] = np.where(df.speed_pxps<edges[0],"low",
                  np.where(df.speed_pxps<edges[1],"med","high"))
    return df, dict(v_fix=v_fix, v_sacc=v_sacc, vbin_edges=edges.tolist())

def agg(df, by, order):
    rows=[]
    for key in order:
        s = df[df[by]==key].err
        sf = s[s<FILTER_PX]
        if len(s)==0: continue
        rows.append(dict(group=key, n=len(s),
            mean=round(s.mean(),2), median=round(s.median(),2),
            p95=round(np.percentile(s,95),2),
            n_filt=len(sf), mean_filt=round(sf.mean(),2) if len(sf) else np.nan,
            median_filt=round(sf.median(),2) if len(sf) else np.nan,
            p95_filt=round(np.percentile(sf,95),2) if len(sf) else np.nan))
    return pd.DataFrame(rows)

def run(pred_df=None, method_name="cv2_classical"):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    df, thr = load_labeled(pred_df, method_name)
    print(f"method={method_name}  GT rows={len(df)}  users={sorted(df.user.unique())}")
    print(f"thresholds: {thr}")
    tA = agg(df, "motion", ["saccade","smooth"])
    tB = agg(df, "ivt", ["fixation","saccade","smooth"])
    tC = agg(df, "vbin", ["low","med","high"])
    print("\n(A) session-level motion (px error):\n", tA.to_string(index=False))
    print("\n(B) I-VT per-frame motion (px error):\n", tB.to_string(index=False))
    print("\n(C) velocity bin (px error):\n", tC.to_string(index=False))
    for name,t in [("A_session",tA),("B_ivt",tB),("C_vbin",tC)]:
        t.insert(0,"method",method_name)
        t.to_csv(os.path.join(REPORTS_DIR, f"error_{name}_{method_name}.csv"), index=False)
    return df, tA, tB, tC

if __name__ == "__main__":
    run()
