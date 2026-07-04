"""Extract REAL EX-Gaze predictions (frame-based img_detect + event-based ev) at
each labeled GT frame from the origin_labelled pickle, then per-motion error.
Session: user48/left/201 (saccade). No torch needed."""
import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from common import load_gt_ellipses, REPORTS_DIR, CACHE_DIR, parse_frame_filename

PK_DIR = ("/sessions/busy-happy-newton/mnt/eveye/codes/EX-Gaze/data/user48/left/"
          "session_2_0_1/events/end_to_end_tracking/"
          "config-eye_crop_mbv3spreX_multi_anchor_det-s1_f2_n4_trans-pre_accum10_50_blink_exp5_0.5_rand")
PKL = os.path.join(PK_DIR, "pre_accum_origin_labelled_pupil_dataset-track_pre_1frames_with_0.8similarity.pickle")
USER, EYE, CODE = 48, "left", 201

def extract():
    obj = pickle.load(open(PKL, "rb"))
    gt = load_gt_ellipses(USER, EYE, str(CODE))
    gt_by_idx = {int(r.frame_idx): r for _, r in gt.iterrows()}
    rows = []
    for entry in obj:
        idx, _ = parse_frame_filename(entry["tracking_end_frame_name"])
        if idx not in gt_by_idx:
            continue
        g = gt_by_idx[idx]; gtt = int(g.ts)
        tr = entry["tracking_result"]
        frame_pred = None; ev_best = None; ev_dt = 1e18
        for e in tr:
            pup = e.get("pupil")
            if pup is None or pup[0] is None:
                continue
            if e["method"] == "img_detect":
                frame_pred = (float(pup[0]), float(pup[1]))
            if e["method"] in ("ev_tracking_model", "ev_accum_threshold"):
                t = e.get("result_timestamp", e.get("accum_end_timestamp"))
                if t is not None and abs(int(t) - gtt) < ev_dt:
                    ev_dt = abs(int(t) - gtt); ev_best = (float(pup[0]), float(pup[1]))
        rows.append(dict(frame_idx=idx, gt_cx=g.cx, gt_cy=g.cy,
                         frame_x=frame_pred[0] if frame_pred else np.nan,
                         frame_y=frame_pred[1] if frame_pred else np.nan,
                         ev_x=ev_best[0] if ev_best else np.nan,
                         ev_y=ev_best[1] if ev_best else np.nan))
    return pd.DataFrame(rows)

def main():
    df = extract()
    vel = pd.read_csv(os.path.join(CACHE_DIR, "velocity_at_gt.csv"))
    vsub = vel[(vel.user==USER)&(vel.eye==EYE)&(vel.code==CODE)][["frame_idx","speed_pxps"]]
    df = df.merge(vsub, on="frame_idx", how="left")
    sacc = vel[vel.motion=="saccade"].speed_pxps.dropna()
    v_fix=np.percentile(sacc,60); v_sacc=np.percentile(sacc,90)
    df["ivt"]=np.where(df.speed_pxps<v_fix,"fixation",np.where(df.speed_pxps>v_sacc,"saccade","smooth"))
    df["err_frame"]=np.hypot(df.frame_x-df.gt_cx, df.frame_y-df.gt_cy)
    df["err_event"]=np.hypot(df.ev_x-df.gt_cx, df.ev_y-df.gt_cy)
    df["err_cv2"]=np.nan
    # cv2 baseline same session from cache
    cv = vel[(vel.user==USER)&(vel.eye==EYE)&(vel.code==CODE)][["frame_idx","det_x","det_y","gt_cx","gt_cy"]].copy()
    cv["err_cv2"]=np.hypot(cv.det_x-cv.gt_cx, cv.det_y-cv.gt_cy)
    df = df.merge(cv[["frame_idx","err_cv2"]], on="frame_idx", how="left", suffixes=("","_c"))
    df["err_cv2"]=df["err_cv2_c"]; df.drop(columns=["err_cv2_c"], inplace=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(REPORTS_DIR,"exgaze_user48_201_per_frame.csv"), index=False)

    def stats(s):
        s=s.dropna()
        return f"n={len(s):2d} mean={s.mean():6.2f} median={s.median():6.2f} p95={np.percentile(s,95):6.2f}" if len(s) else "n=0"
    print("=== user48/left/201 (saccade session, 37 GT frames) — REAL EX-Gaze model ===")
    for col,lab in [("err_frame","EX-Gaze frame-based"),("err_event","EX-Gaze event-based"),("err_cv2","cv2 baseline")]:
        print(f"  {lab:22s}: {stats(df[col])}")
    print("\n--- by I-VT motion class (median px error) ---")
    tab = df.groupby("ivt").agg(n=("frame_idx","count"),
        frame=("err_frame","median"), event=("err_event","median"), cv2=("err_cv2","median")).round(2)
    print(tab.reindex(["fixation","saccade","smooth"]).to_string())
    tab.to_csv(os.path.join(REPORTS_DIR,"error_exgaze_user48_201_by_ivt.csv"))

if __name__ == "__main__":
    main()
