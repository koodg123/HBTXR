"""Aggregate REAL EV-Eye precomputed pixel errors (processed_data/Pixel_error_evaluation)
by motion at SESSION level (pattern: 102/202=smooth, 201=saccade; 101 has no GT/PE).
Both frame-based (UNet) and event-based tracks, all 48 users x both eyes."""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, scipy.io as sio
from common import USERS, EYES, SESSIONS, PE_DIR, REPORTS_DIR

def collect(track):  # track in {"frame","event"}
    rows = []
    for u in USERS:
        for eye in EYES:
            for code in ["102", "201", "202"]:
                folder = SESSIONS[code][0]              # session_X_0_Y
                motion = SESSIONS[code][1]
                f = os.path.join(PE_DIR, track, eye, f"user{u}_{folder}.mat")
                if not os.path.exists(f):
                    continue
                pe = sio.loadmat(f)["matcell"].ravel().astype(float)
                pe = pe[np.isfinite(pe)]
                for v in pe:
                    rows.append((u, eye, code, motion, track, float(v)))
    return pd.DataFrame(rows, columns=["user","eye","code","motion","track","err"])

def summarize(df, track):
    out = []
    for mot in ["saccade","smooth"]:
        s = df[df.motion==mot].err
        sf = s[s<34]
        out.append(dict(track=track, motion=mot, n=len(s),
            mean=round(s.mean(),3), median=round(s.median(),3),
            p95=round(np.percentile(s,95),3),
            mean_filt=round(sf.mean(),3), p95_filt=round(np.percentile(sf,95),3)))
    return pd.DataFrame(out)

def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    allsum=[]
    bigdf=[]
    for track in ["frame","event"]:
        df = collect(track); bigdf.append(df)
        s = summarize(df, track); allsum.append(s)
        print(f"\n=== EV-Eye {track}-based pixel error by motion (REAL, all users) ===")
        print(s.to_string(index=False))
    summ = pd.concat(allsum, ignore_index=True)
    summ.to_csv(os.path.join(REPORTS_DIR,"eveye_real_permotion_session_level.csv"), index=False)
    big = pd.concat(bigdf, ignore_index=True)
    # per-subject medians for rigor
    ps = big.groupby(["track","motion","user"]).err.median().reset_index()
    psg = ps.groupby(["track","motion"]).err.agg(['mean','std','min','max']).round(3)
    print("\n=== per-subject median (mean+-std over users) ===")
    print(psg.to_string())
    big.to_csv(os.path.join(REPORTS_DIR,"eveye_real_pe_allframes.csv"), index=False)
    print(f"\nwrote eveye_real_permotion_session_level.csv  (total PE samples: {len(big)})")

if __name__ == "__main__":
    main()
