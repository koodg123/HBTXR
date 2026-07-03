"""Frame-aligned EVENT-track center error per GT frame, via Frame_event_pupil_track
matcell (col2=rel time us, col3,4 = pupil x,y). Fast single pass. Cache."""
import os, sys, glob; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, scipy.io as sio
from common import USERS, EYES, SESSIONS, CACHE_DIR, PROCESSED_DIR, load_gt_ellipses, list_frames, parse_frame_filename
FE = os.path.join(PROCESSED_DIR, "Frame_event_pupil_track_result")

def session_start_ts(u, eye, code):
    fr = list_frames(u, eye, code)
    return parse_frame_filename(fr[0])[1] if fr else None

def build(budget=38):
    import time; t0=time.time()
    out=os.path.join(CACHE_DIR,"event_aligned_err.csv")
    done=set(); rows=[]
    if os.path.exists(out):
        prev=pd.read_csv(out); rows=[prev]; done=set(zip(prev.user,prev.eye,prev.code.astype(str)))
    todo=[(u,e,c) for u in USERS for e in EYES for c in ["102","201","202"] if (u,e,str(c)) not in done]
    newrows=[]
    for (u,eye,code) in todo:
        if time.time()-t0>budget: break
        f = os.path.join(FE, eye, f"update_20_point_user{u}_{SESSIONS[code][0]}.mat")
        if not os.path.exists(f): continue
        gt = load_gt_ellipses(u, eye, code)
        if len(gt) == 0: continue
        st = session_start_ts(u, eye, code)
        if st is None: continue
        m = sio.loadmat(f)["matcell"]
        rel = m[:,2]; xs=m[:,3]; ys=m[:,4]; ok=m[:,5]>0
        rel,xs,ys=rel[ok],xs[ok],ys[ok]
        if len(rel)==0: continue
        o=np.argsort(rel); rel,xs,ys=rel[o],xs[o],ys[o]
        for _, r in gt.iterrows():
            target=int(r.ts)-st; j=int(np.searchsorted(rel,target)); j=min(max(j,0),len(rel)-1)
            if j>0 and abs(rel[j-1]-target)<abs(rel[j]-target): j-=1
            newrows.append(dict(user=u,eye=eye,code=int(code),frame_idx=int(r.frame_idx),track="event",
                pred_x=float(xs[j]),pred_y=float(ys[j]),dt_ms=round(abs(rel[j]-target)/1e3,1),
                err=round(float(np.hypot(xs[j]-r.cx,ys[j]-r.cy)),3)))
    if newrows: rows.append(pd.DataFrame(newrows))
    df=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()
    df.to_csv(out,index=False)
    return df, len(todo)-sum(1 for _ in []) , len(todo)

if __name__ == "__main__":
    df,_,ntodo = build()
    done = set(zip(df.user,df.eye,df.code.astype(str))) if len(df) else set()
    print(f"event-aligned rows={len(df)} sessions_done={len(done)}/288 median_err={df.err.median():.3f}px median_dt={df.dt_ms.median():.1f}ms")
