"""Per-subject velocity recomputed on 100ms windows (RAW, before aug), from event-track
matcell: resample pupil position to a 100ms grid, speed = displacement/0.1s.
Output per-subject histogram over velocity bins (columns) + percentiles. Resumable."""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, scipy.io as sio
from multiprocessing import Pool
EVR=os.environ.get("EVEYE_ROOT","/sessions/busy-happy-newton/mnt/eveye")
FE=os.path.join(EVR,"processed_data","Frame_event_pupil_track_result")
MT=os.environ.get("MT_CODES","/sessions/busy-happy-newton/mnt/DATASET/codes")
DD=os.path.join(MT,"data-distribution"); CACHE=os.path.join(DD,"cache","velocity_100ms_hist.csv")
SESS={"101":"session_1_0_1","102":"session_1_0_2","201":"session_2_0_1","202":"session_2_0_2"}
BINS=[0,50,100,200,300,500,800,1200,2000,1e9]
LAB=["0-50","50-100","100-200","200-300","300-500","500-800","800-1200","1200-2000",">2000"]
WIN_US=100000

def one(args):
    u,eye,code=args; folder=SESS[code]
    f=os.path.join(FE,eye,f"update_20_point_user{u}_{folder}.mat")
    if not os.path.exists(f): return None
    m=sio.loadmat(f)["matcell"]; t=m[:,2].astype(float); x=m[:,3].astype(float); y=m[:,4].astype(float)
    o=np.argsort(t); t,x,y=t[o],x[o],y[o]
    grid=np.arange(t.min(),t.max(),WIN_US)
    if len(grid)<3: return None
    j=np.searchsorted(t,grid); j=np.clip(j,0,len(t)-1)
    gx=x[j]; gy=y[j]
    sp=np.hypot(np.diff(gx),np.diff(gy))/(WIN_US/1e6)   # px/s per 100ms window
    sp=sp[np.isfinite(sp)]
    h,_=np.histogram(sp,bins=BINS)
    rec={"user":u,"eye":eye,"code":int(code),"n_win":len(sp)}
    for lab,c in zip(LAB,h): rec[lab]=int(c)
    # store a few percentiles too (pooled later via samples not kept; approx from hist omitted)
    rec["sum_sp"]=float(sp.sum()); rec["sumsq"]=float((sp**2).sum())
    rec["p50_sess"]=float(np.median(sp)); rec["p90_sess"]=float(np.percentile(sp,90))
    return rec

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--budget",type=float,default=38); a=ap.parse_args()
    os.makedirs(os.path.dirname(CACHE),exist_ok=True)
    done=set(); rows=[]
    if os.path.exists(CACHE):
        prev=pd.read_csv(CACHE); rows=prev.to_dict('records'); done=set(zip(prev.user,prev.eye,prev.code.astype(str)))
    todo=[(u,e,c) for u in range(1,49) for e in ["left","right"] for c in SESS if (u,e,str(c)) not in done]
    print(f"done={len(done)} todo={len(todo)}")
    if todo:
        t0=time.time(); nd=0
        with Pool(2) as pool:
            for r in pool.imap_unordered(one,todo):
                if r: rows.append(r); nd+=1
                if nd%10==0 or time.time()-t0>a.budget: pd.DataFrame(rows).to_csv(CACHE,index=False)
                if time.time()-t0>a.budget: break
        pd.DataFrame(rows).to_csv(CACHE,index=False)
        print(f"+{nd}, remaining={len(todo)-nd}")
        if len(todo)-nd>0: return
    print("DONE all sessions; rows", len(rows))

if __name__=="__main__": main()
