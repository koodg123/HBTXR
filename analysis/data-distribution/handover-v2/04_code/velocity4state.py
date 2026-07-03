"""Mutually-exclusive 4-state per subject, BEFORE(40ms) vs AFTER(5ms) augmentation.
Same method, only temporal resolution differs (=augmentation effect):
  Blink   = UNet no-pupil frames (progress_state skip_no_ellipse) -- exact, fraction same before/after
  Saccade = pupil-present samples with event-track speed>493 px/s at the grid resolution
  Fixation= non-saccade pupil-present in saccade-session(101/201)
  Smooth  = non-saccade pupil-present in smooth-session(102/202)
Before counts @25Hz(40ms); After counts @200Hz(5ms, x8). Resumable."""
import os, sys, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, scipy.io as sio
from multiprocessing import Pool
EVR=os.environ.get("EVEYE_ROOT","/sessions/busy-happy-newton/mnt/eveye")
FE=os.path.join(EVR,"processed_data","Frame_event_pupil_track_result")
MT=os.environ.get("MT_CODES","/sessions/busy-happy-newton/mnt/DATASET/codes")
DD=os.path.join(MT,"data-distribution"); CACHE=os.path.join(DD,"cache","vel4state_sessions.csv")
F2C={"session_1_0_1":"101","session_1_0_2":"102","session_2_0_1":"201","session_2_0_2":"202"}
SESS={"101":"session_1_0_1","102":"session_1_0_2","201":"session_2_0_1","202":"session_2_0_2"}
MOT={"101":"saccade","102":"smooth","201":"saccade","202":"smooth"}
VSACC=493.0
# progress_state blink/total
PS=json.load(open(os.path.join(EVR,"DeanDataset_full_unet","progress_state.json")))
PT={}
for s in PS["session_summaries"]:
    p=s["session"].replace("\\","/").split("/"); u=int([q for q in p if q.startswith("user")][-1][4:]); eye=p[-2]; folder=p[-1]
    PT[(u,eye,F2C[folder])]=(s["frames"], s.get("skip_no_ellipse",0))

def frac_sacc(t,x,y,xs,ys,win_us):
    grid=np.arange(t.min(),t.max(),win_us)
    if len(grid)<3: return np.nan
    j=np.clip(np.searchsorted(t,grid),0,len(t)-1)
    gx=xs[j]; gy=ys[j]
    sp=np.hypot(np.diff(gx),np.diff(gy))/(win_us/1e6)
    sp=sp[np.isfinite(sp)]
    return float((sp>VSACC).mean()) if len(sp) else np.nan

def one(args):
    u,eye,code=args; folder=SESS[code]
    f=os.path.join(FE,eye,f"update_20_point_user{u}_{folder}.mat")
    if not os.path.exists(f) or (u,eye,code) not in PT: return None
    m=sio.loadmat(f)["matcell"]; t=m[:,2].astype(float); x=m[:,3].astype(float); y=m[:,4].astype(float)
    o=np.argsort(t); t,x,y=t[o],x[o],y[o]
    xs=pd.Series(x).rolling(9,center=True,min_periods=1).median().values
    ys=pd.Series(y).rolling(9,center=True,min_periods=1).median().values
    total,blink=PT[(u,eye,code)]
    return dict(user=u,eye=eye,code=int(code),motion=MOT[code],total=total,blink=blink,
                valid=total-blink, fsacc40=frac_sacc(t,x,y,xs,ys,40000), fsacc5=frac_sacc(t,x,y,xs,ys,5000))

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
        pd.DataFrame(rows).to_csv(CACHE,index=False); print(f"+{nd}, remaining={len(todo)-nd}")
        if len(todo)-nd>0: return
    print("DONE", len(rows))

if __name__=="__main__": main()
