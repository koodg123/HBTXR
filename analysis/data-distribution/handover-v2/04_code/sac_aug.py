"""Saccade-region augmentation footprint (event-guided, NOT TimeLens):
- detect saccades from event-track velocity (>493 px/s runs >=10ms)
- window = saccade +-5 frames (+-200ms), merged
- frame interpolation 40ms->5ms (8x) within windows: orig vs interp frame counts
- event densification: real events in those windows (synthetic-event basis), per 5ms frame
Per session -> per subject. Parallel(2), resumable."""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, scipy.io as sio
from multiprocessing import Pool
EVR=os.environ.get("EVEYE_ROOT","/sessions/busy-happy-newton/mnt/eveye")
FE=os.path.join(EVR,"processed_data","Frame_event_pupil_track_result")
RAW=os.path.join(EVR,"raw_data","Data_davis")
MT=os.environ.get("MT_CODES","/sessions/busy-happy-newton/mnt/DATASET/codes")
DD=os.path.join(MT,"data-distribution"); CACHE=os.path.join(DD,"cache","sac_aug_sessions.csv")
SESS={"101":"session_1_0_1","102":"session_1_0_2","201":"session_2_0_1","202":"session_2_0_2"}
VSACC=493.0; HALF=200000; MINDUR=10000; FR=40000; INTERP=5000  # us

def merge(iv):
    if not iv: return []
    iv=sorted(iv); out=[list(iv[0])]
    for a,b in iv[1:]:
        if a<=out[-1][1]: out[-1][1]=max(out[-1][1],b)
        else: out.append([a,b])
    return out

def one(args):
    u,eye,code=args; folder=SESS[code]
    f=os.path.join(FE,eye,f"update_20_point_user{u}_{folder}.mat")
    ef=os.path.join(RAW,f"user{u}",eye,folder,"events","events.txt")
    if not os.path.exists(f): return None
    m=sio.loadmat(f)["matcell"]; t=m[:,2].astype(float); x=m[:,3].astype(float); y=m[:,4].astype(float)
    o=np.argsort(t); t,x,y=t[o],x[o],y[o]
    xs=pd.Series(x).rolling(9,center=True,min_periods=1).median().values
    ys=pd.Series(y).rolling(9,center=True,min_periods=1).median().values
    sp=np.full(len(t),0.0)
    dt=(t[2:]-t[:-2]); ok=dt>0
    sp[1:-1][ok]=np.hypot(xs[2:]-xs[:-2],ys[2:]-ys[:-2])[ok]/(dt[ok]/1e6)
    above=sp>VSACC
    # runs of 'above' lasting >= MINDUR
    idx=np.where(np.diff(np.concatenate([[0],above.view(np.int8),[0]]))!=0)[0]
    runs=idx.reshape(-1,2); nsac=0; wins=[]
    for a,b in runs:
        if b>=len(t): b=len(t)-1
        if t[min(b,len(t)-1)]-t[a] >= MINDUR:
            nsac+=1; c=(t[a]+t[min(b,len(t)-1)])/2
            wins.append((c-HALF,c+HALF))
    wins=merge(wins)
    windur=sum(b-a for a,b in wins)/1e6
    orig=int(round(windur/ (FR/1e6))); interp=int(round(windur/(INTERP/1e6)))
    # events in windows
    ev_in=0
    if os.path.exists(ef) and wins:
        et=pd.read_csv(ef,sep=r'\s+',header=None,usecols=[0],names=['t'],dtype=np.int64)['t'].values; et.sort()
        for a,b in wins: ev_in+=int(np.searchsorted(et,b)-np.searchsorted(et,a))
    return dict(user=u,eye=eye,code=int(code),n_saccade=nsac,win_dur_s=round(windur,2),
                orig_frames_40ms=orig, interp_frames_5ms=interp, added_frames=interp-orig,
                events_in_win=ev_in)

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
                if nd%8==0 or time.time()-t0>a.budget: pd.DataFrame(rows).to_csv(CACHE,index=False)
                if time.time()-t0>a.budget: break
        pd.DataFrame(rows).to_csv(CACHE,index=False)
        print(f"+{nd}, remaining={len(todo)-nd}")
        if len(todo)-nd>0: return
    print("DONE", len(rows))

if __name__=="__main__": main()
