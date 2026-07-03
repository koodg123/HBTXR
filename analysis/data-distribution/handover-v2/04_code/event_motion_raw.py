import scipy.io as sio, numpy as np, pandas as pd, os, time, sys
EVR="/sessions/busy-happy-newton/mnt/eveye"
DAV=EVR+"/raw_data/Data_davis"; FE=EVR+"/processed_data/Frame_event_pupil_track_result"
DD="/sessions/busy-happy-newton/mnt/DATASET/codes/data-distribution"
CACHE=DD+"/cache/event_motion_counts.csv"
SESS={101:"session_1_0_1",102:"session_1_0_2",201:"session_2_0_1",202:"session_2_0_2"}
SACC_SESS={101,201}; VSACC=493.0; BUDGET=32
done=set()
if os.path.exists(CACHE):
    for ln in open(CACHE).read().splitlines()[1:]:
        p=ln.split(","); done.add((int(p[0]),p[1],int(p[2])))
else:
    open(CACHE,"w").write("user,eye,code,Fixation,Saccade,Smooth,Blink,total\n")
def one(u,eye,code):
    mf=f"{FE}/{eye}/update_20_point_user{u}_{SESS[code]}.mat"
    sp=f"{DAV}/user{u}/{eye}/{SESS[code]}"
    if not (os.path.exists(mf) and os.path.exists(sp+"/events/events.txt")): return None
    M=np.array(sio.loadmat(mf)['matcell'],dtype=float); M=M[np.argsort(M[:,2])]; t=M[:,2]; flag=M[:,5]
    xs=pd.Series(M[:,3]).rolling(9,center=True,min_periods=1).median().values
    ys=pd.Series(M[:,4]).rolling(9,center=True,min_periods=1).median().values
    g=np.arange(t.min(),t.max(),5000.0); j=np.clip(np.searchsorted(t,g),0,len(t)-1)
    spd=np.hypot(np.diff(xs[j]),np.diff(ys[j]))/(5000/1e6); sacc=np.concatenate([[False],spd>VSACC])
    fa=np.sort(t[flag==1]); gp=np.where(np.diff(fa)>60000)[0]; blink_iv=[(fa[i],fa[i+1]) for i in gp]
    est=int(open(sp+"/events/event_startime.txt").read().split()[0])
    ev=pd.read_csv(sp+"/events/events.txt",sep=r'\s+',header=None,usecols=[0]).values[:,0].astype(np.int64)
    rel=ev-est
    gi=np.clip(np.searchsorted(g,rel),0,len(sacc)-1); is_s=sacc[gi]
    is_b=np.zeros(len(rel),bool)
    for a,b in blink_iv: is_b|=(rel>=a)&(rel<b)
    n=len(ev); nb=int(is_b.sum()); ns=int((is_s&~is_b).sum()); rem=n-nb-ns
    fix=rem if code in SACC_SESS else 0; smo=0 if code in SACC_SESS else rem
    return dict(user=u,eye=eye,code=code,Fixation=fix,Saccade=ns,Smooth=smo,Blink=nb,total=n)
t0=time.time(); cnt=0
for u in range(1,49):
    for eye in ["left","right"]:
        for code in [101,102,201,202]:
            if (u,eye,code) in done: continue
            r=one(u,eye,code)
            if r: open(CACHE,"a").write(f"{u},{eye},{code},{r['Fixation']},{r['Saccade']},{r['Smooth']},{r['Blink']},{r['total']}\n"); cnt+=1
            if time.time()-t0>BUDGET: print(f"chunk done +{cnt}, elapsed {time.time()-t0:.0f}s"); sys.exit()
print(f"ALL DONE this run +{cnt}")
